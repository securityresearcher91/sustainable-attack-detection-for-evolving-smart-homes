#!/usr/bin/env python3
import os
import time
import json
import socket
import argparse
import signal
import sys
import threading
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, Ether
from scapy.layers.inet import TCP, UDP

import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.copod import COPOD
from river.anomaly import HalfSpaceTrees

# ================= Configuration =================

FEATURES = [
    'flow_duration', 'flow_byts_s', 'flow_pkts_s',
    'pkt_len', 'pkt_size', 'iat',
    'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
]

MODEL_DIR = "models_rpi"
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = "anomalies_inference_rpi.log"
METRICS_CSV = "metrics_rpi.csv"
MISSED_ANOMALY_LOG = "missed_anomalies_rpi.log"
MISCLASSIFIED_BENIGN_LOG = "misclassified_benigns_rpi.log"

BATCH_INTERVAL = 30
MIN_FLOW_DURATION = 0.1
LEARNING_WINDOW_DEFAULT = 300  # 5 minutes learning phase

# Detection thresholds (updated from supervised thresholds 2025-09-21T09:37:57.044751)
HST_THRESHOLD = 0
OCSVM_THRESHOLD = -0.90
LOF_THRESHOLD = -16.121545
COPOD_THRESHOLD = 18.400942

# Network functionality protocols that should always be allowed
# DHCP (UDP 67/68), DNS (UDP 53), ARP is handled separately
ALLOWED_NETWORK_PROTOCOLS = {
    67,   # DHCP server
    68,   # DHCP client
    53,   # DNS
}

class AbuseIPDBDenyList:
    """
    Handles IP reputation checking using a local AbuseIPDB deny list CSV file.
    CSV format: ip,country_code,abuse_confidence_score,last_reported_at
    
    Logic:
    - If IP is in deny list: Flag as malicious (confirmed anomaly)
    - If IP is private/local: Flag as suspicious (confirmed anomaly)
    - If IP is public but not in deny list: Don't flag (likely false positive)
    - Network functionality traffic (DHCP, DNS to router): Always whitelist
    """
    
    def __init__(self, denylist_file=None, router_ip=None):
        self.denylist_file = denylist_file
        self.router_ip = router_ip
        self.enabled = bool(denylist_file)
        self.deny_list = {}  # {ip: {'country_code': str, 'score': int, 'last_reported': str}}
        
        if self.enabled:
            self.load_denylist()
        else:
            print("AbuseIPDB Deny List: DISABLED (no file specified)")
    
    def load_denylist(self):
        """Load AbuseIPDB deny list from CSV file"""
        try:
            if not os.path.exists(self.denylist_file):
                print(f"⚠️  Deny list file not found: {self.denylist_file}")
                self.enabled = False
                return
            
            import csv
            with open(self.denylist_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ip = row['ip'].strip()
                    self.deny_list[ip] = {
                        'country_code': row.get('country_code', '').strip(),
                        'score': int(row.get('abuse_confidence_score', 0)),
                        'last_reported': row.get('last_reported_at', '').strip()
                    }
            
            print(f"AbuseIPDB Deny List: ENABLED")
            print(f"  File: {self.denylist_file}")
            print(f"  Loaded {len(self.deny_list)} malicious IPs")
            if self.router_ip:
                print(f"  Router IP: {self.router_ip} (whitelisted for network functionality)")
        
        except Exception as e:
            print(f"⚠️  Error loading deny list: {e}")
            self.enabled = False
            self.deny_list = {}
    
    def is_network_functionality(self, src_ip, dst_ip, src_port, dst_port):
        """
        Check if traffic is network functionality (DHCP, DNS to router).
        These should always be allowed regardless of anomaly detection.
        
        IMPORTANT: Be very specific to avoid whitelisting attack traffic!
        - DHCP: Only whitelist if BOTH IPs are in expected ranges
        - DNS: Only whitelist if destination is the router
        """
        # DHCP traffic (ports 67/68)
        # Only whitelist if it's actual DHCP communication (server:67, client:68)
        if (src_port == 67 and dst_port == 68) or (src_port == 68 and dst_port == 67):
            return True, "DHCP"
        
        # DNS to router (port 53)
        # Only whitelist if destination is router and port is 53
        # Do NOT whitelist if source port is 53 (that could be spoofed)
        if self.router_ip and dst_ip == self.router_ip and dst_port == 53:
            return True, "DNS_to_router"
        
        return False, None
    
    def check_ip(self, src_ip, dst_ip, src_port, dst_port):
        """
        Check IP against deny list and private IP ranges.
        
        IMPORTANT: We need to identify the REMOTE IP (non-camera IP) and check that.
        - If camera is source: check destination IP (remote endpoint)
        - If camera is destination: check source IP (remote endpoint)
        
        Returns dict with:
        - in_denylist: bool (IP is in AbuseIPDB deny list)
        - is_private: bool (Remote IP is in private/local range)
        - is_network_function: bool (DHCP, DNS to router)
        - should_flag: bool (True if anomaly should be confirmed)
        - reason: str (explanation)
        - score: int (abuse confidence score if in deny list)
        """
        result = {
            'in_denylist': False,
            'is_private': False,
            'is_network_function': False,
            'should_flag': False,
            'reason': 'unknown',
            'score': 0,
            'country_code': '',
            'remote_ip': None
        }
        
        # Check if this is network functionality traffic (always whitelist)
        is_net_func, func_type = self.is_network_functionality(src_ip, dst_ip, src_port, dst_port)
        if is_net_func:
            result['is_network_function'] = True
            result['should_flag'] = False
            result['reason'] = f'network_functionality_{func_type}'
            return result
        
        # Determine remote IP (the endpoint the camera is communicating with)
        # The camera is typically on a private IP (10.x, 192.168.x, etc.)
        # We want to check the reputation of the OTHER endpoint (public internet IP)
        
        src_is_private = self.is_private_ip(src_ip)
        dst_is_private = self.is_private_ip(dst_ip)
        
        # Determine which IP is the "remote endpoint" to check
        if src_is_private and not dst_is_private:
            # Traffic from camera (private) to internet (public)
            # Check the destination (remote server)
            remote_ip = dst_ip
            result['remote_ip'] = remote_ip
        elif not src_is_private and dst_is_private:
            # Traffic from internet (public) to camera (private)
            # Check the source (remote server)
            remote_ip = src_ip
            result['remote_ip'] = remote_ip
        elif src_is_private and dst_is_private:
            # Both private - suspicious local network communication
            # This could be lateral movement or local attacks
            result['is_private'] = True
            result['should_flag'] = True
            result['reason'] = 'private_to_private'
            result['remote_ip'] = f"both_private_{src_ip}_to_{dst_ip}"
            return result
        else:
            # Both public - unusual for IoT camera, but check destination
            remote_ip = dst_ip
            result['remote_ip'] = remote_ip
        
        # At this point, remote_ip is the public endpoint to validate
        # If deny list is disabled, don't flag legitimate public IPs
        if not self.enabled:
            result['reason'] = 'public_ip_no_denylist_check'
            result['should_flag'] = False
            return result
        
        # Check if remote IP is in the deny list
        if remote_ip in self.deny_list:
            entry = self.deny_list[remote_ip]
            result['in_denylist'] = True
            result['should_flag'] = True
            result['reason'] = 'in_denylist'
            result['score'] = entry['score']
            result['country_code'] = entry['country_code']
            print(f"🚨 Remote IP {remote_ip} found in deny list (score: {entry['score']}, country: {entry['country_code']})")
            return result
        
        # Remote IP is public but NOT in deny list
        # This is likely a legitimate cloud service (DNS, API, streaming, etc.)
        # Suppress the anomaly
        result['reason'] = 'public_ip_not_in_denylist'
        result['should_flag'] = False
        print(f"✓ Remote IP {remote_ip} not in deny list - likely legitimate (suppressing anomaly)")
        return result
    
    def is_private_ip(self, ip):
        """Check if IP is in private/local range"""
        try:
            parts = list(map(int, ip.split('.')))
            
            # Private IP ranges:
            # 10.0.0.0/8
            if parts[0] == 10:
                return True
            # 172.16.0.0/12
            if parts[0] == 172 and 16 <= parts[1] <= 31:
                return True
            # 192.168.0.0/16
            if parts[0] == 192 and parts[1] == 168:
                return True
            # 127.0.0.0/8 (loopback)
            if parts[0] == 127:
                return True
            # Link-local 169.254.0.0/16
            if parts[0] == 169 and parts[1] == 254:
                return True
            
            return False
        except:
            return False

class OnlineDetectorRPi:
    def __init__(self, args):
        self.args = args
        self.stopped = False
        self.learning_phase = True
        self.learning_start = None
        self.learning_window = args.learning_window  # Store learning window from args
        
        # Initialize flow tracking
        self.flow_stats = {}
        self.packet_times = defaultdict(lambda: deque(maxlen=2))
        self.last_batch_time = time.time()
        self.learning_data = []
        
        # Initialize models
        self.hst = HalfSpaceTrees(
            n_trees=10, 
            height=8,
            window_size=100,
            seed=42
        )
        
        self.models = {
            'ocsvm': OneClassSVM(kernel='rbf', nu=0.1),
            'lof': LocalOutlierFactor(n_neighbors=20, novelty=True),
            'copod': COPOD()
        }
        
        self.scaler = StandardScaler()
        
        # Setup evaluator (only used after learning phase)
        self.evaluator = PerAlgorithmEvaluator(['HalfSpaceTrees', 'OC-SVM', 'LOF', 'COPOD'])
        
        # Store MAC addresses
        self.target_mac = args.target_mac.lower()
        self.attacker_mac = args.attacker_mac.lower()
        self.camera_mac = args.camera_mac.lower()
        
        # Initialize AbuseIPDB deny list checker
        denylist_file = getattr(args, 'denylist_file', None)
        router_ip = getattr(args, 'router_ip', None)
        self.denylist = AbuseIPDBDenyList(denylist_file=denylist_file, router_ip=router_ip)
        self.enable_denylist = bool(denylist_file)
        
        # Setup MacBook connection
        self.macbook_sock = None
        if args.macbook_ip:
            self.connect_to_macbook(args.macbook_ip, args.macbook_port)
        
        # Setup graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup periodic metrics saving
        self._setup_periodic_save(args.metrics_interval)

        # Add threshold monitoring
        self.score_history = {
            'hst': deque(maxlen=2000),
            'ocsvm': deque(maxlen=2000), 
            'lof': deque(maxlen=2000),
            'copod': deque(maxlen=2000)
        }
        self.recent_samples = deque(maxlen=2000)  # Store recent samples with ground truth
        self.last_threshold_update = time.time()
        self.threshold_update_interval = 300  # 5 minutes
        
        # Initialize threshold log files
        self.unsupervised_threshold_log = "thresholds_unsupervised.log"
        self.supervised_threshold_log = "thresholds_supervised.log"
        
        # Create log files with headers (always rewrite)
        with open(self.unsupervised_threshold_log, 'w') as f:
            f.write("timestamp,hst_threshold,ocsvm_threshold,lof_threshold,copod_threshold\n")
        with open(self.supervised_threshold_log, 'w') as f:
            f.write("timestamp,hst_threshold,ocsvm_threshold,lof_threshold,copod_threshold\n")

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping gracefully...")
        self.stop()
        
        # Compute and save final thresholds if we have enough data
        if not self.learning_phase and len(self.recent_samples) >= 50:
            print("\n=== Computing Final Thresholds on Termination ===")
            try:
                self.compute_and_save_thresholds()
            except Exception as e:
                print(f"⚠️ Error computing final thresholds: {e}")
        
        self.save_models()
        self.save_metrics()
        sys.exit(0)

    def _setup_periodic_save(self, interval):
        def periodic_save():
            while not self.stopped:
                time.sleep(interval)
                if not self.stopped:
                    print(f"Periodic save at {datetime.now()}")
                    if not self.learning_phase:
                        self.save_metrics()
                    self.save_models()
        
        t = threading.Thread(target=periodic_save, daemon=True)
        t.start()

    def connect_to_macbook(self, ip, port):
        try:
            self.macbook_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.macbook_sock.connect((ip, port))
            print(f"✅ Connected to DL server at {ip}:{port}")
        except Exception as e:
            print(f"⚠️ DL server connection failed: {e}")
            self.macbook_sock = None

    def send_to_macbook(self, flow_key_5t, src_mac, dst_mac, features):
        if not self.macbook_sock:
            return
        try:
            # Use the same field names that learning_rpi.py used to use
            msg = json.dumps({
                'flow_key': flow_key_5t,  # Changed from 'flow_5tuple' to 'flow_key'
                'features': features,
                'src_mac': src_mac,       # Keep these fields for online_detector_dl.py
                'dst_mac': dst_mac,
                'timestamp': datetime.now().isoformat()
            }) + '\n'
            
            # Debug output
            if hasattr(self, '_send_count'):
                self._send_count += 1
            else:
                self._send_count = 1
                
            if self._send_count <= 5 or self._send_count % 50 == 0:
                print(f"📤 Sending sample #{self._send_count} to DL server")
                
            self.macbook_sock.send(msg.encode())
        except Exception as e:
            print(f"⚠️ Error sending to DL server: {e}")
            # Try to reconnect
            if self.args.macbook_ip:
                try:
                    self.macbook_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.macbook_sock.connect((self.args.macbook_ip, self.args.macbook_port))
                    print("✅ Reconnected to DL server")
                except Exception as reconnect_error:
                    print(f"❌ Reconnection failed: {reconnect_error}")
                    self.macbook_sock = None

    def save_models(self):
        print("Saving models...")
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        with open(os.path.join(MODEL_DIR, "halfspacetrees.pkl"), 'wb') as f:
            pickle.dump(self.hst, f)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print("Models saved")

    def stop(self):
        self.stopped = True

    def save_metrics(self):
        if not self.learning_phase:
            # Metrics CSV will be overwritten each time
            self.evaluator.summarize(METRICS_CSV)

    def extract_features(self, stats):
        duration = stats['last_seen'] - stats['first_seen']
        return {
            'flow_duration': duration,
            'flow_byts_s': stats['bytes'] / duration if duration > 0 else 0.0,
            'flow_pkts_s': stats['packets'] / duration if duration > 0 else 0.0,
            'pkt_len': np.mean(stats['pkt_lens']) if stats['pkt_lens'] else 0.0,
            'pkt_size': np.std(stats['pkt_lens']) if stats['pkt_lens'] else 0.0,
            'iat': np.mean(stats['iat']) if stats['iat'] else 0.0,
            'tot_bwd_pkts': stats['bwd_packets'],
            'tot_fwd_pkts': stats['fwd_packets'],
            'totlen_bwd_pkts': stats['bwd_bytes'],
            'totlen_fwd_pkts': stats['fwd_bytes']
        }

    def train_models(self):
        """Train models on collected data during learning phase"""
        # Extract feature vectors from learning data
        X = []
        
        for key5t, stats in list(self.flow_stats.items()):
            duration = stats['last_seen'] - stats['first_seen']
            
            # Skip flows that are too short
            if duration < MIN_FLOW_DURATION:
                continue
                
            # Extract features
            features = self.extract_features(stats)
            X.append([features[f] for f in FEATURES])
        
        if not X:
            print("No data for training")
            return
            
        X = np.array(X)
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train models
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_scaled)
            except Exception as e:
                print(f"⚠️ Error training {name}: {e}")
        
        # Calculate and print initial thresholds
        self.calculate_and_print_initial_thresholds(X_scaled)
        
        # Save initial models
        self.save_models()
        print("Initial training complete")

    def calculate_and_print_initial_thresholds(self, X_train):
        """Calculate and display thresholds based on training data"""
        print("\n=== Initial Threshold Calculation ===")
        contamination_rate = 0.05
        
        # HST threshold calculation
        hst_scores = []
        for row in X_train:
            feature_dict = {j: float(v) for j, v in enumerate(row)}
            score = self.hst.score_one(feature_dict)
            hst_scores.append(score)
            self.hst.learn_one(feature_dict)
        
        if hst_scores:
            calculated_hst = np.percentile(hst_scores, contamination_rate * 100)
            print(f"HST - Current: {HST_THRESHOLD:.4f}, Calculated: {calculated_hst:.4f}")
        
        # OC-SVM threshold
        if 'ocsvm' in self.models:
            try:
                ocsvm_scores = self.models['ocsvm'].decision_function(X_train)
                calculated_ocsvm = np.percentile(ocsvm_scores, contamination_rate * 100)
                print(f"OC-SVM - Current: {OCSVM_THRESHOLD:.4f}, Calculated: {calculated_ocsvm:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating OC-SVM threshold: {e}")
        
        # LOF threshold
        if 'lof' in self.models:
            try:
                lof_scores = self.models['lof'].decision_function(X_train)
                calculated_lof = np.percentile(lof_scores, contamination_rate * 100)
                print(f"LOF - Current: {LOF_THRESHOLD:.4f}, Calculated: {calculated_lof:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating LOF threshold: {e}")
        
        # COPOD threshold
        if 'copod' in self.models:
            try:
                copod_scores = self.models['copod'].decision_function(X_train)
                calculated_copod = np.percentile(copod_scores, (1 - contamination_rate) * 100)
                print(f"COPOD - Current: {COPOD_THRESHOLD:.4f}, Calculated: {calculated_copod:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating COPOD threshold: {e}")
        
        print("=== Initial Threshold Calculation Complete ===\n")

    def detect(self, records, X_scaled):
        """Process flows and evaluate each individually"""
        
        # HST detection (stream processing)
        for i, (key5t, src_mac, dst_mac, features) in enumerate(records):
            now = datetime.now()
            # Ground truth: Consider traffic as attack if it involves attacker and camera
            # This includes both attacker->camera and camera->attacker (responses)
            gt_attack = (
                (src_mac == self.attacker_mac and dst_mac == self.camera_mac) or
                (src_mac == self.camera_mac and dst_mac == self.attacker_mac)
            )
            
            # Store sample for threshold updates
            sample_data = {
                'features': [features[f] for f in FEATURES],
                'gt_attack': gt_attack,
                'timestamp': now,
                'src_mac': src_mac,
                'dst_mac': dst_mac
            }
            self.recent_samples.append(sample_data)
            
            # Note sample for HST
            self.evaluator.note_sample(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
            
            try:
                # Convert features to dict with integer keys for HST
                feature_dict = {j: float(v) for j, v in enumerate([features[f] for f in FEATURES])}
                
                # HST processes and learns from one sample at a time
                score = self.hst.score_one(feature_dict)
                
                # Store score for threshold calculation
                self.score_history['hst'].append(score)
                
                # Update HST with the sample
                self.hst.learn_one(feature_dict)

                if score > HST_THRESHOLD:
                    # Check deny list if enabled
                    if self.enable_denylist:
                        src_ip, dst_ip, proto, src_port, dst_port = key5t
                        denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                        
                        # Whitelist network functionality traffic
                        if denylist_result['is_network_function']:
                            self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                            if gt_attack:
                                log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
                            continue
                        
                        # Only log anomaly if deny list confirms it OR if not in public-not-in-denylist case
                        if denylist_result['should_flag']:
                            self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                            reason = denylist_result['reason']
                            print(f"✓ CONFIRMED anomaly (HST) - {reason}: {key5t} (score={score:.3f})")
                            log_anomaly(key5t, features, score, "HalfSpaceTrees")
                            if not gt_attack:
                                log_misclassified_benign(key5t, features, score, "HalfSpaceTrees")
                        else:
                            # Suppressed - public IP not in deny list
                            self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                            print(f"✗ SUPPRESSED anomaly (HST) - {denylist_result['reason']}: {key5t}")
                            if gt_attack:
                                log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
                    else:
                        # Deny list disabled, use standard detection
                        self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                        log_anomaly(key5t, features, score, "HalfSpaceTrees")
                        if not gt_attack:
                            log_misclassified_benign(key5t, features, score, "HalfSpaceTrees")
                else:
                    self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                    if gt_attack:
                        log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
            except Exception as e:
                print(f"⚠️ HST scoring error: {e}")
            
            # Batch-trained models predict on individual flows
            x = np.array([[features[f] for f in FEATURES]])
            
            # OCSVM prediction
            if 'ocsvm' in self.models and self.models['ocsvm'] is not None:
                self.evaluator.note_sample(now, gt_attack=gt_attack, algorithm='OC-SVM')
                try:
                    score = self.models['ocsvm'].decision_function(x)[0]
                    self.score_history['ocsvm'].append(score)
                    
                    if score < OCSVM_THRESHOLD:
                        # Check deny list if enabled
                        if self.enable_denylist:
                            src_ip, dst_ip, proto, src_port, dst_port = key5t
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            
                            # Whitelist network functionality traffic
                            if denylist_result['is_network_function']:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "OCSVM")
                                continue
                            
                            if denylist_result['should_flag']:
                                self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                                reason = denylist_result['reason']
                                print(f"✓ CONFIRMED anomaly (OC-SVM) - {reason}: {key5t} (score={score:.3f})")
                                log_anomaly(key5t, features, score, "OCSVM")
                                if not gt_attack:
                                    log_misclassified_benign(key5t, features, score, "OCSVM")
                            else:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                                print(f"✗ SUPPRESSED anomaly (OC-SVM) - {denylist_result['reason']}: {key5t}")
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "OCSVM")
                        else:
                            # Deny list disabled
                            self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                            log_anomaly(key5t, features, score, "OCSVM")
                            if not gt_attack:
                                log_misclassified_benign(key5t, features, score, "OCSVM")
                    else:
                        self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                        if gt_attack:
                            log_missed_anomaly(key5t, features, score, "OCSVM")
                except Exception as e:
                    print(f"⚠️ OCSVM scoring error: {e}")
                
            # LOF prediction
            if 'lof' in self.models and self.models['lof'] is not None:
                self.evaluator.note_sample(now, gt_attack=gt_attack, algorithm='LOF')
                try:
                    score = self.models['lof'].decision_function(x)[0]
                    self.score_history['lof'].append(score)
                    
                    if score < LOF_THRESHOLD:
                        # Check deny list if enabled
                        if self.enable_denylist:
                            src_ip, dst_ip, proto, src_port, dst_port = key5t
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            
                            # Whitelist network functionality traffic
                            if denylist_result['is_network_function']:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='LOF')
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "LOF")
                                continue
                            
                            if denylist_result['should_flag']:
                                self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='LOF')
                                reason = denylist_result['reason']
                                print(f"✓ CONFIRMED anomaly (LOF) - {reason}: {key5t} (score={score:.3f})")
                                log_anomaly(key5t, features, score, "LOF")
                                if not gt_attack:
                                    log_misclassified_benign(key5t, features, score, "LOF")
                            else:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='LOF')
                                print(f"✗ SUPPRESSED anomaly (LOF) - {denylist_result['reason']}: {key5t}")
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "LOF")
                        else:
                            # Deny list disabled
                            self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='LOF')
                            log_anomaly(key5t, features, score, "LOF")
                            if not gt_attack:
                                log_misclassified_benign(key5t, features, score, "LOF")
                    else:
                        self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='LOF')
                        if gt_attack:
                            log_missed_anomaly(key5t, features, score, "LOF")
                except Exception as e:
                    print(f"⚠️ LOF scoring error: {e}")
                
            # COPOD prediction
            if 'copod' in self.models and self.models['copod'] is not None:
                self.evaluator.note_sample(now, gt_attack=gt_attack, algorithm='COPOD')
                try:
                    score = self.models['copod'].decision_function(x)[0]
                    self.score_history['copod'].append(score)
                    
                    if score > COPOD_THRESHOLD:
                        # Check deny list if enabled
                        if self.enable_denylist:
                            src_ip, dst_ip, proto, src_port, dst_port = key5t
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            
                            # Whitelist network functionality traffic
                            if denylist_result['is_network_function']:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "COPOD")
                                continue
                            
                            if denylist_result['should_flag']:
                                self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                                reason = denylist_result['reason']
                                print(f"✓ CONFIRMED anomaly (COPOD) - {reason}: {key5t} (score={score:.3f})")
                                log_anomaly(key5t, features, score, "COPOD")
                                if not gt_attack:
                                    log_misclassified_benign(key5t, features, score, "COPOD")
                            else:
                                self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                                print(f"✗ SUPPRESSED anomaly (COPOD) - {denylist_result['reason']}: {key5t}")
                                if gt_attack:
                                    log_missed_anomaly(key5t, features, score, "COPOD")
                        else:
                            # Deny list disabled
                            self.evaluator.note_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                            log_anomaly(key5t, features, score, "COPOD")
                            if not gt_attack:
                                log_misclassified_benign(key5t, features, score, "COPOD")
                    else:
                        self.evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                        if gt_attack:
                            log_missed_anomaly(key5t, features, score, "COPOD")
                except Exception as e:
                    print(f"⚠️ COPOD scoring error: {e}")

    def process_batch(self):
        """Process current batch of flows"""
        if not self.flow_stats:
            return
            
        records = []
        X = []

        # Extract flow records and feature vectors
        for key5t, stats in list(self.flow_stats.items()):
            duration = stats['last_seen'] - stats['first_seen']
            
            # Extract features
            features = self.extract_features(stats)
            
            # ALWAYS send to MacBook for DL processing (regardless of duration)
            # DL models can learn from short flows too
            self.send_to_macbook(key5t, stats['src_mac'], stats['dst_mac'], features)
            
            # Skip flows that are too short for LOCAL detection only
            # Still sent to DL server above
            if duration < MIN_FLOW_DURATION:
                continue
            
            # Add to batch records for LOCAL RPi detection
            records.append((key5t, stats['src_mac'], stats['dst_mac'], features))
            X.append([features[f] for f in FEATURES])
                
        if not records:
            return
            
        X = np.array(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Training phase
        if self.learning_phase:
            now = datetime.now()
            elapsed = (now - self.learning_start).total_seconds()
            
            if elapsed > self.learning_window:
                print("Initial training complete")
                self.learning_phase = False
                self.train_models()
        # Detection phase
        else:
            # Train batch models on the new data
            if len(X) >= 10:  # Minimum batch size
                for name, model in self.models.items():
                    try:
                        model.fit(X_scaled)
                    except Exception as e:
                        print(f"⚠️ {name} training error: {e}")
            
            # Check if we should update thresholds (every 5 minutes)
            current_time = time.time()
            if current_time - self.last_threshold_update >= self.threshold_update_interval:
                self.compute_and_save_thresholds()
                self.last_threshold_update = current_time
            
            # Detect anomalies in individual flows
            self.detect(records, X_scaled)
            
        # Clean up old flows
        self.flow_stats = {}
        for k in self.packet_times.keys():
            self.packet_times[k] = deque(maxlen=50)

    def compute_and_save_thresholds(self):
        """Compute thresholds every 5 minutes using both supervised and unsupervised approaches"""
        if len(self.recent_samples) < 50:
            print("Not enough samples for threshold calculation")
            return
        
        print("\n=== Computing Updated Thresholds ===")
        timestamp = datetime.now().isoformat()
        contamination_rate = 0.05
        
        # Unsupervised approach: Use only benign samples based on score distributions
        unsupervised_thresholds = self.compute_unsupervised_thresholds(contamination_rate)
        
        # Supervised approach: Use ground truth labels
        supervised_thresholds = self.compute_supervised_thresholds(contamination_rate)
        
        # Log unsupervised thresholds (append to the freshly created file)
        with open(self.unsupervised_threshold_log, 'a') as f:
            f.write(f"{timestamp},{unsupervised_thresholds['hst']:.6f},{unsupervised_thresholds['ocsvm']:.6f},{unsupervised_thresholds['lof']:.6f},{unsupervised_thresholds['copod']:.6f}\n")
        
        # Log supervised thresholds (append to the freshly created file)
        with open(self.supervised_threshold_log, 'a') as f:
            f.write(f"{timestamp},{supervised_thresholds['hst']:.6f},{supervised_thresholds['ocsvm']:.6f},{supervised_thresholds['lof']:.6f},{supervised_thresholds['copod']:.6f}\n")
        
        print(f"Thresholds computed and saved at {timestamp}")
        print("=== Threshold Update Complete ===\n")

    def compute_unsupervised_thresholds(self, contamination_rate):
        """Compute thresholds using unsupervised approach (score distributions only)"""
        thresholds = {}
        
        # HST threshold
        if len(self.score_history['hst']) > 100:
            scores = list(self.score_history['hst'])
            thresholds['hst'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised HST threshold: {thresholds['hst']:.6f} (current: {HST_THRESHOLD:.6f})")
        else:
            thresholds['hst'] = HST_THRESHOLD
        
        # OC-SVM threshold
        if len(self.score_history['ocsvm']) > 100:
            scores = list(self.score_history['ocsvm'])
            thresholds['ocsvm'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised OC-SVM threshold: {thresholds['ocsvm']:.6f} (current: {OCSVM_THRESHOLD:.6f})")
        else:
            thresholds['ocsvm'] = OCSVM_THRESHOLD
        
        # LOF threshold
        if len(self.score_history['lof']) > 100:
            scores = list(self.score_history['lof'])
            thresholds['lof'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised LOF threshold: {thresholds['lof']:.6f} (current: {LOF_THRESHOLD:.6f})")
        else:
            thresholds['lof'] = LOF_THRESHOLD
        
        # COPOD threshold
        if len(self.score_history['copod']) > 100:
            scores = list(self.score_history['copod'])
            thresholds['copod'] = np.percentile(scores, (1 - contamination_rate) * 100)
            print(f"Unsupervised COPOD threshold: {thresholds['copod']:.6f} (current: {COPOD_THRESHOLD:.6f})")
        else:
            thresholds['copod'] = COPOD_THRESHOLD
        
        return thresholds

    def compute_supervised_thresholds(self, contamination_rate):
        """Compute thresholds using supervised approach (ground truth labels)"""
        thresholds = {}
        
        # Filter recent samples to get only benign ones
        benign_samples = [s for s in self.recent_samples if not s['gt_attack']]
        
        if len(benign_samples) < 50:
            print("Not enough benign samples for supervised threshold calculation")
            return {
                'hst': HST_THRESHOLD,
                'ocsvm': OCSVM_THRESHOLD,
                'lof': LOF_THRESHOLD,
                'copod': COPOD_THRESHOLD
            }
        
        # Extract features for benign samples
        X_benign = np.array([s['features'] for s in benign_samples[-200:]])  # Use last 200 benign samples
        X_benign_scaled = self.scaler.transform(X_benign)
        
        # HST threshold on benign samples
        hst_scores_benign = []
        for row in X_benign:
            feature_dict = {j: float(v) for j, v in enumerate(row)}
            score = self.hst.score_one(feature_dict)
            hst_scores_benign.append(score)
        
        if hst_scores_benign:
            thresholds['hst'] = np.percentile(hst_scores_benign, contamination_rate * 100)
            print(f"Supervised HST threshold: {thresholds['hst']:.6f} (current: {HST_THRESHOLD:.6f})")
        else:
            thresholds['hst'] = HST_THRESHOLD
        
        # OC-SVM threshold on benign samples
        try:
            ocsvm_scores = self.models['ocsvm'].decision_function(X_benign_scaled)
            thresholds['ocsvm'] = np.percentile(ocsvm_scores, contamination_rate * 100)
            print(f"Supervised OC-SVM threshold: {thresholds['ocsvm']:.6f} (current: {OCSVM_THRESHOLD:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised OC-SVM threshold: {e}")
            thresholds['ocsvm'] = OCSVM_THRESHOLD
        
        # LOF threshold on benign samples
        try:
            lof_scores = self.models['lof'].decision_function(X_benign_scaled)
            thresholds['lof'] = np.percentile(lof_scores, contamination_rate * 100)
            print(f"Supervised LOF threshold: {thresholds['lof']:.6f} (current: {LOF_THRESHOLD:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised LOF threshold: {e}")
            thresholds['lof'] = LOF_THRESHOLD
        
        # COPOD threshold on benign samples
        try:
            copod_scores = self.models['copod'].decision_function(X_benign_scaled)
            thresholds['copod'] = np.percentile(copod_scores, (1 - contamination_rate) * 100)
            print(f"Supervised COPOD threshold: {thresholds['copod']:.6f} (current: {COPOD_THRESHOLD:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised COPOD threshold: {e}")
            thresholds['copod'] = COPOD_THRESHOLD
        
        return thresholds

    def run(self):
        self.learning_start = datetime.now()
        print(f"Starting online detector (learning phase: {self.learning_window}s)")
        
        # Initialize log files - always rewrite them
        for log_file in [LOG_FILE, MISSED_ANOMALY_LOG, MISCLASSIFIED_BENIGN_LOG]:
            with open(log_file, 'w') as f:
                f.write("")

        def packet_callback(pkt):
            if self.stopped:
                return
            
            if not (pkt.haslayer(Ether) and 
                   (pkt[Ether].src.lower() == self.target_mac or 
                    pkt[Ether].dst.lower() == self.target_mac)):
                return

            if not (IP in pkt and (TCP in pkt or UDP in pkt)):
                return

            now = time.time()
            
            # Extract packet info
            eth = pkt[Ether]
            ip = pkt[IP]
            sport = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
            dport = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
            
            key5t = (ip.src, ip.dst, ip.proto, sport, dport)
            length = len(pkt)
            src_mac = eth.src.lower()
            dst_mac = eth.dst.lower()

            # Update flow stats
            if key5t not in self.flow_stats:
                self.flow_stats[key5t] = {
                    'first_seen': now,
                    'last_seen': now,
                    'bytes': 0,
                    'packets': 0,
                    'fwd_bytes': 0,
                    'fwd_packets': 0,
                    'bwd_bytes': 0,
                    'bwd_packets': 0,
                    'pkt_lens': [],
                    'iat': [],
                    'src_mac': src_mac,
                    'dst_mac': dst_mac
                }
            
            stats = self.flow_stats[key5t]
            stats['last_seen'] = now
            stats['bytes'] += length
            stats['packets'] += 1
            
            if ip.src == key5t[0]:
                stats['fwd_bytes'] += length
                stats['fwd_packets'] += 1
            else:
                stats['bwd_bytes'] += length
                stats['bwd_packets'] += 1
            
            stats['pkt_lens'].append(length)
            
            prev_time = self.packet_times[key5t][-1] if self.packet_times[key5t] else now
            stats['iat'].append(now - prev_time)
            self.packet_times[key5t].append(now)

            # Check if batch interval elapsed
            if now - self.last_batch_time > BATCH_INTERVAL:
                self.process_batch()
                self.last_batch_time = now

        while not self.stopped:
            try:
                sniff(
                    iface=self.args.interface,
                    prn=packet_callback,
                    store=False,
                    count=100  # Process in smaller batches
                )
            except Exception as e:
                print(f"⚠️ Socket error: {e}")
                if not self.stopped:
                    print(f"Reopening interface {self.args.interface}...")
                    time.sleep(1)  # Brief pause before retry
                    continue

        # Save final state
        if not self.learning_phase:
            self.save_metrics()
        self.save_models()


class PerAlgorithmEvaluator:
    def __init__(self, algorithms):
        self.algorithms = algorithms
        self.metrics = {alg: {
            'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
            'latency_sum': 0.0, 'latency_count': 0
        } for alg in algorithms}
        self.last_alert = {alg: None for alg in algorithms}
        self.last_sample = {alg: None for alg in algorithms}

    def note_sample(self, ts, gt_attack=False, algorithm=None):
        """Record a new sample timestamp and ground truth"""
        if algorithm and algorithm in self.algorithms:
            # Update for specific algorithm
            if self.last_sample[algorithm] is None or self.last_sample[algorithm][0] != ts:
                self.last_sample[algorithm] = (ts, gt_attack)
        else:
            # Update for all algorithms (backward compatibility)
            for alg in self.algorithms:
                if self.last_sample[alg] is None or self.last_sample[alg][0] != ts:
                    self.last_sample[alg] = (ts, gt_attack)

    def note_alert(self, ts, gt_attack=False, algorithm=None):
        """Record an alert and update metrics"""
        if not algorithm or algorithm not in self.algorithms:
            return

        m = self.metrics[algorithm]
        if gt_attack:
            m['tp'] += 1
            # Compute detection latency
            if self.last_sample[algorithm]:
                sample_ts, _ = self.last_sample[algorithm]
                latency = (ts - sample_ts).total_seconds()
                m['latency_sum'] += latency
                m['latency_count'] += 1
        else:
            m['fp'] += 1
        self.last_alert[algorithm] = ts

    def note_no_alert(self, ts, gt_attack=False, algorithm=None):
        """Record lack of alert and update metrics"""
        if not algorithm or algorithm not in self.algorithms:
            return

        m = self.metrics[algorithm]
        if gt_attack:
            m['fn'] += 1
        else:
            m['tn'] += 1

    def get_metrics(self, algorithm):
        """Get precision, recall, FPR, and average latency for an algorithm"""
        m = self.metrics[algorithm]
        
        precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
        recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
        fpr = m['fp'] / (m['fp'] + m['tn']) if (m['fp'] + m['tn']) > 0 else 0
        avg_latency = m['latency_sum'] / m['latency_count'] if m['latency_count'] > 0 else 0
        
        return precision, recall, fpr, avg_latency

    def summarize(self, csv_file):
        """Save metrics summary to CSV file"""
        with open(csv_file, 'w') as f:
            f.write("Algorithm,TP,FP,TN,FN,Precision,Recall,FPR,Avg_Latency\n")
            for alg in self.algorithms:
                m = self.metrics[alg]
                precision, recall, fpr, avg_latency = self.get_metrics(alg)
                f.write(f"{alg},{m['tp']},{m['fp']},{m['tn']},{m['fn']},{precision:.4f},{recall:.4f},{fpr:.4f},{avg_latency:.4f}\n")
        
        print(f"Metrics saved to {csv_file}")


def log_anomaly(flow_key_5t, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] Anomaly: {flow_key_5t} (score={score:.3f})")


def log_missed_anomaly(flow_key_5t, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISSED_ANOMALY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] Missed Anomaly: {flow_key_5t} (score={score:.3f})")


def log_misclassified_benign(flow_key_5t, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISCLASSIFIED_BENIGN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] False Positive: {flow_key_5t} (score={score:.3f})")


def get_args():
    parser = argparse.ArgumentParser(description='Online anomaly detector for IoT networks')
    parser.add_argument('--interface', '-i', required=True, help='Network interface to monitor')
    parser.add_argument('--target-mac', '-t', required=True, help='Target device MAC address')
    parser.add_argument('--attacker-mac', '-a', required=True, help='Attacker MAC address')
    parser.add_argument('--camera-mac', '-c', required=True, help='Camera MAC address')
    parser.add_argument('--macbook-ip', help='MacBook IP for DL processing')
    parser.add_argument('--macbook-port', type=int, default=8888, help='MacBook port for DL processing')
    parser.add_argument('--metrics-interval', type=int, default=300, help='Metrics save interval (seconds)')
    parser.add_argument('--learning-window', type=int, default=LEARNING_WINDOW_DEFAULT, 
                        help=f'Learning phase duration in seconds (default: {LEARNING_WINDOW_DEFAULT}s)')
    parser.add_argument('--denylist-file', type=str, default=None,
                        help='Path to AbuseIPDB deny list CSV file (format: ip,country_code,abuse_confidence_score,last_reported_at)')
    parser.add_argument('--router-ip', type=str, default=None,
                        help='Router IP address for whitelisting network functionality traffic (DHCP, DNS)')
    return parser.parse_args()


def main():
    args = get_args()
    detector = OnlineDetectorRPi(args)
    detector.run()


if __name__ == "__main__":
    main()
