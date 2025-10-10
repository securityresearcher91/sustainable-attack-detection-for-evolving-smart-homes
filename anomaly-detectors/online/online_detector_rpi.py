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

# Initial detection thresholds (will be adapted during runtime)
INIT_HST_THRESHOLD = 0.48 
INIT_OCSVM_THRESHOLD = -1.034788
INIT_LOF_THRESHOLD = -16.121545
INIT_COPOD_THRESHOLD = 18.400942

# Adaptive threshold settings
ADAPTIVE_THRESHOLD_ENABLED = True
ADAPTIVE_WINDOW_SIZE = 500  # Number of scores to keep for percentile calculation
ADAPTIVE_PERCENTILE = 99    # Percentile to use for threshold calculation (higher = fewer alerts)
ADAPTIVE_UPDATE_FREQ = 50   # Update threshold every N samples
ADAPTIVE_SMOOTHING = 0.9    # Smoothing factor (higher = slower adaptation)

# Global variables for tracking thresholds - don't access directly, use OnlineDetectorRPi.adaptive_thresholds
HST_THRESHOLD = INIT_HST_THRESHOLD
OCSVM_THRESHOLD = INIT_OCSVM_THRESHOLD
LOF_THRESHOLD = INIT_LOF_THRESHOLD
COPOD_THRESHOLD = INIT_COPOD_THRESHOLD

# Debug log helper functions
def debug_log(message):
    """Print debug messages only when DEBUG flag is enabled"""
    if DEBUG:
        print(f"[DEBUG] {message}")
        
def info_log(message):
    """Print important info messages regardless of debug setting"""
    print(message)

class OnlineDetectorRPi:
    def __init__(self, args):
        self.args = args
        self.stopped = False
        self.learning_phase = True
        self.learning_start = None
        self.learning_window = args.learning_window  # Store learning window from args
        
        # Set global debug flag
        global DEBUG
        DEBUG = args.debug
        
        # Initialize flow tracking
        self.flow_stats = {}
        self.packet_times = defaultdict(lambda: deque(maxlen=2))
        self.last_batch_time = time.time()
        self.learning_data = []
        
        # Initialize adaptive thresholds
        self.adaptive_thresholds = {
            'hst': INIT_HST_THRESHOLD,
            'ocsvm': INIT_OCSVM_THRESHOLD,
            'lof': INIT_LOF_THRESHOLD,
            'copod': INIT_COPOD_THRESHOLD
        }
        self.sample_count_since_update = 0
        
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
        self.unsupervised_threshold_log = "thresholds_unsupervised_rpi.log"
        self.supervised_threshold_log = "thresholds_supervised_rpi.log"
        self.adaptive_threshold_log = "thresholds_adaptive_rpi.log"
        
        # Create log files with headers (always rewrite)
        with open(self.unsupervised_threshold_log, 'w') as f:
            f.write("timestamp,hst_threshold,ocsvm_threshold,lof_threshold,copod_threshold\n")
        with open(self.supervised_threshold_log, 'w') as f:
            f.write("timestamp,hst_threshold,ocsvm_threshold,lof_threshold,copod_threshold\n")
        with open(self.adaptive_threshold_log, 'w') as f:
            f.write("timestamp,hst_threshold,ocsvm_threshold,lof_threshold,copod_threshold\n")
            # Write initial values
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp},{self.adaptive_thresholds['hst']:.6f},{self.adaptive_thresholds['ocsvm']:.6f},{self.adaptive_thresholds['lof']:.6f},{self.adaptive_thresholds['copod']:.6f}\n")

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
            print(f"Attempting to connect to DL server at {ip}:{port}...")
            
            # Close existing connection if any
            if self.macbook_sock:
                try:
                    self.macbook_sock.close()
                except:
                    pass
                self.macbook_sock = None
                
            # Create new socket
            self.macbook_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Set a timeout so we don't hang forever
            self.macbook_sock.settimeout(10)  # Increased timeout for networks with higher latency
            
            # Print address info for debugging
            try:
                print(f"Local address: {socket.gethostbyname(socket.gethostname())}")
                print(f"Connecting to remote address: {ip}:{port}")
            except:
                pass
                
            # Try to connect
            self.macbook_sock.connect((ip, port))
            
            # Set back to blocking mode for normal operation
            self.macbook_sock.settimeout(None)
            print(f"✅ Connected to DL server at {ip}:{port}")
            
            # Send a ping message to test the connection
            try:
                ping_msg = json.dumps({
                    'type': 'ping',
                    'timestamp': datetime.now().isoformat(),
                    'client_ip': socket.gethostbyname(socket.gethostname())
                }) + '\n'
                self.macbook_sock.send(ping_msg.encode())
                print("✅ Successfully sent test message to DL server")
                
                # Try to receive a ping response
                self.macbook_sock.settimeout(5)
                try:
                    response = self.macbook_sock.recv(1024)
                    print(f"Received response from server: {response.decode('utf-8', errors='replace')}")
                except socket.timeout:
                    print("No ping response received (timeout), but connection established")
                except Exception as e:
                    print(f"Error receiving ping response: {e}")
                finally:
                    # Reset to blocking mode
                    self.macbook_sock.settimeout(None)
                
            except Exception as e:
                print(f"⚠️ Failed to send test message: {e}")
                self.macbook_sock = None
                
        except ConnectionRefusedError:
            print(f"⚠️ Connection refused - DL server at {ip}:{port} is not accepting connections")
            print("Please ensure the DL server is running and the port is correct")
            self.macbook_sock = None
        except socket.gaierror:
            print(f"⚠️ Address error - Could not resolve {ip}")
            print("Please check the IP address of the DL server")
            self.macbook_sock = None
        except Exception as e:
            print(f"⚠️ DL server connection failed: {e}")
            self.macbook_sock = None
            
        # Schedule a reconnection attempt if we failed
        if not self.macbook_sock and not self.stopped:
            def reconnect_attempt():
                if not self.macbook_sock and not self.stopped:
                    print(f"Attempting to reconnect to DL server...")
                    self.connect_to_macbook(ip, port)
            
            # Try to reconnect in the background after a delay (increased to 60s)
            print(f"Will attempt to reconnect in 60 seconds")
            threading.Timer(60, reconnect_attempt).start()

    def send_to_macbook(self, flow_key_5t, src_mac, dst_mac, features):
        if not self.macbook_sock:
            # Try to reconnect if we have connection info
            if hasattr(self.args, 'macbook_ip') and self.args.macbook_ip:
                # Don't reconnect on every message, use a timer
                if not hasattr(self, '_last_reconnect_attempt') or time.time() - self._last_reconnect_attempt > 60:
                    self._last_reconnect_attempt = time.time()
                    print(f"No active connection. Attempting to reconnect to DL server...")
                    self.connect_to_macbook(self.args.macbook_ip, self.args.macbook_port)
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
            if not hasattr(self, '_send_count'):
                self._send_count = 0
                self._last_status_time = 0
                
            self._send_count += 1
            
            # Print status periodically
            current_time = time.time()
            if self._send_count <= 5 or current_time - self._last_status_time > 30:
                print(f"📤 Sending sample #{self._send_count} to DL server")
                self._last_status_time = current_time
                
            # Send heartbeat/ping every 100 messages
            if self._send_count % 100 == 0:
                try:
                    ping_msg = json.dumps({
                        'type': 'ping',
                        'timestamp': datetime.now().isoformat()
                    }) + '\n'
                    self.macbook_sock.send(ping_msg.encode())
                    print("✅ Sent heartbeat ping to DL server")
                except Exception as ping_error:
                    print(f"⚠️ Error sending ping: {ping_error}")
                
            # Send the actual data
            self.macbook_sock.send(msg.encode())
            
        except ConnectionResetError:
            print(f"⚠️ Connection reset by DL server")
            self.macbook_sock = None
            # Schedule a reconnection
            if hasattr(self.args, 'macbook_ip') and self.args.macbook_ip:
                threading.Timer(10, lambda: self.connect_to_macbook(
                    self.args.macbook_ip, self.args.macbook_port)).start()
                
        except BrokenPipeError:
            print(f"⚠️ Broken pipe - DL server has closed the connection")
            self.macbook_sock = None
            # Schedule a reconnection
            if hasattr(self.args, 'macbook_ip') and self.args.macbook_ip:
                threading.Timer(10, lambda: self.connect_to_macbook(
                    self.args.macbook_ip, self.args.macbook_port)).start()
                
        except Exception as e:
            print(f"⚠️ Error sending to DL server: {e}")
            
            # Check if socket is still connected
            try:
                # Try sending a small ping to check connection
                self.macbook_sock.settimeout(1)
                self.macbook_sock.send(b'{"type":"ping"}\n')
                self.macbook_sock.settimeout(None)
            except:
                print("Socket appears to be disconnected. Attempting to reconnect...")
                self.macbook_sock = None
                # Try to reconnect
                if hasattr(self.args, 'macbook_ip') and self.args.macbook_ip:
                    threading.Timer(5, lambda: self.connect_to_macbook(
                        self.args.macbook_ip, self.args.macbook_port)).start()

    def save_models(self):
        print("Saving models...")
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
        with open(os.path.join(MODEL_DIR, "halfspacetrees.pkl"), 'wb') as f:
            pickle.dump(self.hst, f)
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
            
        # Save current adaptive thresholds
        with open(os.path.join(MODEL_DIR, "adaptive_thresholds.json"), 'w') as f:
            json.dump({
                "hst": self.adaptive_thresholds['hst'],
                "ocsvm": self.adaptive_thresholds['ocsvm'],
                "lof": self.adaptive_thresholds['lof'],
                "copod": self.adaptive_thresholds['copod'],
                "timestamp": datetime.now().isoformat()
            }, f, indent=4)
            
        print("Models and thresholds saved")

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
            print(f"HST - Current: {self.adaptive_thresholds['hst']:.4f}, Calculated: {calculated_hst:.4f}")
            # Update adaptive threshold if calculated value is better
            if calculated_hst > 0:
                self.adaptive_thresholds['hst'] = calculated_hst
                global HST_THRESHOLD
                HST_THRESHOLD = calculated_hst
                print(f"HST threshold updated to {calculated_hst:.4f}")
        
        # OC-SVM threshold
        if 'ocsvm' in self.models:
            try:
                ocsvm_scores = self.models['ocsvm'].decision_function(X_train)
                calculated_ocsvm = np.percentile(ocsvm_scores, contamination_rate * 100)
                print(f"OC-SVM - Current: {self.adaptive_thresholds['ocsvm']:.4f}, Calculated: {calculated_ocsvm:.4f}")
                # Update threshold
                self.adaptive_thresholds['ocsvm'] = calculated_ocsvm
                global OCSVM_THRESHOLD
                OCSVM_THRESHOLD = calculated_ocsvm
                print(f"OC-SVM threshold updated to {calculated_ocsvm:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating OC-SVM threshold: {e}")
        
        # LOF threshold
        if 'lof' in self.models:
            try:
                lof_scores = self.models['lof'].decision_function(X_train)
                calculated_lof = np.percentile(lof_scores, contamination_rate * 100)
                print(f"LOF - Current: {self.adaptive_thresholds['lof']:.4f}, Calculated: {calculated_lof:.4f}")
                # Update threshold
                self.adaptive_thresholds['lof'] = calculated_lof
                global LOF_THRESHOLD
                LOF_THRESHOLD = calculated_lof
                print(f"LOF threshold updated to {calculated_lof:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating LOF threshold: {e}")
        
        # COPOD threshold
        if 'copod' in self.models:
            try:
                copod_scores = self.models['copod'].decision_function(X_train)
                calculated_copod = np.percentile(copod_scores, (1 - contamination_rate) * 100)
                print(f"COPOD - Current: {self.adaptive_thresholds['copod']:.4f}, Calculated: {calculated_copod:.4f}")
                # Update threshold
                self.adaptive_thresholds['copod'] = calculated_copod
                global COPOD_THRESHOLD
                COPOD_THRESHOLD = calculated_copod
                print(f"COPOD threshold updated to {calculated_copod:.4f}")
            except Exception as e:
                print(f"⚠️ Error calculating COPOD threshold: {e}")
        
        print("=== Initial Threshold Calculation Complete ===\n")

    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent scores"""
        if not ADAPTIVE_THRESHOLD_ENABLED:
            return
            
        self.sample_count_since_update += 1
        
        # Only update after collecting enough samples
        if self.sample_count_since_update < ADAPTIVE_UPDATE_FREQ:
            return
            
        # Reset counter
        self.sample_count_since_update = 0
        
        # Update each threshold based on recent scores
        for algo, scores in self.score_history.items():
            if len(scores) < ADAPTIVE_WINDOW_SIZE // 2:  # Need at least half the window size
                continue
                
            # Calculate new threshold based on percentile
            if algo == 'hst' or algo == 'copod':  # Higher values are anomalous
                new_threshold = np.percentile(list(scores), ADAPTIVE_PERCENTILE)
            else:  # ocsvm, lof - lower values are anomalous
                new_threshold = np.percentile(list(scores), 100 - ADAPTIVE_PERCENTILE)
                
            # Apply smoothing
            current = self.adaptive_thresholds[algo]
            updated = current * ADAPTIVE_SMOOTHING + new_threshold * (1 - ADAPTIVE_SMOOTHING)
            self.adaptive_thresholds[algo] = updated
            
            # Update global variable
            global HST_THRESHOLD, OCSVM_THRESHOLD, LOF_THRESHOLD, COPOD_THRESHOLD
            if algo == 'hst':
                HST_THRESHOLD = updated
            elif algo == 'ocsvm':
                OCSVM_THRESHOLD = updated
            elif algo == 'lof':
                LOF_THRESHOLD = updated
            elif algo == 'copod':
                COPOD_THRESHOLD = updated
                
            print(f"Adaptive {algo.upper()} threshold updated: {current:.4f} -> {updated:.4f}")
    
    def detect(self, records, X_scaled):
        """Process flows and evaluate each individually"""
        
        # HST detection (stream processing)
        for i, (key5t, src_mac, dst_mac, features) in enumerate(records):
            now = datetime.now()
            gt_attack = (src_mac == self.attacker_mac and dst_mac == self.camera_mac)
            
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

                if score > self.adaptive_thresholds['hst']:
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
                    
                    if score < self.adaptive_thresholds['ocsvm']:
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
                    
                    if score < self.adaptive_thresholds['lof']:
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
                    
                    if score > self.adaptive_thresholds['copod']:
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
                    
            # Update adaptive thresholds after processing each sample
            self.update_adaptive_thresholds()

    def process_batch(self):
        """Process current batch of flows"""
        if not self.flow_stats:
            return
            
        records = []
        X = []

        # Extract flow records and feature vectors
        for key5t, stats in list(self.flow_stats.items()):
            duration = stats['last_seen'] - stats['first_seen']
            
            # Skip flows that are too short
            if duration < MIN_FLOW_DURATION:
                continue
                
            # Extract features
            features = self.extract_features(stats)
            
            # Send to MacBook for DL processing
            self.send_to_macbook(key5t, stats['src_mac'], stats['dst_mac'], features)
            
            # Add to batch records
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
        
        # Apply supervised thresholds to adaptive thresholds
        # Supervised takes precedence over unsupervised when available
        global HST_THRESHOLD, OCSVM_THRESHOLD, LOF_THRESHOLD, COPOD_THRESHOLD
        
        for algo in ['hst', 'ocsvm', 'lof', 'copod']:
            if len(self.recent_samples) > 200:
                # Use supervised when we have enough data
                new_threshold = supervised_thresholds[algo]
                print(f"Updating {algo} threshold using supervised approach: {self.adaptive_thresholds[algo]:.6f} -> {new_threshold:.6f}")
            else:
                # Fall back to unsupervised when we don't have enough labeled data
                new_threshold = unsupervised_thresholds[algo]
                print(f"Updating {algo} threshold using unsupervised approach: {self.adaptive_thresholds[algo]:.6f} -> {new_threshold:.6f}")
                
            # Apply smoothing
            self.adaptive_thresholds[algo] = self.adaptive_thresholds[algo] * ADAPTIVE_SMOOTHING + new_threshold * (1 - ADAPTIVE_SMOOTHING)
            
            # Update global variables for backwards compatibility
            if algo == 'hst':
                HST_THRESHOLD = self.adaptive_thresholds[algo]
            elif algo == 'ocsvm':
                OCSVM_THRESHOLD = self.adaptive_thresholds[algo]
            elif algo == 'lof':
                LOF_THRESHOLD = self.adaptive_thresholds[algo]
            elif algo == 'copod':
                COPOD_THRESHOLD = self.adaptive_thresholds[algo]
        
        # Log unsupervised thresholds (append to the freshly created file)
        with open(self.unsupervised_threshold_log, 'a') as f:
            f.write(f"{timestamp},{unsupervised_thresholds['hst']:.6f},{unsupervised_thresholds['ocsvm']:.6f},{unsupervised_thresholds['lof']:.6f},{unsupervised_thresholds['copod']:.6f}\n")
        
        # Log supervised thresholds (append to the freshly created file)
        with open(self.supervised_threshold_log, 'a') as f:
            f.write(f"{timestamp},{supervised_thresholds['hst']:.6f},{supervised_thresholds['ocsvm']:.6f},{supervised_thresholds['lof']:.6f},{supervised_thresholds['copod']:.6f}\n")
        
        # Log current adaptive thresholds
        with open("thresholds_adaptive_rpi.log", 'a') as f:
            f.write(f"{timestamp},{self.adaptive_thresholds['hst']:.6f},{self.adaptive_thresholds['ocsvm']:.6f},{self.adaptive_thresholds['lof']:.6f},{self.adaptive_thresholds['copod']:.6f}\n")
        
        print(f"Thresholds computed and saved at {timestamp}")
        print("=== Threshold Update Complete ===\n")

    def compute_unsupervised_thresholds(self, contamination_rate):
        """Compute thresholds using unsupervised approach (score distributions only)"""
        thresholds = {}
        
        # HST threshold
        if len(self.score_history['hst']) > 100:
            scores = list(self.score_history['hst'])
            thresholds['hst'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised HST threshold: {thresholds['hst']:.6f} (current: {self.adaptive_thresholds['hst']:.6f})")
        else:
            thresholds['hst'] = self.adaptive_thresholds['hst']
        
        # OC-SVM threshold
        if len(self.score_history['ocsvm']) > 100:
            scores = list(self.score_history['ocsvm'])
            thresholds['ocsvm'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised OC-SVM threshold: {thresholds['ocsvm']:.6f} (current: {self.adaptive_thresholds['ocsvm']:.6f})")
        else:
            thresholds['ocsvm'] = self.adaptive_thresholds['ocsvm']
        
        # LOF threshold
        if len(self.score_history['lof']) > 100:
            scores = list(self.score_history['lof'])
            thresholds['lof'] = np.percentile(scores, contamination_rate * 100)
            print(f"Unsupervised LOF threshold: {thresholds['lof']:.6f} (current: {self.adaptive_thresholds['lof']:.6f})")
        else:
            thresholds['lof'] = self.adaptive_thresholds['lof']
        
        # COPOD threshold
        if len(self.score_history['copod']) > 100:
            scores = list(self.score_history['copod'])
            thresholds['copod'] = np.percentile(scores, (1 - contamination_rate) * 100)
            print(f"Unsupervised COPOD threshold: {thresholds['copod']:.6f} (current: {self.adaptive_thresholds['copod']:.6f})")
        else:
            thresholds['copod'] = self.adaptive_thresholds['copod']
        
        return thresholds

    def compute_supervised_thresholds(self, contamination_rate):
        """Compute thresholds using supervised approach (ground truth labels)"""
        thresholds = {}
        
        # Filter recent samples to get only benign ones
        benign_samples = [s for s in self.recent_samples if not s['gt_attack']]
        
        if len(benign_samples) < 50:
            print("Not enough benign samples for supervised threshold calculation")
            return {
                'hst': self.adaptive_thresholds['hst'],
                'ocsvm': self.adaptive_thresholds['ocsvm'],
                'lof': self.adaptive_thresholds['lof'],
                'copod': self.adaptive_thresholds['copod']
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
            print(f"Supervised HST threshold: {thresholds['hst']:.6f} (current: {self.adaptive_thresholds['hst']:.6f})")
        else:
            thresholds['hst'] = self.adaptive_thresholds['hst']
        
        # OC-SVM threshold on benign samples
        try:
            ocsvm_scores = self.models['ocsvm'].decision_function(X_benign_scaled)
            thresholds['ocsvm'] = np.percentile(ocsvm_scores, contamination_rate * 100)
            print(f"Supervised OC-SVM threshold: {thresholds['ocsvm']:.6f} (current: {self.adaptive_thresholds['ocsvm']:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised OC-SVM threshold: {e}")
            thresholds['ocsvm'] = self.adaptive_thresholds['ocsvm']
        
        # LOF threshold on benign samples
        try:
            lof_scores = self.models['lof'].decision_function(X_benign_scaled)
            thresholds['lof'] = np.percentile(lof_scores, contamination_rate * 100)
            print(f"Supervised LOF threshold: {thresholds['lof']:.6f} (current: {self.adaptive_thresholds['lof']:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised LOF threshold: {e}")
            thresholds['lof'] = self.adaptive_thresholds['lof']
        
        # COPOD threshold on benign samples
        try:
            copod_scores = self.models['copod'].decision_function(X_benign_scaled)
            thresholds['copod'] = np.percentile(copod_scores, (1 - contamination_rate) * 100)
            print(f"Supervised COPOD threshold: {thresholds['copod']:.6f} (current: {self.adaptive_thresholds['copod']:.6f})")
        except Exception as e:
            print(f"⚠️ Error computing supervised COPOD threshold: {e}")
            thresholds['copod'] = self.adaptive_thresholds['copod']
        
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
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug logging')
    return parser.parse_args()


def main():
    args = get_args()
    detector = OnlineDetectorRPi(args)
    detector.run()


if __name__ == "__main__":
    main()
