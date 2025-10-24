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
import ipaddress

import numpy as np
import pandas as pd
from scapy.all import sniff, IP, Ether
from scapy.layers.inet import TCP, UDP

import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================

FEATURES = [
    'flow_duration', 'flow_byts_s', 'flow_pkts_s',
    'pkt_len', 'pkt_size', 'iat',
    'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
]

MODEL_DIR = "models_rpi"
HST_FILE = os.path.join(MODEL_DIR, "halfspacetrees.pkl")
OCSVM_FILE = os.path.join(MODEL_DIR, "ocsvm.joblib")
LOF_FILE = os.path.join(MODEL_DIR, "lof.joblib")
COPOD_FILE = os.path.join(MODEL_DIR, "copod.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")

LOG_FILE = "anomalies_inference_rpi.log"
METRICS_CSV = "metrics_rpi.csv"
MISSED_ANOMALY_LOG = "missed_anomalies_rpi.log"
MISCLASSIFIED_BENIGN_LOG = "misclassified_benigns_rpi.log"

BATCH_INTERVAL = 30
MIN_FLOW_DURATION = 0.1
HST_THRESHOLD = 0.8  # Updated threshold for HalfSpaceTrees
OCSVM_THRESHOLD = -0.03
LOF_THRESHOLD = -1.0
COPOD_THRESHOLD = 0.8  # You may want to tune this value

# Network functionality protocols that should always be allowed
ALLOWED_NETWORK_PROTOCOLS = {
    67,   # DHCP server
    68,   # DHCP client
    53,   # DNS
}

# ================= AbuseIPDB Deny List =================

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
        self.deny_list = {}
        
        if self.enabled:
            self.load_denylist()
        else:
            print("ℹ️ AbuseIPDB deny list not enabled (no file provided)")
    
    def load_denylist(self):
        """Load AbuseIPDB deny list from CSV file"""
        try:
            with open(self.denylist_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 4:
                    ip = parts[0]
                    country_code = parts[1]
                    abuse_score = int(parts[2]) if parts[2].isdigit() else 0
                    last_reported = parts[3]
                    
                    self.deny_list[ip] = {
                        'country_code': country_code,
                        'score': abuse_score,
                        'last_reported': last_reported
                    }
            
            print(f"✅ Loaded {len(self.deny_list)} IPs from AbuseIPDB deny list")
            
        except Exception as e:
            print(f"⚠️ Error loading deny list from {self.denylist_file}: {e}")
            self.enabled = False
    
    def is_network_functionality(self, src_ip, dst_ip, src_port, dst_port):
        """Check if traffic is network functionality (DHCP, DNS to router)"""
        if (src_port == 67 and dst_port == 68) or (src_port == 68 and dst_port == 67):
            return True, "dhcp"
        
        if self.router_ip and dst_ip == self.router_ip and dst_port == 53:
            return True, "dns_to_router"
        
        return False, None
    
    def check_ip(self, src_ip, dst_ip, src_port, dst_port):
        """Check IP against deny list and private IP ranges"""
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
        
        is_net_func, func_type = self.is_network_functionality(src_ip, dst_ip, src_port, dst_port)
        if is_net_func:
            result['is_network_function'] = True
            result['should_flag'] = False
            result['reason'] = f'network_functionality_{func_type}'
            return result
        
        src_is_private = self.is_private_ip(src_ip)
        dst_is_private = self.is_private_ip(dst_ip)
        
        if src_is_private and not dst_is_private:
            remote_ip = dst_ip
        elif not src_is_private and dst_is_private:
            remote_ip = src_ip
        elif src_is_private and dst_is_private:
            result['is_private'] = True
            result['should_flag'] = True
            result['reason'] = 'private_to_private'
            result['remote_ip'] = f"both_private_{src_ip}_to_{dst_ip}"
            return result
        else:
            remote_ip = dst_ip
        
        result['remote_ip'] = remote_ip
        
        if not self.enabled:
            result['should_flag'] = False
            result['reason'] = 'denylist_disabled'
            return result
        
        if remote_ip in self.deny_list:
            result['in_denylist'] = True
            result['should_flag'] = True
            result['reason'] = 'in_denylist'
            result['score'] = self.deny_list[remote_ip]['score']
            result['country_code'] = self.deny_list[remote_ip]['country_code']
            return result
        
        result['reason'] = 'public_ip_not_in_denylist'
        result['should_flag'] = False
        return result
    
    def is_private_ip(self, ip):
        """Check if IP is in private/local range"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            return (
                ip_obj.is_private or
                ip_obj.is_loopback or
                ip_obj.is_link_local or
                ip_obj.is_multicast
            )
        except:
            return False

# ================= Args =================

def get_args():
    p = argparse.ArgumentParser(description="RPi Inference-only (classical) + per-algorithm metrics + graceful exit")
    p.add_argument('--target-mac', required=True, help='MAC to monitor')
    p.add_argument('--interface', default='wlan0', help='Interface (default: wlan0)')
    p.add_argument('--macbook-ip', required=False, help='DL server IP')
    p.add_argument('--macbook-port', type=int, default=8888)
    p.add_argument('--attacker-mac', required=True, help='Attacker laptop MAC')
    p.add_argument('--camera-mac', required=True, help='Camera/server MAC')
    p.add_argument('--metrics-interval', type=int, default=60, help='Metrics save interval (seconds)')
    p.add_argument('--denylist-file', type=str, default=None,
                   help='Path to AbuseIPDB deny list CSV file')
    p.add_argument('--router-ip', type=str, default=None,
                   help='Router IP address for whitelisting network functionality traffic (DHCP, DNS)')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()

# ================= Networking to MacBook =================

macbook_sock = None

def connect_to_macbook(ip, port):
    global macbook_sock
    if not ip:
        return False
    try:
        macbook_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        macbook_sock.connect((ip, port))
        print(f"✅ Connected to DL server at {ip}:{port}")
        return True
    except Exception as e:
        print(f"⚠️ DL server connection failed: {e}")
        macbook_sock = None
        return False

def send_to_macbook(flow_key_5t, src_mac, dst_mac, features):
    if not macbook_sock:
        return
    try:
        msg = json.dumps({
            'flow_5tuple': flow_key_5t,
            'src_mac': src_mac,
            'dst_mac': dst_mac,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }) + '\n'
        macbook_sock.send(msg.encode())
    except Exception as e:
        print(f"⚠️ Error sending to DL server: {e}")

# ================= Load models =================

def load_models(verbose=False):
    models = {}
    scaler = None

    try:
        scaler = joblib.load(SCALER_FILE)
        if verbose: print(f"Loaded scaler: {SCALER_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load scaler: {e}")
        scaler = StandardScaler()

    try:
        with open(HST_FILE, "rb") as f:
            models['hst'] = pickle.load(f)
        if verbose: print(f"Loaded HST: {HST_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load HalfSpaceTrees: {e}")

    try:
        models['ocsvm'] = joblib.load(OCSVM_FILE)
        if verbose: print(f"Loaded OC-SVM: {OCSVM_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load OC-SVM: {e}")

    try:
        models['lof'] = joblib.load(LOF_FILE)
        if verbose: print(f"Loaded LOF: {LOF_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load LOF: {e}")

    try:
        models['copod'] = joblib.load(COPOD_FILE)
        if verbose: print(f"Loaded COPOD: {COPOD_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load COPOD: {e}")

    return models, scaler

# ================= Feature extraction =================

def flow_5tuple(pkt):
    if IP not in pkt:
        return None
    ip = pkt[IP]
    sport = pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else None)
    dport = pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else None)
    return (ip.src, ip.dst, ip.proto, sport, dport)

def extract_all_records(flow_stats):
    """Extract ALL records without filtering by duration - for sending to DL server"""
    records = []
    for k, v in flow_stats.items():
        duration = v['last_seen'] - v['first_seen']
        record = {
            'flow_duration': duration,
            'flow_byts_s': v['bytes'] / duration if duration > 0 else 0.0,
            'flow_pkts_s': v['packets'] / duration if duration > 0 else 0.0,
            'pkt_len': np.mean(v['pkt_lens']) if v['pkt_lens'] else 0.0,
            'pkt_size': np.std(v['pkt_lens']) if v['pkt_lens'] else 0.0,
            'iat': np.mean(v['iat']) if v['iat'] else 0.0,
            'tot_bwd_pkts': v['bwd_packets'],
            'tot_fwd_pkts': v['fwd_packets'],
            'totlen_bwd_pkts': v['bwd_bytes'],
            'totlen_fwd_pkts': v['fwd_bytes']
        }
        records.append((k, v['src_mac'], v['dst_mac'], record))
    return records

def extract_records(flow_stats):
    """Extract records filtered by duration - for local RPi detection"""
    records = []
    for k, v in flow_stats.items():
        duration = v['last_seen'] - v['first_seen']
        if duration < MIN_FLOW_DURATION:
            continue
        record = {
            'flow_duration': duration,
            'flow_byts_s': v['bytes'] / duration if duration > 0 else 0.0,
            'flow_pkts_s': v['packets'] / duration if duration > 0 else 0.0,
            'pkt_len': np.mean(v['pkt_lens']) if v['pkt_lens'] else 0.0,
            'pkt_size': np.std(v['pkt_lens']) if v['pkt_lens'] else 0.0,
            'iat': np.mean(v['iat']) if v['iat'] else 0.0,
            'tot_bwd_pkts': v['bwd_packets'],
            'tot_fwd_pkts': v['fwd_packets'],
            'totlen_bwd_pkts': v['bwd_bytes'],
            'totlen_fwd_pkts': v['fwd_bytes']
        }
        records.append((k, v['src_mac'], v['dst_mac'], record))
    return records

# ================= Logging =================

def log_anomaly(flow_key_5t, features, score, model_name):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[{model_name}] Anomaly: {flow_key_5t} (score={score:.3f})")

def log_missed_anomaly(flow_key_5t, features, score, model_name):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISSED_ANOMALY_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[{model_name}] Missed anomaly: {flow_key_5t} (score={score:.3f})")

def log_misclassified_benign(flow_key_5t, features, score, model_name):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_key_5t,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISCLASSIFIED_BENIGN_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"[{model_name}] Misclassified benign: {flow_key_5t} (score={score:.3f})")

# ================= Per-Algorithm Evaluator =================

class PerAlgorithmEvaluator:
    def __init__(self, algorithms=['HalfSpaceTrees', 'OC-SVM', 'LOF', 'COPOD']):
        self.algorithms = algorithms
        self.metrics = {}
        self.first_attack_seen_time = None
        
        for alg in algorithms:
            self.metrics[alg] = {
                'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0,
                'first_tp_time': None
            }

    def note_sample(self, t, gt_attack: bool):
        if gt_attack and self.first_attack_seen_time is None:
            self.first_attack_seen_time = t

    def note_alert(self, t, gt_attack: bool, algorithm: str):
        if algorithm in self.metrics:
            if gt_attack:
                self.metrics[algorithm]['tp'] += 1
                if self.metrics[algorithm]['first_tp_time'] is None:
                    self.metrics[algorithm]['first_tp_time'] = t
            else:
                self.metrics[algorithm]['fp'] += 1

    def note_no_alert(self, t, gt_attack: bool, algorithm: str):
        if algorithm in self.metrics:
            if gt_attack:
                self.metrics[algorithm]['fn'] += 1
            else:
                self.metrics[algorithm]['tn'] += 1

    def summarize(self, csv_path=METRICS_CSV):
        results = []
        for alg in self.algorithms:
            m = self.metrics[alg]
            precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0.0
            recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0.0
            fpr = m['fp'] / (m['fp'] + m['tn']) if (m['fp'] + m['tn']) > 0 else 0.0
            latency_s = None
            if m['first_tp_time'] and self.first_attack_seen_time:
                latency_s = (m['first_tp_time'] - self.first_attack_seen_time).total_seconds()
            
            results.append({
                'algorithm': alg,
                'tp': m['tp'], 'fp': m['fp'], 'tn': m['tn'], 'fn': m['fn'],
                'precision': precision, 'recall': recall, 'fpr': fpr,
                'first_detection_latency_s': latency_s
            })
        
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        print(f"Saved per-algorithm metrics to {csv_path}")
        
        # Print summary
        print("\n=== Per-Algorithm Metrics Summary ===")
        for _, row in df.iterrows():
            latency_str = f"{row['first_detection_latency_s']:.1f}s" if row['first_detection_latency_s'] is not None else "N/A"
            print(f"{row['algorithm']:>15}: P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"FPR={row['fpr']:.3f} Latency={latency_str}")
        return df

# ================= Detection =================

def detect_hst(hst, records, evaluator, attacker_mac, camera_mac, denylist=None):
    if not hst:
        return
    for key5t, src_mac, dst_mac, features in records:
        now = datetime.now()
        # Ground truth: traffic is attack if it involves attacker AND camera
        # This includes both attacker->camera and camera->attacker (responses)
        gt_attack = (
            (src_mac == attacker_mac and dst_mac == camera_mac) or
            (src_mac == camera_mac and dst_mac == attacker_mac)
        )
        evaluator.note_sample(now, gt_attack=gt_attack)
        score = None
        try:
            score = hst.score_one(features)
            # Mark as anomaly if score < threshold
            is_anom = score < HST_THRESHOLD
            
            if is_anom:
                should_flag = True
                if denylist and denylist.enabled:
                    src_ip, dst_ip, proto, src_port, dst_port = key5t
                    denylist_result = denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                    
                    # Step 2: Suppress network functionality
                    if denylist_result['is_network_function']:
                        should_flag = False
                    # Step 3: Private IP = vulnerability (flag it)
                    elif denylist_result['is_private']:
                        should_flag = True
                    # Step 4: Public IP - check deny list
                    elif denylist_result['in_denylist']:
                        should_flag = True
                    else:
                        should_flag = False  # Public IP not in deny list
                
                if should_flag:
                    evaluator.note_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                    log_anomaly(key5t, features, score, "HalfSpaceTrees")
                    if not gt_attack:
                        log_misclassified_benign(key5t, features, score, "HalfSpaceTrees")
                else:
                    evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                    if gt_attack:
                        log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
            else:
                evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                if gt_attack:
                    log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
        except Exception as e:
            print(f"⚠️ HST scoring error: {e}")

def check_denylist_suppression(key5t, denylist):
    """
    Helper to check if anomaly should be flagged based on deny list logic.
    Returns: (should_flag, reason)
    
    Logic:
    1. ML detected anomaly
    2. Check if network functionality → suppress
    3. If private IP → flag as vulnerability
    4. If public IP in deny list → flag as attack
    5. If public IP not in deny list → suppress (likely FP)
    """
    if not denylist or not denylist.enabled:
        return True, "denylist_disabled"  # No deny list, trust ML
    
    src_ip, dst_ip, proto, src_port, dst_port = key5t
    result = denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
    
    # Step 2: Suppress network functionality
    if result['is_network_function']:
        return False, result['reason']
    
    # Step 3: Private IP = vulnerability (flag it)
    if result['is_private']:
        return True, result['reason']
    
    # Step 4: Public IP - check deny list
    if result['in_denylist']:
        return True, result['reason']
    
    # Step 5: Public IP not in deny list = likely false positive
    return False, result['reason']

def detect_classical(models, df_scaled, records, evaluator, attacker_mac, camera_mac, denylist=None):
    # OC-SVM
    if 'ocsvm' in models:
        m = models['ocsvm']
        preds = m.predict(df_scaled)
        scores = m.decision_function(df_scaled)
        for ((key5t, src_mac, dst_mac, features), pred, score) in zip(records, preds, scores):
            now = datetime.now()
            # Ground truth: traffic is attack if it involves attacker AND camera
            gt_attack = (
                (src_mac == attacker_mac and dst_mac == camera_mac) or
                (src_mac == camera_mac and dst_mac == attacker_mac)
            )
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score < threshold
            is_anom = score < OCSVM_THRESHOLD
            
            if is_anom:
                should_flag, reason = check_denylist_suppression(key5t, denylist)
                if should_flag:
                    evaluator.note_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                    log_anomaly(key5t, features, score, "OC-SVM")
                    if not gt_attack:
                        log_misclassified_benign(key5t, features, score, "OC-SVM")
                else:
                    evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                    if gt_attack:
                        log_missed_anomaly(key5t, features, score, "OC-SVM")
            else:
                evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                if gt_attack:
                    log_missed_anomaly(key5t, features, score, "OC-SVM")

    # LOF
    if 'lof' in models:
        m = models['lof']
        preds = m.predict(df_scaled)
        scores = m.decision_function(df_scaled)
        for ((key5t, src_mac, dst_mac, features), pred, score) in zip(records, preds, scores):
            now = datetime.now()
            gt_attack = (
                (src_mac == attacker_mac and dst_mac == camera_mac) or
                (src_mac == camera_mac and dst_mac == attacker_mac)
            )
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score < threshold
            is_anom = score < LOF_THRESHOLD
            
            if is_anom:
                should_flag, reason = check_denylist_suppression(key5t, denylist)
                if should_flag:
                    evaluator.note_alert(now, gt_attack=gt_attack, algorithm='LOF')
                    log_anomaly(key5t, features, score, "LOF")
                    if not gt_attack:
                        log_misclassified_benign(key5t, features, score, "LOF")
                else:
                    evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='LOF')
                    if gt_attack:
                        log_missed_anomaly(key5t, features, score, "LOF")
            else:
                evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='LOF')
                if gt_attack:
                    log_missed_anomaly(key5t, features, score, "LOF")

    # COPOD
    if 'copod' in models:
        m = models['copod']
        preds = m.predict(df_scaled)
        scores = m.decision_function(df_scaled)
        for ((key5t, src_mac, dst_mac, features), pred, score) in zip(records, preds, scores):
            now = datetime.now()
            gt_attack = (
                (src_mac == attacker_mac and dst_mac == camera_mac) or
                (src_mac == camera_mac and dst_mac == attacker_mac)
            )
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score > threshold
            is_anom = score > COPOD_THRESHOLD
            
            if is_anom:
                should_flag, reason = check_denylist_suppression(key5t, denylist)
                if should_flag:
                    evaluator.note_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                    log_anomaly(key5t, features, score, "COPOD")
                    if not gt_attack:
                        log_misclassified_benign(key5t, features, score, "COPOD")
                else:
                    evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                    if gt_attack:
                        log_missed_anomaly(key5t, features, score, "COPOD")
            else:
                evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                if gt_attack:
                    log_missed_anomaly(key5t, features, score, "COPOD")

# ================= Graceful Exit =================

class DetectorRPi:
    def __init__(self, args):
        self.args = args
        self.stopped = False
        self.evaluator = PerAlgorithmEvaluator()
        self.models, self.scaler = load_models(args.verbose)
        
        self.attacker_mac = args.attacker_mac.lower()
        self.camera_mac = args.camera_mac.lower()
        self.target_mac = args.target_mac.lower()
        
        # Initialize deny list
        self.denylist = AbuseIPDBDenyList(
            denylist_file=args.denylist_file,
            router_ip=args.router_ip
        )
        
        # Setup graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Setup periodic metrics saving
        self._setup_periodic_save(args.metrics_interval)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping gracefully...")
        self.stop()
        self.save_metrics()
        sys.exit(0)

    def _setup_periodic_save(self, interval):
        def periodic_save():
            while not self.stopped:
                time.sleep(interval)
                if not self.stopped:
                    print(f"Periodic metrics save at {datetime.now()}")
                    self.save_metrics()
        
        t = threading.Thread(target=periodic_save, daemon=True)
        t.start()

    def stop(self):
        self.stopped = True

    def save_metrics(self):
        self.evaluator.summarize(METRICS_CSV)

    def run(self):
        if self.args.macbook_ip:
            connect_to_macbook(self.args.macbook_ip, self.args.macbook_port)
        else:
            print("ℹ️ No DL server IP provided; only local detection + evaluation will run.")

        FLOW_STATS = {}
        packet_times = defaultdict(lambda: deque(maxlen=2))
        last_batch_time = time.time()

        def process_packet(pkt):
            if self.stopped:
                return
            
            if not (pkt.haslayer(Ether) and (pkt[Ether].src.lower() == self.target_mac or pkt[Ether].dst.lower() == self.target_mac)):
                return

            key5t = flow_5tuple(pkt)
            if key5t is None:
                return

            now = time.time()
            length = len(pkt)
            ip = pkt[IP]
            eth = pkt[Ether]
            src_mac = eth.src.lower()
            dst_mac = eth.dst.lower()

            stats = FLOW_STATS.get(key5t, {
                'first_seen': now, 'last_seen': now,
                'bytes': 0, 'packets': 0,
                'fwd_bytes': 0, 'fwd_packets': 0,
                'bwd_bytes': 0, 'bwd_packets': 0,
                'pkt_lens': [], 'iat': [],
                'src_mac': src_mac, 'dst_mac': dst_mac,
                'src_ip': ip.src, 'dst_ip': ip.dst
            })

            direction = 'fwd' if ip.src == key5t[0] else 'bwd'
            stats['last_seen'] = now
            stats['bytes'] += length
            stats['packets'] += 1
            stats[f'{direction}_bytes'] += length
            stats[f'{direction}_packets'] += 1
            stats['pkt_lens'].append(length)

            prev_time = packet_times[key5t][-1] if packet_times[key5t] else now
            stats['iat'].append(now - prev_time)
            packet_times[key5t].append(now)

            FLOW_STATS[key5t] = stats

        def packet_cb(pkt):
            process_packet(pkt)

            nonlocal last_batch_time
            now = time.time()
            if now - last_batch_time > BATCH_INTERVAL:
                # CRITICAL FIX: Send ALL flows to DL server first (before duration filtering)
                # This ensures DL server receives continuous data regardless of flow duration
                if macbook_sock:
                    all_records = extract_all_records(FLOW_STATS)
                    for (key5t, src_mac, dst_mac, features) in all_records:
                        send_to_macbook(list(key5t), src_mac, dst_mac, features)
                
                # Now extract filtered records for LOCAL RPi detection only
                # Short flows (<0.1s) are skipped for local detection but were already sent to DL
                records = extract_records(FLOW_STATS)
                if records:
                    df = pd.DataFrame([rec for (_, _, _, rec) in records], columns=FEATURES)

                    try:
                        X_scaled = self.scaler.transform(df.values)
                    except Exception:
                        self.scaler.fit(df.values)
                        X_scaled = self.scaler.transform(df.values)

                    detect_hst(self.models.get('hst'), records, self.evaluator, self.attacker_mac, self.camera_mac, self.denylist)
                    detect_classical(self.models, X_scaled, records, self.evaluator, self.attacker_mac, self.camera_mac, self.denylist)

                FLOW_STATS.clear()
                last_batch_time = now

        print(f"RPi detector on {self.args.interface}, MAC-based GT (attacker={self.attacker_mac}, camera={self.camera_mac})")
        try:
            sniff(iface=self.args.interface, prn=packet_cb, store=False, stop_filter=lambda x: self.stopped)
        finally:
            self.save_metrics()

# ================= Main =================

def main():
    args = get_args()
    detector = DetectorRPi(args)
    detector.run()

if __name__ == "__main__":
    main()

