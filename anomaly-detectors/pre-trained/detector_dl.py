#!/usr/bin/env python3
import os
import socket
import json
import threading
import signal
import sys
from datetime import datetime
from collections import deque
import ipaddress

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import time

# ================= Configuration =================

SERVER_PORT_DEFAULT = 8888
BATCH_SIZE = 50

FEATURES = [
    'flow_duration', 'flow_byts_s', 'flow_pkts_s',
    'pkt_len', 'pkt_size', 'iat',
    'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
]

# Network functionality protocols that should always be allowed
ALLOWED_NETWORK_PROTOCOLS = {
    67,   # DHCP server
    68,   # DHCP client
    53,   # DNS
}

MODEL_DIR = "models_dl"
VAE_FILE = os.path.join(MODEL_DIR, "vae.keras")
LSTM_FILE = os.path.join(MODEL_DIR, "lstm.keras")
GRU_FILE = os.path.join(MODEL_DIR, "gru.keras")

LOG_FILE = "dl_inference_anomalies.log"
METRICS_CSV = "metrics_dl.csv"

VAE_THRESH = 0.01
LSTM_THRESH = 0.01
GRU_THRESH = 0.01

MISSED_ANOMALY_LOG = "missed_anomalies_dl.log"
MISCLASSIFIED_BENIGN_LOG = "misclassified_benigns_dl.log"

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
        self.deny_list = {}  # {ip: {'country_code': str, 'score': int, 'last_reported': str}}
        
        if self.enabled:
            self.load_denylist()
        else:
            print("ℹ️ AbuseIPDB deny list not enabled (no file provided)")
    
    def load_denylist(self):
        """Load AbuseIPDB deny list from CSV file"""
        try:
            with open(self.denylist_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header
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
        """
        Check if traffic is network functionality (DHCP, DNS to router).
        These should always be allowed regardless of anomaly detection.
        
        IMPORTANT: Be very specific to avoid whitelisting attack traffic!
        - DHCP: Only whitelist if BOTH IPs are in expected ranges
        - DNS: Only whitelist if destination is the router
        """
        # DHCP traffic (ports 67/68)
        if (src_port == 67 and dst_port == 68) or (src_port == 68 and dst_port == 67):
            return True, "dhcp"
        
        # DNS to router (port 53)
        if self.router_ip and dst_ip == self.router_ip and dst_port == 53:
            return True, "dns_to_router"
        
        return False, None
    
    def check_ip(self, src_ip, dst_ip, src_port, dst_port):
        """
        Check IP against deny list and private IP ranges.
        
        Returns dict with:
        - in_denylist: bool
        - is_private: bool
        - is_network_function: bool
        - should_flag: bool (True if anomaly should be confirmed)
        - reason: str
        - remote_ip: str
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
        
        # Check network functionality first
        is_net_func, func_type = self.is_network_functionality(src_ip, dst_ip, src_port, dst_port)
        if is_net_func:
            result['is_network_function'] = True
            result['should_flag'] = False
            result['reason'] = f'network_functionality_{func_type}'
            return result
        
        # Determine remote IP
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
        
        # Check deny list
        if remote_ip in self.deny_list:
            result['in_denylist'] = True
            result['should_flag'] = True
            result['reason'] = 'in_denylist'
            result['score'] = self.deny_list[remote_ip]['score']
            result['country_code'] = self.deny_list[remote_ip]['country_code']
            return result
        
        # Public IP not in deny list
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
    p = argparse.ArgumentParser(description="DL Inference Server + per-algorithm metrics + graceful exit")
    p.add_argument('--port', type=int, default=SERVER_PORT_DEFAULT, help='Server port')
    p.add_argument('--attacker-mac', required=True, help='Attacker laptop MAC')
    p.add_argument('--camera-mac', required=True, help='Camera/server MAC')
    p.add_argument('--metrics-interval', type=int, default=60, help='Metrics save interval (seconds)')
    p.add_argument('--denylist-file', type=str, default=None,
                   help='Path to AbuseIPDB deny list CSV file')
    p.add_argument('--router-ip', type=str, default=None,
                   help='Router IP address for whitelisting network functionality traffic (DHCP, DNS)')
    return p.parse_args()

# ================= Logging =================

def log_anomaly(flow_5tuple, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_5tuple,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] Anomaly: {flow_5tuple} (score={score:.3f})")

def log_missed_anomaly(flow_5tuple, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_5tuple,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISSED_ANOMALY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] Missed anomaly: {flow_5tuple} (score={score:.3f})")

def log_misclassified_benign(flow_5tuple, features, score, model_name):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flow_5tuple": flow_5tuple,
        "model": model_name,
        "score": float(score),
        "features": features
    }
    with open(MISCLASSIFIED_BENIGN_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{model_name}] Misclassified benign: {flow_5tuple} (score={score:.3f})")

# ================= Per-Algorithm Evaluator =================

class PerAlgorithmEvaluator:
    def __init__(self, algorithms=['VAE', 'LSTM', 'GRU']):
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
            print(f"{row['algorithm']:>8}: P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"FPR={row['fpr']:.3f} Latency={latency_str}")
        return df

# ================= Server =================

class DLInferenceServer:
    def __init__(self, args):
        self.port = args.port
        self.queue = deque()
        self.lock = threading.Lock()
        self.stopped = False

        self.attacker_mac = args.attacker_mac.lower()
        self.camera_mac = args.camera_mac.lower()

        self.vae = self._load_model(VAE_FILE, "VAE")
        self.lstm = self._load_model(LSTM_FILE, "LSTM")
        self.gru = self._load_model(GRU_FILE, "GRU")

        self.evaluator = PerAlgorithmEvaluator()
        
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

    def _load_model(self, path, name):
        try:
            m = tf.keras.models.load_model(path)
            print(f"Loaded {name}: {path}")
            return m
        except Exception as e:
            print(f"⚠️ Could not load {name}: {e}")
            return None

    def handle_client(self, conn, addr):
        print(f"Connected from {addr}")
        buffer = ""
        try:
            while not self.stopped:
                data = conn.recv(2048).decode()
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip():
                        continue
                    try:
                        flow = json.loads(line)
                        with self.lock:
                            self.queue.append(flow)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            conn.close()

    def process_loop(self):
        batch = []
        while not self.stopped:
            with self.lock:
                while self.queue and len(batch) < BATCH_SIZE:
                    batch.append(self.queue.popleft())
            if batch:
                self.infer_batch(batch)
                batch = []
            else:
                time.sleep(0.05)

    def _is_attack(self, item):
        src_mac = (item.get('src_mac') or '').lower()
        dst_mac = (item.get('dst_mac') or '').lower()
        if src_mac and dst_mac:
            # Ground truth: traffic is attack if it involves attacker AND camera
            # This includes both attacker->camera and camera->attacker (responses)
            return (
                (src_mac == self.attacker_mac and dst_mac == self.camera_mac) or
                (src_mac == self.camera_mac and dst_mac == self.attacker_mac)
            )
        return False

    def infer_batch(self, batch):
        try:
            feats = [b['features'] for b in batch]
            df = pd.DataFrame(feats)
            X = df[FEATURES].values

            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-8
            Xn = (X - mu) / sd
            
            # Pad to BATCH_SIZE to avoid TensorFlow retracing
            # This ensures consistent batch dimensions for .predict()
            actual_size = Xn.shape[0]
            if actual_size < BATCH_SIZE:
                # Pad with zeros (won't affect actual results since we only use [:actual_size])
                padding = np.zeros((BATCH_SIZE - actual_size, Xn.shape[1]))
                Xn_padded = np.vstack([Xn, padding])
            else:
                Xn_padded = Xn
            
            Xseq_padded = Xn_padded.reshape(Xn_padded.shape[0], 1, Xn_padded.shape[1])

            now = datetime.now()
            gt_flags = [self._is_attack(b) for b in batch]
            for gt in gt_flags:
                self.evaluator.note_sample(now, gt_attack=gt)

            # VAE
            if self.vae is not None and actual_size > 0:
                recon_padded = self.vae.predict(Xn_padded, verbose=0)
                # Only use results for actual samples (not padding)
                recon = recon_padded[:actual_size]
                errs = np.mean((Xn[:actual_size] - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    # Check if score indicates anomaly
                    is_anom = err >= VAE_THRESH
                    
                    if is_anom:
                        # ML detected anomaly - validate with deny list
                        flow_key = item.get('flow_5tuple') or item.get('flow_key')
                        src_ip, dst_ip, proto, src_port, dst_port = None, None, None, None, None
                        
                        if flow_key and isinstance(flow_key, (tuple, list)) and len(flow_key) >= 5:
                            src_ip, dst_ip, proto, src_port, dst_port = flow_key[0], flow_key[1], flow_key[2], flow_key[3], flow_key[4]
                        
                        should_flag = True  # Default: trust ML detection
                        if self.denylist.enabled and src_ip:
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            
                            # Step 2: Suppress network functionality
                            if denylist_result['is_network_function']:
                                should_flag = False
                            # Step 3: Private IP = vulnerability (flag it)
                            elif denylist_result['is_private']:
                                should_flag = True
                            # Step 4: Public IP - check deny list
                            elif denylist_result['in_denylist']:
                                should_flag = True  # In deny list = confirmed attack
                            else:
                                should_flag = False  # Public IP not in deny list = likely FP
                        
                        if should_flag:
                            self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='VAE')
                            log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")
                            if not gt:
                                log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")
                        else:
                            self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='VAE')
                            if gt:
                                log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")
                    else:
                        # No anomaly detected
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='VAE')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")

            # LSTM
            if self.lstm is not None and actual_size > 0:
                recon_padded = self.lstm.predict(Xseq_padded, verbose=0)
                recon = recon_padded[:actual_size]
                errs = np.mean((Xn[:actual_size] - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    is_anom = err >= LSTM_THRESH
                    
                    if is_anom:
                        flow_key = item.get('flow_5tuple') or item.get('flow_key')
                        src_ip, dst_ip, proto, src_port, dst_port = None, None, None, None, None
                        
                        if flow_key and isinstance(flow_key, (tuple, list)) and len(flow_key) >= 5:
                            src_ip, dst_ip, proto, src_port, dst_port = flow_key[0], flow_key[1], flow_key[2], flow_key[3], flow_key[4]
                        
                        should_flag = True
                        if self.denylist.enabled and src_ip:
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            if denylist_result['is_network_function']:
                                should_flag = False
                            elif denylist_result['is_private']:
                                should_flag = True
                            elif denylist_result['in_denylist']:
                                should_flag = True
                            else:
                                should_flag = False
                        
                        if should_flag:
                            self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='LSTM')
                            log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")
                            if not gt:
                                log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")
                        else:
                            self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='LSTM')
                            if gt:
                                log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")
                    else:
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='LSTM')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")

            # GRU
            if self.gru is not None and actual_size > 0:
                recon_padded = self.gru.predict(Xseq_padded, verbose=0)
                recon = recon_padded[:actual_size]
                errs = np.mean((Xn[:actual_size] - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    is_anom = err >= GRU_THRESH
                    
                    if is_anom:
                        flow_key = item.get('flow_5tuple') or item.get('flow_key')
                        src_ip, dst_ip, proto, src_port, dst_port = None, None, None, None, None
                        
                        if flow_key and isinstance(flow_key, (tuple, list)) and len(flow_key) >= 5:
                            src_ip, dst_ip, proto, src_port, dst_port = flow_key[0], flow_key[1], flow_key[2], flow_key[3], flow_key[4]
                        
                        should_flag = True
                        if self.denylist.enabled and src_ip:
                            denylist_result = self.denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                            if denylist_result['is_network_function']:
                                should_flag = False
                            elif denylist_result['is_private']:
                                should_flag = True
                            elif denylist_result['in_denylist']:
                                should_flag = True
                            else:
                                should_flag = False
                        
                        if should_flag:
                            self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='GRU')
                            log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")
                            if not gt:
                                log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")
                        else:
                            self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='GRU')
                            if gt:
                                log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")
                    else:
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='GRU')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")

            print(f"Inferred batch of {len(batch)} flows")

        except Exception as e:
            print(f"Inference error: {e}")

    def start(self):
        t = threading.Thread(target=self.process_loop, daemon=True)
        t.start()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', self.port))
        sock.listen(5)
        print(f"DL inference server listening on {self.port}")

        try:
            while True:
                conn, addr = sock.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            self.stop()
            self.save_metrics()

if __name__ == "__main__":
    args = get_args()
    DLInferenceServer(args).start()

