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

def extract_records(flow_stats):
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
            print(f"{row['algorithm']:>15}: P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"FPR={row['fpr']:.3f} Latency={row['first_detection_latency_s']:.1f}s")
        return df

# ================= Detection =================

def detect_hst(hst, records, evaluator, attacker_mac, camera_mac):
    if not hst:
        return
    for key5t, src_mac, dst_mac, features in records:
        now = datetime.now()
        gt_attack = (src_mac == attacker_mac and dst_mac == camera_mac)
        evaluator.note_sample(now, gt_attack=gt_attack)
        score = None
        try:
            score = hst.score_one(features)
            # Mark as anomaly if score < threshold
            if score < HST_THRESHOLD:
                evaluator.note_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                log_anomaly(key5t, features, score, "HalfSpaceTrees")
                if not gt_attack:
                    log_misclassified_benign(key5t, features, score, "HalfSpaceTrees")
            else:
                evaluator.note_no_alert(now, gt_attack=gt_attack, algorithm='HalfSpaceTrees')
                if gt_attack:
                    log_missed_anomaly(key5t, features, score, "HalfSpaceTrees")
        except Exception as e:
            print(f"⚠️ HST scoring error: {e}")

def detect_classical(models, df_scaled, records, evaluator, attacker_mac, camera_mac):
    # OC-SVM
    if 'ocsvm' in models:
        m = models['ocsvm']
        preds = m.predict(df_scaled)
        scores = m.decision_function(df_scaled)
        for ((key5t, src_mac, dst_mac, features), pred, score) in zip(records, preds, scores):
            now = datetime.now()
            gt_attack = (src_mac == attacker_mac and dst_mac == camera_mac)
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score < threshold
            if score < OCSVM_THRESHOLD:
                evaluator.note_alert(now, gt_attack=gt_attack, algorithm='OC-SVM')
                log_anomaly(key5t, features, score, "OC-SVM")
                if not gt_attack:
                    log_misclassified_benign(key5t, features, score, "OC-SVM")
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
            gt_attack = (src_mac == attacker_mac and dst_mac == camera_mac)
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score < threshold
            if score < LOF_THRESHOLD:
                evaluator.note_alert(now, gt_attack=gt_attack, algorithm='LOF')
                log_anomaly(key5t, features, score, "LOF")
                if not gt_attack:
                    log_misclassified_benign(key5t, features, score, "LOF")
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
            gt_attack = (src_mac == attacker_mac and dst_mac == camera_mac)
            evaluator.note_sample(now, gt_attack=gt_attack)
            # Mark as anomaly if score > threshold
            if score > COPOD_THRESHOLD:
                evaluator.note_alert(now, gt_attack=gt_attack, algorithm='COPOD')
                log_anomaly(key5t, features, score, "COPOD")
                if not gt_attack:
                    log_misclassified_benign(key5t, features, score, "COPOD")
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
                records = extract_records(FLOW_STATS)
                if records:
                    df = pd.DataFrame([rec for (_, _, _, rec) in records], columns=FEATURES)

                    try:
                        X_scaled = self.scaler.transform(df.values)
                    except Exception:
                        self.scaler.fit(df.values)
                        X_scaled = self.scaler.transform(df.values)

                    detect_hst(self.models.get('hst'), records, self.evaluator, self.attacker_mac, self.camera_mac)
                    detect_classical(self.models, X_scaled, records, self.evaluator, self.attacker_mac, self.camera_mac)

                    if macbook_sock:
                        for (key5t, src_mac, dst_mac, features) in records:
                            send_to_macbook(list(key5t), src_mac, dst_mac, features)

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

