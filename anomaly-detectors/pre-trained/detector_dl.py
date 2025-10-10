#!/usr/bin/env python3
import os
import socket
import json
import threading
import signal
import sys
from datetime import datetime
from collections import deque

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

MODEL_DIR = "models_dl"
VAE_FILE = os.path.join(MODEL_DIR, "vae.keras")
LSTM_FILE = os.path.join(MODEL_DIR, "lstm.keras")
GRU_FILE = os.path.join(MODEL_DIR, "gru.keras")

LOG_FILE = "dl_inference_anomalies.log"
METRICS_CSV = "metrics_dl.csv"

VAE_THRESH = 0.0844
LSTM_THRESH = 0.1001
GRU_THRESH = 0.0684

MISSED_ANOMALY_LOG = "missed_anomalies_dl.log"
MISCLASSIFIED_BENIGN_LOG = "misclassified_benigns_dl.log"

# ================= Args =================

def get_args():
    p = argparse.ArgumentParser(description="DL Inference Server + per-algorithm metrics + graceful exit")
    p.add_argument('--port', type=int, default=SERVER_PORT_DEFAULT, help='Server port')
    p.add_argument('--attacker-mac', required=True, help='Attacker laptop MAC')
    p.add_argument('--camera-mac', required=True, help='Camera/server MAC')
    p.add_argument('--metrics-interval', type=int, default=60, help='Metrics save interval (seconds)')
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
            print(f"{row['algorithm']:>8}: P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"FPR={row['fpr']:.3f} Latency={row['first_detection_latency_s']:.1f}s")
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
            return (src_mac == self.attacker_mac and dst_mac == self.camera_mac)
        return False

    def infer_batch(self, batch):
        try:
            feats = [b['features'] for b in batch]
            df = pd.DataFrame(feats)
            X = df[FEATURES].values

            mu = X.mean(axis=0)
            sd = X.std(axis=0) + 1e-8
            Xn = (X - mu) / sd
            Xseq = Xn.reshape(Xn.shape[0], 1, Xn.shape[1])

            now = datetime.now()
            gt_flags = [self._is_attack(b) for b in batch]
            for gt in gt_flags:
                self.evaluator.note_sample(now, gt_attack=gt)

            # VAE
            if self.vae is not None and len(Xn) > 0:
                recon = self.vae.predict(Xn, verbose=0)
                errs = np.mean((Xn - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    if err >= VAE_THRESH:
                        self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='VAE')
                        log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")
                    else:
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='VAE')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")
                        else:
                            log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "VAE")

            # LSTM
            if self.lstm is not None and len(Xseq) > 0:
                recon = self.lstm.predict(Xseq, verbose=0)
                errs = np.mean((Xn - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    if err >= LSTM_THRESH:
                        self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='LSTM')
                        log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")
                    else:
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='LSTM')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")
                        else:
                            log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "LSTM")

            # GRU
            if self.gru is not None and len(Xseq) > 0:
                recon = self.gru.predict(Xseq, verbose=0)
                errs = np.mean((Xn - recon) ** 2, axis=1)
                for (item, err, gt) in zip(batch, errs, gt_flags):
                    if err >= GRU_THRESH:
                        self.evaluator.note_alert(datetime.now(), gt_attack=gt, algorithm='GRU')
                        log_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")
                    else:
                        self.evaluator.note_no_alert(datetime.now(), gt_attack=gt, algorithm='GRU')
                        if gt:
                            log_missed_anomaly(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")
                        else:
                            log_misclassified_benign(item.get('flow_5tuple'), item.get('features'), float(err), "GRU")

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

