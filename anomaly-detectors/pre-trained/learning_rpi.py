#!/usr/bin/env python3
import os
import time
import json
import socket
import argparse
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scapy.all import sniff, IP

from river import anomaly
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.copod import COPOD
from sklearn.preprocessing import StandardScaler

import joblib
import pickle

# ================= Configuration =================

FEATURES = [
    'flow_duration', 'flow_byts_s', 'flow_pkts_s',
    'pkt_len', 'pkt_size', 'iat',
    'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
]

# Streaming/learning cadence
INTERFACE_DEFAULT = "wlan0"
BATCH_INTERVAL = 30             # seconds between batch processing windows
MIN_FLOW_DURATION = 0.1         # ignore ultra-short flows

# Batch learning policy (consistent across OC-SVM, LOF, COPOD)
BATCH_MIN_SAMPLES = 50          # require at least this many samples to fit batch models
K_TARGET = 20                   # LOF target neighborhood size
K_MIN = 5                       # LOF minimum k to avoid very noisy neighborhoods

# Model persistence
MODEL_DIR = "models_rpi"
HST_FILE = os.path.join(MODEL_DIR, "halfspacetrees.pkl")
OCSVM_FILE = os.path.join(MODEL_DIR, "ocsvm.joblib")
LOF_FILE = os.path.join(MODEL_DIR, "lof.joblib")
COPOD_FILE = os.path.join(MODEL_DIR, "copod.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= CLI =================

def get_args():
    p = argparse.ArgumentParser(description='RPi Learn-only Network Model Trainer (consistent batch learning)')
    p.add_argument('--target-mac', required=True, help='Target MAC address to monitor (required)')
    p.add_argument('--macbook-ip', help='MacBook IP address for deep learning server (optional)')
    p.add_argument('--macbook-port', type=int, default=8888, help='MacBook port (default: 8888)')
    p.add_argument('--interface', default=INTERFACE_DEFAULT, help=f'Network interface (default: {INTERFACE_DEFAULT})')
    p.add_argument('--batch-interval', type=int, default=BATCH_INTERVAL, help='Batch interval seconds')
    p.add_argument('--batch-min-samples', type=int, default=BATCH_MIN_SAMPLES, help='Minimum samples to fit batch models')
    p.add_argument('--lof-k-target', type=int, default=K_TARGET, help='LOF target n_neighbors')
    p.add_argument('--lof-k-min', type=int, default=K_MIN, help='LOF minimum n_neighbors')
    return p.parse_args()

args = get_args()
TARGET_MAC = args.target_mac
MACBOOK_IP = args.macbook_ip
MACBOOK_PORT = args.macbook_port
INTERFACE = args.interface
BATCH_INTERVAL = args.batch_interval
BATCH_MIN_SAMPLES = args.batch_min_samples
K_TARGET = args.lof_k_target
K_MIN = args.lof_k_min

# ================= Runtime state =================

FLOW_STATS = {}
packet_times = defaultdict(lambda: deque(maxlen=2))

# Online streaming learner
river_model = anomaly.HalfSpaceTrees(seed=42, window_size=30)

# Batch buffer and scaler
X_batch = []
scaler = StandardScaler()

# MacBook streaming socket
macbook_socket = None

# ================= Networking to MacBook (feature forward) =================

def connect_to_macbook():
    """Connect to MacBook server for feature streaming (optional)."""
    global macbook_socket
    if not MACBOOK_IP:
        return False
    try:
        macbook_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        macbook_socket.connect((MACBOOK_IP, MACBOOK_PORT))
        print(f"✅ Connected to MacBook at {MACBOOK_IP}:{MACBOOK_PORT}")
        return True
    except Exception as e:
        print(f"⚠️ MacBook connection failed: {e}")
        macbook_socket = None
        return False

def send_to_macbook(flow_key, features):
    """Send feature data to MacBook learn-only DL server."""
    global macbook_socket
    if not macbook_socket:
        return
    try:
        data = {
            'flow_key': flow_key,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        message = json.dumps(data) + '\n'
        macbook_socket.send(message.encode())
    except Exception as e:
        print(f"⚠️ Error sending to MacBook: {e}")
        connect_to_macbook()

# ================= Flow handling =================

def flow_key(pkt):
    if IP in pkt:
        ip = pkt[IP]
        return (ip.src, ip.dst, ip.proto)
    return None

def process_packet(pkt):
    key = flow_key(pkt)
    if key is None:
        return
    now = time.time()
    length = len(pkt)
    ip = pkt[IP]
    stats = FLOW_STATS.get(key, {
        'first_seen': now, 'last_seen': now,
        'bytes': 0, 'packets': 0,
        'fwd_bytes': 0, 'fwd_packets': 0,
        'bwd_bytes': 0, 'bwd_packets': 0,
        'pkt_lens': [], 'iat': []
    })
    direction = 'fwd' if ip.src == key[0] else 'bwd'
    stats['last_seen'] = now
    stats['bytes'] += length
    stats['packets'] += 1
    stats[f'{direction}_bytes'] += length
    stats[f'{direction}_packets'] += 1
    stats['pkt_lens'].append(length)

    prev_time = packet_times[key][-1] if packet_times[key] else now
    stats['iat'].append(now - prev_time)
    packet_times[key].append(now)

    FLOW_STATS[key] = stats

def extract_features():
    records = []
    for k, v in FLOW_STATS.items():
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
        records.append((k, record))
    return records

# ================= Learning (no detection) =================

def learn_streaming(records):
    """Online learn with HalfSpaceTrees and save; forward features to DL server."""
    for _, features in records:
        river_model.learn_one(features)

    # Save HST
    try:
        with open(HST_FILE, "wb") as f:
            pickle.dump(river_model, f)
    except Exception as e:
        print(f"⚠️ Failed to save HalfSpaceTrees: {e}")

    # Forward features to MacBook for DL learn-only
    for key, features in records:
        send_to_macbook(key, features)

def learn_batch(records):
    """Fit OC-SVM, LOF, COPOD on a consistent, sufficient batch; save models + scaler."""
    global X_batch
    X_batch.extend([features for _, features in records])

    n_samples = len(X_batch)
    if n_samples < BATCH_MIN_SAMPLES:
        print(f"Batch models: waiting for more samples ({n_samples}/{BATCH_MIN_SAMPLES})")
        return

    df = pd.DataFrame(X_batch)
    X = df[FEATURES].values

    # Fit scaler on the full batch buffer (consistent normalization)
    X_scaled = scaler.fit_transform(X)

    # Persist scaler for inference parity
    try:
        joblib.dump(scaler, SCALER_FILE)
    except Exception as e:
        print(f"⚠️ Failed to save scaler: {e}")

    # Fit OC-SVM on scaled batch
    try:
        ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
        ocsvm.fit(X_scaled)
        joblib.dump(ocsvm, OCSVM_FILE)
        print(f"Saved OC-SVM: {OCSVM_FILE} on {n_samples} samples")
    except Exception as e:
        print(f"⚠️ OC-SVM fit/save error: {e}")

    # Fit LOF with safe k based on available samples
    try:
        # n_neighbors cannot exceed n_samples - 1
        k_safe = min(K_TARGET, max(K_MIN, n_samples - 1))
        lof = LocalOutlierFactor(n_neighbors=k_safe, novelty=True)
        lof.fit(X_scaled)
        joblib.dump(lof, LOF_FILE)
        print(f"Saved LOF (k={k_safe}): {LOF_FILE} on {n_samples} samples")
    except Exception as e:
        print(f"⚠️ LOF fit/save error: {e}")

    # Fit COPOD on scaled batch
    try:
        copod = COPOD()
        copod.fit(X_scaled)
        joblib.dump(copod, COPOD_FILE)
        print(f"Saved COPOD: {COPOD_FILE} on {n_samples} samples")
    except Exception as e:
        print(f"⚠️ COPOD fit/save error: {e}")

    # Clear buffer after a successful consistent batch fit
    X_batch.clear()

# ================= Main =================

def run():
    print(f"Learn-only trainer (consistent batch) on {INTERFACE} for MAC {TARGET_MAC}")
    if MACBOOK_IP:
        connect_to_macbook()
    else:
        print("ℹ️ No MacBook IP specified. Use --macbook-ip to stream features for DL learning.")

    last_batch_time = time.time()

    def packet_callback(pkt):
        if not (pkt.haslayer("Ether") and (pkt.src == TARGET_MAC or pkt.dst == TARGET_MAC)):
            return
        process_packet(pkt)

        nonlocal last_batch_time
        now = time.time()
        if now - last_batch_time > BATCH_INTERVAL:
            records = extract_features()
            if records:
                learn_streaming(records)
                learn_batch(records)
            FLOW_STATS.clear()
            last_batch_time = now

    bpf_filter = f"ether host {TARGET_MAC}"
    sniff(iface=INTERFACE, prn=packet_callback, store=False, filter=bpf_filter)

if __name__ == "__main__":
    run()

