#!/usr/bin/env python3
"""
online_detector_dl.py

Deep learning anomaly detectors (VAE, LSTM, GRU) with adaptive thresholding.
Thresholds are updated online using the 99th percentile of benign scores.
"""

import argparse
import json
import csv
import socket
import threading
import time
import signal
import sys
import os
import traceback
from datetime import datetime
from collections import Counter, deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Online DL anomaly detector with adaptive thresholding")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--attacker-mac", required=True)
    p.add_argument("--camera-mac", required=True)
    p.add_argument("--batch-train-every", type=int, default=50)
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    p.add_argument("--adaptive-thresholds", action="store_true", 
                   help="Enable adaptive thresholds (default: use static thresholds)")
    p.add_argument("--denylist-file", type=str, default=None,
                   help="Path to AbuseIPDB deny list CSV file")
    p.add_argument("--router-ip", type=str, default=None,
                   help="Router IP address for whitelisting network functionality traffic")
    return p.parse_args()

args = parse_args()
PORT = args.port
ATTACKER_MAC = args.attacker_mac.lower()
CAMERA_MAC = args.camera_mac.lower()
BATCH_TRAIN_EVERY = args.batch_train_every
DEBUG = args.debug  # Global debug flag
USE_ADAPTIVE_THRESHOLDS = args.adaptive_thresholds

# -------------------------
# Static Thresholds (used when adaptive thresholds are disabled)
# -------------------------
STATIC_THRESHOLDS = {
    "VAE": 0.01,      # Recalibrated for normalized features (was 15.552058)
    "LSTM": 0.01,     # Recalibrated for normalized features (was 10.902546)
    "GRU": 0.01       # Recalibrated for normalized features (was 8.423924)
}
# NOTE: These thresholds work with normalized features (0-1 range).
# The old thresholds (8-15) were calibrated for unnormalized data.
# After warm-up, monitor performance and adjust if needed:
# - If FPR too high: increase thresholds (e.g., 0.05)
# - If recall too low: decrease thresholds (e.g., 0.005)

# Network functionality protocols that should always be allowed
ALLOWED_NETWORK_PROTOCOLS = {
    67,   # DHCP server
    68,   # DHCP client
    53,   # DNS
}

# -------------------------
# Logs & metrics
# -------------------------
ANOMALY_LOG = "anomalies_inference.log"
MISCLASS_LOG = "misclassified_benigns.log"
MISSED_LOG = "missed_anomalies.log"
METRICS_FILE = "metrics.csv"
anomaly_f = open(ANOMALY_LOG, "a")
misclass_f = open(MISCLASS_LOG, "a")
missed_f = open(MISSED_LOG, "a")

# Helper function for debug logging
def debug_log(message):
    """Print debug messages only when DEBUG flag is enabled"""
    if DEBUG:
        print(f"[DEBUG] {message}")
        
def info_log(message):
    """Print important info messages regardless of debug setting"""
    print(message)

# -------------------------
# AbuseIPDB Deny List Integration
# -------------------------
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
            info_log("AbuseIPDB Deny List: DISABLED (no file specified)")
    
    def load_denylist(self):
        """Load AbuseIPDB deny list from CSV file"""
        try:
            if not os.path.exists(self.denylist_file):
                info_log(f"⚠️  Deny list file not found: {self.denylist_file}")
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
            
            info_log(f"AbuseIPDB Deny List: ENABLED")
            info_log(f"  File: {self.denylist_file}")
            info_log(f"  Loaded {len(self.deny_list)} malicious IPs")
            if self.router_ip:
                info_log(f"  Router IP: {self.router_ip} (whitelisted for network functionality)")
        
        except Exception as e:
            info_log(f"⚠️  Error loading deny list: {e}")
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
        
        IMPORTANT: We check the REMOTE IP (the endpoint the camera is communicating with).
        
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
            info_log(f"🚨 Remote IP {remote_ip} found in deny list (score: {entry['score']}, country: {entry['country_code']})")
            return result
        
        # Remote IP is public but NOT in deny list
        # This is likely a legitimate cloud service (DNS, API, streaming, etc.)
        # Suppress the anomaly
        result['reason'] = 'public_ip_not_in_denylist'
        result['should_flag'] = False
        debug_log(f"✓ Remote IP {remote_ip} not in deny list - likely legitimate (suppressing anomaly)")
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

# Initialize deny list
denylist = AbuseIPDBDenyList(
    denylist_file=args.denylist_file,
    router_ip=args.router_ip
)

metrics = {"VAE": Counter(), "LSTM": Counter(), "GRU": Counter()}

# -------------------------
# Adaptive thresholds & score buffers
# -------------------------
# Configuration for adaptive thresholds
ADAPTIVE_WINDOW_SIZE = 500     # Number of scores to keep for percentile calculation
ADAPTIVE_PERCENTILE = 99       # Percentile to use for threshold calculation (higher = fewer alerts)
ADAPTIVE_UPDATE_FREQ = 50      # Update threshold every N samples
ADAPTIVE_SMOOTHING = 0.9       # Smoothing factor (higher = slower adaptation)

# Initial thresholds (use static thresholds as starting point)
INIT_THRESHOLDS = STATIC_THRESHOLDS.copy()

# Score buffers for benign samples (for threshold adaptation)
score_buffers = {m: deque(maxlen=ADAPTIVE_WINDOW_SIZE) for m in ["VAE", "LSTM", "GRU"]}

# Current adaptive thresholds
adaptive_thresholds = dict(INIT_THRESHOLDS)

# For tracking threshold updates
samples_since_update = {"VAE": 0, "LSTM": 0, "GRU": 0}
last_update_time = datetime.now()

# Log file for tracking threshold changes
THRESHOLD_LOG = "thresholds_adaptive_dl.log"
with open(THRESHOLD_LOG, "w") as f:
    f.write("timestamp,model,old_threshold,new_threshold\n")

# -------------------------
# Models (dynamic init after first sample)
# -------------------------
initialized = False
INPUT_DIM = None
vae_model = None
lstm_model = None
gru_model = None

vae_opt = optimizers.Adam(1e-3)
lstm_opt = optimizers.Adam(1e-3)
gru_opt = optimizers.Adam(1e-3)

buffers = {"VAE": [], "LSTM": [], "GRU": []}
accepted_since_train = {"VAE": 0, "LSTM": 0, "GRU": 0}
pre_init_buffer = []

# Connection monitoring
connection_established = False
first_message_received = False
connection_time = None
first_message_time = None

# Warm-up mode: Train on all data for the first N minutes regardless of anomaly score
WARMUP_DURATION_MINUTES = 5  # Train on everything for first 5 minutes to bootstrap models
warmup_start_time = None  # Will be set when models are initialized
warmup_complete = {"VAE": False, "LSTM": False, "GRU": False}

# Feature normalization statistics (computed during training)
feature_stats = {
    "VAE": {"min": None, "range": None},
    "LSTM": {"min": None, "range": None},
    "GRU": {"min": None, "range": None}
}

# Try to load existing thresholds
def load_existing_thresholds():
    global adaptive_thresholds
    try:
        if os.path.exists("models_dl/adaptive_thresholds.json"):
            with open("models_dl/adaptive_thresholds.json", "r") as f:
                saved_thresholds = json.load(f)
                # Update thresholds if they exist
                for model in ["VAE", "LSTM", "GRU"]:
                    if model in saved_thresholds:
                        adaptive_thresholds[model] = saved_thresholds[model]
                        print(f"Loaded saved threshold for {model}: {adaptive_thresholds[model]:.6f}")
                
                # Log loaded thresholds
                with open(THRESHOLD_LOG, "a") as log_f:
                    for model in ["VAE", "LSTM", "GRU"]:
                        log_f.write(f"{datetime.now().isoformat()},{model},{adaptive_thresholds[model]:.6f},{adaptive_thresholds[model]:.6f},LOADED\n")
                
                return True
    except Exception as e:
        print(f"Failed to load existing thresholds: {e}")
    return False

# Try to load existing models
def try_load_models():
    global vae_model, lstm_model, gru_model, INPUT_DIM, initialized
    
    try:
        if os.path.exists("models_dl/vae.keras") and os.path.exists("models_dl/lstm.keras") and os.path.exists("models_dl/gru.keras"):
            # Determine input dimension from one of the models
            temp_model = models.load_model("models_dl/vae.keras")
            INPUT_DIM = temp_model.input_shape[1]
            
            # Load all models
            vae_model = models.load_model("models_dl/vae.keras")
            lstm_model = models.load_model("models_dl/lstm.keras")
            gru_model = models.load_model("models_dl/gru.keras")
            
            print(f"Successfully loaded existing models with INPUT_DIM={INPUT_DIM}")
            initialized = True
            return True
    except Exception as e:
        print(f"Failed to load existing models: {e}")
        # Reset to uninitialized state
        vae_model = None
        lstm_model = None
        gru_model = None
        initialized = False
        INPUT_DIM = None
    return False

# -------------------------
# Model builders
# -------------------------
def build_vae(input_dim, latent_dim=4):
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(32, activation="relu")(inputs)
    mu = layers.Dense(latent_dim)(h)
    logvar = layers.Dense(latent_dim)(h)
    def sample(args):
        mu, logvar = args
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5*logvar) * eps
    z = layers.Lambda(sample)([mu, logvar])
    h2 = layers.Dense(32, activation="relu")(z)
    outputs = layers.Dense(input_dim)(h2)
    return models.Model(inputs, outputs)

def build_seq_autoencoder(input_dim, seq_len=1, hidden=16):
    inputs = layers.Input(shape=(seq_len, input_dim))
    x = layers.LSTM(hidden, return_sequences=False)(inputs)
    x = layers.RepeatVector(seq_len)(x)
    outputs = layers.LSTM(input_dim, return_sequences=True)(x)
    return models.Model(inputs, outputs)

def build_gru_forecaster(input_dim, seq_len=1, hidden=16):
    inputs = layers.Input(shape=(seq_len, input_dim))
    x = layers.GRU(hidden)(inputs)
    outputs = layers.Dense(input_dim)(x)
    return models.Model(inputs, outputs)

# -------------------------
# Score functions
# -------------------------
def normalize_features(x, model_name):
    """Normalize features using saved statistics from training"""
    # Always convert to numpy array first
    x_arr = np.array(x, dtype=np.float32)
    
    # Check for nan/inf in input features
    if not np.all(np.isfinite(x_arr)):
        # Replace nan/inf with 0 to avoid propagating them
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if feature_stats[model_name]["min"] is None:
        # Not trained yet, return unnormalized but as numpy array
        return x_arr
    
    x_min = feature_stats[model_name]["min"]
    x_range = feature_stats[model_name]["range"]
    
    # Compute normalized features
    x_norm = (x_arr - x_min) / x_range
    
    # Final safety check: replace any nan/inf that might have been introduced
    if not np.all(np.isfinite(x_norm)):
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return x_norm

def vae_score(x):
    x_norm = normalize_features(x, "VAE")
    x_tf = tf.convert_to_tensor(x_norm)
    recon = vae_model(x_tf, training=False)
    return tf.reduce_mean(tf.math.square(recon - x_tf), axis=1).numpy()

def lstm_score(x):
    x_norm = normalize_features(x, "LSTM")
    x_arr = x_norm.reshape((-1, 1, INPUT_DIM))
    recon = lstm_model.predict(x_arr, verbose=0).reshape((-1, INPUT_DIM))
    return np.mean((recon - x_norm)**2, axis=1)

def gru_score(x):
    x_norm = normalize_features(x, "GRU")
    x_arr = x_norm.reshape((-1, 1, INPUT_DIM))
    pred = gru_model.predict(x_arr, verbose=0)
    return np.mean((pred - x_norm)**2, axis=1)

# -------------------------
# Training
# -------------------------
def train_dl_model(model_name):
    global vae_model, lstm_model, gru_model, feature_stats
    
    X = np.array(buffers[model_name], dtype=np.float32)
    if X.shape[0] < 10:
        return
    
    # Check for nan/inf in training data
    if not np.all(np.isfinite(X)):
        info_log(f"⚠️ Warning: Training data for {model_name} contains nan/inf values, skipping training")
        return
    
    # Normalize data to prevent numerical instability
    # Use min-max normalization for stability
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0  # Avoid division by zero
    X_normalized = (X - X_min) / X_range
    
    # Save normalization stats for inference
    feature_stats[model_name]["min"] = X_min
    feature_stats[model_name]["range"] = X_range
    
    try:
        if model_name == "VAE":
            # Save model state before training in case of failure
            weights_backup = vae_model.get_weights()
            
            vae_model.compile(optimizer=vae_opt, loss="mse")
            history = vae_model.fit(X_normalized, X_normalized, epochs=3, batch_size=16, verbose=0)
            final_loss = history.history['loss'][-1]
            
            # Check if training produced valid loss
            if not np.isfinite(final_loss):
                info_log(f"⚠️ {model_name} training produced nan/inf loss, restoring previous weights")
                vae_model.set_weights(weights_backup)
                return
            
            info_log(f"[TRAIN] {model_name} trained on {X.shape[0]} samples, final loss: {final_loss:.4f}")
            
        elif model_name == "LSTM":
            weights_backup = lstm_model.get_weights()
            
            seq = X_normalized.reshape((-1, 1, X_normalized.shape[1]))
            lstm_model.compile(optimizer=lstm_opt, loss="mse")
            history = lstm_model.fit(seq, seq, epochs=3, batch_size=8, verbose=0)
            final_loss = history.history['loss'][-1]
            
            if not np.isfinite(final_loss):
                info_log(f"⚠️ {model_name} training produced nan/inf loss, restoring previous weights")
                lstm_model.set_weights(weights_backup)
                return
            
            info_log(f"[TRAIN] {model_name} trained on {X.shape[0]} samples, final loss: {final_loss:.4f}")
            
        elif model_name == "GRU":
            weights_backup = gru_model.get_weights()
            
            seq = X_normalized.reshape((-1, 1, X_normalized.shape[1]))
            gru_model.compile(optimizer=gru_opt, loss="mse")
            history = gru_model.fit(seq, X_normalized, epochs=3, batch_size=8, verbose=0)
            final_loss = history.history['loss'][-1]
            
            if not np.isfinite(final_loss):
                info_log(f"⚠️ {model_name} training produced nan/inf loss, restoring previous weights")
                gru_model.set_weights(weights_backup)
                return
            
            info_log(f"[TRAIN] {model_name} trained on {X.shape[0]} samples, final loss: {final_loss:.4f}")
            
    except Exception as e:
        info_log(f"⚠️ Error training {model_name}: {e}")

# -------------------------
# Processing entries
# -------------------------
def process_entry(entry):
    global initialized, INPUT_DIM, vae_model, lstm_model, gru_model, warmup_start_time

    # Add debugging to help diagnose issues
    if isinstance(entry, dict):
        debug_log(f"Processing entry with keys: {list(entry.keys())}")
    else:
        info_log(f"Warning: Entry is not a dictionary: {type(entry)}")
        return

    ts = entry.get("timestamp", datetime.now().isoformat())
    src_mac = entry.get("src_mac", "").lower()
    dst_mac = entry.get("dst_mac", "").lower()
    
    # Check for flow_key which might be used instead of flow_5tuple
    flow_key = entry.get("flow_key", entry.get("flow_5tuple", "unknown"))
    
    # Extract features - handle different feature formats
    features = entry.get("features", {})
    if not features:
        info_log("Warning: Entry has no features")
        return
        
    # Print some debug info about the first few entries
    if not hasattr(process_entry, "entry_count"):
        process_entry.entry_count = 0
    
    process_entry.entry_count += 1
    if process_entry.entry_count <= 5:
        debug_log(f"Entry #{process_entry.entry_count} details:")
        debug_log(f"  - src_mac: {src_mac}, dst_mac: {dst_mac}")
        debug_log(f"  - flow_key: {flow_key}")
        debug_log(f"  - features type: {type(features).__name__}")
        if isinstance(features, dict):
            debug_log(f"  - feature keys: {list(features.keys())}")
        elif isinstance(features, list):
            debug_log(f"  - feature length: {len(features)}")
        else:
            debug_log(f"  - unexpected feature type: {type(features).__name__}")

    # Initialize models if not done yet
    if not initialized:
        # Define the standard features we expect
        FEATURES = [
            'flow_duration', 'flow_byts_s', 'flow_pkts_s',
            'pkt_len', 'pkt_size', 'iat',
            'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
        ]
        
        # Set input dimension based on the feature vector
        if isinstance(features, dict):
            # We use the standard feature set length
            INPUT_DIM = len(FEATURES)
        else:
            # If it's already a list, use its length
            INPUT_DIM = len(features)
            
        info_log(f"Features type: {type(features).__name__}, setting INPUT_DIM={INPUT_DIM}")
        
        # Create models directory if it doesn't exist
        os.makedirs("models_dl", exist_ok=True)
        
        # Build new models with the correct input dimension
        info_log(f"Building new models with input dimension {INPUT_DIM}")
        vae_model = build_vae(INPUT_DIM)
        lstm_model = build_seq_autoencoder(INPUT_DIM, seq_len=1)
        gru_model = build_gru_forecaster(INPUT_DIM, seq_len=1)
        initialized = True
        
        # Start warm-up timer
        warmup_start_time = datetime.now()
        info_log(f"[WARMUP] Starting {WARMUP_DURATION_MINUTES}-minute warm-up period - training on all data")
        
        # Process any entries that came in before initialization
        for e in pre_init_buffer:
            process_entry(e)
        pre_init_buffer.clear()
        
        # Log the initial thresholds
        for model_name, threshold in adaptive_thresholds.items():
            with open(THRESHOLD_LOG, "a") as f:
                f.write(f"{datetime.now().isoformat()},{model_name},{threshold:.6f},{threshold:.6f},INITIAL\n")

    # Convert features to the format we need
    feature_list = []
    
    # Handle different feature formats
    if isinstance(features, dict):
        # Define the same feature order as in the RPi script
        FEATURES = [
            'flow_duration', 'flow_byts_s', 'flow_pkts_s',
            'pkt_len', 'pkt_size', 'iat',
            'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
        ]
        # Extract features in the correct order
        feature_list = [features.get(f, 0.0) for f in FEATURES]
    elif isinstance(features, list):
        # If already a list, use it directly
        feature_list = features
    else:
        print(f"Warning: Unsupported features type: {type(features)}")
        return
    
    # Ensure we have the right dimension
    feats = (feature_list + [0.0]*INPUT_DIM)[:INPUT_DIM]

    # Scores (with protection against nan/inf from untrained models)
    v_score = float(vae_score([feats])[0]) if initialized else 0.0
    l_score = float(lstm_score([feats])[0]) if initialized else 0.0
    g_score = float(gru_score([feats])[0]) if initialized else 0.0
    
    # Replace nan/inf with a very high value to treat as anomaly but allow training
    v_score = v_score if np.isfinite(v_score) else 1e6
    l_score = l_score if np.isfinite(l_score) else 1e6
    g_score = g_score if np.isfinite(g_score) else 1e6
    
    scores = {"VAE": v_score, "LSTM": l_score, "GRU": g_score}
    
    # Ground truth: Consider traffic as attack if it involves attacker and camera
    # This includes both attacker->camera and camera->attacker (responses)
    gt_anomaly = (
        (src_mac == ATTACKER_MAC and dst_mac == CAMERA_MAC) or
        (src_mac == CAMERA_MAC and dst_mac == ATTACKER_MAC)
    )
    
    # Extract flow key for deny list checking
    flow_key = entry.get("flow_key", entry.get("flow_5tuple", None))
    
    # Parse flow key if available (format: (src_ip, dst_ip, proto, src_port, dst_port))
    src_ip, dst_ip, proto, src_port, dst_port = None, None, None, None, None
    if flow_key and isinstance(flow_key, (tuple, list)) and len(flow_key) >= 5:
        src_ip, dst_ip, proto, src_port, dst_port = flow_key[0], flow_key[1], flow_key[2], flow_key[3], flow_key[4]

    for model_name, score in scores.items():
        # Choose threshold based on mode
        if USE_ADAPTIVE_THRESHOLDS:
            thr = adaptive_thresholds[model_name]
        else:
            thr = STATIC_THRESHOLDS[model_name]
            
        is_anom = score > thr

        # Process detection with deny list validation if enabled
        if is_anom:
            # ML detected anomaly - now validate with deny list if enabled
            if denylist.enabled and src_ip is not None:
                denylist_result = denylist.check_ip(src_ip, dst_ip, src_port, dst_port)
                
                # Whitelist network functionality traffic
                if denylist_result['is_network_function']:
                    # Suppressed as network functionality
                    if gt_anomaly:
                        metrics[model_name]["FN"] += 1
                        missed_f.write(json.dumps({
                            "timestamp": ts, 
                            "model": model_name, 
                            "score": score,
                            "reason": denylist_result['reason']
                        }) + "\n")
                    else:
                        metrics[model_name]["TN"] += 1
                    
                    # Add score to buffer for threshold calculation
                    score_buffers[model_name].append(score)
                    samples_since_update[model_name] += 1
                    
                    # CRITICAL FIX #9: Don't skip training during warm-up!
                    # During warm-up, we need to train on ALL data including network functionality
                    # to build initial model weights. After warm-up, we can skip these samples.
                    if warmup_start_time is not None:
                        elapsed_minutes = (datetime.now() - warmup_start_time).total_seconds() / 60.0
                        if elapsed_minutes >= WARMUP_DURATION_MINUTES:
                            # Warm-up complete - skip network functionality for training
                            continue
                        # else: Still in warm-up - don't skip, continue to training logic below
                    else:
                        # No warm-up timer (shouldn't happen) - skip for safety
                        continue
                
                # Check if anomaly should be confirmed or suppressed
                if denylist_result['should_flag']:
                    # Confirmed anomaly (in deny list or private-to-private)
                    if gt_anomaly:
                        metrics[model_name]["TP"] += 1
                    else:
                        metrics[model_name]["FP"] += 1
                        misclass_f.write(json.dumps({
                            "timestamp": ts,
                            "model": model_name,
                            "score": score,
                            "reason": denylist_result['reason']
                        }) + "\n")
                    
                    anomaly_f.write(json.dumps({
                        "timestamp": ts,
                        "model": model_name,
                        "score": score,
                        "reason": denylist_result['reason'],
                        "remote_ip": denylist_result.get('remote_ip')
                    }) + "\n")
                    
                    debug_log(f"✓ CONFIRMED anomaly ({model_name}) - {denylist_result['reason']}: score={score:.4f}")
                else:
                    # Suppressed (public IP not in deny list)
                    if gt_anomaly:
                        metrics[model_name]["FN"] += 1
                        missed_f.write(json.dumps({
                            "timestamp": ts,
                            "model": model_name,
                            "score": score,
                            "reason": denylist_result['reason']
                        }) + "\n")
                    else:
                        metrics[model_name]["TN"] += 1
                    
                    debug_log(f"✗ SUPPRESSED anomaly ({model_name}) - {denylist_result['reason']}: score={score:.4f}")
            else:
                # Deny list disabled or no flow info - use standard detection
                if gt_anomaly:
                    metrics[model_name]["TP"] += 1
                else:
                    metrics[model_name]["FP"] += 1
                    misclass_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
                
                anomaly_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
        else:
            # No anomaly detected
            if gt_anomaly:
                metrics[model_name]["FN"] += 1
                missed_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
            else:
                metrics[model_name]["TN"] += 1

        # Add score to buffer for adaptive threshold calculations (only if valid)
        # Filter out nan/inf values that can occur with untrained models
        if USE_ADAPTIVE_THRESHOLDS and np.isfinite(score):
            score_buffers[model_name].append(score)
            samples_since_update[model_name] += 1
        
        # Update adaptive threshold if we've collected enough samples (only in adaptive mode)
        if USE_ADAPTIVE_THRESHOLDS and samples_since_update[model_name] >= ADAPTIVE_UPDATE_FREQ and len(score_buffers[model_name]) >= ADAPTIVE_WINDOW_SIZE // 2:
            # Calculate new threshold from percentile of recent scores
            new_threshold = np.percentile(list(score_buffers[model_name]), ADAPTIVE_PERCENTILE)
            
            # Apply smoothing to avoid abrupt changes
            old_threshold = adaptive_thresholds[model_name]
            smoothed_threshold = old_threshold * ADAPTIVE_SMOOTHING + new_threshold * (1 - ADAPTIVE_SMOOTHING)
            
            # Update the threshold
            adaptive_thresholds[model_name] = smoothed_threshold
            
            # Log the threshold change
            with open(THRESHOLD_LOG, "a") as f:
                f.write(f"{datetime.now().isoformat()},{model_name},{old_threshold:.6f},{smoothed_threshold:.6f}\n")
                
            # Print info about the update
            info_log(f"[ADAPTIVE] {model_name} threshold updated: {old_threshold:.6f} -> {smoothed_threshold:.6f}")
            debug_log(f"[ADAPTIVE] Based on {len(score_buffers[model_name])} samples, {ADAPTIVE_PERCENTILE}th percentile: {new_threshold:.6f}")
            
            # Calculate FP/TP impact
            benign_scores = [s for s in score_buffers[model_name]]
            old_fp_rate = sum(1 for s in benign_scores if s > old_threshold) / len(benign_scores) if benign_scores else 0
            new_fp_rate = sum(1 for s in benign_scores if s > smoothed_threshold) / len(benign_scores) if benign_scores else 0
            debug_log(f"[ADAPTIVE] Estimated FP rate: {old_fp_rate:.2%} -> {new_fp_rate:.2%}")
            
            # Reset the counter
            samples_since_update[model_name] = 0

        # Training logic: 
        # - During warm-up (first WARMUP_DURATION_MINUTES), train on ALL samples
        # - After warm-up, only train on benign samples (not is_anom)
        should_train = False
        
        # Check if we're still in warm-up period
        if warmup_start_time is not None:
            elapsed_minutes = (datetime.now() - warmup_start_time).total_seconds() / 60.0
            
            if elapsed_minutes < WARMUP_DURATION_MINUTES:
                # Still in warm-up mode: train on everything
                should_train = True
            else:
                # Warm-up just finished
                if not warmup_complete[model_name]:
                    warmup_complete[model_name] = True
                    info_log(f"[WARMUP] {model_name} completed {WARMUP_DURATION_MINUTES}-minute warm-up phase")
                # Normal mode: only train on benign samples
                if not is_anom:
                    should_train = True
        elif not is_anom:
            # Fallback if warmup_start_time not set (shouldn't happen)
            should_train = True
        
        if should_train:
            buffers[model_name].append(feats)
            accepted_since_train[model_name] += 1
            if accepted_since_train[model_name] >= BATCH_TRAIN_EVERY:
                train_dl_model(model_name)
                accepted_since_train[model_name] = 0

# -------------------------
# Server
# -------------------------
def client_thread(conn):
    global connection_established, first_message_received, connection_time, first_message_time
    
    client_addr = conn.getpeername()
    info_log(f"Connection established with client {client_addr}")
    
    connection_established = True
    connection_time = datetime.now()
    
    # Set socket timeout to detect stalled connections
    conn.settimeout(30.0)  # 30 second timeout for receiving data
    
    with conn:
        buf = b""
        message_count = 0
        last_message_time = datetime.now()
        
        try:
            while True:
                try:
                    data = conn.recv(4096)
                    if not data:
                        info_log(f"Client {client_addr} disconnected (received {message_count} messages)")
                        break
                    
                    buf += data
                    last_message_time = datetime.now()
                    
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            entry = json.loads(line.decode("utf-8"))
                            message_count += 1
                            
                            # Mark first message received
                            if message_count == 1:
                                first_message_received = True
                                first_message_time = datetime.now()
                                elapsed = (first_message_time - connection_time).total_seconds()
                                info_log(f"✅ First message received after {elapsed:.1f}s")
                            
                            # Log every 50 messages or the first few
                            if message_count <= 5 or message_count % 50 == 0:
                                info_log(f"Received message #{message_count} from {client_addr}")
                            
                            # Log first message details to help diagnose issues
                            if message_count == 1:
                                info_log(f"First message keys: {list(entry.keys())}")
                                if 'features' in entry:
                                    info_log(f"Features present: type={type(entry['features']).__name__}")
                            
                            # Check if this is a ping message
                            if entry.get('type') == 'ping':
                                debug_log(f"Received ping from client {client_addr}")
                                # Send a ping response back
                                try:
                                    response = json.dumps({
                                        'type': 'ping_response',
                                        'timestamp': datetime.now().isoformat()
                                    }) + '\n'
                                    conn.send(response.encode())
                                except Exception as e:
                                    debug_log(f"Failed to send ping response: {e}")
                                continue
                            
                            # Always process entries - process_entry will handle initialization
                            try:
                                process_entry(entry)
                            except Exception as e:
                                info_log(f"Error processing entry: {e}")
                                debug_log(f"Entry structure: {entry.keys()}")
                                if 'features' in entry:
                                    debug_log(f"Features type: {type(entry['features']).__name__}")
                                    if isinstance(entry['features'], dict):
                                        debug_log(f"Feature keys: {entry['features'].keys()}")
                        except json.JSONDecodeError:
                            debug_log(f"Invalid JSON received: {line[:100].decode('utf-8', errors='replace')}...")
                        except Exception as e:
                            debug_log(f"Error processing message: {e}")
                            
                except socket.timeout:
                    # No data received for 30 seconds
                    elapsed = (datetime.now() - last_message_time).total_seconds()
                    
                    # Only warn if we're in critical phases
                    if message_count == 0:
                        info_log(f"⚠️ WARNING: No data received from {client_addr} for {elapsed:.0f} seconds")
                        info_log(f"   Check that RPi detector is running and sending data")
                        info_log(f"   Expected format: {{'features': {{...}}, 'src_mac': '...', 'dst_mac': '...'}}")
                    elif not initialized:
                        info_log(f"⚠️ WARNING: Still waiting for initialization data (received {message_count} messages)")
                        info_log(f"   Need at least one valid message to initialize models")
                    elif message_count < 50:
                        info_log(f"⚠️ WARNING: Low traffic - only {message_count} messages received in {(datetime.now() - connection_time).total_seconds():.0f}s")
                        info_log(f"   Models need at least 50 samples to train. Continue waiting...")
                    else:
                        # After training has started, low traffic is normal
                        debug_log(f"No data for {elapsed:.0f}s (received {message_count} messages so far)")
                    
                    # Continue waiting instead of disconnecting
                    last_message_time = datetime.now()
                    continue
                    
                except ConnectionResetError:
                    info_log(f"Connection reset by client {client_addr} (received {message_count} messages)")
                    break
                except Exception as e:
                    info_log(f"Socket error with client {client_addr}: {e}")
                    break
        except Exception as outer_e:
            info_log(f"Outer exception in client thread for {client_addr}: {outer_e}")
            if DEBUG:
                traceback.print_exc()
        finally:
            debug_log(f"Client thread for {client_addr} ending after processing {message_count} messages")

def server_loop():
    if DEBUG:
        # Print out all network interfaces to help with debugging
        debug_log("\n=== Available Network Interfaces ===")
        
        # Just use socket to get network info since netifaces might not be available
        hostname = socket.gethostname()
        try:
            debug_log(f"Hostname: {hostname}")
            primary_ip = socket.gethostbyname(hostname)
            debug_log(f"Primary IP: {primary_ip}")
            
            # Try to get all addresses
            try:
                all_ips = socket.gethostbyname_ex(hostname)
                if len(all_ips) > 2 and isinstance(all_ips[2], list):
                    debug_log(f"All IPs: {', '.join(all_ips[2])}")
            except Exception:
                pass
        except Exception as e:
            debug_log(f"Could not determine IP addresses: {e}")
        
        # Check if we can get interface information without netifaces
        try:
            import subprocess
            if sys.platform == "darwin":  # macOS
                result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                if result.returncode == 0:
                    # Simple parsing for demonstration - could be more sophisticated
                    lines = result.stdout.split('\n')
                    current_if = None
                    for line in lines:
                        if line and not line.startswith('\t') and not line.startswith(' '):
                            current_if = line.split(':')[0] if ':' in line else line.split()[0]
                        elif 'inet ' in line and current_if:
                            ip = line.strip().split('inet ')[1].split(' ')[0]
                            debug_log(f"Interface: {current_if}, IP: {ip}")
        except Exception as e:
            debug_log(f"Could not get detailed network info: {e}")
            
        debug_log("=====================================\n")
    
    # Create and bind the socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Set timeout to prevent indefinite blocking
    srv.settimeout(60)
    
    global PORT
    try:
        srv.bind(("0.0.0.0", PORT))
        srv.listen(4)
        info_log(f"Listening for features on 0.0.0.0:{PORT}")
    except OSError as e:
        info_log(f"Failed to bind to port {PORT}: {e}")
        # Try an alternate port if the default is in use
        alternate_port = PORT + 1
        info_log(f"Attempting to use alternate port {alternate_port}")
        try:
            srv.bind(("0.0.0.0", alternate_port))
            srv.listen(4)
            info_log(f"Listening for features on 0.0.0.0:{alternate_port}")
            PORT = alternate_port
        except Exception as e2:
            info_log(f"Critical error: Cannot bind to any port: {e2}")
            sys.exit(1)
    
    info_log("Waiting for RPi connection...")
    
    while True:
        try:
            conn, addr = srv.accept()
            info_log(f"New connection from {addr[0]}:{addr[1]}")
            threading.Thread(target=client_thread, args=(conn,), daemon=True).start()
        except socket.timeout:
            # This is normal with a timeout set, just continue
            debug_log("Listening for new connections...")
            continue
        except KeyboardInterrupt:
            info_log("Server shutdown requested")
            break
        except Exception as e:
            info_log(f"Error accepting connection: {e}")
            if DEBUG:
                traceback.print_exc()
            # Small pause before trying again
            time.sleep(1)

# -------------------------
# Exit
# -------------------------
def on_exit(sig, frame):
    print("Exiting: saving models, thresholds and metrics")
    try:
        if initialized:
            vae_model.save("models_dl/vae.keras")
            lstm_model.save("models_dl/lstm.keras")
            gru_model.save("models_dl/gru.keras")
            
            # Save thresholds
            with open("models_dl/adaptive_thresholds.json", "w") as f:
                json.dump({
                    "VAE": adaptive_thresholds["VAE"],
                    "LSTM": adaptive_thresholds["LSTM"],
                    "GRU": adaptive_thresholds["GRU"],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4)
    except Exception as e:
        print("Warning saving models and thresholds:", e)
        
    # Save final metrics
    with open(METRICS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model","TP","FP","FN","TN","precision","recall","f1","fpr"])
        writer.writeheader()
        for model_name, c in metrics.items():
            TP,FP,FN,TN = c["TP"], c["FP"], c["FN"], c["TN"]
            precision = TP/(TP+FP) if (TP+FP) else 0
            recall = TP/(TP+FN) if (TP+FN) else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
            fpr = FP/(FP+TN) if (FP+TN) else 0
            writer.writerow(dict(model=model_name, TP=TP, FP=FP, FN=FN, TN=TN,
                                 precision=precision, recall=recall, f1=f1, fpr=fpr))
    
    # Save final thresholds to the log
    with open(THRESHOLD_LOG, "a") as f:
        for model_name, threshold in adaptive_thresholds.items():
            f.write(f"{datetime.now().isoformat()},{model_name},{threshold:.6f},{threshold:.6f},FINAL\n")
            
    anomaly_f.close(); misclass_f.close(); missed_f.close()
    sys.exit(0)

signal.signal(signal.SIGINT, on_exit)

# Function to periodically save models and thresholds
def periodic_save():
    while True:
        time.sleep(300)  # Save every 5 minutes
        print(f"Periodic save at {datetime.now().isoformat()}")
        
        if not initialized:
            continue
            
        try:
            # Save models
            vae_model.save("models_dl/vae.keras")
            lstm_model.save("models_dl/lstm.keras")
            gru_model.save("models_dl/gru.keras")
            
            # Save thresholds
            with open("models_dl/adaptive_thresholds.json", "w") as f:
                json.dump({
                    "VAE": adaptive_thresholds["VAE"],
                    "LSTM": adaptive_thresholds["LSTM"],
                    "GRU": adaptive_thresholds["GRU"],
                    "timestamp": datetime.now().isoformat()
                }, f, indent=4)
                
            # Save current metrics
            with open(METRICS_FILE, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["model","TP","FP","FN","TN","precision","recall","f1","fpr"])
                writer.writeheader()
                for model_name, c in metrics.items():
                    TP,FP,FN,TN = c["TP"], c["FP"], c["FN"], c["TN"]
                    precision = TP/(TP+FP) if (TP+FP) else 0
                    recall = TP/(TP+FN) if (TP+FN) else 0
                    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
                    fpr = FP/(FP+TN) if (FP+TN) else 0
                    writer.writerow(dict(model=model_name, TP=TP, FP=FP, FN=FN, TN=TN,
                                     precision=precision, recall=recall, f1=f1, fpr=fpr))
        except Exception as e:
            print(f"Error during periodic save: {e}")

# Function to monitor connection and data flow
def connection_monitor():
    """Monitor connection status and warn if no data is received"""
    while True:
        time.sleep(60)  # Check every minute
        
        if not connection_established:
            info_log("⚠️ WARNING: No RPi connection established yet")
            info_log("   Make sure the RPi detector is running with --macbook-ip <this_machine_ip>")
            continue
        
        if not first_message_received and connection_time is not None:
            elapsed = (datetime.now() - connection_time).total_seconds()
            if elapsed > 60:  # Connected for >1 minute but no data
                info_log(f"⚠️ WARNING: Connected for {elapsed:.0f}s but no data received!")
                info_log("   Check that:")
                info_log("   1. RPi detector is capturing packets (check interface name)")
                info_log("   2. There is network traffic to capture")
                info_log("   3. RPi detector is successfully sending features")

if __name__ == "__main__":
    # Check for required dependencies
    missing_deps = []
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import tensorflow as tf
    except ImportError:
        missing_deps.append("tensorflow")
    
    if missing_deps:
        print("=" * 60)
        print("ERROR: Missing required dependencies:", ", ".join(missing_deps))
        print("Please install the missing packages with:")
        print(f"pip install {' '.join(missing_deps)}")
        print("=" * 60)
        sys.exit(1)
    
    # Create models directory if it doesn't exist
    os.makedirs("models_dl", exist_ok=True)
    
    info_log("\n=== Starting DL Anomaly Detector Server ===")
    info_log(f"Port: {PORT}")
    info_log(f"Attacker MAC: {ATTACKER_MAC}")
    info_log(f"Camera MAC: {CAMERA_MAC}")
    info_log(f"Debug mode: {'Enabled' if DEBUG else 'Disabled'}")
    info_log(f"Threshold mode: {'Adaptive' if USE_ADAPTIVE_THRESHOLDS else 'Static'}")
    if not USE_ADAPTIVE_THRESHOLDS:
        info_log(f"  Static thresholds: {STATIC_THRESHOLDS}")
    info_log(f"Deny list: {'Enabled' if denylist.enabled else 'Disabled'}")
    if denylist.enabled:
        info_log(f"  Loaded {len(denylist.deny_list)} malicious IPs")
    info_log("=" * 40)
    
    # Models will be trained from scratch (not loaded from disk)
    info_log("Models will be trained from scratch on incoming data")
    info_log("⚠️ Models will be initialized on first data sample")
    info_log("⚠️ Ensure RPi detector is running and connected!")
    info_log("")
    
    if USE_ADAPTIVE_THRESHOLDS:
        thresholds_loaded = load_existing_thresholds()
        if thresholds_loaded:
            info_log("✅ Successfully loaded existing adaptive thresholds")
        else:
            info_log(f"⚠️ Using initial adaptive thresholds (from static): {adaptive_thresholds}")
    else:
        info_log(f"⚠️ Adaptive thresholds disabled - using static thresholds: {STATIC_THRESHOLDS}")
    
    # Start server thread
    server_thread = threading.Thread(target=server_loop, daemon=True)
    server_thread.start()
    
    # Start periodic save thread
    save_thread = threading.Thread(target=periodic_save, daemon=True)
    save_thread.start()
    
    # Start connection monitoring thread
    monitor_thread = threading.Thread(target=connection_monitor, daemon=True)
    monitor_thread.start()
    
    info_log(f"\n✅ DL anomaly detector started. Listening on port {PORT}")
    info_log(f"📋 Expected connection format: {{'flow_key': '...', 'features': {...}, 'src_mac': '...', 'dst_mac': '...'}}")
    info_log(f"💻 Press Ctrl+C to stop")
    info_log(f"📊 Connection monitoring active - will warn if no data received")
    info_log("")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown requested. Calling cleanup function...")
        on_exit(signal.SIGINT, None)
