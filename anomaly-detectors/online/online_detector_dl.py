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
    return p.parse_args()

args = parse_args()
PORT = args.port
ATTACKER_MAC = args.attacker_mac.lower()
CAMERA_MAC = args.camera_mac.lower()
BATCH_TRAIN_EVERY = args.batch_train_every
DEBUG = args.debug  # Global debug flag

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

metrics = {"VAE": Counter(), "LSTM": Counter(), "GRU": Counter()}

# -------------------------
# Adaptive thresholds & score buffers
# -------------------------
# Configuration for adaptive thresholds
ADAPTIVE_WINDOW_SIZE = 500     # Number of scores to keep for percentile calculation
ADAPTIVE_PERCENTILE = 99       # Percentile to use for threshold calculation (higher = fewer alerts)
ADAPTIVE_UPDATE_FREQ = 50      # Update threshold every N samples
ADAPTIVE_SMOOTHING = 0.9       # Smoothing factor (higher = slower adaptation)

# Initial thresholds
INIT_THRESHOLDS = {"VAE": 0.1, "LSTM": 0.1, "GRU": 0.1}

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
def vae_score(x):
    x_tf = tf.convert_to_tensor(np.array(x, dtype=np.float32))
    recon = vae_model(x_tf, training=False)
    return tf.reduce_mean(tf.math.square(recon - x_tf), axis=1).numpy()

def lstm_score(x):
    x_arr = np.array(x, dtype=np.float32).reshape((-1, 1, INPUT_DIM))
    recon = lstm_model.predict(x_arr, verbose=0).reshape((-1, INPUT_DIM))
    return np.mean((recon - np.array(x, dtype=np.float32))**2, axis=1)

def gru_score(x):
    x_arr = np.array(x, dtype=np.float32).reshape((-1, 1, INPUT_DIM))
    pred = gru_model.predict(x_arr, verbose=0)
    return np.mean((pred - np.array(x, dtype=np.float32))**2, axis=1)

# -------------------------
# Training
# -------------------------
def train_dl_model(model_name):
    X = np.array(buffers[model_name], dtype=np.float32)
    if X.shape[0] < 10:
        return
    if model_name == "VAE":
        vae_model.compile(optimizer=vae_opt, loss="mse")
        vae_model.fit(X, X, epochs=3, batch_size=16, verbose=0)
    elif model_name == "LSTM":
        seq = X.reshape((-1, 1, X.shape[1]))
        lstm_model.compile(optimizer=lstm_opt, loss="mse")
        lstm_model.fit(seq, seq, epochs=3, batch_size=8, verbose=0)
    elif model_name == "GRU":
        seq = X.reshape((-1, 1, X.shape[1]))
        gru_model.compile(optimizer=gru_opt, loss="mse")
        gru_model.fit(seq, X, epochs=3, batch_size=8, verbose=0)

# -------------------------
# Processing entries
# -------------------------
def process_entry(entry):
    global initialized, INPUT_DIM, vae_model, lstm_model, gru_model

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

    # Scores
    v_score = float(vae_score([feats])[0]) if initialized else 0.0
    l_score = float(lstm_score([feats])[0]) if initialized else 0.0
    g_score = float(gru_score([feats])[0]) if initialized else 0.0
    scores = {"VAE": v_score, "LSTM": l_score, "GRU": g_score}
    gt_anomaly = (src_mac == ATTACKER_MAC and dst_mac == CAMERA_MAC)

    for model_name, score in scores.items():
        thr = adaptive_thresholds[model_name]
        is_anom = score > thr

        if is_anom and gt_anomaly:
            metrics[model_name]["TP"] += 1
            anomaly_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
        elif is_anom and not gt_anomaly:
            metrics[model_name]["FP"] += 1
            misclass_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
        elif (not is_anom) and gt_anomaly:
            metrics[model_name]["FN"] += 1
            missed_f.write(json.dumps({"timestamp": ts, "model": model_name, "score": score}) + "\n")
        else:
            metrics[model_name]["TN"] += 1

        # Add score to buffer for future threshold calculations
        score_buffers[model_name].append(score)
        samples_since_update[model_name] += 1
        
        # Update adaptive threshold if we've collected enough samples
        if samples_since_update[model_name] >= ADAPTIVE_UPDATE_FREQ and len(score_buffers[model_name]) >= ADAPTIVE_WINDOW_SIZE // 2:
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

        if not is_anom:
            buffers[model_name].append(feats)
            accepted_since_train[model_name] += 1
            if accepted_since_train[model_name] >= BATCH_TRAIN_EVERY:
                train_dl_model(model_name)
                accepted_since_train[model_name] = 0

# -------------------------
# Server
# -------------------------
def client_thread(conn):
    client_addr = conn.getpeername()
    info_log(f"Connection established with client {client_addr}")
    with conn:
        buf = b""
        message_count = 0
        try:
            while True:
                try:
                    data = conn.recv(4096)
                    if not data:
                        info_log(f"Client {client_addr} disconnected")
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            entry = json.loads(line.decode("utf-8"))
                            message_count += 1
                            
                            # Log every 50 messages or the first few
                            if message_count <= 5 or message_count % 50 == 0:
                                debug_log(f"Received message #{message_count} from {client_addr}")
                            
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
                                
                            if not initialized:
                                info_log(f"Models not yet initialized, buffering data")
                                pre_init_buffer.append(entry)
                            else:
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
                except ConnectionResetError:
                    info_log(f"Connection reset by client {client_addr}")
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
    info_log("=" * 40)
    
    # Try to load existing models and thresholds
    info_log("Checking for existing models and thresholds...")
    models_loaded = try_load_models()
    thresholds_loaded = load_existing_thresholds()
    
    if models_loaded:
        info_log("✅ Successfully loaded existing models")
    else:
        info_log("⚠️ No existing models found - will initialize on first data sample")
    
    if thresholds_loaded:
        info_log("✅ Successfully loaded existing adaptive thresholds")
    else:
        info_log(f"⚠️ Using default initial thresholds: {adaptive_thresholds}")
    
    # Start server thread
    server_thread = threading.Thread(target=server_loop, daemon=True)
    server_thread.start()
    
    # Start periodic save thread
    save_thread = threading.Thread(target=periodic_save, daemon=True)
    save_thread.start()
    
    info_log(f"\n✅ DL anomaly detector started. Listening on port {PORT}")
    info_log(f"📋 Expected connection format: {{'flow_key': '...', 'features': {...}, 'src_mac': '...', 'dst_mac': '...'}}")
    info_log(f"💻 Press Ctrl+C to stop")
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutdown requested. Calling cleanup function...")
        on_exit(signal.SIGINT, None)
