#!/usr/bin/env python3
import os
import socket
import json
import threading
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from collections import deque
import queue

# ================= Configuration =================

SERVER_PORT = 8888
BATCH_SIZE = 50

FEATURES = [
    'flow_duration', 'flow_byts_s', 'flow_pkts_s',
    'pkt_len', 'pkt_size', 'iat',
    'tot_bwd_pkts', 'tot_fwd_pkts', 'totlen_bwd_pkts', 'totlen_fwd_pkts'
]

MODEL_DIR = "models_dl"
os.makedirs(MODEL_DIR, exist_ok=True)
VAE_FILE = os.path.join(MODEL_DIR, "vae.keras")
LSTM_FILE = os.path.join(MODEL_DIR, "lstm.keras")
GRU_FILE = os.path.join(MODEL_DIR, "gru.keras")

class DeepLearningServer:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.feature_buffer = deque(maxlen=1000)

        # Initialize deep learning models
        self.vae_model = self.build_vae()
        self.lstm_model = self.build_lstm()
        self.gru_model = self.build_gru()

        print("Deep learning models initialized (learn-only)")

    def build_vae(self):
        """Build VAE model"""
        encoder_input = tf.keras.Input(shape=(len(FEATURES),))
        encoded = tf.keras.layers.Dense(16, activation='relu')(encoder_input)
        latent = tf.keras.layers.Dense(4)(encoded)
        decoded = tf.keras.layers.Dense(16, activation='relu')(latent)
        decoder_output = tf.keras.layers.Dense(len(FEATURES))(decoded)
        model = tf.keras.Model(encoder_input, decoder_output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_lstm(self):
        """Build LSTM autoencoder"""
        inputs = tf.keras.Input(shape=(1, len(FEATURES)))
        lstm = tf.keras.layers.LSTM(16, return_sequences=True)(inputs)
        outputs = tf.keras.layers.Dense(len(FEATURES))(lstm)
        outputs = tf.keras.layers.Flatten()(outputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def build_gru(self):
        """Build GRU model"""
        inputs = tf.keras.Input(shape=(1, len(FEATURES)))
        gru = tf.keras.layers.GRU(16)(inputs)
        outputs = tf.keras.layers.Dense(len(FEATURES))(gru)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def handle_client(self, conn, addr):
        """Handle incoming data from Raspberry Pi"""
        print(f"Connected to RPi at {addr}")
        buffer = ""
        try:
            while True:
                data = conn.recv(1024).decode()
                if not data:
                    break
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            flow_data = json.loads(line)
                            self.data_queue.put(flow_data)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            conn.close()

    def process_deep_learning(self):
        """Process data with deep learning models (learn-only)"""
        batch_data = []
        while True:
            try:
                flow_data = self.data_queue.get(timeout=1)
                batch_data.append(flow_data)
                if len(batch_data) >= BATCH_SIZE:
                    self.update_models(batch_data)
                    batch_data = []
            except queue.Empty:
                if batch_data:
                    self.update_models(batch_data)
                    batch_data = []

    def update_models(self, batch_data):
        """Learn only: fit one epoch and save models"""
        try:
            features_list = [data['features'] for data in batch_data]
            df = pd.DataFrame(features_list)
            X = df[FEATURES].values

            # Per-batch normalization (consistent with prior design)
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-8
            X_normalized = (X - X_mean) / X_std

            # VAE learn
            self.vae_model.fit(X_normalized, X_normalized, epochs=1, verbose=0)
            try:
                self.vae_model.save(VAE_FILE, overwrite=True)
            except Exception as e:
                print(f"⚠️ Failed to save VAE: {e}")

            # LSTM learn
            X_lstm = X_normalized.reshape(X_normalized.shape[0], 1, -1)
            self.lstm_model.fit(X_lstm, X_normalized, epochs=1, verbose=0)
            try:
                self.lstm_model.save(LSTM_FILE, overwrite=True)
            except Exception as e:
                print(f"⚠️ Failed to save LSTM: {e}")

            # GRU learn
            self.gru_model.fit(X_lstm, X_normalized, epochs=1, verbose=0)
            try:
                self.gru_model.save(GRU_FILE, overwrite=True)
            except Exception as e:
                print(f"⚠️ Failed to save GRU: {e}")

            print(f"Updated DL models with batch of {len(batch_data)} flows")
        except Exception as e:
            print(f"Error in DL update: {e}")

    def start_server(self):
        """Start TCP server"""
        # Start DL processing thread
        dl_thread = threading.Thread(target=self.process_deep_learning, daemon=True)
        dl_thread.start()

        # Start TCP server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', SERVER_PORT))
        sock.listen(5)
        print(f"DL learn-only server listening on port {SERVER_PORT}")

        while True:
            conn, addr = sock.accept()
            client_thread = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            client_thread.start()

if __name__ == "__main__":
    server = DeepLearningServer()
    server.start_server()

