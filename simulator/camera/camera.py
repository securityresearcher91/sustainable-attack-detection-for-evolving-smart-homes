#!/usr/bin/env python3
"""
Simulated ONVIF-style smart camera with:
- Local MJPEG streaming (/video_feed) served by Flask
- Runtime-configurable cloud upload
- Optional Cloudflare Quick Tunnel
- Continuous "cloud recording" uplink
"""

import os
import cv2
import time
import socket
import threading
import subprocess
import base64
import itertools
import uuid
from datetime import datetime, timezone
from typing import Iterable
from threading import Lock

from flask import Flask, Response, request, jsonify

# Optional HTTP/2-capable client
try:
    import httpx  # enables HTTP/2 and better connection reuse
    _USE_HTTPX = True
except Exception:
    import requests  # fallback
    _USE_HTTPX = False

# ---------------------------
# Configuration (env overridable)
# ---------------------------

VIDEO_PATH = os.environ.get("CAMERA_VIDEO", "videos/sample_video.mp4")
HOST = os.environ.get("CAMERA_HOST", "0.0.0.0")
PORT = int(os.environ.get("CAMERA_PORT", "0"))  # 0 => auto-select

TUNNEL_ENABLED = os.environ.get("CAMERA_TUNNEL", "1") == "1"
CLOUDFLARED_BIN = os.environ.get("CLOUDFLARED_BIN", "cloudflared")

CLOUD_ENABLED = os.environ.get("CAMERA_CLOUD_ENABLED", "1") == "1"
CLOUD_BASE_URL = os.environ.get("CAMERA_CLOUD_BASE_URL", "https://rec.example-cloud.com")
CLOUD_INGEST_PATH = os.environ.get("CAMERA_CLOUD_INGEST_PATH", "/v1/ingest")
CLOUD_HEARTBEAT_PATH = os.environ.get("CAMERA_CLOUD_HEARTBEAT_PATH", "/v1/heartbeat")
DEVICE_ID = os.environ.get("CAMERA_DEVICE_ID", str(uuid.uuid4()))

TARGET_FPS = float(os.environ.get("CAMERA_TARGET_FPS", "3.0"))
FRAMES_PER_SEGMENT = int(os.environ.get("CAMERA_FRAMES_PER_SEGMENT", "6"))
UPLOAD_INTERVAL_S = float(os.environ.get("CAMERA_UPLOAD_INTERVAL_S", "2.0"))
JPEG_QUALITY = int(os.environ.get("CAMERA_JPEG_QUALITY", "60"))
MAX_BACKOFF_S = float(os.environ.get("CAMERA_MAX_BACKOFF_S", "60.0"))
HEARTBEAT_PERIOD_S = float(os.environ.get("CAMERA_HEARTBEAT_S", "45.0"))

EXTRA_HEADERS = {
    "X-Device-ID": DEVICE_ID,
    "User-Agent": "TestbedCam/1.0",
}

# ---------------------------
# Utilities
# ---------------------------

def get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port_ = s.getsockname()[1]
    s.close()
    return port_

def downscale_if_needed(frame, max_dim=640):
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = float(max_dim) / float(max(h, w))
    return cv2.resize(frame, (int(w * scale), int(h * scale)))

# ---------------------------
# ONVIF Device Model
# ---------------------------

class SimulatedONVIFCamera:
    def __init__(self):
        self.hostname = socket.gethostname()
        try:
            self.ip = socket.gethostbyname(self.hostname)
        except Exception:
            self.ip = "127.0.0.1"
        self.port = None
        self.username = "admin"
        self.password = "admin"

    def get_device_information(self):
        return {
            "Manufacturer": "SimuCam Inc.",
            "Model": "SC1000",
            "FirmwareVersion": "1.0.0",
            "SerialNumber": "123456789",
            "HardwareId": "SC1000-Dev"
        }

# ---------------------------
# Cloud Upload Management
# ---------------------------

class CloudUploader:
    def __init__(self, base_url: str, ingest_path: str, hb_path: str):
        self.ingest_url = base_url.rstrip("/") + ingest_path
        self.heartbeat_url = base_url.rstrip("/") + hb_path
        self.stop_evt = threading.Event()
        self.backoff = 1.0
        self.enabled = CLOUD_ENABLED
        self.lock = Lock()
        self.media_t = None
        self.hb_t = None
        
        if _USE_HTTPX:
            self.client = httpx.Client(http2=True, timeout=15.0)
        else:
            self.client = requests.Session()

    def toggle(self, enabled: bool) -> bool:
        """Toggle cloud upload state"""
        with self.lock:
            if enabled == self.enabled:
                return self.enabled
            
            self.enabled = enabled
            if enabled:
                if not (self.media_t and self.media_t.is_alive()):
                    self.start(VIDEO_PATH)
                    print("[cloud] Uplink enabled")
            else:
                self.stop()
                print("[cloud] Uplink disabled")
            return self.enabled

    def start(self, video_path: str):
        self.stop_evt.clear()
        self.media_t = threading.Thread(
            target=self._media_loop, 
            args=(video_path,), 
            name="uplink-media", 
            daemon=True
        )
        self.hb_t = threading.Thread(
            target=self._heartbeat_loop,
            name="uplink-heartbeat",
            daemon=True
        )
        self.media_t.start()
        self.hb_t.start()

    def stop(self):
        self.stop_evt.set()
        try:
            self.client.close()
        except Exception:
            pass

    def _media_loop(self, video_path: str):
        segs = segment_generator(video_path)
        last_push = 0.0
        while not self.stop_evt.is_set():
            if not self.enabled:
                time.sleep(1)
                continue
                
            seg = next(segs)
            now = time.time()
            since = now - last_push
            if since < UPLOAD_INTERVAL_S:
                time.sleep(UPLOAD_INTERVAL_S - since)
            last_push = time.time()
            
            try:
                r = self.client.post(self.ingest_url, json=seg, headers=EXTRA_HEADERS)
                _ = getattr(r, "status_code", 204)
                self.backoff = 1.0
            except Exception:
                time.sleep(self.backoff)
                self.backoff = min(self.backoff * 2.0, MAX_BACKOFF_S)

    def _heartbeat_loop(self):
        while not self.stop_evt.is_set():
            if not self.enabled:
                time.sleep(1)
                continue
                
            payload = {
                "device_id": DEVICE_ID,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "status": "ok"
            }
            try:
                self.client.post(self.heartbeat_url, json=payload, headers=EXTRA_HEADERS)
            except Exception:
                pass
            
            jitter = (uuid.uuid4().int % 500) / 1000.0
            time.sleep(HEARTBEAT_PERIOD_S + jitter)

# ---------------------------
# Frame Generation
# ---------------------------

def frame_source(video_path: str, target_fps: float) -> Iterable[bytes]:
    frame_interval = 1.0 / max(target_fps, 0.1)
    last = 0.0
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[uplink] Cannot open video: {video_path}")
            time.sleep(5)
            continue
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = downscale_if_needed(frame, max_dim=640)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ok:
                    continue
                now = time.time()
                sleep_s = (last + frame_interval) - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                last = time.time()
                yield buf.tobytes()
        finally:
            cap.release()

def segment_generator(video_path: str) -> Iterable[dict]:
    frames = frame_source(video_path, TARGET_FPS)
    batch = []
    seq = itertools.count()
    for jpg in frames:
        batch.append(base64.b64encode(jpg).decode("ascii"))
        if len(batch) >= FRAMES_PER_SEGMENT:
            seg = {
                "device_id": DEVICE_ID,
                "segment_id": next(seq),
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "fps": TARGET_FPS,
                "jpeg_quality": JPEG_QUALITY,
                "frames_b64": batch,
            }
            batch = []
            yield seg

def mjpeg_generator(video_path: str) -> Iterable[bytes]:
    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[CAMERA] Cannot open video: {video_path}")
            time.sleep(5)
            continue
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = downscale_if_needed(frame, max_dim=640)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if not ok:
                    continue
                jpg = buf.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                       jpg + b"\r\n")
        finally:
            cap.release()

# ---------------------------
# Flask App & Routes
# ---------------------------

app = Flask("camera")
app.secret_key = "camera-key"
onvif_cam = SimulatedONVIFCamera()

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(VIDEO_PATH),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/admin")
def admin():
    return "<h2>Camera Admin</h2><p>Firmware Version: 1.0.0</p>"

@app.route("/upload_firmware", methods=["POST"])
def upload_firmware():
    if "firmware" not in request.files:
        return "No file part", 400
    file = request.files["firmware"]
    os.makedirs("uploads", exist_ok=True)
    save_path = os.path.join("uploads", file.filename)
    file.save(save_path)
    print(f"[CAMERA] Firmware uploaded: {save_path}")
    return "Firmware uploaded."

@app.route("/onvif/device_info")
def device_info():
    info = onvif_cam.get_device_information()
    info.update({
        "Host": onvif_cam.hostname,
        "IP": onvif_cam.ip,
        "Port": onvif_cam.port
    })
    return jsonify(info)

@app.route("/cloud/toggle", methods=["POST"])
def toggle_cloud():
    try:
        data = request.get_json()
        enabled = bool(data.get("enabled", False))
        current_state = app.uploader.toggle(enabled)
        return jsonify({
            "success": True,
            "enabled": current_state
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ---------------------------
# Cloudflare Tunnel
# ---------------------------

def start_cloudflared(port: int) -> str:
    try:
        proc = subprocess.Popen(
            [CLOUDFLARED_BIN, "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
    except FileNotFoundError:
        print("[TUNNEL] cloudflared not found; skipping tunnel")
        return ""

    public_url = ""
    def reader():
        nonlocal public_url
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            line = line.strip()
            print(f"[cloudflared] {line}")
            if "trycloudflare.com" in line:
                for word in line.split():
                    if word.startswith("https://") and "trycloudflare.com" in word:
                        public_url = word.strip()
                        return

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    
    for _ in range(30):
        if public_url:
            break
        time.sleep(0.2)
    return public_url

# ---------------------------
# Main
# ---------------------------

def main():
    global PORT
    if PORT == 0:
        PORT = get_free_port()
    onvif_cam.port = PORT

    # Initialize uploader and make accessible to Flask
    app.uploader = CloudUploader(CLOUD_BASE_URL, CLOUD_INGEST_PATH, CLOUD_HEARTBEAT_PATH)
    
    # Start cloud uplink if initially enabled
    if CLOUD_ENABLED:
        app.uploader.start(VIDEO_PATH)
        print(f"[cloud] uplink -> {CLOUD_BASE_URL} as device {DEVICE_ID}")

    # Start Cloudflare tunnel if enabled
    public_url = ""
    if TUNNEL_ENABLED:
        public_url = start_cloudflared(PORT)
        if public_url:
            print(f"\n📹 Camera Feed: {public_url}/video_feed")
            print(f"🛠  ONVIF Device Info: {public_url}/onvif/device_info\n")
        else:
            print("[TUNNEL] No public URL obtained.")
    else:
        print(f"[TUNNEL] Disabled. Local feed: http://{HOST}:{PORT}/video_feed")

    # Run Flask app
    app.run(host=HOST, port=PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()

