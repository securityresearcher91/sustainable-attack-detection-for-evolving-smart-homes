import os
import cv2
import time
import socket
import threading
import subprocess

from flask import Flask, Response, request
# from onvif import ONVIFCamera  # Not needed for simulation

# --- CAMERA DEVICE ---
def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# ONVIF simulation endpoint
class SimulatedONVIFCamera:
    def __init__(self):
        self.hostname = socket.gethostname()
        self.ip = socket.gethostbyname(self.hostname)
        self.port = 8899
        self.username = "admin"
        self.password = "admin"

    def get_device_information(self):
        return {
            'Manufacturer': 'SimuCam Inc.',
            'Model': 'SC1000',
            'FirmwareVersion': '1.0.0',
            'SerialNumber': '123456789',
            'HardwareId': 'SC1000-Dev'
        }

def run_camera():
    app = Flask("camera")
    app.secret_key = "camera-key"
    VIDEO_PATH = "videos/sample_video.mp4"

    onvif_cam = SimulatedONVIFCamera()

    def generate_video_feed():
        while True:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                print(f"[CAMERA ERROR] Cannot open video: {VIDEO_PATH}", flush=True)
                time.sleep(5)
                continue
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            cap.release()

    @app.route('/video_feed')
    def video_feed():
        return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/admin')
    def admin():
        return '''<h2>Camera Admin</h2><p>Firmware Version: 1.0.0</p>'''

    @app.route('/upload_firmware', methods=['POST'])
    def upload_firmware():
        file = request.files['firmware']
        save_path = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        print(f"[CAMERA] Firmware uploaded: {save_path}", flush=True)
        return "Firmware uploaded."

    @app.route('/onvif/device_info')
    def device_info():
        return onvif_cam.get_device_information()

    try:
        port = get_free_port()
        public_url = None
        tunnel_proc = subprocess.Popen([
            "cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(tunnel_proc.stdout.readline, ''):
            print(line.strip(), flush=True)
            if "trycloudflare.com" in line:
                for word in line.split():
                    if word.startswith("https://") and "trycloudflare.com" in word:
                        public_url = word.strip()
                        break
                if public_url:
                    break

        if public_url:
            print(f"\n📹 Camera Feed: {public_url}/video_feed\n🛠 Admin Panel: {public_url}/admin\n🧠 ONVIF Device Info: {public_url}/onvif/device_info", flush=True)
        else:
            print("[CAMERA ERROR] Cloudflared tunnel did not return a public URL", flush=True)

        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except Exception as e:
        print(f"[CAMERA ERROR] {e}", flush=True)

if __name__ == '__main__':
    run_camera()

