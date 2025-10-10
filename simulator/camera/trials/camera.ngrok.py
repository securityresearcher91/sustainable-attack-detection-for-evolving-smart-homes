import os
import cv2
import time
import socket
import threading
from flask import Flask, Response, request, render_template_string, redirect, url_for
from pyngrok import conf, ngrok

app = Flask(__name__)

# --- Configuration ---
VIDEO_PATH = "videos/sample_video.mp4"
DEFAULT_USERNAME = "admin"
DEFAULT_PASSWORD = "admin"

# --- Insecure HTML login form ---
login_form = '''
<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body>
<h2>Login (Default: admin/admin)</h2>
<form method="post">
    Username: <input type="text" name="username"><br>
    Password: <input type="password" name="password"><br>
    <input type="submit" value="Login">
</form>
</body>
</html>
'''

# --- Insecure Admin Dashboard ---
admin_panel = '''
<!DOCTYPE html>
<html>
<head><title>Admin Panel</title></head>
<body>
<h2>Welcome, admin!</h2>
<p>Device status: <b>Recording</b></p>
<p>Firmware Version: v1.0.0</p>
<form method="post" action="/upload_firmware" enctype="multipart/form-data">
    Upload Firmware:<br>
    <input type="file" name="firmware">
    <input type="submit" value="Upload">
</form>
</body>
</html>
'''

# --- Vulnerable MJPEG Stream (no auth required) ---
def generate_video_feed():
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

@app.route('/')
def home():
    return redirect(url_for('login'))

# --- Simulates CVE-2017-8225 (Default creds) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            return redirect(url_for('admin'))
        else:
            return "Login failed", 403
    return render_template_string(login_form)

# --- Simulates CVE-2018-9995 (Unauthenticated access) ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Simulates CVE-2021-32934 (Malware upload) ---
@app.route('/upload_firmware', methods=['POST'])
def upload_firmware():
    file = request.files['firmware']
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)
    print(f"[!] Firmware uploaded: {save_path}")
    return "Firmware uploaded."

# --- Simulates CVE-2017-17106 (Admin panel exposure) ---
@app.route('/admin')
def admin():
    return render_template_string(admin_panel)

# --- Utility to get free port ---
def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# --- Main Entrypoint ---
if __name__ == '__main__':
    conf.get_default().auth_token = "2viIvvVdB2ZYCJbiYmgYVBts5jR_fRvr85rmvrZZzx8CSLtD"  # Replace with your token
    port = get_free_port()
    public_url = ngrok.connect(port)
    print(f"\n🔓 Public camera feed available at: {public_url}/video_feed")
    print(f"🛠 Admin panel: {public_url}/admin (use admin/admin)")
    print(f"📂 Firmware upload: {public_url}/upload_firmware")
    print(f"🌐 Local: http://localhost:{port}/video_feed")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

