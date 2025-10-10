# PRE-REQUISITES:
Install cloudflared
```
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb -o cloudflared.deb
sudo dpkg -i cloudflared.deb
```

Check that it is working
```
cloudflared tunnel --url http://localhost:5000
```

Install virtualenv if you don't have it
```
sudo apt install python3-venv
```

Create a virtual environment
```
python3 -m venv cam_env
```

Activate it
```
source cam_env/bin/activate
```

Now install your packages
```
pip install -r requirements.txt
```

Copy a sample video to $PWD/videos
```
cp video.mp4 ./videos
```

# Running the Camera Simulator
Initialise Cloudflare environment
```
source exports.sh
```

Run the application
```
python3 camera.py
```

Open the URL in a brower to access the video feed or use the scenario_simulator script
```
python scenario_simulator.py [-h] --scenario {1,2,3} [--hours HOURS] --lan-url LAN_URL --public-url PUBLIC_URL
```