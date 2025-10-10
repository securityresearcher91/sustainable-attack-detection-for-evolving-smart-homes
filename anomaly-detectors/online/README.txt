# PRE-REQUISITES:
Install redis-server on laptop (https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

Install python packages
```
pip install -r requirements.txt
```

# Running the Online Anomaly Detector
Terminal 1: Run camera simulator on the device simulator Raspberry Pi (refer to simulator/camera/README)
Terminal 2: Run the scenario_simulator on the monitor Raspberry Pi (refer to simulator/camera/README)

Terminal 3: Run online anomaly detector on the laptop before the RPi
python online_detector_dl.py --attacker-mac MACBOOK_IP --camera-mac SIMULATOR_MAC

Terminal 4: Run online anomaly detector on the Raspberry Pi
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" /home/sas/dev/env/bin/python online_detector_rpi.py --target-mac SIMULATOR_MAC --macbook-ip MACBOOK_IP --attacker-mac MACBOOK_IP --camera-mac SIMULATOR_MAC

NOTE: DO NOT EXECUTE THE DoS ATTACK AGAINST THE PUBLIC_URL HOSTED ON CLOUDFLARE WITHOUT PERMISSION FROM CLOUDFLARE. ONLY RUN THE ATTACK ON THE LOCAL URL.
Terminal 5: Run the DoS attack (in supplementary_material) on the laptop using the LOCAL URL printed by the camera simulator
./dos.sh http://SIMULATOR_IP:SIMULATOR_PORT
