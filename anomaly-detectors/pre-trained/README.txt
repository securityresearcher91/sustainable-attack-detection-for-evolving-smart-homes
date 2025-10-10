# PRE-REQUISITES:
Install redis-server on laptop (https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/)

Install python packages
```
pip install -r requirements.txt
```

# Learning the Pre-Trained Models
Terminal 1: Run camera simulator on the device simulator Raspberry Pi (refer to simulator/camera/README)
Terminal 2: Run the scenario_simulator on the monitor Raspberry Pi (refer to simulator/camera/README)

Terminal 3: Run learning_rpi.py on the monitor Raspberry Pi (models will be saved periodically in models_rpi)
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" /home/sas/dev/env/bin/python learning_rpi.py --target-mac SIMULATOR_MAC --macbook-ip MACBOOK_IP

Terminal 4: Run learning_dl.py on the laptop (models will be saved periodically in models_dl)
python learning_dl.py

# Running an Anomaly Detector using Pre-Trained Models
Copy models_rpi to the same folder as detector_rpi.py and models_dl to the same folder as detector_dl.py

Terminal 1: Run camera simulator on the device simulator Raspberry Pi (refer to simulator/camera/README)
Terminal 2: Run the scenario_simulator on the monitor Raspberry Pi (refer to simulator/camera/README)
Terminal 3: Run the anomaly detector on the monitor Raspberry Pi
sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" /home/sas/dev/env/bin/python detector_rpi.py --target-mac SIMULATOR_MAC --macbook-ip MACBOOK_IP --attacker-mac MACBOOK_IP --camera-mac SIMULATOR_MAC

Terminal 4: Run the anomaly detector on the laptop
python detector_dl.py --attacker-mac MACBOOK_IP --camera-mac SIMULATOR_MAC

NOTE: DO NOT EXECUTE THE DoS ATTACK AGAINST THE PUBLIC_URL HOSTED ON CLOUDFLARE WITHOUT PERMISSION FROM CLOUDFLARE. ONLY RUN THE ATTACK ON THE LOCAL URL.
Terminal 5: Run the DoS attack (in supplementary_material) on the laptop using the local URL printed by the camera simulator
./dos.sh http://SIMULATOR_IP:SIMULATOR_PORT