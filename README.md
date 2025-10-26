# Sustainable Attack Detection for Evolving Smart Homes

This repository contains the replication package for our paper **“Sustainable Attack Detection for Evolving Smart Homes.”**

The project investigates how network-based anomaly detection can remain effective as smart-home systems evolve through software updates, changing device behaviours, and new usage patterns. 
It provides a complete experimental setup that combines a smart-camera simulator with machine-learning and deep-learning anomaly detectors to study **sustainable security** over time.

---

## Repository Structure

### [`simulator/`](./simulator)
Implements the **smart-camera simulator** and the **evolution scenarios** used in the experiments.  
The simulator reproduces benign behaviour, software updates, and attack events (e.g., denial-of-service) under controlled conditions.

See the [simulator README](simulator/camera/README.md) for setup and usage instructions.

---

### [`anomaly-detectors/`](./anomaly-detectors)
Contains the **network-based anomaly detection framework**, including implementations of seven algorithms:
- Machine learning: Half-Space Trees, One-Class SVM, Local Outlier Factor, COPOD  
- Deep learning: Variational Autoencoder (VAE), LSTM, GRU  

These models are evaluated in both **pre-trained** and **online** learning configurations.  
Refer to the [anomaly-detectors README](anomaly-detectors/pre-trained/README.md and anomaly-detectors/online/README.md) for details on training, evaluation, and adaptive thresholding.

---

### [`supplementary_material/`](./supplementary_material)
Includes supplementary scripts used for data processing and performing a Denial-of-Service (DoS) attack.

---

*GitHub Copilot was used as a coding assistant for implementation.*
