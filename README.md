# 🚀 Towards On-Orbit Machine Learning in Picosatellites-WIP

This project explores deploying lightweight object detection ML models onboard Picosatellites to minimize data transmission to reduce power consumption, latency and increase security by processing data on-orbit.
 
 
<p align="center">
  <img src="assets/Introdiagram.JPG" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b> Left – challenges with traditional satellite data collection; Right – benefits of On-Orbit processing using edge ML.
</p>
📌 Key Features

- ✅ Evaluates Object Detection models on constrained edge devices in the context of a a full deployment for example data reading, model image slicing .
- ✅ Hardware and Software Optimisation **quantisation (INT8, FP16)** and hardware acceleration (TPU, GPU, etc.).
- ✅ Software for automating the testing process on edge devices
- ✅ Framework for recording important information, complete breakdown of OS, Architectures and Software Packages.

---

## 📁 Project Structure

```bash
.
├── models/             # YOLOv5 & YOLOv8 trained models
├── data/               # DIOR dataset preprocessing and conversion
├── scripts/            # Training, quantisation, benchmarking
├── results/            # Logged metrics and plots
├── docs/               # Full technical documentation
├── README.md
└── LICENSE
``` 
## 🧪 Benchmark Platforms

<p align="center">
  <img src="assets/SmartpowerandBoard.jpeg" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b> Left – challenges with traditional satellite data collection; Right – benefits of On-Orbit processing using edge ML.
</p>

The following embedded devices were used to evaluate on-orbit ML performance:

**Raspberry Pi 3B+** — with Coral Edge TPU  
**Odroid N2+** — with ARM Mali-G52 GPU acceleration  
**Odroid XU4** — CPU-only testing

### Metrics Measured

- 🔋 Peak Power (mW)
- 💾 Memory usage (MB)
- ⏱️ Inference and Full Process time (s)
- 🎯 Accuracy (mAP50, Recall)
