# 🚀 Towards On-Orbit Machine Learning in Picosatellites

This project explores deploying lightweight object detection ML models onboard Picosatellites to minimize data transmission and power consumption by processing data on-orbit.

---

## 📌 Key Features

- ✅ Evaluates **YOLOv5/YOLOv8** models on constrained edge devices.
- ✅ Benchmarks **power, memory, CPU usage, Accelerator usage** across platforms.
- ✅ Supports **quantisation (INT8, FP16)** and hardware acceleration (TPU, GPU, etc.).
- ✅ Tests conducted on Raspberry Pi 3B+, Odroid XU4, Odroid N2+.

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
