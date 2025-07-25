# 🚀 Towards On-Orbit Machine Learning in Picosatellites

The current method of downlinking the data collected by satellites to Earth causes high latency and power consumption and therefore is unsuitable for Picosatellites with limited system resources. On-Orbit Machine Learning (on-orbit ML) offers a solution by decreasing the amount of data to transmit. However, while ML methods exist for use in Picosatellites, they lack suitable recordings of system measurements such as power, memory, and processing time, which hinders their implementation. We address this issue by testing various ML models with compression methods and lightweight architectures on edge setups, using a mix of hardware and software optimisations, while focusing on different processing architectures, accelerators, and delegate improvements. We found that different combinations of models and edge setups for various goals and limitations that Picosatellites can have, providing valuable insights for practical applications. 
 
 
<p align="center">
  <img src="assets/Introdiagram.JPG" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b> Left – challenges with traditional satellite data collection; Right – benefits of On-Orbit processing using edge ML.
</p>


## What This Project Demonstrates

- Deployment of **lightweight ML models** on edge devices for real-time satellite image processing.
- Integration of **hardware accelerators (TPU, GPU)** and software optimizations (TensorRT, TF Lite).
- Benchmarking of object detection pipelines with **quantization and compression**.
- A **data logging framework** to record power, latency, and memory usage during inference.
- Insights into trade-offs between **accuracy, energy efficiency, and runtime** for different mission constraints.


## Key Results

- **Power consumption reduced by up to XX%** using INT8 quantization vs. FP32 precision.
- **YOLOv5n + Coral TPU** achieved the **fastest inference** (XX ms) with mAP > XX%.  
- Identified **optimal combinations of hardware and models** for Picosatellite constraints.  
- Developed a testbench enabling **real-time recording of power and memory usage** during edge inference.
- Showed that **on-orbit ML can cut downlink data needs by XX%**, improving autonomy.


## Skills & Techniques

- **Edge AI Optimization:** Quantization (INT8, FP16), pruning, lightweight CNN architectures.
- **Object Detection:** YOLOv5, YOLOv8 (PyTorch, ONNX, TensorRT).
- **Hardware & Embedded Systems:** Raspberry Pi 3B+, Odroid N2+, Coral Edge TPU.
- **Profiling & Benchmarking:** Power (W), Memory (MB), Inference Time (ms).
- **Computer Vision Tools:** OpenCV for preprocessing and image slicing.
- **Tech Stack:** Python, C++, TensorFlow Lite, PyTorch, ONNX Runtime.


phon
## 📁 Setting up edge boards
<p align="center">
  <img src="assets/Dependencies.JPG" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b> Dependencies for setting up an edge board with hardware acceleration
</p>
Each edge board introduces its own constraints. These diagrams highlight how hardware, accelerators, software packages, and progmraming languages environments interconnect. Use this as a guide to avoid common pitfalls when debugging or porting across devices.

<p align="center">
  <img src="assets/testingboardsetup.JPG" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b> Overview of the process for setting up an edge board, OS, tools and packages, the ML framework, recordings software, and hardware acceleration.
</p>
This shows a strategy of setting up the boards detailing the common steps needed and where you might also find the solutions for them.

### 🧪 Setup used in this project

<p align="center">
  <img src="assets/SmartpowerandBoard.jpeg" alt="On-Orbit ML diagram" width="400"/>
  <br>
  <b>Figure:</b>1) Is the Smart Power 3 used in the power recording 2) USB-Uart cable for transferring power data to the main computer 3) Split cable power cable split leads in to red and black so it can draw power from Smart Power 3 4)Odroid Xu4 board 5) Raspberry Pi3b+ board 6) Odroid N2+ Board.
</p>

The following embedded devices were used to evaluate on-orbit ML performance:

**Raspberry Pi 3B+** — with Coral Edge TPU  
**Odroid N2+** — with ARM Mali-G52 GPU acceleration  
**Odroid XU4** — CPU-only testing

## 📁 Setting up test bench
You will need the odroid smart power 3 for real time data logging, however other 

Depending on what operating system your edge board is running the filepaths can be read in diffrently, because of this it should be best to set all the absolute file paths in x file, 

To record power effectively you will need a second device, to run the reciver code which will start the recording 
simply run the files after this 

### Metrics Measured
In order to demonstrate the application of on-orbit ML, a simulation process is created. The process involves reading 100 images, adding bounding boxes to them, and saving the modified images. The diagram below provides an overview of this process, outlining the steps and sub-tasks involved. The entire process can be divided into the following four stages. %Note that the pre-processing to post-processing steps (steps 2 to 4) are repeated 100 times, once for each image:


<p align="center">
  <img src="assets/recordingprocess.JPG" alt="ProcessFlowchart" width="400"/>
  <br>
  <b>Figure:</b> A diagram of the full process, defining each subsection.
</p>




The green-colored boxes in represent steps that are required when using the TensorFlow Lite (TF Lite) framework, as well as when using the ArmNN inference engine or Coral TPU extensions. The blue-colored boxes, on the other hand, can be customized, e.g., images can be loaded using the OpenCV library.
To find points of failures such as peak memory, and to test elements of the development that can not be feasibly changed, for example diffrent code be used to load and run and edit the images however the inference of the models will stay the same across all implementations, so this collection method looks at the full process and the unchanging parts to give the results broader relevence to the industry.


- 🔋 Peak and average Power at inference and full process (W)
- 💾 Memory usage at inference and full process (MB)
- ⏱️ Inference and Full Process time (ms)
- 🎯 Accuracy (mAP50, Recall)

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
