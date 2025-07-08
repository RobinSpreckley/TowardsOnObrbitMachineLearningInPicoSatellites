🛠️ How to Reproduce This Project
This section outlines how to train object detection models, convert them to edge-compatible formats, and evaluate them on embedded devices for on-orbit ML deployment.

📦 1. Setup the Environment
Make sure you have:

Python 3.8 or 3.9

PyTorch (for model training)

TensorFlow Lite runtime

Supporting packages (e.g., psutil, opencv-python, numpy, tflite-support)

bash
Copy
Edit
pip install -r requirements.txt
⚠️ Each board may require additional setup. See docs/hardware.md for per-device instructions (e.g., PyCoral on Raspberry Pi, ArmNN on Odroid).

🧠 2. Train the YOLO Models
You’ll train multiple versions of YOLO (v5 and v8) using the DIOR dataset.

DIOR dataset includes 23,000+ satellite images and 192,000+ object annotations.

We test two subsets:

ALL: all 20 classes

SHIP: single-class "ship" detection

Training:

bash
Copy
Edit
python scripts/train_yolo.py --model yolov5n --dataset ALL
python scripts/train_yolo.py --model yolov8n --dataset SHIP
Models were trained for 100 epochs in batches of 4 on a Titan XP GPU.

🔁 3. Convert Models for Edge Deployment
After training, convert the PyTorch models to TensorFlow Lite with quantisation:

bash
Copy
Edit
python scripts/convert_to_tflite.py --model yolov5n --quant INT8
python scripts/convert_to_tflite.py --model yolov8s --quant FP16
This generates:

FP32 baseline models

FP16 models for GPU

INT8 models for CPU and TPU

🤖 4. Optional: Prepare for Edge TPU
To run models on Coral Edge TPU:

Ensure all model ops are compatible with TPU delegates

Use the Google Coral API

Convert TFLite INT8 model to TPU binary:

bash
Copy
Edit
edgetpu_compiler model_int8.tflite
⚠️ Partial ops offloading can reduce performance. Full mapping is ideal to avoid CPU–TPU sync bottlenecks.

🧪 5. Run Benchmark Tests
Each model is benchmarked on a test board using this command:

bash
Copy
Edit
python scripts/run_benchmark.py --model model.tflite --device N2+ --accelerator GPU
The test pipeline does:

Loads 100 images

Runs inference

Measures time, power, memory, and CPU

Adds bounding boxes and saves outputs

All results are logged in the results/ folder.

📊 6. Evaluation Metrics
Metrics captured per test run include:

🔋 Power (Peak & FPS/Watt)

🧠 Memory (MB)

⏱️ Wall-clock & CPU time

🎯 Accuracy: mAP50, Precision, Recall

See the results in docs/results.md or explore the test logs and graphs in the results/ directory.

📎 Sample Hardware Used
Raspberry Pi 3B+ (with Coral TPU)

Odroid N2+ (CPU + ARM Mali GPU)

Odroid XU4 (CPU only)

Detailed setup instructions per device are in docs/hardware.md.
