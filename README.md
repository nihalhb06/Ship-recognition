# 🚢 Real-Time Ship Detection using YOLOv8

A real-time **Ship Detection System** using a custom-trained **YOLOv8 object detection model**. The system can detect ships from a webcam, video file, or RTSP stream and display the detected objects with bounding boxes and confidence scores.

## 📌 Features

* 🚢 Real-time ship detection
* 🎥 Webcam support
* 📹 Video file support
* 📡 RTSP stream support
* 🤖 Custom-trained YOLOv8 model
* ⚡ Automatic GPU/CPU detection
* 🎯 Configurable confidence threshold
* 📦 Configurable IoU threshold
* 📊 Real-time FPS display
* 💾 Option to save annotated detection videos

The detection script uses YOLO inference with configurable image size, confidence, IoU and device parameters.

---

## 🛠️ Technologies Used

* **Python**
* **YOLOv8**
* **Ultralytics**
* **OpenCV**
* **PyTorch**
* **Torchvision**

The project requirements include Ultralytics, OpenCV, PyTorch and Torchvision.

---

## 📂 Project Structure

```text
Ship-Detection/
│
├── best.pt
├── webcam_detect.py
├── requirements.txt
└── README.md
```

### Files

| File               | Description                     |
| ------------------ | ------------------------------- |
| `best.pt`          | Trained YOLOv8 model weights    |
| `webcam_detect.py` | Real-time ship detection script |
| `requirements.txt` | Required Python packages        |
| `README.md`        | Project documentation           |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ship-detection.git
cd ship-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The provided requirements specify:

```text
ultralytics>=8.2.0
opencv-python>=4.9.0
torch>=2.2.0
torchvision>=0.17.0
```

For GPU acceleration, install the CUDA-enabled PyTorch build appropriate for your system.

---

## 🚀 Usage

### Webcam Detection

Place `best.pt` in the project directory and run:

```bash
python webcam_detect.py --weights best.pt
```

By default, the program uses camera `0`.

```bash
python webcam_detect.py --weights best.pt --camera 0
```

The program displays the detection window and shows the current FPS. Press **`q`** to stop detection.

---

## 🎥 Detect Ships in a Video

You can also use a video file instead of a webcam:

```bash
python webcam_detect.py --weights best.pt --source video.mp4
```

The script also supports RTSP streams through the `--source` option.

---

## 💾 Save Detection Output

To save the annotated video:

```bash
python webcam_detect.py --weights best.pt --source video.mp4 --save output.mp4
```

The program automatically creates a video writer when the `--save` option is provided.

---

## 🎯 Detection Parameters

The detection settings can be customized using command-line arguments.

```bash
python webcam_detect.py \
    --weights best.pt \
    --camera 0 \
    --imgsz 640 \
    --conf 0.5 \
    --iou 0.45
```

| Parameter   | Description                    | Default                                   |
| ----------- | ------------------------------ | ----------------------------------------- |
| `--weights` | Path to trained YOLO model     | `runs/train/ship_yolov8s/weights/best.pt` |
| `--camera`  | Camera index                   | `0`                                       |
| `--source`  | Video/RTSP source              | None                                      |
| `--imgsz`   | Input image size               | `640`                                     |
| `--conf`    | Detection confidence threshold | `0.5`                                     |
| `--iou`     | IoU threshold for NMS          | `0.45`                                    |
| `--device`  | CPU/GPU device                 | Auto                                      |
| `--save`    | Output video path              | None                                      |

These parameters are defined directly in the detection script.

---

## 🧠 How It Works

```text
Camera / Video / RTSP
          ↓
      Video Frame
          ↓
       YOLOv8
          ↓
   Ship Detection
          ↓
 Bounding Boxes + Confidence
          ↓
     Display Result
          ↓
 Optional Video Saving
```

For every frame, the system runs YOLO inference and plots the detection results onto the frame before displaying them.

---

## ⚡ GPU Support

The program automatically checks whether CUDA is available.

If a CUDA-compatible GPU is available:

```text
GPU → YOLO inference
```

Otherwise:

```text
CPU → YOLO inference
```

You can also manually select the device:

```bash
python webcam_detect.py --weights best.pt --device cpu
```

or:

```bash
python webcam_detect.py --weights best.pt --device 0
```

The device is automatically selected in the script when `--device` is not specified.

---

## 📊 Output

The system displays:

* Detected ship bounding boxes
* Detection confidence
* Real-time FPS
* Annotated video feed

Example:

```text
┌─────────────────────────────────┐
│                                 │
│       ┌──────────────┐          │
│       │    SHIP      │  0.91    │
│       └──────────────┘          │
│                                 │
│ FPS: 28.5                       │
└─────────────────────────────────┘
```

---

## 🔮 Future Improvements

* Add multiple ship categories
* Improve detection accuracy with a larger dataset
* Add object tracking
* Count detected ships
* Add distance estimation
* Add automatic ship classification
* Create a web-based monitoring dashboard
* Add alert notifications for detected ships
* Deploy the model on edge devices
* Improve performance for low-light and adverse weather conditions

---

## 👨‍💻 Author

**Nihal H B**

Computer Science & Data Science
CMR Institute of Technology

---

## 📜 License

This project is intended for educational and research purposes.
