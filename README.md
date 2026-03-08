# Real-Time Object Tracking System

This project demonstrates two different approaches to solving the same computer vision problem:

**Tracking a selected object in real time from a webcam feed.**

The goal of this repository is not only to implement object tracking, but also to show the **evolution of the solution**, starting from a simple traditional computer vision tracker and moving toward a modern deep learning–based tracking pipeline.

---

# Project Overview

The system allows a user to:

* Open a live webcam feed
* Select an object using a bounding box
* Track that object in real time as it moves

Two different implementations were developed:

1. **Traditional Tracking (OpenCV + CSRT)**
2. **Deep Learning Tracking (YOLOv8 + ByteTrack)**

---

# Repository Structure

```
real-time-object-tracking
│
├── opencv-csrt-tracker
│   ├── tracker.py
│   ├── requirements.txt
│   └── demo_csrt.mp4
│
├── yolo-bytetrack-tracker
│   ├── yolo_bytetrack_live.py
│   ├── requirements.txt
│   └── demo_yolo_bytetrack.mp4
│
└── README.md
```

---

# Implementation 1: OpenCV + CSRT Tracker

### Description

The first implementation uses the **CSRT tracker from OpenCV**, which is a classical object tracking algorithm.

The user selects an object in the live webcam stream, and the tracker follows the object across frames.

### Key Characteristics

* No deep learning required
* Lightweight and simple
* Works well for single-object tracking
* Fast to implement

### Workflow

1. Start webcam feed
2. Press **S** to select an object
3. Draw a bounding box around the object
4. The **CSRT tracker** follows the object in subsequent frames

### Demo

`opencv-csrt-tracker/demo_csrt.mp4`

### Limitations

Although CSRT works well in simple scenarios, it has several limitations:

* It **does not understand object classes**
* It can **lose the object during occlusion**
* It struggles when **multiple similar objects appear**
* It cannot recover if the object leaves and re-enters the frame

These limitations motivated the second implementation.

---

# Implementation 2: YOLOv8 + ByteTrack

### Description

The second implementation uses a modern **deep learning detection + tracking pipeline**.

* **YOLOv8** performs object detection.
* **ByteTrack** performs multi-object tracking by assigning persistent IDs to detected objects.

This approach allows the system to track objects more robustly.

### Key Characteristics

* Deep learning–based detection
* Multi-object tracking
* Unique ID for each object
* More robust tracking performance

### Workflow

1. YOLO detects objects in each frame
2. ByteTrack assigns a **tracking ID** to each object
3. The user selects an object
4. The system tracks the **same object ID** across frames

### Demo

`yolo-bytetrack-tracker/demo_yolo_bytetrack.mp4`

---

# Why We Improved the System

The project intentionally started with a **simple baseline implementation** before moving to a more advanced solution.

This approach allowed us to:

* Understand the fundamentals of object tracking
* Identify limitations of traditional trackers
* Improve the system using modern deep learning techniques

| Feature               | OpenCV CSRT | YOLO + ByteTrack |
| --------------------- | ----------- | ---------------- |
| Object Detection      | ❌           | ✔                |
| Multi-object Tracking | ❌           | ✔                |
| Robustness            | Medium      | High             |
| Object Identity       | ❌           | ✔                |
| Complexity            | Low         | Higher           |

The deep learning pipeline provides **more reliable and scalable tracking**, especially in real-world scenarios.

---

# Installation

Create a Python environment:

```
conda create -n tracker_env python=3.10
conda activate tracker_env
```

Install dependencies depending on the implementation.

---

## For CSRT Version

```
cd opencv-csrt-tracker
pip install -r requirements.txt
```

Run:

```
python tracker.py
```

---

## For YOLO + ByteTrack Version

```
cd yolo-bytetrack-tracker
pip install -r requirements.txt
```

Run:

```
python yolo_bytetrack_live.py
```

---

# Results

Both implementations successfully track a user-selected object from a webcam feed.

The second implementation demonstrates how integrating **deep learning detection with tracking algorithms** significantly improves robustness and flexibility.

---

# Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* ByteTrack
* NumPy

---

# Future Improvements

Possible improvements include:

* Object re-identification
* GPU acceleration
* Web interface for interaction
* Tracking multiple selected objects
* Deploying as a real-time monitoring system

---

# Conclusion

This project demonstrates the progression from **classical computer vision tracking** to **modern deep learning–based tracking systems**.

By implementing both approaches, we gain a deeper understanding of the trade-offs between **simplicity, performance, and scalability** in real-time object tracking systems.

