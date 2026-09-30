# Vehicle Traffic Detection and Counting

## Overview

This project is a Python-based vehicle detection, tracking, and counting application using a pre-trained YOLO11 model and ByteTrack.

The application processes a traffic video and detects, tracks, and counts three vehicle categories:

- Cars
- Bikes / Motorcycles
- Buses

A virtual counting line is used to count tracked vehicles when they cross the line.

## Features

- YOLO11 object detection
- ByteTrack vehicle tracking
- Unique tracking IDs
- Bounding boxes around detected vehicles
- Confidence scores
- Virtual vehicle counting line
- Counting of cars, bikes, and buses
- Processed output video
- Frame-by-frame video processing

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLO11
- ByteTrack
- NumPy

## Vehicle Classes

The project uses the following YOLO COCO classes:

| Class | YOLO Class ID |
|---|---:|
| Car | 2 |
| Motorcycle / Bike | 3 |
| Bus | 5 |

## How It Works

1. The traffic video is loaded using OpenCV.
2. YOLO11 detects vehicles in each frame.
3. ByteTrack assigns a unique tracking ID to each detected vehicle.
4. Bounding boxes, vehicle labels, IDs, and confidence scores are displayed.
5. A virtual horizontal counting line is placed in the video.
6. When a tracked vehicle crosses the counting line, it is counted.
7. The processed video is saved as an output video.

## Project Structure

```text
vehicle tracking/
│
├── input/
│   ├── traffic.mp4
│   └── traffic_35sec.mp4
│
├── output/
│   └── traffic_count.avi
│
├── runs/
│
├── venv/
│
├── count.py
├── main.py
├── track.py
├── yolo11n.pt
├── requirements.txt
└── README.md