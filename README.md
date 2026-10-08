# Smart Home Security System with Edge AI

A Raspberry Pi-based intelligent home security prototype that combines motion sensing, computer vision, local object detection, physical alarms, event logging, and Telegram notifications.

The system monitors a protected area using a PIR motion sensor. When motion is detected, the Raspberry Pi captures an image using the Pi Camera and performs local object detection using a TensorFlow Lite model. If a person is identified, the system activates an audible alarm and sends an image-based security alert through Telegram.

---

## Overview

Traditional motion sensors can trigger alarms whenever movement occurs, regardless of whether the source is actually a person.

This project improves on a basic motion-triggered alarm by introducing an AI-based verification stage.

Instead of immediately treating every movement as an intrusion, the system:

1. Detects movement using a PIR sensor.
2. Captures an image using the Raspberry Pi Camera.
3. Runs TensorFlow Lite object detection locally.
4. Determines whether a person has been detected.
5. Activates the buzzer when a person is identified.
6. Sends a Telegram notification with the detected image.
7. Logs security events to a CSV file.

This creates a simple edge-AI security system in which sensing, image processing, and decision-making take place directly on the Raspberry Pi.

---

## System Architecture

```text
        PIR Motion Sensor
               │
               ▼
         Raspberry Pi
               │
               ▼
          Pi Camera
      Captures an Image
               │
               ▼
      TensorFlow Lite Model
        Object Detection
               │
               ▼
        Is it a Person?
          /          \
        Yes           No
         │             │
         ▼             ▼
   Activate Buzzer   Normal Motion
         │            Notification
         ▼
 Telegram Alert
 + Detected Image
         │
         ▼
      CSV Logging
```

---

## Key Features

- PIR-based motion detection
- Raspberry Pi Camera image capture
- TensorFlow Lite edge inference
- OpenCV image processing
- Person detection with confidence filtering
- Bounding-box visualization
- GPIO-controlled buzzer alarm
- LED activity indicator
- Telegram security notifications
- Image-based intrusion alerts
- Asynchronous Telegram message handling
- CSV event logging
- In-memory storage of recent security events
- Graceful GPIO cleanup when the program exits

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main application logic |
| Raspberry Pi | Edge-computing platform |
| RPi.GPIO | PIR sensor, LED and buzzer control |
| Picamera2 | Raspberry Pi Camera interface |
| TensorFlow Lite | Lightweight object-detection inference |
| OpenCV | Image processing and annotation |
| NumPy | Image-array processing |
| Telegram Bot API | Remote security notifications |
| AsyncIO | Non-blocking Telegram communication |
| CSV | Local security-event logging |

---

## Hardware Components

The current implementation is designed around:

- Raspberry Pi
- Raspberry Pi Camera
- PIR motion sensor
- Piezo buzzer
- LED
- Appropriate resistors and connecting wires
- Breadboard or equivalent prototyping connections

The application currently uses physical GPIO board numbering.

| Component | Physical Pin |
|---|---:|
| PIR Sensor | 11 |
| Piezo Buzzer | 7 |
| LED | 13 |

Hardware configuration can be changed in the `gpio_components` dictionary in the Python application.

---

## Detection Workflow

When the system starts, it continuously monitors the PIR sensor.

When new motion is detected:

```text
Motion Detected
      ↓
LED Activated
      ↓
Capture Camera Frame
      ↓
Resize Image
      ↓
TensorFlow Lite Inference
      ↓
Confidence > 65%?
      ↓
Identify Detected Object
      ↓
Person Detected?
```

If the detected object is a person:

```text
Buzzer ON
    ↓
Telegram Alert
    ↓
Detected Image Sent
    ↓
Event Logged
```

If the detected object is not a person, the system sends a normal-motion notification without activating the intrusion alarm.

---

## Object Detection

The application loads a TensorFlow Lite object-detection model using:

```python
Interpreter(model_path="detect.tflite")
```

Object labels are loaded from:

```text
coco_labels.txt
```

Only detections with a confidence score greater than `0.65` are currently accepted.

Detected objects are annotated using OpenCV with:

- Bounding boxes
- Object labels
- Detection confidence

The annotated image is then saved locally and can be sent through Telegram when a person is detected.

---

## Telegram Notifications

The system uses a Telegram bot to provide remote security notifications.

Two configuration values are required:

```python
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "CHAT_ID_HERE"
```

Real credentials should never be committed to a public repository.

A future improvement to this project will move these values into environment variables or a local configuration file excluded through `.gitignore`.

---

## Event Logging

Security events are stored in:

```text
motion_log.csv
```

Each event contains:

```text
Timestamp, Event
```

Example events include:

```text
MOTION DETECTED
ALERT: Person Detected
Normal Motion Detected
MOTION ENDED
```

The system also keeps the most recent 100 events in memory using Python's `deque`.

---

## Repository Structure

The repository is being reorganized toward the following structure:

```text
Home-Security-System/
│
├── src/
│   └── home_security.py
│
├── models/
│   ├── detect.tflite
│   └── coco_labels.txt
│
├── docs/
│   ├── architecture.png
│   ├── wiring-diagram.png
│   └── system-demo.jpg
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

The current repository contains the original Python implementation. Additional project files and documentation are being added as part of repository cleanup.

---

## Running the System

The application is intended to run directly on a Raspberry Pi with the required hardware and software dependencies installed.

The TensorFlow Lite model and class-label file must also be available to the application.

Once configured, the system can be started using:

```bash
python home_security.py
```

The application then begins monitoring for motion continuously until it is stopped.

---

## Current Project Status

The core system logic has been implemented, including:

- Motion detection
- Raspberry Pi Camera capture
- TensorFlow Lite object detection
- Person-based alarm decisions
- Telegram notifications
- Event logging
- GPIO cleanup

The original system was developed as a Raspberry Pi security prototype.

The repository is currently being improved for reproducibility and documentation. The complete hardware setup is not presently available for a fresh end-to-end hardware validation, so future refactoring will preserve the intended system behaviour until physical re-testing can be completed.

---

## Future Improvements

Potential improvements include:

- Environment-variable based credential management
- More robust camera lifecycle management
- Configurable detection thresholds
- Improved TensorFlow Lite model configuration
- Multiple-camera support
- Web-based monitoring dashboard
- Remote arming and disarming
- Event-image history
- Database-backed security logs
- Push notification alternatives
- Improved false-positive filtering
- Hardware watchdog and automatic recovery

---

## Project Purpose

This project demonstrates the integration of:

**Embedded Systems + Internet of Things + Computer Vision + Edge AI + Automation**

It shows how lightweight machine-learning inference can be combined with physical sensors and remote communication to build an intelligent monitoring system.

---

## Author

**Oscar Muchiri**

Computer Scientist | Machine Learning • Software Engineering • Geospatial Systems
