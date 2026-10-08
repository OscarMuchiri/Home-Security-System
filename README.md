# Smart Home Security System with Edge AI

[![Python syntax check](https://github.com/OscarMuchiri/Home-Security-System/actions/workflows/syntax-check.yml/badge.svg)](https://github.com/OscarMuchiri/Home-Security-System/actions/workflows/syntax-check.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Raspberry Pi security prototype that combines **motion sensing, computer vision, TensorFlow Lite edge inference, physical alarms, Telegram notifications, and local event logging**.

The system uses a PIR sensor to detect movement, captures an image with the Raspberry Pi Camera, runs local object detection, and escalates the event when the detected class is `person`.

> **Project status:** the original prototype logic is implemented. The repository has been cleaned and refactored for portfolio use, but a fresh end-to-end hardware validation is still pending because the complete hardware setup is not currently available.

---

## What the system does

1. Monitors a PIR motion sensor.
2. Turns on an LED when motion begins.
3. Captures an image using the Raspberry Pi Camera.
4. Runs TensorFlow Lite object detection locally on the Raspberry Pi.
5. Accepts detections above a `0.65` confidence threshold.
6. If a person is detected:
   - activates the buzzer;
   - sends a Telegram alert with the detected image;
   - logs the event.
7. If the detection is not a person, sends a normal-motion notification.
8. Records events in a CSV log.
9. Cleans up GPIO resources when the application exits.

---

## System architecture

```text
PIR Motion Sensor
       │
       ▼
  Raspberry Pi
       │
       ▼
   Pi Camera
       │
       ▼
TensorFlow Lite
Object Detection
       │
       ▼
  Person detected?
    /        \
  Yes         No
   │           │
   ▼           ▼
Buzzer      Normal-motion
+ Telegram  notification
photo
    \         /
     \       /
      ▼     ▼
      CSV event log
```

---

## Technology stack

| Technology | Role |
|---|---|
| Python | Application logic |
| Raspberry Pi | Edge-computing platform |
| RPi.GPIO | PIR sensor, LED and buzzer control |
| Picamera2 | Raspberry Pi Camera interface |
| TensorFlow Lite | Local object-detection inference |
| OpenCV | Image processing and annotation |
| NumPy | Image tensor handling |
| python-telegram-bot | Remote Telegram notifications |
| AsyncIO / threading | Non-blocking notification dispatch |
| CSV | Local event logging |

---

## Hardware configuration

The implementation is designed for:

- Raspberry Pi
- Raspberry Pi Camera
- PIR motion sensor
- Piezo buzzer
- LED
- Appropriate resistors, jumper wires and breadboard/prototyping connections

The code uses **physical BOARD pin numbering**.

| Component | Physical pin |
|---|---:|
| PIR sensor | 11 |
| Piezo buzzer | 7 |
| LED | 13 |

The pin mapping is defined in `GPIO_COMPONENTS` inside `home_security.py`.

---

## Repository structure

```text
Home-Security-System/
├── docs/
│   └── MODEL_SETUP.md
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── home_security.py
└── requirements.txt
```

The TensorFlow Lite model and matching label file are runtime dependencies and are intentionally not committed until their exact source and redistribution terms are verified.

---

## Required local runtime files

Place these files beside `home_security.py`:

```text
detect.tflite
coco_labels.txt
```

The current implementation expects the detector outputs to be ordered as:

1. bounding boxes;
2. class IDs;
3. confidence scores.

See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md) for the model-interface assumptions and validation notes.

---

## Telegram configuration

The application reads credentials from environment variables rather than storing secrets in source control.

Required variables:

```text
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

A reference template is provided in `.env.example`.

For example, in a shell session:

```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
python home_security.py
```

> `.env.example` is documentation only. The application reads process environment variables directly and does not automatically load a `.env` file.

---

## Installation

### 1. Prepare Raspberry Pi OS

Use a Raspberry Pi OS installation with camera support enabled and Picamera2 available.

Picamera2 is commonly provided through Raspberry Pi OS system packages rather than ordinary PyPI installation, so it is intentionally not listed in `requirements.txt`.

### 2. Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 3. Add the detector files

Place:

```text
detect.tflite
coco_labels.txt
```

in the repository root.

### 4. Set Telegram environment variables

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in the environment used to start the application.

### 5. Run

```bash
python3 home_security.py
```

---

## Event logging

Runtime events are written to:

```text
~/Desktop/motion_log.csv
```

The application also keeps the most recent 100 events in memory.

Example event types include:

```text
MOTION DETECTED
ALERT: Person Detected
Normal Motion Detected: <label>
MOTION ENDED
```

Captured runtime images are stored as:

```text
~/Desktop/frame.jpg
~/Desktop/detected.jpg
```

These generated files are excluded from Git.

---

## Software improvements made during repository cleanup

The portfolio version now includes:

- professional project documentation;
- safe Telegram credential handling through environment variables;
- explicit runtime validation for missing credentials/model files;
- model and label paths resolved relative to the application file;
- model input dimensions read dynamically from the TensorFlow Lite input tensor;
- safer camera shutdown using `try/finally`;
- GPIO setup and cleanup separated into dedicated functions;
- a conventional `main()` entry point;
- clearer dependency documentation;
- runtime output and secret exclusions through `.gitignore`;
- TensorFlow Lite model-interface documentation;
- MIT licensing.

These changes are software/documentation improvements and intentionally preserve the original hardware pin mapping and overall system workflow.

---

## Current limitations

- The original `detect.tflite` model is not included in the public repository.
- The matching label file is not included until the original model package is verified.
- Detector output ordering is preserved from the original implementation and should be verified when the original model is recovered.
- The latest repository refactor has been syntax-reviewed, but has not yet been re-tested on the complete physical hardware setup.
- This is a prototype and should not be treated as a certified or sole security system.

---

## Future improvements

Potential extensions include:

- configurable output directory;
- remote arming/disarming;
- event history dashboard;
- persistent database logging;
- camera lifecycle optimization for continuous operation;
- configurable detection thresholds;
- model metadata validation;
- multiple-camera support;
- hardware watchdog and recovery;
- improved false-positive handling;
- deployment as a Raspberry Pi system service.

---

## Portfolio relevance

This project demonstrates the integration of:

**Embedded Systems · IoT · Computer Vision · Edge AI · Automation · Python**

It complements software-only machine-learning projects by showing how AI inference can be connected to physical sensors, actuators and real-time notifications.

---

## Author

**Oscar Muchiri**

Computer Scientist | Machine Learning • Software Engineering • Geospatial Systems
