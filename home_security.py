"""Raspberry Pi edge-AI home security prototype."""

import asyncio
import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from telegram import Bot
from tflite_runtime.interpreter import Interpreter


# Paths and runtime configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "detect.tflite"
LABELS_PATH = BASE_DIR / "coco_labels.txt"

DESKTOP_PATH = Path.home() / "Desktop"
CSV_FILE = DESKTOP_PATH / "motion_log.csv"
FRAME_PATH = DESKTOP_PATH / "frame.jpg"
DETECTED_IMAGE_PATH = DESKTOP_PATH / "detected.jpg"

DETECTION_THRESHOLD = 0.65

# Physical BOARD pin numbering, preserved from the original prototype.
GPIO_COMPONENTS = {
    "pir_sensor": 11,
    "piezo": 7,
    "led": 13,
}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def validate_configuration():
    """Fail early when required runtime inputs are missing."""
    missing = []

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN environment variable")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID environment variable")
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH))
    if not LABELS_PATH.exists():
        missing.append(str(LABELS_PATH))

    if missing:
        raise RuntimeError(
            "Missing required configuration/files:\n- " + "\n- ".join(missing)
        )


validate_configuration()


# Telegram setup
bot = Bot(token=TELEGRAM_BOT_TOKEN)

event_loop = asyncio.new_event_loop()
threading.Thread(target=event_loop.run_forever, daemon=True).start()


async def send_alert_async(message, image_path=None):
    """Send a Telegram message and optionally attach an image."""
    if image_path:
        with open(image_path, "rb") as photo:
            await bot.send_photo(
                chat_id=TELEGRAM_CHAT_ID,
                photo=photo,
                caption=message,
            )
    else:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)


def send_telegram_alert(message, image_path=None):
    """Queue a Telegram alert without blocking the monitoring loop."""
    return asyncio.run_coroutine_threadsafe(
        send_alert_async(message, image_path),
        event_loop,
    )


# TensorFlow Lite setup
interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with LABELS_PATH.open("r", encoding="utf-8") as label_file:
    labels = [line.strip() for line in label_file if line.strip()]


# Event logging
DESKTOP_PATH.mkdir(parents=True, exist_ok=True)
motion_log = deque(maxlen=100)

if not CSV_FILE.exists():
    with CSV_FILE.open("w", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow(["Timestamp", "Event"])


def log_event(event_type):
    """Log an event to CSV and recent in-memory history."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    motion_log.append((timestamp, event_type))

    with CSV_FILE.open("a", newline="", encoding="utf-8") as log_file:
        csv.writer(log_file).writerow([timestamp, event_type])

    print(f"{timestamp} - {event_type} (Logged)")


# Object detection
def detect_object():
    """Capture one image and return the first detection above the threshold."""
    picam2 = Picamera2()
    camera_started = False

    try:
        picam2.start()
        camera_started = True
        time.sleep(2)
        picam2.capture_file(str(FRAME_PATH))
    finally:
        if camera_started:
            picam2.stop()
        picam2.close()

    image = cv2.imread(str(FRAME_PATH))
    if image is None:
        raise RuntimeError(f"Camera frame could not be read from {FRAME_PATH}")

    height, width, _ = image.shape

    # Read the required input dimensions directly from the TFLite model.
    input_shape = input_details[0]["shape"]
    input_height = int(input_shape[1])
    input_width = int(input_shape[2])

    resized = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(resized, axis=0)

    expected_dtype = input_details[0]["dtype"]
    if input_data.dtype != expected_dtype:
        input_data = input_data.astype(expected_dtype)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    # The original detector is expected to expose boxes, classes and scores
    # at output indexes 0, 1 and 2 respectively.
    boxes = interpreter.get_tensor(output_details[0]["index"])[0]
    classes = interpreter.get_tensor(output_details[1]["index"])[0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]["index"])[0]

    for i, score in enumerate(scores):
        if float(score) <= DETECTION_THRESHOLD:
            continue

        class_id = int(classes[i])
        confidence = int(float(score) * 100)
        label = (
            labels[class_id]
            if 0 <= class_id < len(labels)
            else f"Unknown ({class_id})"
        )

        print(f"Detected: {label} ({confidence}%)")

        ymin, xmin, ymax, xmax = boxes[i]
        left = int(xmin * width)
        top = int(ymin * height)
        right = int(xmax * width)
        bottom = int(ymax * height)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({confidence}%)",
            (left, max(top - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cv2.imwrite(str(DETECTED_IMAGE_PATH), image)
        return label, DETECTED_IMAGE_PATH

    return None, None


# GPIO lifecycle and monitoring loop
def configure_gpio():
    """Configure GPIO pins using physical BOARD numbering."""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_COMPONENTS["pir_sensor"], GPIO.IN)
    GPIO.setup(GPIO_COMPONENTS["piezo"], GPIO.OUT)
    GPIO.setup(GPIO_COMPONENTS["led"], GPIO.OUT)
    GPIO.output(GPIO_COMPONENTS["led"], False)
    GPIO.output(GPIO_COMPONENTS["piezo"], False)


def cleanup_gpio():
    """Return outputs to a safe state and release GPIO resources."""
    GPIO.output(GPIO_COMPONENTS["led"], False)
    GPIO.output(GPIO_COMPONENTS["piezo"], False)
    GPIO.cleanup()


def main():
    """Run the continuous motion-monitoring loop."""
    configure_gpio()

    try:
        print("System ready - monitoring for motion...")
        last_state = False

        while True:
            current_state = GPIO.input(GPIO_COMPONENTS["pir_sensor"])

            if current_state and not last_state:
                GPIO.output(GPIO_COMPONENTS["led"], True)
                log_event("MOTION DETECTED")

                label, image_path = detect_object()

                if label == "person":
                    GPIO.output(GPIO_COMPONENTS["piezo"], True)
                    send_telegram_alert(
                        "Intruder detected! Check the attached image.",
                        image_path,
                    )
                    time.sleep(2)
                    GPIO.output(GPIO_COMPONENTS["piezo"], False)
                    log_event("ALERT: Person Detected")
                else:
                    GPIO.output(GPIO_COMPONENTS["piezo"], False)
                    send_telegram_alert("Normal motion detected.")
                    log_event(f"Normal Motion Detected: {label}")

            elif not current_state and last_state:
                GPIO.output(GPIO_COMPONENTS["led"], False)
                log_event("MOTION ENDED")

            last_state = current_state
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        cleanup_gpio()

        print("\nLast 5 Events:")
        for timestamp, event in list(motion_log)[-5:]:
            print(f"{timestamp}: {event}")

        print(f"\nFull log available at: {CSV_FILE}")
        print("GPIO cleanup complete.")


if __name__ == "__main__":
    main()
