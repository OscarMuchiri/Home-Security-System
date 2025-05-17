
# Libraries

import RPi.GPIO as GPIO         # For controlling Raspberry Pi GPIO pins
import time
import os
import csv             # time: delays & timestamps, os: file paths, csv: event logging
from datetime import datetime    # For readable timestamps in logs
from collections import deque    # To keep recent events in memory
from pathlib import Path         # For handling file system paths
from picamera2 import Picamera2  # For controlling Pi Camera
import cv2
import numpy as np          # OpenCV and Numpy for image processing and AI inference
from telegram import Bot         # Telegram Bot API for sending alerts
import threading
import asyncio        # For asynchronous Telegram notifications
# TensorFlow Lite for AI object detection
from tflite_runtime.interpreter import Interpreter


#  GPIO Hardware Configuration

gpio_components = {
    'pir_sensor': 11,   # PIR motion sensor connected to GPIO pin 11
    'piezo': 7,         # Buzzer connected to GPIO pin 7
    'led': 13            # LED indicator connected to GPIO pin 13
}

GPIO.setmode(GPIO.BOARD)  # Using the physical pin numbering
GPIO.setup(gpio_components['pir_sensor'], GPIO.IN)
GPIO.setup(gpio_components['piezo'], GPIO.OUT)
GPIO.setup(gpio_components['led'], GPIO.OUT)
GPIO.output(gpio_components['led'], False)    # Ensure LED is initially off
GPIO.output(gpio_components['piezo'], False)  # Ensure buzzer is initially off


# Telegram Bot Configuration

TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN_HERE' 

CHAT_ID = 'YOUR_CHAT_ID_HERE'
bot = Bot(token=TELEGRAM_BOT_TOKEN)

# Set up an asynchronous loop for non-blocking message sending
loop = asyncio.new_event_loop()
threading.Thread(target=loop.run_forever, daemon=True).start()


async def send_alert_async(message, image_path=None):
    """Asynchronous function to send a Telegram message and optionally an image."""
    if image_path:
        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
    else:
        await bot.send_message(chat_id=CHAT_ID, text=message)


def send_telegram_alert(message, image_path=None):
    """Trigger async Telegram alert function from main code."""
    asyncio.run_coroutine_threadsafe(
        send_alert_async(message, image_path), loop
    )



# TensorFlow Lite AI Setup

interpreter = Interpreter(model_path="detect.tflite")  # Load AI model
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load object detection labels (e.g., person, cat, dog)
with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]


#Event Logging Configuration

DESKTOP_PATH = os.path.join(os.path.expanduser('~'), 'Desktop')
CSV_FILE = os.path.join(DESKTOP_PATH, "motion_log.csv")
motion_log = deque(maxlen=100)  # Store last 100 events in memory

# Creating the log file if it doesn't already exist
Path(DESKTOP_PATH).mkdir(parents=True, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "Event"])
    print(f"Created new log file at: {CSV_FILE}")


def log_event(event_type):
    """Log event to CSV and memory log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    motion_log.append((timestamp, event_type))
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([timestamp, event_type])
    print(f"{timestamp} - {event_type} (Logged)")


#  Object Detection Function

def detect_object():
    """Capture image and run object detection. Returns detected label and image path."""
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # Let camera adjust lighting
    image_path = os.path.join(DESKTOP_PATH, "frame.jpg")
    picam2.capture_file(image_path)
    picam2.stop()
    picam2.close()

    # Prepare image for AI model
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    resized = cv2.resize(image, (300, 300))
    input_data = np.expand_dims(resized, axis=0)

    # Run AI inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[
        0].astype(np.int32)
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Check detections with confidence > 65%
    for i in range(len(scores)):
        if scores[i] > 0.65:
            class_id = int(classes[i])
            confidence = int(scores[i] * 100)
            label = labels[class_id] if class_id < len(
                labels) else f"Unknown ({class_id})"
            print(f"Detected: {label} ({confidence}%)")

            # Draw detection box on image
            ymin, xmin, ymax, xmax = boxes[i]
            left, top = int(xmin * width), int(ymin * height)
            right, bottom = int(xmax * width), int(ymax * height)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({confidence}%)", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_image = os.path.join(DESKTOP_PATH, "detected.jpg")
            cv2.imwrite(detected_image, image)
            return label, detected_image

    return None, None  # No confident detections found



#  Main Monitoring Loop

try:
    print("✅ System ready - Monitoring for motion...")
    last_state = False  # Keeps track of previous motion state

    while True:
        current_state = GPIO.input(gpio_components['pir_sensor'])

        # If motion detected
        if current_state and not last_state:
            GPIO.output(gpio_components['led'], True)  # Turn on LED
            log_event("MOTION DETECTED")
            label, image_path = detect_object()

            if label == "person":
                GPIO.output(gpio_components['piezo'], True)  # Activate buzzer
                send_telegram_alert(
                    "Intruder detected! Check the attached image.", image_path)
                time.sleep(2)  # Keep buzzer on for 2 seconds
                GPIO.output(gpio_components['piezo'], False)
                log_event("ALERT: Person Detected")
            else:
                GPIO.output(gpio_components['piezo'], False)
                send_telegram_alert("Normal Motion detected.")  # Not a person
                log_event(f"Normal Motion Detected: {label}")

        # If motion ended
        elif not current_state and last_state:
            GPIO.output(gpio_components['led'], False)  # Turn off LED
            log_event("MOTION ENDED")

        last_state = current_state  # Update state for next loop
        time.sleep(0.05)  # Reduce CPU usage

except KeyboardInterrupt:
    print("\n🛑 Shutting down...")

finally:
    # Cleanup GPIO before exiting
    GPIO.output(gpio_components['led'], False)
    GPIO.output(gpio_components['piezo'], False)
    GPIO.cleanup()

    # Print last few events
    print("\n📋 Last 5 Events:")
    for event in list(motion_log)[-5:]:
        print(f"{event[0]}: {event[1]}")
    print(f"\n📂 Full log available at: {CSV_FILE}")
    print("✅ GPIO cleanup complete.")
