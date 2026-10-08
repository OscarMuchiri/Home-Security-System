# Hardware Revalidation Checklist

This checklist is for the next time the complete Raspberry Pi hardware setup is available. It is intentionally kept separate from the current software-only repository cleanup.

## 1. Physical setup

- Confirm Raspberry Pi model and OS version.
- Confirm Raspberry Pi Camera is detected by the operating system.
- Confirm PIR sensor is connected to physical BOARD pin 11 as expected.
- Confirm buzzer is connected to physical BOARD pin 7 as expected.
- Confirm LED is connected to physical BOARD pin 13 as expected.
- Verify power, grounding, resistor values and component ratings against the actual hardware before energizing the circuit.

## 2. Camera

- Confirm Picamera2 can initialize the camera.
- Capture a standalone test image.
- Confirm the image is written correctly to the configured output location.

## 3. TensorFlow Lite model

- Restore the original `detect.tflite` and matching `coco_labels.txt`.
- Inspect input tensor shape and dtype.
- Inspect output tensor ordering.
- Confirm output index 0 contains boxes.
- Confirm output index 1 contains class IDs.
- Confirm output index 2 contains confidence scores.
- Confirm the label mapping resolves a human detection to exactly `person`.
- Test detections above and below the 0.65 threshold.

## 4. Motion workflow

- Confirm the PIR sensor transitions correctly between motion/no-motion states.
- Confirm LED activation when motion begins.
- Confirm the camera captures only on a new motion event.
- Confirm non-person motion does not activate the buzzer.
- Confirm person detection activates the buzzer for the expected duration.

## 5. Telegram alerts

- Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as environment variables.
- Confirm a text-only notification can be delivered.
- Confirm a person-detection image can be delivered.
- Confirm no credentials are written to Git or logs.

## 6. Logging and shutdown

- Confirm `motion_log.csv` is created.
- Confirm timestamps and event labels are correct.
- Confirm `frame.jpg` and `detected.jpg` are written.
- Stop the application with Ctrl+C.
- Confirm GPIO cleanup completes and outputs return to the safe/off state.

## 7. Stability test

After individual checks pass, run a longer monitoring session and verify:

- repeated motion events do not cause camera initialization failures;
- Telegram notifications continue to send;
- the event log remains writable;
- the process does not accumulate obvious errors or lock GPIO resources.

## Completion

Only after the checks above pass should the repository status be changed from **hardware revalidation pending** to **hardware revalidated**.
