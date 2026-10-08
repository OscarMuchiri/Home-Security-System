# TensorFlow Lite Model Setup

The security application requires two local runtime files:

```text
detect.tflite
coco_labels.txt
```

Place both files in the repository root beside `home_security.py`.

## Why the model is not committed

The original TensorFlow Lite model is not currently redistributed in this repository because its exact source and redistribution terms have not yet been verified. The repository therefore documents the required model interface without publishing an unverified third-party model artifact.

## Expected model interface

The current application expects an object-detection TensorFlow Lite model with:

- one image input tensor;
- detection boxes at output index 0;
- class IDs at output index 1;
- confidence scores at output index 2.

The application reads the required input image width and height directly from the model. It uses a confidence threshold of `0.65`.

The label file must match the class-index ordering used by the model. For the intrusion path to activate, the relevant class label must resolve exactly to:

```text
person
```

## Validation before hardware use

When the original model files are recovered, verify the model input/output tensor layout before relying on the system for person detection. Different TensorFlow Lite object detectors can expose their outputs in a different order.

The repository cleanup preserves the original detector-output assumption rather than guessing a new model format without the original model artifact.
