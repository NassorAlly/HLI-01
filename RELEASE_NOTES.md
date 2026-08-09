# HLI-01 Version 0.9.0 Release Notes

**Release:** Version 0.9.0  
**Release Theme:** Real-Time Inference Pipeline  
**Status:** Stable Development Release  
**Release Date:** 9 August 2026

---

## Overview

HLI-01 Version 0.9.0 introduces the real-time inference pipeline for
the HLI-01 Sign Language Recognition Framework.

This release extends the training pipeline established in Version 0.8.0
by enabling trained HLI-01 models to perform both offline and real-time
sign-language inference.

The framework can now execute the operational pipeline:

Webcam
→ Hand Detection
→ MediaPipe Landmark Extraction
→ 30-Frame Sequence Buffering
→ Trained Model
→ Predictor
→ Class Probabilities
→ Sign Prediction
→ Confidence Display
→ Real-Time Recognition

---

## Major Highlights

Version 0.9.0 introduces:

- Reusable Predictor
- Trained-checkpoint loading for inference
- Real-time webcam inference
- MediaPipe hand-landmark extraction
- 30-frame sequence buffering
- Live sign prediction
- Prediction confidence display
- Real-time switching between recognized signs
- Input-shape validation
- NaN-value validation
- Infinite-value validation
- Inference unit tests
- Real-time inference tests
- Protected checkpoint handling during tests
- Reproducible MediaPipe/OpenCV/NumPy environment
- Full model training and checkpoint validation
- Expanded automated test suite

---

## Inference Pipeline

The inference subsystem provides reusable prediction functionality
for trained HLI-01 models.

The Predictor:

- Accepts a sequence with shape (30, 63)
- Validates sequence dimensions
- Rejects NaN and infinite values
- Converts input data to PyTorch tensors
- Places input on the configured device
- Runs inference without gradient computation
- Applies Softmax to model outputs
- Determines the predicted class
- Returns the class ID
- Returns the sign label
- Returns prediction confidence
- Returns class probabilities

---

## Real-Time Recognition

Version 0.9.0 introduces webcam-based real-time sign recognition.

The system:

- Captures live video frames
- Detects hand landmarks using MediaPipe
- Extracts numerical landmark features
- Accumulates temporal information across 30 frames
- Sends completed sequences to the trained model
- Produces live sign predictions
- Displays prediction confidence
- Responds to transitions between supported signs

The real-time inference pipeline was functionally verified using the
four currently supported HLI-01 sign classes:

- hello
- no
- peace
- yes

---

## Model Checkpoint

The trained model checkpoint was successfully loaded and verified for
inference.

During the final training run:

- Best checkpoint epoch: 40
- Best validation loss: approximately 0.0449

A sample inference check using a `peace` sequence produced the correct
class prediction with approximately 0.999 confidence.

This individual confidence result should not be interpreted as overall
model accuracy.

---

## Environment Compatibility

The real-time inference environment was validated using:

- MediaPipe 0.10.21
- OpenCV Contrib Python 4.11.0.86
- NumPy 1.26.4

These versions are pinned in `requirements.txt` to improve
reproducibility and prevent dependency incompatibilities.

---

## Testing

The complete HLI-01 automated test suite was executed after the
Version 0.9.0 changes.

Final result:

86 tests passed.

The test suite covers components including:

- Dataset processing
- Training
- Checkpoint management
- Resume training
- Prediction
- Input validation
- Model inference behavior
- Real-time inference utilities

---

## Version Progression

Version 0.6.0
→ Core Framework

Version 0.7.0
→ Data Pipeline

Version 0.8.0
→ Training Pipeline

Version 0.9.0
→ Real-Time Inference Pipeline

Version 1.0.0
→ Final Research-Ready Release