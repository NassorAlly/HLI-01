# Version 0.9.0 — Real-Time Inference Pipeline

## Added

- Reusable inference `Predictor`
- Dedicated real-time inference pipeline
- Trained-checkpoint loading for inference
- Webcam-based real-time sign recognition
- MediaPipe hand-landmark extraction
- 30-frame temporal sequence buffering
- Live sign prediction
- Prediction confidence reporting
- Class-probability output
- Input-shape validation for inference sequences
- NaN-value validation
- Infinite-value validation
- Automated Predictor tests
- Automated real-time inference tests

## Improved

- Extended HLI-01 from model training to operational inference
- Integrated the trained model checkpoint with the inference pipeline
- Improved checkpoint handling and loading behavior
- Protected the trained checkpoint from test-side modification
- Updated training entry-point references to Version 0.9.0
- Improved real-time transition between supported sign predictions
- Standardized inference output reporting
- Updated dependency configuration for reproducible real-time inference

## Real-Time Inference

Version 0.9.0 introduces webcam-based real-time sign-language
recognition.

The inference workflow is:

Webcam
→ Hand Detection
→ MediaPipe Landmark Extraction
→ 30-Frame Sequence Buffer
→ Trained Model
→ Predictor
→ Class Probabilities
→ Sign Prediction
→ Confidence Display

The real-time pipeline was functionally verified with the currently
supported sign classes:

- `hello`
- `no`
- `peace`
- `yes`

The system successfully responds to transitions between supported
signs during live webcam operation.

## Model Checkpoint

A fully trained checkpoint was used for Version 0.9.0 inference.

- Best checkpoint epoch: 40
- Best validation loss: approximately 0.0449
- Checkpoint successfully loaded for offline and real-time inference

A sample from the `peace` class was correctly predicted with
approximately 0.999 confidence.

This single-sample confidence value represents prediction confidence
for that inference example and should not be interpreted as overall
model accuracy.

## Environment Compatibility

The real-time inference environment was validated using:

- MediaPipe: 0.10.21
- OpenCV Contrib Python: 4.11.0.86
- NumPy: 1.26.4

These versions are pinned in `requirements.txt` to improve
reproducibility and prevent dependency incompatibilities.

## Testing

- Added Predictor test coverage
- Added real-time inference utility tests
- Verified checkpoint management behavior
- Verified inference input validation
- Verified model evaluation-mode behavior during inference
- Verified inference does not modify model parameters
- Full automated regression suite executed after Version 0.9.0 integration
- **86 tests passed**
- No regression detected in existing dataset, preprocessing, model,
  training, evaluation, or visualization components

---