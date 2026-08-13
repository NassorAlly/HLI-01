# HLI-01 Sign Language Recognition Framework

**Current Version:** 0.9.0  
**Release Theme:** Real-Time Inference Pipeline  
**Status:** Stable Development Release

---

## Overview

HLI-01 is a modular research framework for sign-language recognition
using temporal hand-landmark sequences and deep-learning models.

The framework provides an end-to-end workflow covering:

- Sign-language dataset collection
- Dataset validation and preprocessing
- Stratified train/validation/test splitting
- PyTorch dataset and DataLoader management
- Deep-learning model construction
- Model training and validation
- Best-model checkpointing
- Checkpoint restoration and resume training
- Early stopping
- Learning-rate scheduling
- Held-out test-set evaluation
- Metrics and confusion-matrix generation
- Training and evaluation visualization
- Offline sign inference
- Real-time webcam-based sign recognition
- MediaPipe hand-landmark extraction
- Prediction confidence and class-probability reporting
- Automated regression testing

Version 0.9.0 extends the training capabilities introduced in Version
0.8.0 by adding the real-time inference pipeline.

A trained HLI-01 model can now process live webcam input, extract hand
landmarks, accumulate temporal sequences, and produce real-time sign
predictions.

The currently supported sign classes are:

- `hello`
- `no`
- `peace`
- `yes`

---

## Development Progression

HLI-01 is being developed incrementally, with each release introducing
a major layer of the complete sign-language recognition framework.

### Version 0.5.0 — Visualization System

Established the visualization and experiment-output infrastructure
required to inspect model behavior and research results.

Key capabilities included:

- Training-history visualization
- Confusion-matrix visualization
- Classification-metrics visualization
- Prediction visualization
- Experiment reporting
- Dashboard generation

↓

### Version 0.6.0 — Core Framework

Established the modular software architecture of HLI-01 and integrated
the major framework components.

Key capabilities included:

- Dataset module
- Models module
- Training module
- Evaluation module
- Metrics computation
- Confusion-matrix generation
- Evaluator
- Model factory and registry
- Modular visualization framework

↓

### Version 0.7.0 — Data Pipeline

Introduced the complete data-management pipeline required to prepare
sign-language sequences for model training and evaluation.

Key capabilities included:

- Dataset collection
- Dataset loading
- Dataset validation
- Automatic label management
- Stratified dataset splitting
- PyTorch Dataset integration
- Landmark normalization
- Feature scaling
- Temporal smoothing
- Dataset statistics
- Training DataLoader management

Dataset configuration:

- Sign classes: 4
- Classes: `hello`, `no`, `peace`, `yes`
- Samples per class: 100
- Total samples: 400
- Sequence length: 30 frames
- Features per frame: 63
- Training split: 70%
- Validation split: 15%
- Testing split: 15%

↓

### Version 0.8.0 — Training Pipeline

Connected the data pipeline to the HLI-01 model architecture and
introduced the complete model-training and evaluation lifecycle.

Key capabilities included:

- Batch-based PyTorch training
- Training and validation loops
- Cross-entropy loss
- Adam optimization
- Training and validation accuracy tracking
- Training-history management
- Best-model checkpointing
- Checkpoint restoration
- Resume training
- Early stopping
- Learning-rate scheduling
- Learning-rate history tracking
- Training-history visualization
- Held-out test evaluation
- Automatic evaluation artifact generation

Version 0.8.0 held-out test performance:

- Test samples: 60
- Accuracy: 98.33%
- Weighted Precision: 98.44%
- Weighted Recall: 98.33%
- Weighted F1-score: 98.33%
- Correct predictions: 59 / 60

↓

### Version 0.9.0 — Real-Time Inference Pipeline

**PREVIOUS RELEASE**

Version 0.9.0 extends HLI-01 from model development and training into
an operational real-time sign-language recognition framework.

Key capabilities include:

- Reusable Predictor
- Trained-checkpoint loading for inference
- Offline sequence inference
- Webcam-based real-time inference
- MediaPipe hand-landmark extraction
- 30-frame temporal sequence buffering
- Live sign prediction
- Prediction confidence reporting
- Class-probability output
- Input-shape validation
- NaN-value validation
- Infinite-value validation
- Real-time transition between supported signs
- Automated Predictor testing
- Automated real-time inference testing
- Reproducible MediaPipe/OpenCV/NumPy environment

The real-time pipeline has been functionally verified with the four
currently supported signs:

- `hello`
- `no`
- `peace`
- `yes`

A fully trained checkpoint was successfully used for offline and
real-time inference.

Training checkpoint information:

- Best checkpoint epoch: 40
- Best validation loss: approximately 0.0449

A sample from the `peace` class was correctly predicted with
approximately 0.999 confidence.

This single-sample confidence value represents model confidence for
that individual prediction and should not be interpreted as overall
model accuracy.

Automated regression status:

- 86 tests passed
- No regression detected in existing framework components

↓

### Version 1.0.0 — Research-Ready Framework

**CURRENT RELEASE**

Version 1.0.0 transforms HLI-01 into a reproducible and
research-ready experimental framework.

Version 1.0.0 includes:

- Reproducible experiment initialization
- Experiment metadata and logging
- Research metrics
- Model comparison
- Ablation support
- Experiment aggregation
- CSV comparison export
- Publication-oriented experiment comparison visualization
- Comparison integration into the PDF research report
- Comparison integration into the HTML dashboard
- Automated regression testing with 126 tests passing

---

## System Architecture

The current HLI-01 framework follows the general pipeline:

```text
Dataset
   ↓
Dataset Validation
   ↓
Preprocessing
   ↓
Train / Validation / Test Split
   ↓
PyTorch DataLoaders
   ↓
BiLSTM + Attention Model
   ↓
Training
   ↓
Validation
   ↓
Best-Model Checkpoint
   ↓
Held-Out Test Evaluation
   ↓
Offline / Real-Time Inference
   ↓
Prediction + Confidence
```

For real-time operation:

```text
Webcam
   ↓
Video Frame Capture
   ↓
MediaPipe Hand Detection
   ↓
Hand-Landmark Extraction
   ↓
30-Frame Sequence Buffer
   ↓
Trained HLI-01 Model
   ↓
Predictor
   ↓
Class Probabilities
   ↓
Predicted Sign
   ↓
Confidence Display
```

---

## Dataset

The current HLI-01 experimental dataset contains four sign classes:

```text
hello
no
peace
yes
```

Each class contains:

```text
100 sequences
```

Each sequence contains:

```text
30 frames
```

Each frame contains:

```text
63 hand-landmark features
```

Therefore, the current dataset contains:

```text
4 classes × 100 samples = 400 sequences
```

The dataset is divided using stratified splitting:

```text
Training   : 70%
Validation : 15%
Testing    : 15%
```

This produces:

```text
Training   : 280 samples
Validation : 60 samples
Testing    : 60 samples
```

---

## Model

HLI-01 currently uses a temporal deep-learning architecture based on:

```text
Hand-Landmark Sequence
        ↓
Bidirectional LSTM
        ↓
Attention
        ↓
Classification Layer
        ↓
Sign Prediction
```

The model processes temporal sequences of hand-landmark features and
learns discriminative temporal patterns for sign classification.

---

## Training Pipeline

The HLI-01 training pipeline provides:

- Batch-based training
- Cross-entropy loss computation
- Backpropagation
- Adam optimization
- Validation after every epoch
- Training-loss tracking
- Validation-loss tracking
- Training-accuracy tracking
- Validation-accuracy tracking
- Best-epoch identification
- Best-model checkpointing
- Early stopping
- Learning-rate scheduling
- Resume-training support
- Training-history recording

The main training entry point is:

```bash
python train.py
```

---

## Evaluation

The evaluation pipeline loads the best trained checkpoint and evaluates
the model using the held-out test dataset.

The evaluation system provides:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Prediction analysis
- Persistent evaluation artifacts

The main evaluation entry point is:

```bash
python evaluate.py
```

The Version 0.8.0 held-out test evaluation produced:

```text
Test samples       : 60
Correct predictions: 59 / 60
Accuracy           : 98.33%
Weighted Precision : 98.44%
Weighted Recall    : 98.33%
Weighted F1-score  : 98.33%
```

Confusion matrix:

```text
[[14  1  0  0]
 [ 0 15  0  0]
 [ 0  0 15  0]
 [ 0  0  0 15]]
```

These results are retained as the historical held-out evaluation
results associated with the Version 0.8.0 training-pipeline release.

---

## Inference

Version 0.9.0 introduces a reusable inference subsystem.

The Predictor accepts a sign sequence and:

1. Validates the input sequence.
2. Converts the input to `float32`.
3. Checks the expected sequence shape.
4. Rejects NaN values.
5. Rejects infinite values.
6. Converts the sequence to a PyTorch tensor.
7. Places the tensor on the configured device.
8. Runs the trained model in evaluation mode.
9. Applies Softmax to model logits.
10. Determines the predicted class.
11. Returns the prediction confidence.
12. Returns the complete class-probability distribution.

The prediction output includes:

```text
class_id
label
confidence
probabilities
```

---

## Real-Time Inference

Version 0.9.0 introduces live webcam-based sign recognition.

The real-time system performs:

- Webcam frame capture
- Hand detection
- MediaPipe hand-landmark extraction
- Temporal sequence construction
- 30-frame sequence buffering
- Model inference
- Live sign prediction
- Confidence reporting
- Transition between recognized signs

The real-time inference module can be launched from the project root
using:

```bash
python -m src.inference.realtime_inference
```

Press:

```text
Q
```

while the webcam window is active to close the real-time inference
application.

---

## Checkpoint Management

HLI-01 automatically saves the best-performing model checkpoint during
training.

The default checkpoint is stored at:

```text
outputs/checkpoints/best_model.pth
```

The checkpoint contains training information including:

- Model state
- Optimizer state
- Epoch
- Validation loss

For the trained checkpoint verified during Version 0.9.0 development:

```text
Best checkpoint epoch : 40
Best validation loss  : approximately 0.0449
```

Checkpoint loading is supported for:

- Evaluation
- Offline prediction
- Real-time prediction
- Resume training

---

## Visualization

HLI-01 provides visualization utilities for research analysis and
experiment inspection.

Visualization capabilities include:

- Training-loss curves
- Validation-loss curves
- Training-accuracy curves
- Validation-accuracy curves
- Learning-rate curves
- Confusion matrices
- Classification metrics
- Prediction visualization
- Experiment reporting
- Dashboard generation

---

## Generated Artifacts

Training and evaluation may generate artifacts including:

```text
outputs/checkpoints/best_model.pth
outputs/figures/loss_curve.png
outputs/figures/accuracy_curve.png
outputs/figures/learning_rate_curve.png
outputs/evaluation/evaluation_results.txt
outputs/evaluation/confusion_matrix.npy
```

These artifacts support experiment tracking, analysis, reporting, and
reproducibility.

---

## Project Structure

The HLI-01 repository follows a modular structure similar to:

```text
HLI-01/
│
├── configs/
├── dataset/
├── docs/
├── evaluation/
├── experiments/
├── outputs/
├── paper/
│
├── src/
│   ├── config/
│   ├── dataset/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── preprocessing/
│   ├── registry/
│   ├── training/
│   └── visualization/
│
├── tests/
│
├── .gitignore
├── CHANGELOG.md
├── evaluate.py
├── LICENSE.txt
├── predict.py
├── README.md
├── RELEASE_NOTES.md
├── requirements.txt
├── train.py
└── VERSION.txt
```

---

## Environment

HLI-01 Version 0.9.0 is developed using Python and PyTorch.

The real-time inference environment has been validated with:

```text
MediaPipe             0.10.21
OpenCV Contrib Python 4.11.0.86
NumPy                 1.26.4
```

These versions are pinned where necessary in `requirements.txt` to
maintain compatibility with the real-time inference pipeline.

---

## Installation

Create and activate a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check dependency compatibility:

```powershell
python -m pip check
```

---

## Running Tests

Run the complete automated regression suite from the project root:

```powershell
python -m pytest -q
```

At the completion of Version 0.9.0 development, the regression suite
reported:

```text
86 passed
```

The automated tests cover major framework components including:

- Dataset loading
- Dataset validation
- Dataset splitting
- Preprocessing
- Dataset statistics
- Label management
- Models
- Forward passes
- Training
- Early stopping
- Learning-rate scheduling
- Checkpoint management
- Resume training
- Evaluation
- Metrics
- Visualization
- Predictor behavior
- Inference input validation
- Real-time inference utilities

---

## Reproducibility

HLI-01 incorporates several mechanisms to support reproducible
experimentation:

- Centralized configuration
- Fixed random-seed configuration
- Stratified dataset splitting
- Explicit train/validation/test separation
- Best-model checkpointing
- Versioned source code
- Versioned dependency configuration
- Automated regression testing
- Persistent evaluation artifacts
- Version-specific release notes
- Project changelog

---

## Current Status

HLI-01 Version 1.0.0 provides a research-ready pipeline covering:

```text
Dataset Collection
        ↓
Dataset Validation
        ↓
Preprocessing
        ↓
Dataset Splitting
        ↓
Model Construction
        ↓
Training
        ↓
Validation
        ↓
Checkpointing
        ↓
Held-Out Evaluation
        ↓
Offline Inference
        ↓
Real-Time Webcam Inference
```

Current development status:

- Dataset pipeline: Operational
- Model architecture: Operational
- Training pipeline: Operational
- Evaluation pipeline: Operational
- Checkpoint management: Operational
- Offline inference: Operational
- Real-time webcam inference: Operational
- Automated regression suite: 126 tests passing

Version 1.0.0 is the current research-ready release.

---

## Version History

### Version 0.5.0

**Theme:** Visualization System

Established the visualization and experiment-output infrastructure.

### Version 0.6.0

**Theme:** Building the Core Framework

Established the modular framework architecture, evaluation utilities,
visualization framework, model infrastructure, and automated testing.

### Version 0.7.0

**Theme:** Building the Data Pipeline

Introduced dataset collection, loading, validation, stratified
splitting, preprocessing, statistics, label management, and PyTorch
dataset integration.

### Version 0.8.0

**Theme:** Building the Training Pipeline

Introduced the complete training, validation, checkpointing,
learning-rate scheduling, early-stopping, resume-training, and
held-out evaluation pipeline.

Historical held-out test performance:

- Accuracy: 98.33%
- Weighted Precision: 98.44%
- Weighted Recall: 98.33%
- Weighted F1-score: 98.33%

### Version 0.9.0

**Theme:** Real-Time Inference Pipeline

Introduced reusable prediction, trained-checkpoint inference,
MediaPipe-based hand-landmark extraction, webcam-based real-time
recognition, live confidence reporting, inference validation, and
expanded automated testing.

Historical automated regression status:

```text
86 tests passed
```

---

## Current Release

### Version 1.0.0 — Research-Ready Framework

Version 1.0.0 establishes the research-ready HLI-01 framework with
reproducible experiment management, research metrics, model comparison,
experiment aggregation, research visualization, PDF reporting, and
HTML dashboard integration.

---

## HLI-01 Development Path

Version 0.5.0  
**Visualization System**

↓

Version 0.6.0  
**Core Framework**

↓

Version 0.7.0  
**Data Pipeline**

↓

Version 0.8.0  
**Training Pipeline**

↓

Version 0.9.0
**Real-Time Inference Pipeline**

↓

Version 1.0.0
**Research-Ready Framework — CURRENT RELEASE**

---

## License

HLI-01 is released under the MIT License.

See:

```text
LICENSE.txt
```

for the complete license terms.

---

## Author

**Dr. Nassor Ally Nassor**

HLI-01 Sign Language Recognition Framework

---

## Citation

A formal research citation will be provided with associated HLI-01
research publications.