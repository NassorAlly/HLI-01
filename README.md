# HLI-01: Holistic Language Intelligence

*A Modular Deep Learning Framework for Dynamic Sign Language Recognition*

**Current Version:** 0.8.0  
**Release Theme:** Building the Training Pipeline  
**Status:** Stable Development Release

---

## Overview

HLI-01 (Holistic Language Intelligence) is a modular deep learning framework
for dynamic hand gesture and sign language recognition.

The framework is designed for research, education, experimentation, and the
development of intelligent sign-language recognition systems using sequential
hand-landmark data.

HLI-01 processes temporal hand-landmark sequences extracted using MediaPipe
and provides an extensible architecture covering the complete machine-learning
workflow.

The current framework includes:

- Dataset collection
- Dataset validation
- Dataset preprocessing
- Label management
- Dataset statistics
- Stratified dataset splitting
- PyTorch dataset and DataLoader management
- LSTM-based sequence modelling
- Model training
- Validation
- Early stopping
- Learning-rate scheduling
- Model checkpointing
- Resume-training support
- Held-out test evaluation
- Performance metrics
- Confusion matrix analysis
- Training visualization
- Prediction visualization
- Experiment reporting
- HTML dashboard generation
- Automated regression testing

Version **0.8.0** establishes the complete HLI-01 training pipeline and connects
the data, model, training, evaluation, and visualization subsystems into an
integrated experimental workflow.

---

# Development Progress

HLI-01 is being developed incrementally through clearly defined framework
milestones.

```text
v0.5.0
Visualization System
        ↓
v0.6.0
Core Framework
        ↓
v0.7.0
Data Pipeline
        ↓
v0.8.0
Training Pipeline
        ↓
Future Development
        ↓
v1.0.0
Research-Ready Framework
```

---

# Current Dataset Configuration

The current HLI-01 experimental dataset contains four sign classes:

```text
hello
no
peace
yes
```

Dataset configuration:

| Parameter | Value |
|---|---:|
| Number of classes | 4 |
| Samples per class | 100 |
| Total samples | 400 |
| Sequence length | 30 frames |
| Features per frame | 63 |
| Data type | float32 |
| Training split | 70% |
| Validation split | 15% |
| Testing split | 15% |
| Training samples | 280 |
| Validation samples | 60 |
| Test samples | 60 |

A fixed random seed is used to support reproducible dataset splitting.

---

# Framework Architecture

The current HLI-01 workflow can be represented as:

```text
Raw Sign Samples
       │
       ▼
Dataset Collection
       │
       ▼
Dataset Validation
       │
       ▼
Preprocessing
       │
       ▼
Label Management
       │
       ▼
Stratified Dataset Split
       │
       ├─────────────┬─────────────┐
       ▼             ▼             ▼
    Training      Validation      Testing
       │             │             │
       └──────┬──────┘             │
              ▼                    │
        Model Training             │
              │                    │
              ▼                    │
         Validation                │
              │                    │
              ▼                    │
     Learning-Rate Scheduler       │
              │                    │
              ▼                    │
        Early Stopping             │
              │                    │
              ▼                    │
       Best Checkpoint             │
              │                    │
              └──────────┬─────────┘
                         ▼
                  Final Evaluation
                         │
                         ▼
              Metrics + Confusion Matrix
                         │
                         ▼
                 Evaluation Artifacts
```

---

# Features

## Dataset Module

The dataset subsystem provides:

- Dataset Collector
- Dataset Loader
- Dataset Validator
- Dataset Statistics
- Dataset Splitter
- Label Manager
- PyTorch `SignDataset`
- DataLoader management

The framework validates dataset structure and supports reproducible,
stratified train-validation-test splitting.

---

## Preprocessing Module

The preprocessing pipeline includes:

- Landmark normalization
- Landmark scaling
- Temporal smoothing
- Batch preprocessing
- Configurable preprocessing stages
- Input-dimension validation

These components prepare landmark sequences before they are supplied to the
learning pipeline.

---

## Model Module

The model subsystem provides reusable sequence-classification components.

Current capabilities include:

- LSTM-based sequence model
- Configurable input dimensions
- Configurable hidden dimensions
- Configurable number of recurrent layers
- Dropout
- Multi-class classification output
- Base model architecture
- Model Factory
- Model Registry

The modular architecture is intended to support additional sequence models in
future HLI-01 experiments.

---

# Training Module

Version 0.8.0 introduces the complete training subsystem.

Major capabilities include:

- Batch-based PyTorch training
- Cross-entropy loss
- Adam optimization
- Training-loss tracking
- Training-accuracy tracking
- Validation-loss tracking
- Validation-accuracy tracking
- Training-history management
- Best-epoch tracking
- Automatic best-model checkpointing
- Early stopping
- Learning-rate scheduling
- Resume-training support
- Automatic training visualization

---

## Training Configuration

Central training parameters are maintained in:

```text
src/config/settings.py
```

Core configuration includes:

```text
SEQUENCE_LENGTH = 30
NUM_FEATURES    = 63
NUM_CLASSES     = 4

TRAIN_RATIO     = 0.70
VALID_RATIO     = 0.15
TEST_RATIO      = 0.15

RANDOM_SEED     = 42

BATCH_SIZE      = 32
NUM_EPOCHS      = 50
LEARNING_RATE   = 0.001

HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
DROPOUT         = 0.30
```

Additional training settings control early stopping, checkpoint management,
learning-rate scheduling, and output directories.

---

# Early Stopping

HLI-01 monitors validation loss during training.

If validation performance stops improving for the configured number of epochs,
training can terminate automatically.

This reduces unnecessary training and helps limit overfitting.

---

# Learning-Rate Scheduling

Version 0.8.0 integrates a validation-aware learning-rate scheduler based on
PyTorch `ReduceLROnPlateau`.

The scheduler monitors validation loss and reduces the optimizer learning rate
when improvement stalls.

The learning-rate scheduler has been independently tested and verified.

---

# Checkpoint Management

HLI-01 automatically saves the best model according to validation loss.

The primary checkpoint is stored at:

```text
outputs/checkpoints/best_model.pth
```

Checkpoint information includes:

- Model state
- Optimizer state
- Epoch
- Validation loss

The checkpoint system also supports restoration of model and optimizer state
for resumed training.

---

# Training the Model

Run the complete training pipeline from the project root:

```bash
python train.py
```

The training pipeline performs:

```text
Load Dataset
    ↓
Create Train / Validation / Test DataLoaders
    ↓
Create Model
    ↓
Create Optimizer
    ↓
Train Model
    ↓
Validate Model
    ↓
Adjust Learning Rate
    ↓
Apply Early Stopping
    ↓
Save Best Checkpoint
    ↓
Generate Training Figures
```

The maximum number of epochs is configured centrally, while early stopping can
terminate training sooner when appropriate.

---

# Training Outputs

Training automatically generates:

```text
outputs/
│
├── checkpoints/
│   └── best_model.pth
│
└── figures/
    ├── loss_curve.png
    ├── accuracy_curve.png
    └── learning_rate_curve.png
```

The figures show:

- Training loss
- Validation loss
- Training accuracy
- Validation accuracy
- Learning-rate evolution

---

# Evaluation Module

The evaluation subsystem computes classification performance using the
held-out test dataset.

Supported metrics include:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Confusion Matrix

The evaluation components are implemented under:

```text
src/evaluation/
```

---

# Evaluating the Best Model

After training, run:

```bash
python evaluate.py
```

The evaluation entry point:

1. Loads the test DataLoader.
2. Reconstructs the model architecture.
3. Loads the best saved checkpoint.
4. Performs inference on the held-out test split.
5. Computes classification metrics.
6. Computes the confusion matrix.
7. Displays the results.
8. Saves the official evaluation artifacts.

---

# Version 0.8.0 Test Results

The best saved checkpoint was evaluated using the held-out 60-sample test
split.

| Metric | Result |
|---|---:|
| Accuracy | **98.33%** |
| Weighted Precision | **98.44%** |
| Weighted Recall | **98.33%** |
| Weighted F1-score | **98.33%** |
| Correct Predictions | **59 / 60** |

The resulting confusion matrix was:

```text
[[14  1  0  0]
 [ 0 15  0  0]
 [ 0  0 15  0]
 [ 0  0  0 15]]
```

The model correctly classified **59 of the 60 held-out test samples**.

---

# Evaluation Outputs

Final evaluation artifacts are automatically saved under:

```text
outputs/evaluation/
```

Current artifacts:

```text
outputs/evaluation/
├── evaluation_results.txt
└── confusion_matrix.npy
```

`evaluation_results.txt` provides a human-readable record of the experiment
results.

`confusion_matrix.npy` preserves the confusion matrix as a NumPy array for
future programmatic analysis.

---

# Visualization Module

The HLI-01 visualization framework includes:

- Training-loss visualization
- Training-accuracy visualization
- Learning-rate visualization
- Confusion-matrix visualization
- Classification-metrics visualization
- Prediction visualization
- Experiment report generation
- HTML dashboard generation

The visualization system uses a non-interactive Matplotlib backend to support
automated testing and headless execution.

---

# Additional Generated Outputs

Depending on the executed visualization and reporting components, HLI-01 can
produce:

```text
outputs/
│
├── checkpoints/
│   └── best_model.pth
│
├── figures/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── learning_rate_curve.png
│
├── evaluation/
│   ├── evaluation_results.txt
│   └── confusion_matrix.npy
│
├── confusion_matrix/
│   └── confusion_matrix.png
│
├── metrics/
│   └── classification_metrics.png
│
├── predictions/
│   └── prediction_summary.txt
│
├── reports/
│   └── experiment_report.pdf
│
└── dashboard/
    └── index.html
```

---

# Project Structure

The HLI-01 repository follows a modular `src/` architecture.

```text
HLI-01/
│
├── dataset/
│
├── src/
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── dataset/
│   │
│   ├── evaluation/
│   │
│   ├── models/
│   │
│   ├── preprocessing/
│   │
│   ├── registries/
│   │
│   ├── training/
│   │
│   └── visualization/
│
├── tests/
│   └── data/
│
├── outputs/
│   ├── checkpoints/
│   ├── evaluation/
│   ├── figures/
│   ├── confusion_matrix/
│   ├── metrics/
│   ├── predictions/
│   ├── reports/
│   └── dashboard/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
└── RELEASE_NOTES.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/NassorAlly/HLI-01.git
cd HLI-01
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify Python

```bash
python --version
```

HLI-01 v0.8.0 has been developed and tested using Python 3.11.

---

# Basic Usage

A typical HLI-01 experiment can be executed using:

```bash
python train.py
```

followed by:

```bash
python evaluate.py
```

This provides the basic workflow:

```text
Dataset
   ↓
Training
   ↓
Best Checkpoint
   ↓
Held-Out Test Evaluation
   ↓
Metrics and Evaluation Artifacts
```

---

# Running Tests

Run the complete regression suite:

```bash
pytest -q
```

Final v0.8.0 regression result:

```text
69 passed
```

The final regression run completed with all tests passing.

For detailed test names:

```bash
pytest -v
```

---

# Selected Component Tests

Training engine:

```bash
python tests/test_trainer.py
```

Early stopping:

```bash
python tests/test_early_stopping.py
```

Checkpoint management:

```bash
python tests/test_checkpoint_manager.py
```

DataLoader management:

```bash
python tests/test_data_loader.py
```

Training visualization:

```bash
python tests/test_training_plots.py
```

Evaluation:

```bash
python tests/test_evaluator.py
```

Visualization pipeline:

```bash
python tests/test_visualization_pipeline.py
```

---

# Testing Coverage

The current automated test suite verifies major components including:

- Dataset collection
- Dataset loading
- Dataset validation
- Dataset statistics
- Dataset splitting
- Label management
- SignDataset
- Preprocessing
- Model components
- Training engine
- Early stopping
- Checkpoint management
- DataLoader management
- Evaluation metrics
- Confusion matrix
- Training plots
- Metrics plots
- Prediction visualization
- Experiment reports
- Dashboard generation
- Visualization pipeline integration

At the completion of Version 0.8.0:

```text
69 tests passed
```

---

# Version History

## Version 0.5.0 — Visualization System

Introduced the visualization and experiment-reporting subsystem.

Major additions included:

- Training visualization
- Confusion matrix visualization
- Classification metrics visualization
- Prediction summaries
- PDF experiment reports
- HTML dashboard

---

## Version 0.6.0 — Core Framework

Introduced the modular software-engineering architecture.

Major additions included:

- Standardized `src/` architecture
- Registry framework
- Evaluation framework
- Visualization framework
- Model infrastructure
- Automated regression testing

---

## Version 0.7.0 — Data Pipeline

Introduced the structured data-processing subsystem.

Major additions included:

- Dataset collection
- Dataset loading
- Dataset validation
- Dataset statistics
- Label management
- Stratified splitting
- PyTorch dataset integration
- Preprocessing pipeline
- DataLoader management

---

## Version 0.8.0 — Training Pipeline

Introduces the complete training and test-evaluation workflow.

Major additions include:

- Training engine
- Validation tracking
- Early stopping
- Best-model checkpointing
- Resume-training support
- Learning-rate scheduling
- Training-history management
- Automatic training plots
- Centralized training configuration
- Held-out test evaluation
- Persistent evaluation artifacts

Official held-out test accuracy:

```text
98.33%
```

---

# Roadmap

Development after v0.8.0 will continue toward a research-ready Version 1.0.0.

Potential future milestones include:

### Experiment Infrastructure

- Structured experiment management
- Experiment metadata
- Configuration snapshots
- Reproducibility tracking
- Model comparison

### Inference

- Dedicated inference pipeline
- Single-sequence prediction
- Confidence reporting
- Real-time recognition support

### Model Research

- Additional recurrent architectures
- Attention-based architectures
- Model comparison experiments
- Hyperparameter experiments
- Ablation studies

### Research Evaluation

- Per-class metrics
- Classification reports
- Additional evaluation visualizations
- Cross-validation experiments
- Statistical model comparison

### Deployment Research

- TorchScript
- ONNX export
- Model optimization
- Real-time performance analysis

### Version 1.0.0

The Version 1.0.0 milestone is intended to represent a stable,
research-ready HLI-01 framework suitable for systematic experiments and
publication-oriented studies.

---

# Reproducibility

HLI-01 is designed around reproducible experimentation.

Current reproducibility mechanisms include:

- Centralized configuration
- Fixed dataset-splitting seed
- Stratified splitting
- Explicit train-validation-test separation
- Saved best-model checkpoint
- Saved optimizer state
- Training-history tracking
- Persisted evaluation results
- Persisted confusion matrix
- Automated regression testing

Future releases will extend experiment metadata and configuration tracking.

---

# Citation

If you use HLI-01 in academic research, please cite the corresponding HLI-01
publication when available.

Formal publication citation information will be added after publication.

---

# License

This project is released under the MIT License.

---

# Acknowledgements

HLI-01 builds on open-source scientific and machine-learning technologies,
including:

- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- ReportLab
- Pytest

---

# Author

**Dr. Nassor Ally Nassor**

Department of Electronics and Telecommunication Engineering  
College of Information and Communication Technologies (CoICT)  
University of Dar es Salaam

---

**HLI-01 Version 0.8.0**

*Building the Training Pipeline*