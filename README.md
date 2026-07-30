# HLI-01: Holistic Language Intelligence

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Status](https://img.shields.io/badge/Version-v0.5.0-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

## Overview

HLI-01 (Holistic Language Intelligence) is a deep learning framework for dynamic hand gesture and sign language recognition. The framework is designed for research, education, and real-time intelligent systems using sequential hand landmark data extracted from MediaPipe.

The project implements a complete machine learning workflow, including:

- Dataset collection
- Dataset preprocessing
- BiLSTM + Attention neural network
- Model training
- Model evaluation
- Visualization
- Experiment reporting
- Dashboard generation

Version **0.5.0** provides a complete visualization subsystem with automated report generation and integration testing.

---

# Features

## Dataset Module

- Dataset Collector
- Dataset Loader
- Dataset Validator
- Dataset Statistics
- Dataset Splitter
- Label Manager

## Model Module

- BiLSTM Network
- Attention Mechanism
- Model Factory
- Checkpoint Manager

## Training Module

- Model Trainer
- Early Stopping
- Training History

## Evaluation Module

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Model Evaluator

## Visualization Module

- Training Loss Curve
- Accuracy Curve
- Confusion Matrix Visualization
- Classification Metrics Visualization
- Prediction Summary
- PDF Experiment Report
- HTML Dashboard

## Testing

- Unit Tests
- Integration Tests
- Visualization Pipeline Test

---

# Project Structure

```text
HLI-01 Version 0.5.0/

│
├── dataset/
├── evaluation/
├── models/
├── training/
├── visualization/
├── outputs/
│
├── checkpoints/
├── confusion_matrix/
├── figures/
├── logs/
├── metrics/
├── predictions/
├── reports/
├── dashboard/
│
├── tests/
│
├── requirements.txt
├── README.md
└── CHANGELOG.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/yourusername/HLI-01.git
```

or copy the project folder.

---

## 2. Create a virtual environment (Recommended)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify installation

```bash
python --version
```

Expected

```
Python 3.11+
```

---

# Training the Model

Train the BiLSTM + Attention model using:

```bash
python train.py
```

After training, the model checkpoint will be stored in:

```text
outputs/checkpoints/
```

Training history will be automatically saved for visualization.

---

# Evaluating the Model

Run the evaluation module:

```bash
python evaluate.py
```

The evaluation module computes:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Generating Visualizations

Generate the visualization outputs using:

Training Curves

```bash
python tests/test_training_plots.py
```

Confusion Matrix

```bash
python tests/test_confusion_plot.py
```

Classification Metrics

```bash
python tests/test_metrics_plot.py
```

Prediction Summary

```bash
python tests/test_prediction_visualizer.py
```

Experiment Report

```bash
python tests/test_experiment_report.py
```

Dashboard

```bash
python tests/test_dashboard.py
```

---

# Generated Outputs

After execution, the following outputs are produced:

```text
outputs/

├── figures/
│     loss_curve.png
│     accuracy_curve.png
│
├── confusion_matrix/
│     confusion_matrix.png
│
├── metrics/
│     classification_metrics.png
│
├── predictions/
│     prediction_summary.txt
│
├── reports/
│     experiment_report.pdf
│
└── dashboard/
      index.html
```

---

# Running Tests

Run individual tests

Training Plot

```bash
python tests/test_training_plots.py
```

Confusion Matrix

```bash
python tests/test_confusion_plot.py
```

Metrics

```bash
python tests/test_metrics_plot.py
```

Prediction

```bash
python tests/test_prediction_visualizer.py
```

Experiment Report

```bash
python tests/test_experiment_report.py
```

Dashboard

```bash
python tests/test_dashboard.py
```

---

## Integration Test

Run the complete visualization pipeline

```bash
python tests/test_visualization_pipeline.py
```

Expected output

```text
ALL VISUALIZATION MODULES PASSED

VISUALIZATION PIPELINE PASSED
```

---

# Current Version

Current Release

```
Version 0.5.0
```

Status

```
Stable Release
```

---

# Roadmap

## Version 0.6.0

Planned improvements

- src/ architecture
- Base classes
- Configuration management
- Logging
- Experiment Manager
- Documentation improvements
- Code refactoring

## Version 0.7.0

- Real-time webcam recognition
- Probability visualization
- FPS counter
- Temporal smoothing

## Version 0.8.0

- ONNX export
- TorchScript export
- Quantization
- Performance optimization

## Version 0.9.0

- Desktop GUI
- Dataset Manager
- Model Manager

## Version 1.0.0

- Production Release
- API
- Installer
- GitHub Release

---

# Author

**Dr. Nassor Ally Nassor**

Department of Electronics and Telecommunication Engineering

College of Information and Communication Technologies (CoICT)

University of Dar es Salaam

---

# Citation

If you use HLI-01 in your research, please cite the corresponding publication when available.

---

# License

This project is released under the MIT License.

---

# Acknowledgements

- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- ReportLab

---

**HLI-01 Version 0.5.0**

*A Modular Deep Learning Framework for Dynamic Sign Language Recognition*