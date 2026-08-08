# HLI-01 Changelog

All notable development milestones of the HLI-01 Sign Language
Recognition Framework are documented here.

---

# Version 0.8.0 — Building the Training Pipeline

## Added

- Complete model training pipeline
- Centralized training configuration
- PyTorch DataLoader management for training, validation, and testing
- Training and validation loops
- Training and validation accuracy tracking
- Training-history recording
- Best-epoch tracking
- Automatic best-model checkpointing
- Checkpoint loading and restoration
- Resume-training support
- Early stopping
- Configurable early-stopping patience and minimum delta
- ReduceLROnPlateau learning-rate scheduler
- Learning-rate history tracking
- Automatic training visualization integration
- Learning-rate curve visualization
- Final held-out test-set evaluation pipeline
- Automatic persistence of evaluation results
- NumPy confusion-matrix artifact

## Improved

- Integrated training parameters with centralized `settings.py`
- Added reproducible random-seed configuration
- Improved Trainer architecture and validation workflow
- Extended checkpoint management for training restoration
- Integrated existing visualization framework directly with Trainer history
- Extended training plots to support:
  - Training and validation loss
  - Training and validation accuracy
  - Learning-rate scheduling
- Converted root `train.py` into the main training entry point
- Converted root `evaluate.py` into the final evaluation entry point
- Preserved separation between training, validation, and held-out testing
- Standardized v0.8.0 training output reporting

## Evaluation

Final evaluation was performed using the best saved checkpoint on the
held-out test split.

- Test samples: 60
- Accuracy: 98.33%
- Weighted Precision: 98.44%
- Weighted Recall: 98.33%
- Weighted F1-score: 98.33%
- Correct predictions: 59 / 60

Confusion Matrix:

    [[14  1  0  0]
     [ 0 15  0  0]
     [ 0  0 15  0]
     [ 0  0  0 15]]

## Generated Training Artifacts

- `outputs/checkpoints/best_model.pth`
- `outputs/figures/loss_curve.png`
- `outputs/figures/accuracy_curve.png`
- `outputs/figures/learning_rate_curve.png`
- `outputs/evaluation/evaluation_results.txt`
- `outputs/evaluation/confusion_matrix.npy`

## Testing

- Full automated regression suite executed after training-pipeline integration
- 69 tests passed
- No regression detected in existing dataset, preprocessing, model,
  evaluation, or visualization components

---

# Version 0.7.0 — Building the Data Pipeline

## Added

- Dataset collection framework
- Dataset loader
- Dataset validator
- Stratified dataset splitter
- PyTorch SignDataset
- Label management
- Dataset statistics utilities
- Data preprocessing pipeline
- Landmark normalization
- Landmark scaling
- Temporal smoothing
- Training DataLoader management

## Improved

- Reproducible dataset splitting
- Dataset integrity validation
- Class-label consistency
- Batch-based PyTorch data access
- Separation of training, validation, and testing datasets
- Modular preprocessing architecture

## Dataset Configuration

- Sign classes: 4
- Classes: `hello`, `no`, `peace`, `yes`
- Samples per class: 100
- Total samples: 400
- Sequence length: 30 frames
- Features per frame: 63
- Data type: `float32`

Dataset split:

- Training: 70%
- Validation: 15%
- Testing: 15%

## Testing

- Added comprehensive automated tests for the data pipeline
- Dataset loading tested
- Dataset validation tested
- Stratified splitting tested
- Preprocessing components tested
- Dataset statistics tested
- Label management tested
- All data-pipeline tests passing

---

# Version 0.6.0 — Building the Core Framework

## Added

- Core framework architecture
- Registry package
- Visualization framework
- Evaluation framework
- Dashboard generator
- Experiment report generator
- Prediction visualizer
- Training plot generator
- Confusion matrix plotter
- Metrics plotter

## Improved

- Standardized `src` package structure
- Unified visualization imports
- Non-interactive Matplotlib backend (`Agg`)
- Consistent Version 0.6.0 headers

## Testing

- Added automated pytest regression suite
- 10 visualization and evaluation tests
- All tests passing