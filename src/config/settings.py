"""
settings.py

Central configuration for the HLI-01 project.
"""

# =====================================================
# DATASET
# =====================================================

DATASET_PATH = "dataset"

SEQUENCE_LENGTH = 30

NUM_FEATURES = 63

NUM_CLASSES = 4

# =====================================================
# DATASET SPLITTING
# =====================================================

TRAIN_RATIO = 0.70

VALID_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 32

NUM_EPOCHS = 50

LEARNING_RATE = 0.001

# =====================================================
# MODEL
# =====================================================

INPUT_SIZE = NUM_FEATURES

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.30

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cpu"

# =====================================================
# FILES
# =====================================================

MODEL_PATH = "models/sign_lstm_model.pth"

CLASSES_PATH = "models/classes.npy"
