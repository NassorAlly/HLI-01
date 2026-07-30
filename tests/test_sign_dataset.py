from src.dataset.dataset_splitter import DatasetSplitter
from src.dataset.sign_dataset import SignDataset

# Split the dataset
splitter = DatasetSplitter()

(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
) = splitter.split()

# Create the PyTorch Dataset
train_dataset = SignDataset(X_train, y_train)

print("=" * 60)

print("Dataset Length :", len(train_dataset))

sample, label = train_dataset[0]

print("Sample Shape   :", sample.shape)
print("Label          :", label)
print("Tensor Type    :", sample.dtype)
print("Input Shape    :", train_dataset.input_shape)
print("Classes        :", train_dataset.num_classes)

print("=" * 60)
