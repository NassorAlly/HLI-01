from src.dataset.dataset_splitter import DatasetSplitter

splitter = DatasetSplitter()

(
    X_train,
    X_valid,
    X_test,
    y_train,
    y_valid,
    y_test,
) = splitter.split()

print("=" * 50)
print("Training Samples  :", len(X_train))
print("Validation Samples:", len(X_valid))
print("Testing Samples   :", len(X_test))

print()

print("Training Shape    :", X_train.shape)
print("Validation Shape  :", X_valid.shape)
print("Testing Shape     :", X_test.shape)

print("=" * 50)
