from src.training.data_loader import DataLoaderManager

manager = DataLoaderManager()

train_loader, valid_loader, test_loader = manager.create()

print("=" * 60)

print("Training Batches  :", len(train_loader))
print("Validation Batches:", len(valid_loader))
print("Testing Batches   :", len(test_loader))

print()

X, y = next(iter(train_loader))

print("Batch Shape       :", X.shape)
print("Labels Shape      :", y.shape)
print("Data Type         :", X.dtype)

print("=" * 60)
