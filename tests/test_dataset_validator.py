from src.dataset.dataset_validator import DatasetValidator

validator = DatasetValidator()

errors = validator.validate()

print()
print(f"Total Errors: {len(errors)}")
