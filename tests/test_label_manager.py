from src.dataset.dataset_collector import DatasetCollector
from src.dataset.label_manager import LabelManager

collector = DatasetCollector("dataset")

manager = LabelManager(
    collector.get_classes()
)

print("Classes:")
print(manager.get_labels())

print()

print("Label Mapping:")
print(manager.label_to_id)

print()

print("Reverse Mapping:")
print(manager.id_to_label)

print()

print("Encode 'hello':", manager.encode("hello"))
print("Decode 0:", manager.decode(0))
