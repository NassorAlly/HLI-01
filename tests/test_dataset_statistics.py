from src.dataset.dataset_statistics import DatasetStatistics

print("DatasetStatistics imported from:")
print(DatasetStatistics.__module__)
print(DatasetStatistics.__init__.__code__.co_filename)

stats = DatasetStatistics("dataset")

stats.summary()
