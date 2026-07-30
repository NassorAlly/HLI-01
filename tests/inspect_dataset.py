import os
import numpy as np

DATASET_PATH = "dataset"

print("=" * 80)

total = 0

for gesture in sorted(os.listdir(DATASET_PATH)):

    folder = os.path.join(DATASET_PATH, gesture)

    if os.path.isdir(folder):

        files = sorted([f for f in os.listdir(folder)
                        if f.endswith(".npy")])

        print(f"\nGesture : {gesture}")
        print(f"Samples : {len(files)}")

        total += len(files)

        sample = np.load(os.path.join(folder, files[0]))

        print("Shape :", sample.shape)
        print("Type  :", sample.dtype)

print("\nTotal Samples :", total)

print("=" * 80)
