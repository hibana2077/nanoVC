import numpy as np

#load data
data = np.load("./tts_dataset.npz", allow_pickle=True)

# NpzFile './tts_dataset.npz' with keys: inputs, ground_truths
inputs = data["inputs"]
ground_truths = data["ground_truths"]

from pprint import pprint

# Print the first 5 pairs
for i in range(min(5, len(inputs))):
    print(f"Pair {i+1}")
    print("Input 1")
    print(f"Shape: {inputs[i][0].shape}, Time: {inputs[i][0].shape[0]/22050:.2f} seconds")
    print("Input 2")
    print(f"Shape: {inputs[i][1].shape}, Time: {inputs[i][1].shape[0]/22050:.2f} seconds")
    print("Ground Truth")
    print(f"Shape: {ground_truths[i].shape}, Time: {ground_truths[i].shape[0]/22050:.2f} seconds")
    print()