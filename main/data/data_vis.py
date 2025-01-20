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
    pprint(inputs[i][0].shape)
    print("Input 2")
    pprint(inputs[i][1].shape)
    print("Ground Truth")
    pprint(ground_truths[i].shape)
    print()