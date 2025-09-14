import numpy as np

targets = np.load("./precomputed_features/targets.npy")
print("Target range:", targets.min(), targets.max())
print("Target mean:", targets.mean())
