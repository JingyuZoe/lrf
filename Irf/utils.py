# lrf/utils.py
import numpy as np

def generate_s(rainfall, alpha):
    S = np.zeros_like(rainfall)
    for t in range(1, len(rainfall)):
        S[t] = alpha * S[t - 1] + (1 - alpha) * rainfall[t]
    return S
