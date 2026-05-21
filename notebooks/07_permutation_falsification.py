import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.symbolic_search import random_symbolic_search
from src.proxies import generic_proxy_frame

from sklearn.datasets import load_breast_cancer, load_diabetes


def prepare_breast():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    y = data.target
    return y, generic_proxy_frame(X)


def prepare_diabetes():
    data = load_diabetes()

    X = pd.DataFrame(data.data)
    y_raw = data.target

    low = np.percentile(y_raw, 33)
    high = np.percentile(y_raw, 67)

    mask = (y_raw <= low) | (y_raw >= high)

    X = X[mask]
    y = (y_raw[mask] >= high).astype(int)

    return y, generic_proxy_frame(X)


datasets = {
    "breast": prepare_breast(),
    "diabetes": prepare_diabetes()
}

best_real, _ = random_symbolic_search(
    datasets,
    n_iter=4000,
    seed=42
)

rng = np.random.default_rng(123)

shuffled = {}

for name, (y, Xdf) in datasets.items():
    shuffled[name] = (
        rng.permutation(y),
        Xdf
    )

best_shuffled, _ = random_symbolic_search(
    shuffled,
    n_iter=4000,
    seed=42
)

print("REAL BEST")
print(best_real)

print("\nSHUFFLED BEST")
print(best_shuffled)

print("\nDELTA MEAN AUC")
print(best_real["mean_auc"] - best_shuffled["mean_auc"])
