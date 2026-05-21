import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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

    low = y_raw <= y_raw.mean()
    y = low.astype(int)

    return y, generic_proxy_frame(X)


datasets = {
    "breast": prepare_breast(),
    "diabetes": prepare_diabetes()
}

best, history = random_symbolic_search(
    datasets,
    n_iter=8000,
    seed=42
)

print("BEST SYMBOLIC STRUCTURE")
print(best)

df = pd.DataFrame(history)
df = df.sort_values("objective", ascending=False)

print("\nTOP 10")
print(
    df[
        [
            "form",
            "mean_auc",
            "min_auc",
            "std_auc",
            "objective"
        ]
    ].head(10)
)
