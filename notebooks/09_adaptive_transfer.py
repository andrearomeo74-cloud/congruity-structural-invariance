import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sklearn.datasets import load_breast_cancer, load_diabetes

from src.proxies import generic_proxy_frame
from src.transfer import leave_one_domain_out


def prepare_breast():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    y = data.target
    return y, generic_proxy_frame(X)


def prepare_diabetes():
    data = load_diabetes()
    X = pd.DataFrame(data.data)

    y_raw = data.target
    y = (y_raw > y_raw.mean()).astype(int)

    return y, generic_proxy_frame(X)


datasets = {
    "breast": prepare_breast(),
    "diabetes": prepare_diabetes()
}

results = leave_one_domain_out(
    datasets,
    n_iter=5000,
    seed=42
)

df = pd.DataFrame(results)

print("LEAVE ONE DOMAIN OUT RESULTS")
print(df)

print("\nSUMMARY")
print(df["test_auc"].describe())
