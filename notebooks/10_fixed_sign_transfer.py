import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_diabetes

from src.proxies import generic_proxy_frame
from src.congruity import congruity_discovered
from src.metrics import fixed_sign_auc
from src.transfer import choose_training_sign


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


training_domains = {
    "breast": prepare_breast()
}

external_y, external_proxy = prepare_diabetes()

sign_info = choose_training_sign(training_domains)

external_score = congruity_discovered(
    external_proxy["V"].values,
    external_proxy["E"].values,
    external_proxy["I"].values,
    external_proxy["S"].values
)

external_auc = fixed_sign_auc(
    external_y,
    external_score,
    sign=sign_info["sign"]
)

print("TRAIN SIGN INFO")
print(sign_info)

print("\nEXTERNAL FIXED SIGN AUC")
print(external_auc)
