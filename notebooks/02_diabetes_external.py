from sklearn.datasets import load_diabetes
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from src.proxies import generic_proxy_frame
from src.congruity import (
    congruity_base,
    congruity_balanced,
    congruity_discovered
)
from src.metrics import orientation_invariant_auc


data = load_diabetes()

X = pd.DataFrame(data.data)
y_raw = data.target

low = np.percentile(y_raw, 33)
high = np.percentile(y_raw, 67)

mask = (y_raw <= low) | (y_raw >= high)

X = X[mask]
y = (y_raw[mask] >= high).astype(int)

proxy = generic_proxy_frame(X)

V = proxy["V"].values
E = proxy["E"].values
I = proxy["I"].values
S = proxy["S"].values

print("CONGRUITY RESULTS")

print(
    "C_base:",
    orientation_invariant_auc(
        y,
        congruity_base(V, E, I, S)
    )
)

print(
    "C_balanced:",
    orientation_invariant_auc(
        y,
        congruity_balanced(V, E, I, S)
    )
)

print(
    "C_discovered:",
    orientation_invariant_auc(
        y,
        congruity_discovered(V, E, I, S)
    )
)

X_train, X_test, y_train, y_test = train_test_split(
    proxy,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

models = {
    "LogReg": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

print("\nML BASELINES")

for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(name, auc)
