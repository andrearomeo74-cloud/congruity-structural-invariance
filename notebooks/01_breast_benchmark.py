import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.proxies import generic_proxy_frame
from src.congruity import (
    congruity_base,
    congruity_balanced,
    congruity_discovered
)
from src.metrics import orientation_invariant_auc


data = load_breast_cancer()

X = data.data
y = data.target

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
