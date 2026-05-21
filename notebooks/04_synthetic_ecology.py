import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.congruity import (
    congruity_base,
    congruity_balanced,
    congruity_discovered
)
from src.metrics import orientation_invariant_auc


rng = np.random.default_rng(42)

steps = 600

x = np.zeros(steps)
y_pop = np.zeros(steps)

x[0] = 40
y_pop[0] = 9

alpha = 1.0
beta = 0.05
delta = 0.02
gamma = 0.8
noise = 0.08

for t in range(1, steps):
    dx = alpha * x[t-1] - beta * x[t-1] * y_pop[t-1]
    dy = delta * x[t-1] * y_pop[t-1] - gamma * y_pop[t-1]

    dx += rng.normal(0, noise * max(x[t-1], 1))
    dy += rng.normal(0, noise * max(y_pop[t-1], 1))

    x[t] = max(0, x[t-1] + dx * 0.05)
    y_pop[t] = max(0, y_pop[t-1] + dy * 0.05)

window = 20
rows = []

for t in range(window, steps):
    xp = x[t-window:t]
    yp = y_pop[t-window:t]

    V = np.mean(xp) / (np.mean(xp) + np.mean(yp) + 1e-9)
    E = np.std(xp) + np.std(yp)

    px = np.abs(np.fft.rfft(xp)) ** 2
    py = np.abs(np.fft.rfft(yp)) ** 2

    p = px + py + 1e-9
    p = p / p.sum()

    I = -np.sum(p * np.log(p))
    S = np.var(np.diff(xp)) + np.var(np.diff(yp))

    future = x[t:min(t+15, steps)]
    collapse = int(np.mean(future) < 15)

    rows.append([V, E, I, S, collapse])

df = pd.DataFrame(
    rows,
    columns=["V", "E", "I", "S", "label"]
)

df[["V", "E", "I", "S"]] = MinMaxScaler().fit_transform(
    df[["V", "E", "I", "S"]]
)

label = df["label"].values

V = df["V"].values
E = df["E"].values
I = df["I"].values
S = df["S"].values

print("SYNTHETIC ECOLOGY CONGRUITY RESULTS")

print(
    "C_base:",
    orientation_invariant_auc(
        label,
        congruity_base(V, E, I, S)
    )
)

print(
    "C_balanced:",
    orientation_invariant_auc(
        label,
        congruity_balanced(V, E, I, S)
    )
)

print(
    "C_discovered:",
    orientation_invariant_auc(
        label,
        congruity_discovered(V, E, I, S)
    )
)

X_train, X_test, y_train, y_test = train_test_split(
    df[["V", "E", "I", "S"]],
    label,
    test_size=0.3,
    random_state=42,
    stratify=label
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
