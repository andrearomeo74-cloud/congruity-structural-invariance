import numpy as np
import pandas as pd
import mne

from mne.datasets import eegbci
from sklearn.preprocessing import MinMaxScaler

from src.congruity import (
    congruity_base,
    congruity_balanced,
    congruity_discovered
)
from src.metrics import orientation_invariant_auc


subjects = range(1, 11)
rows = []

for subject in subjects:
    print("Loading subject", subject)

    runs = [3, 7]

    files = eegbci.load_data(subject, runs)
    raw = mne.io.read_raw_edf(files, preload=True, verbose=False)

    eegbci.standardize(raw)
    raw.set_montage("standard_1005")

    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=2,
        baseline=(0, 0),
        preload=True,
        verbose=False
    )

    X = epochs.get_data()
    y = epochs.events[:, -1]

    y = (y != y.min()).astype(int)

    for i in range(len(X)):
        sample = X[i]

        energy = np.mean(sample ** 2)

        power = np.abs(np.fft.rfft(sample.flatten())) ** 2
        power = power + 1e-9
        power = power / power.sum()

        entropy = -np.sum(power * np.log(power))

        stability = np.std(np.diff(sample, axis=1))

        coherence = 1 / (1 + stability)

        rows.append([
            coherence,
            energy,
            entropy,
            stability,
            y[i]
        ])

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

print("EEG CONGRUITY RESULTS")

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
