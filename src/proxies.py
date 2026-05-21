import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


EPS = 1e-9


def minmax_dataframe(df, columns=None):
    """
    Min-max normalize selected dataframe columns.
    """
    out = df.copy()

    if columns is None:
        columns = out.columns

    out[columns] = MinMaxScaler().fit_transform(out[columns])

    return out


def row_entropy(row, eps=EPS):
    """
    Entropy of a non-negative row vector.
    """
    values = np.abs(np.array(row, dtype=float)) + eps
    p = values / values.sum()
    return -np.sum(p * np.log(p))


def generic_proxy_frame(X):
    """
    Generic V,E,I,S proxy construction from a numeric dataframe.

    V: retained regularity
    E: mean burden magnitude
    I: row entropy
    S: structural heterogeneity
    """
    X = pd.DataFrame(X).copy()

    Xn = pd.DataFrame(
        MinMaxScaler().fit_transform(X),
        columns=X.columns
    )

    V = 1 - Xn.std(axis=1)
    E = Xn.mean(axis=1)
    I = Xn.apply(row_entropy, axis=1)
    S = Xn.diff(axis=1).abs().fillna(0).mean(axis=1)

    proxies = pd.DataFrame({
        "V": V,
        "E": E,
        "I": I,
        "S": S
    })

    return minmax_dataframe(proxies, ["V", "E", "I", "S"])
