import numpy as np

from symbolic_search import random_symbolic_search
from metrics import orientation_invariant_auc, fixed_sign_auc
from congruity import congruity_discovered


def leave_one_domain_out(datasets, n_iter=12000, seed=123):
    """
    Adaptive leave-one-domain-out transfer.

    For each held-out domain:
    - train symbolic parameters on remaining domains
    - evaluate on held-out domain
    """
    rows = []

    domain_names = list(datasets.keys())

    for heldout in domain_names:
        train = {
            name: datasets[name]
            for name in domain_names
            if name != heldout
        }

        best, _ = random_symbolic_search(
            train,
            n_iter=n_iter,
            seed=seed
        )

        y_test, X_test = datasets[heldout]

        from symbolic_search import candidate_formula

        score = candidate_formula(
            X_test["V"].values,
            X_test["E"].values,
            X_test["I"].values,
            X_test["S"].values,
            np.array(best["params"]),
            best["form"]
        )

        test_auc = orientation_invariant_auc(y_test, score)

        rows.append({
            "heldout": heldout,
            "form": best["form"],
            "params": best["params"],
            "train_mean_auc": best["mean_auc"],
            "train_min_auc": best["min_auc"],
            "train_std_auc": best["std_auc"],
            "test_auc": float(test_auc)
        })

    return rows


def choose_training_sign(datasets):
    """
    Choose score orientation from discovery domains.
    """
    scores = []
    labels = []

    for _, (y, Xdf) in datasets.items():
        score = congruity_discovered(
            Xdf["V"].values,
            Xdf["E"].values,
            Xdf["I"].values,
            Xdf["S"].values
        )

        scores.extend(score)
        labels.extend(y)

    scores = np.array(scores)
    labels = np.array(labels)

    auc_normal = fixed_sign_auc(labels, scores, sign=1)
    auc_flipped = fixed_sign_auc(labels, scores, sign=-1)

    sign = 1 if auc_normal >= auc_flipped else -1

    return {
        "sign": sign,
        "auc_normal": float(auc_normal),
        "auc_flipped": float(auc_flipped)
    }
