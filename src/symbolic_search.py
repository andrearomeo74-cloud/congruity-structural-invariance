import numpy as np

from src.metrics import orientation_invariant_auc


EPS = 1e-9


def candidate_formula(V, E, I, S, params, form):
    """
    Candidate symbolic structures used in cross-domain search.
    """
    a, b, c, d = params

    if form == "additive":
        return (V ** a) / ((E ** b) + (I ** c) + (S ** d) + EPS)

    if form == "multiplicative":
        return (V ** a) / (
            ((1 + E) ** b)
            * ((1 + I) ** c)
            * ((1 + S) ** d)
            + EPS
        )

    if form == "hybrid":
        return (V ** a) / (
            ((E ** b) + (I ** c))
            * ((1 + S) ** d)
            + EPS
        )

    if form == "inverse_entropy":
        return (V ** a) / (
            ((1 + E) ** b)
            * ((1 + S) ** c)
            * ((I + EPS) ** d)
            + EPS
        )

    if form == "ratio":
        return ((V ** a) + (I ** b)) / ((E ** c) + (S ** d) + EPS)

    raise ValueError(f"Unknown form: {form}")


def meta_score(form, params, datasets):
    """
    Compute mean, min, std, and all AUCs across datasets.
    """
    aucs = []

    for _, (y, Xdf) in datasets.items():
        V = Xdf["V"].values
        E = Xdf["E"].values
        I = Xdf["I"].values
        S = Xdf["S"].values

        score = candidate_formula(V, E, I, S, params, form)
        auc = orientation_invariant_auc(y, score)
        aucs.append(auc)

    aucs = np.array(aucs)

    return {
        "mean_auc": float(aucs.mean()),
        "min_auc": float(aucs.min()),
        "std_auc": float(aucs.std()),
        "all_aucs": aucs.tolist(),
    }


def random_symbolic_search(
    datasets,
    n_iter=8000,
    seed=42,
    forms=None,
):
    """
    Random symbolic search over interpretable Congruity-like forms.
    """
    rng = np.random.default_rng(seed)

    if forms is None:
        forms = [
            "additive",
            "multiplicative",
            "hybrid",
            "inverse_entropy",
            "ratio",
        ]

    best = None
    history = []

    for _ in range(n_iter):
        params = rng.uniform(0.2, 2.0, 4)
        form = rng.choice(forms)

        result = meta_score(form, params, datasets)
        objective = result["mean_auc"] - 0.5 * result["std_auc"]

        entry = {
            "form": form,
            "params": params.tolist(),
            "objective": float(objective),
            **result,
        }

        history.append(entry)

        if best is None or objective > best["objective"]:
            best = entry

    return best, history
