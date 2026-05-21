import numpy as np


def congruity_base(V, E, I, S, eps=1e-9):
    """
    Canonical additive Congruity.
    """
    return V / (E + I + S + eps)


def congruity_balanced(V, E, I, S, eps=1e-9):
    """
    Canonical multiplicative Congruity.
    """
    return V / ((1 + E) * (1 + I) * (1 + S) + eps)


def congruity_discovered(V, E, I, S, eps=1e-9):
    """
    Symbolically discovered structural form from cross-domain search.
    """
    return (V ** 1.4698) / (
        ((1 + E) ** 0.2023) *
        ((1 + S) ** 0.2419) *
        (I ** 1.4425) +
        eps
    )


def best_auc_orientation(score, y_true, roc_auc_score):
    """
    Orientation-invariant AUC.
    """
    auc1 = roc_auc_score(y_true, score)
    auc2 = roc_auc_score(y_true, -score)
    return max(auc1, auc2)
