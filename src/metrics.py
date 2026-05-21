import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auc(y_true, score):
    """
    Compute ROC AUC with NaN/Inf protection.
    """
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    return roc_auc_score(y_true, score)


def orientation_invariant_auc(y_true, score):
    """
    Best AUC across score polarity.
    """
    auc_normal = safe_auc(y_true, score)
    auc_flipped = safe_auc(y_true, -score)
    return max(auc_normal, auc_flipped)


def fixed_sign_auc(y_true, score, sign=1):
    """
    Fixed semantic orientation evaluation.
    """
    return safe_auc(y_true, sign * score)


def summarize_results(results):
    """
    Basic summary statistics.
    """
    arr = np.array(results)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }
