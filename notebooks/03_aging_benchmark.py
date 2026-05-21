import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.proxies import generic_proxy_frame
from src.congruity import (
    congruity_base,
    congruity_balanced,
    congruity_discovered
)
from src.metrics import orientation_invariant_auc
