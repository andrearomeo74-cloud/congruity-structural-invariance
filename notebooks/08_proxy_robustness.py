import itertools
import pandas as pd

from sklearn.datasets import load_breast_cancer

from src.congruity import congruity_discovered
from src.metrics import orientation_invariant_auc
from src.proxies import generic_proxy_frame


data = load_breast_cancer()

X = pd.DataFrame(data.data)
y = data.target

proxy = generic_proxy_frame(X)

base = ["V", "E", "I", "S"]

results = []

for perm in itertools.permutations(base):
    V = proxy[perm[0]].values
    E = proxy[perm[1]].values
    I = proxy[perm[2]].values
    S = proxy[perm[3]].values

    auc = orientation_invariant_auc(
        y,
        congruity_discovered(V, E, I, S)
    )

    results.append({
        "mapping": perm,
        "auc": auc
    })

df = pd.DataFrame(results)
df = df.sort_values("auc", ascending=False)

true_mapping = ("V", "E", "I", "S")

rank = (
    df.reset_index(drop=True)
      .query("mapping == @true_mapping")
      .index[0]
) + 1

percentile = 1 - ((rank - 1) / len(df))

print("TRUE MAPPING")
print(true_mapping)

print("\nRANK")
print(rank)

print("\nTOTAL")
print(len(df))

print("\nPERCENTILE")
print(percentile)

print("\nTOP 10")
print(df.head(10))
