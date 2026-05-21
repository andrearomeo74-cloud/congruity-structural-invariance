import subprocess
import sys

scripts = [
    "notebooks/01_breast_benchmark.py",
    "notebooks/02_diabetes_external.py",
    "notebooks/04_synthetic_ecology.py",
    "notebooks/06_symbolic_discovery.py",
    "notebooks/07_permutation_falsification.py",
    "notebooks/08_proxy_robustness.py",
    "notebooks/09_adaptive_transfer.py",
    "notebooks/10_fixed_sign_transfer.py"
]

print("CONGRUITY STRUCTURAL INVARIANCE")
print("Running reproducibility benchmark suite\n")

for script in scripts:
    print("=" * 70)
    print("RUNNING:", script)
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script]
    )

    if result.returncode != 0:
        print("\nFAILED:", script)
        sys.exit(result.returncode)

print("\nALL BENCHMARKS COMPLETED")
