import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


SCRIPTS = [
    "notebooks/01_breast_benchmark.py",
    "notebooks/02_diabetes_external.py",
    "notebooks/04_synthetic_ecology.py",
    "notebooks/06_symbolic_discovery.py",
    "notebooks/07_permutation_falsification.py",
    "notebooks/08_proxy_robustness.py",
    "notebooks/09_adaptive_transfer.py",
    "notebooks/10_fixed_sign_transfer.py",
]


OPTIONAL_SCRIPTS = [
    "notebooks/03_aging_benchmark.py",
    "notebooks/05_eeg_benchmark.py",
]


def run_script(script):
    print("=" * 70)
    print(f"RUNNING: {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, str(ROOT / script)],
        cwd=str(ROOT)
    )

    if result.returncode != 0:
        print(f"\nFAILED: {script}")
        return False

    print(f"\nCOMPLETED: {script}")
    return True


def main():
    print("CONGRUITY STRUCTURAL INVARIANCE")
    print("Running public reproducibility benchmark suite\n")

    failed = []

    for script in SCRIPTS:
        ok = run_script(script)
        if not ok:
            failed.append(script)

    print("\nOPTIONAL SCRIPTS")
    print("These may require external downloads or user-provided data.\n")

    for script in OPTIONAL_SCRIPTS:
        print(f"- {script}")

    if failed:
        print("\nFAILED SCRIPTS:")
        for script in failed:
            print("-", script)
        sys.exit(1)

    print("\nALL CORE BENCHMARKS COMPLETED")


if __name__ == "__main__":
    main()
