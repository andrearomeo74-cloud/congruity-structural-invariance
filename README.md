![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Status](https://img.shields.io/badge/status-reproducibility%20build-orange)

# Congruity Structural Invariance

This repository contains reproducible computational benchmarks supporting the preprint:

**Congruity as a Candidate Structural Invariance Class: Cross-Domain Emergence, Falsification, Robustness, and Transfer Evaluation**

Author: Andrea Romeo  
Affiliation: Independent Systems Research, Italy

## Purpose

The repository provides code and notebooks for testing whether heterogeneous systems exhibit proportional value-over-burden structures consistent with a candidate Congruity Invariance Class.

The repository is designed for scientific reproducibility, not for deployment of proprietary Congruity systems.

## Included experiments

- Domain-level Congruity benchmarks
- Symbolic cross-domain discovery
- Permutation falsification
- Proxy permutation robustness
- Blind frozen transfer
- Adaptive leave-one-domain-out transfer
- External unseen diabetes validation
- Fixed-sign transfer stress test
- Comparison against standard machine learning baselines

## Domains

The experiments include:

- Biological aging benchmark
- Breast cancer morphology benchmark
- EEG motor imagery benchmark
- Synthetic stochastic ecological collapse benchmark
- External unseen diabetes clinical benchmark

## Core structural hypothesis

The tested hypothesis is not a universal fixed equation.

The tested hypothesis is that viable systems may belong to a broader proportional structural family:

```math
C = \frac{V^\alpha}{G(E,I,S)}
```

where:

- `V` represents retained functional value
- `E` represents energetic burden
- `I` represents informational disorder
- `S` represents structural instability
- `G(E,I,S)` represents domain-dependent burden interaction geometry

## Important scope limitation

This repository does not contain proprietary Congruity engines, patented implementations, private orchestration logic, or commercial decision systems.

It only contains the public reproducibility code needed to evaluate the preprint experiments.

## Quick start

Clone the repository:

```bash
git clone https://github.com/andrearomeo74-cloud/congruity-structural-invariance.git
cd congruity-structural-invariance
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the public benchmark suite:

```bash
python run_all.py
```

Run individual benchmarks:

```bash
python notebooks/01_breast_benchmark.py
python notebooks/02_diabetes_external.py
python notebooks/04_synthetic_ecology.py
python notebooks/05_eeg_benchmark.py
python notebooks/06_symbolic_discovery.py
python notebooks/07_permutation_falsification.py
python notebooks/08_proxy_robustness.py
python notebooks/09_adaptive_transfer.py
python notebooks/10_fixed_sign_transfer.py
```
Optional benchmark (requires user-provided dataset):

```bash
python notebooks/03_aging_benchmark.py
```

## Expected outputs

The repository reproduces:

- Congruity benchmark comparisons
- symbolic structural discovery
- shuffled-label falsification
- proxy robustness analysis
- adaptive transfer evaluation
- fixed-sign transfer stress testing
- external validation benchmarks

## License

Code is released under the Apache License 2.0.

Text, figures, and manuscript-related materials are released under CC BY-NC 4.0 unless otherwise stated.

## Intellectual property notice

This repository is provided for scientific reproducibility and academic evaluation.

The broader Congruity framework, associated patents, and commercial implementations may be subject to separate intellectual property protections.

This repository does not grant rights to proprietary implementations beyond the code explicitly released here.

See:

[LICENSE_NOTICE.md](LICENSE_NOTICE.md)

## Status

Public reproducibility benchmark repository.

Core benchmark suite is available.

Optional benchmarks may require external datasets or downloads.
