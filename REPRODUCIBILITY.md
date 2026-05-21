# Reproducibility

This repository is the public computational companion to the preprint:

**Congruity as a Candidate Structural Invariance Class: Cross-Domain Emergence, Falsification, Robustness, and Transfer Evaluation**

## Reproducible components

The repository provides public scripts for:

- canonical Congruity formulas
- discovered Congruity structural form
- breast cancer benchmark
- external diabetes benchmark
- synthetic ecological collapse benchmark
- EEG benchmark
- symbolic discovery
- permutation falsification
- proxy permutation robustness
- adaptive transfer
- fixed-sign transfer stress testing
- machine learning baseline comparison

## Non-reproduced components

Some exploratory preprocessing choices may differ from the original research notebooks, especially for biological aging data, where public sources and preprocessing pipelines can vary.

For this reason, the aging benchmark script uses a documented CSV interface:

```text
data/aging_dataset.csv
```

with a required target column:

```text
label
```

## Expected usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all available public scripts:

```bash
python run_all.py
```

## Interpretation

This repository is intended to reproduce the public methodological logic of the preprint.

It should not be interpreted as a release of proprietary Congruity systems, advanced orchestration logic, or commercial implementations.

## Contact

For questions about replication, contact:

Andrea Romeo
