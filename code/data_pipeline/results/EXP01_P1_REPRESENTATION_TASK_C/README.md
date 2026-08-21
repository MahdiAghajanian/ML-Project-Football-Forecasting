# EXP01 — P1 Representation Ablation, Task C

This directory is generated from frozen upstream MD1 and Berrar P1 outputs.

- Berrar selected recency: n=28
- Fixed-recency comparator: n=10
- Common population: 374 matches
- Split: 251 train / 77 validation / 46 test
- Model probe: one fixed Random Forest configuration for every learned representation
- Primary metric: RPS
- Representation selection: validation RPS only
- Test: report only

See `audit/FROZEN_BEFORE_TEST.json` for the pre-test freeze record.
See `metrics/` for validation/test metrics and paired bootstrap results.
