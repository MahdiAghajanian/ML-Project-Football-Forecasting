# EXP01B — Single-Season vs Super-League Berrar Ablation

Status: PASS

- Frozen Berrar recency: n=28
- Feature family: six `total` Berrar features
- Single-season eligible matches: 320
- Super-League eligible matches: 374
- Additional matches made eligible by cross-season history: 54
- Common predictive population: 320
- Common split counts: {'train': 197, 'validation': 77, 'test': 46}
- Validation RPS delta, Super League - Single Season: -0.00975132
- Test RPS delta, Super League - Single Season: 0.016681345422336208

Negative RPS delta favors the Super-League representation.

The test stage is report-only. The experiment does not re-select recency or alter
the feature definition. See `audit/FROZEN_BEFORE_TEST.json` and
`audit/DATA_STAGE_CLOSURE.json`.
