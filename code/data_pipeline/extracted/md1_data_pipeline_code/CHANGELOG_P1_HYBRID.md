# Change log — TA-approved hybrid P1 revision

## Added

- `md1_data_pipeline/p1_hybrid.py`
  - Football-Data historical EPL downloader for 2000/01--2014/15;
  - strict five-column historical ingestion (`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`);
  - 20-team/380-match season-completeness gates;
  - StatsBomb EPL 2015/16 target metadata ingestion;
  - cross-provider club identity mapping with explicit EPL aliases;
  - hybrid same-league Super League history;
  - StatsBomb-only target examples and target-season causal continuation;
  - 13 source/leakage/integrity tests;
  - source-role, fidelity, feature, readiness, completeness, and history audits.

## Changed

- `MD1Config` now exposes the approved P1 history-provider/source-role settings.
- `build_p1_default_config()` now defaults to the hybrid EPL design rather than multi-season La Liga.
- Synthetic P1 recency smoke grid changed from `2..6` to `6..10`, so every candidate can be evaluated under the minimum-six eligibility rule.
- `execute_p1_workflow()` now runs the hybrid P1 branch.
- README and smoke-test documentation now describe the TA-approved source design.

## Preserved

- Existing MD1 EPL 2015/16 baseline execution path.
- Existing 3/5/10 rolling event-derived representation.
- Existing odds integration and market benchmark behavior.
- Existing Task-L snapshot/cutoff logic.
- Existing baseline leakage and readiness checks.

## Legacy

`p1_berrar.py` remains as the reusable Berrar feature/recency engine and contains the earlier standalone StatsBomb-only Super League path for diagnostic/backward compatibility. It is no longer the default P1 workflow.
