# Mid Defence 1 Data Pipeline + Berrar P1

This package preserves the verified Mid Defence 1 StatsBomb EPL 2015/16 data pipeline and adds an opt-in reimplementation of the data-representation method from Berrar, Lopes & Dubitzky (2024).

## Final P1 source design

The P1 branch follows the source design explicitly approved by the TA:

- **StatsBomb EPL 2015/16** remains the primary modeling/target dataset. Match IDs, labels, current-season completed results, events, lineups, and live snapshots remain StatsBomb-derived.
- **Football-Data EPL 2000/01--2014/15** is used only as complete pre-target league history for the Berrar representation. Only `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, and `FTAG` are admitted into this branch.
- **Football-Data EPL 2015/16 odds** remain a separate market baseline in the existing MD1 pipeline. Odds are never used to build Berrar features.

The historical Football-Data files are downloaded from the official season pattern:

```text
https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/E0.csv
```

For example, 2000/01 is `0001/E0.csv` and 2014/15 is `1415/E0.csv`.

## Why the hybrid source design is necessary

Berrar's central contribution is a same-league longitudinal **Super League** representation. Complete seasons are concatenated chronologically; team histories continue across season boundaries and across same-league participation gaps. This requires complete historical results for all clubs. The StatsBomb EPL 2015/16 target season remains the project modeling season, while the TA-approved Football-Data history supplies the missing long pre-target context.

Football-Data historical rows are **context only**. They are never supervised examples in the final project. Final P1 target examples are the same StatsBomb EPL 2015/16 matches used by the baseline pipeline.

## Berrar representation implemented

The paper-aligned core uses only:

- goals scored;
- goals conceded;
- normalized league success.

Two feature views are exported:

- `p1_total`: exactly **6** features = 2 teams × 3 metrics × total history;
- `p1_homeaway`: exactly **18** features = 2 teams × 3 metrics × total/home/away views.

Mean aggregation is used. The minimum-history rule is the paper's **six prior same-league matches per team**. A team with at least six but fewer than the selected recency `n` uses all available prior matches up to `n`.

## Recency selection

Real P1 runs evaluate every integer:

```text
n = 9, 10, ..., 100
```

The primary selector is the paper's Pearson method:

```text
signal_n =
    (home avg goal difference - away avg goal difference)
    +
    (home avg goals scored - away avg goals scored)
```

The selected `n` maximizes the positive Pearson correlation between this signal and observed home-minus-away goal margin.

**Project-required anti-leakage adaptation:** the correlation is calculated on the StatsBomb **training split only**. Validation and test labels cannot select `n`.

The paper's conceptual `n/2` venue rule is ambiguous for odd `n` even though it searches all integers 9..100. The implementation uses explicit `floor(n/2)` and records that as a documented fidelity clarification.

## Chronology and source boundaries

The P1 history seen by a target StatsBomb EPL 2015/16 match is:

```text
complete Football-Data EPL 2000/01--2014/15 results
+
StatsBomb EPL 2015/16 matches that finished before target kickoff
```

Football-Data **2015/16 final results do not enter P1 history**. They remain outside this branch; the 2015/16 Football-Data file is used by the baseline odds integration only.

For old Football-Data matches, which do not consistently include reliable kickoff times, completion is conservatively treated as end-of-day. Since all such rows predate the target season, no target feature can gain same-day future information from this rule.

## Hard completeness gate

Every Football-Data historical EPL season must pass all of the following before P1 results are allowed:

- 20 unique clubs;
- 380 completed matches;
- 38 matches per club;
- no duplicate directed fixture;
- each unordered pair appears exactly twice;
- no missing date/home/away/full-time score values.

StatsBomb EPL 2015/16 is independently subjected to the same 20-team/380-match fixture completeness audit.

A failing season blocks P1 results.

## Leakage and integrity tests

The hybrid P1 branch includes tests for:

1. all feature sources finishing strictly before target kickoff;
2. target-result perturbation;
3. future-result perturbation;
4. a positive control showing an allowed old historical result can affect a later target feature;
5. strict provider-role separation;
6. rejection of Football-Data odds/statistics from the canonical historical table;
7. cross-season history continuity;
8. same-league gap continuity;
9. minimum-six eligibility;
10. validation/test isolation from selected `n`;
11. normalized-rank temporal safety;
12. current-season rank population;
13. exact 6/18 feature-count contracts.

The readiness gate additionally verifies source approval, historical/target season completeness, StatsBomb-only target examples, exclusion of Football-Data target-season results, rank bounds, valid `n`, common train/validation/test populations, and fidelity metadata.

## Existing MD1 baseline is preserved

`build_default_config()` and `execute_complete_workflow()` retain the original EPL 2015/16 pipeline:

- StatsBomb match/event/lineup integration;
- rolling windows `(3, 5, 10)`;
- 128 pre-match model-eligible features in the verified full configuration;
- 21 snapshots per match;
- 189 live model-eligible features;
- Football-Data odds join and de-vigged market probabilities;
- existing leakage and relational checks.

P1 is opt-in and is executed by `build_p1_default_config()` + `execute_p1_workflow()`.

## Main P1 outputs

```text
data/silver/p1_historical_epl_matches.parquet
data/silver/p1_target_statsbomb_matches.parquet
data/silver/p1_super_league_matches.parquet
data/silver/p1_team_match_facts.parquet

data/gold/p1_features_total_all_n.parquet
data/gold/p1_features_total_fixed.parquet
data/gold/p1_features_total_selected.parquet
data/gold/p1_features_homeaway_selected.parquet
data/gold/p1_split_manifest.csv

audit/p1_history_download_manifest.csv
audit/p1_target_download_manifest.csv
audit/p1_season_completeness.csv
audit/p1_target_season_completeness.csv
audit/p1_source_roles.csv
audit/p1_source_roles.json
audit/p1_team_identity_audit.csv
audit/p1_history_coverage.csv
audit/p1_eligibility.csv
audit/p1_pearson_recency_search.csv
audit/p1_selected_recency.json
audit/p1_feature_dictionary.csv
audit/p1_fidelity_table.csv
audit/p1_fidelity_decisions.json
audit/p1_leakage_tests.csv
audit/p1_readiness.csv
audit/p1_run_summary.json
```

## Colab

Upload the delivered `md1_data_pipeline_code.zip` and the Colab runner to Google Drive. The runner installs `requirements_colab.txt`, compiles all modules, runs an optional synthetic preflight, and then runs the real P1 pipeline.

For the final paper-aligned P1 run keep:

```python
P1_RUN_MODE = "real"
P1_FULL_RECENCY_SEARCH = True
P1_RUN_KNN_SEARCH = False
```

The primary Pearson run evaluates all 92 candidate values from 9 through 100. Optional k-NN recency search is substantially more expensive and is not needed for the primary P1 deliverable.

## Important reporting language

The defensible description is:

> The primary modeling dataset is StatsBomb EPL 2015/16. With explicit TA approval, complete Football-Data EPL results from prior seasons are used solely as causal historical context for the Berrar representation. Final supervised examples and labels remain StatsBomb matches, while Football-Data bookmaker odds remain an independent market benchmark.
