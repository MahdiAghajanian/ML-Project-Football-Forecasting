<h1 align="center">
Forecasting Competitive Football
</h1>

<p align="center">
  <img src="assets/github-banner.png" width="100%">
</p>

<p align="center">
Probabilistic Football Forecasting with Leakage-Safe Event Pipelines, Berrar-Style Longitudinal Representations, and Natural Gradient Boosting
</p>

---

## Overview

This repository contains the implementation for the Machine Learning course project **Forecasting Competitive Football**.

The project builds an end-to-end, leakage-aware football forecasting system from raw **StatsBomb Open Data**, with an independent **Football-Data.co.uk** bookmaker baseline. It supports three forecasting settings:

| Task | Prediction time | Input | Output |
|---|---|---|---|
| **Task C** | Pre-match | Causal team-history features | `P(Home)`, `P(Draw)`, `P(Away)` |
| **Task R** | Pre-match | Same pre-match representation | Signed goal margin, clipped to `[-5, +5]` |
| **Task L** | In-play | Pre-match context + event prefix up to time `t` | Updated outcome probabilities and final-margin estimate |

The repository also contains the P1 data-paper reimplementation:

> **Berrar, Lopes & Dubitzky (2024)** — *A data- and knowledge-driven framework for developing machine learning models to predict soccer match outcomes.*

The P1 implementation reproduces the paper's longitudinal **Super League** idea, six-feature `total` representation, 18-feature `homeaway` representation, minimum-history rule, mean aggregation, and Pearson-based recency selection, while adapting the outer evaluation to the project's stricter chronological anti-leakage rules.

The next project phase is predictive modeling. The planned P2 contribution is a reimplementation of **Natural Gradient Boosting (NGBoost)** based on O'Malley et al. (2023), adapted to the football classification and regression tasks.

> **Current status:** the complete data-engineering and P1 stage has passed its final readiness gate and is frozen. The project is now ready to move to Task C / Task R / Task L modeling.

---

## Current Project Status

| Component | Status |
|---|---|
| StatsBomb raw ingestion | ✅ Complete |
| Football-Data odds integration | ✅ Complete |
| Bronze / Silver / Gold data architecture | ✅ Complete |
| Leakage-safe pre-match feature pipeline | ✅ Complete |
| Leakage-safe in-play snapshot pipeline | ✅ Complete |
| Chronological match-level split | ✅ Complete |
| Market odds matching and de-vigging | ✅ Complete |
| Baseline readiness and leakage tests | ✅ Complete |
| Berrar P1 Super League implementation | ✅ Complete |
| Full Pearson recency search `n = 9...100` | ✅ Complete |
| Selected P1 recency `n = 28` | ✅ Frozen |
| P1 six-feature / 18-feature contracts | ✅ Complete |
| P1 leakage and source-role tests | ✅ Complete |
| EXP01 representation ablation | ✅ Complete |
| EXP01B single-season vs Super-League ablation | ✅ Complete |
| Data-stage closure | ✅ `DATA_STAGE_EXPERIMENTS_COMPLETE` |
| Full Task C modeling | ⏳ Next |
| Task R modeling | ⏳ Planned |
| Task L modeling | ⏳ Planned |
| P2 / NGBoost reimplementation | ⏳ Planned |
| Calibration and SHAP analysis | ⏳ Planned |

---

## Data Sources and Roles

### StatsBomb Open Data

StatsBomb is the **primary modeling source**.

It provides:

- match metadata and labels;
- event streams;
- lineups;
- optional 360 data where available;
- the EPL 2015/16 target/modeling season used in the verified real pipeline;
- the causal current-season match history used by P1.

Repository:

```text
https://github.com/statsbomb/open-data
```

### Football-Data.co.uk

Football-Data has two deliberately separated roles.

#### Market baseline

For EPL 2015/16, Football-Data provides bookmaker 1X2 odds. Rows are joined to StatsBomb using pre-match identity only:

```text
date + normalized home team + normalized away team
```

No score or match statistic is used to resolve the join.

Decimal odds are converted to implied probabilities and de-vigged:

\[
q_i = \frac{1}{o_i}
\]

\[
p_i = \frac{q_i}{q_H + q_D + q_A}
\]

The resulting market probabilities are kept **outside the main predictive feature matrix**.

#### TA-approved Berrar historical context

The Berrar Super League requires complete longitudinal same-league results. For this P1 branch, the TA approved using **Football-Data EPL 2000/01–2014/15 basic completed-match results only**:

```text
Date
HomeTeam
AwayTeam
FTHG
FTAG
```

These rows are used only as historical context for the P1 representation.

They are **not** supervised target examples, and bookmaker odds/statistics are not admitted into the P1 historical feature engine.

The final target/modeling season remains **StatsBomb EPL 2015/16**.

---

## Verified Real Data Pipeline

The canonical baseline run is:

```text
Premier_League_2015_16_DATA_ONLY_FULL_REAL
```

### Dataset scale

| Quantity | Verified value |
|---|---:|
| Matches | **380** |
| Teams | **20** |
| Players | **644** |
| StatsBomb event rows | **1,313,773** |
| Lineup / starter rows | **15,926** |
| Team-match instances with 11 starters verified | **760** |
| Available StatsBomb 360 matches | **0** |

### Pre-match Gold representation

The baseline pre-match table contains **128 model-eligible features**:

```text
96 rolling team features
+ 21 home-minus-away difference features
+ 11 supporting/context features
= 128
```

The rolling component uses the baseline windows:

```text
3, 5, 10 previous matches
```

and includes causal historical metrics derived from StatsBomb matches/events such as goals, points, shots, shots on target, xG, passes, pressures, final-third entries, red cards, and possession-event share.

### In-play Gold representation

Each match generates **21 snapshots**:

```text
380 matches × 21 snapshots = 7,980 snapshot rows
```

The live vector contains:

```text
128 pre-match features
+ 61 live-specific features
= 189 model-eligible features
```

Important temporal boundaries are kept distinct:

```text
M45 ≠ HT
M90 ≠ FT
```

`M90` is a strict 90:00 cutoff, while `FT` includes regulation-time stoppage events. The same principle applies to `M45` versus half-time.

### Chronological split

The canonical match-level split is frozen as:

| Split | Matches |
|---|---:|
| Train | **257** |
| Validation | **77** |
| Test | **46** |

All snapshots of one match inherit the same split.

### Market coverage

Football-Data odds were successfully joined for:

```text
380 / 380 matches
```

with:

```text
0 ambiguous
0 unmatched
```

The market probabilities remain an independent benchmark.

### Baseline readiness

The latest verified baseline evidence reports:

```text
15 / 15 readiness checks passed
12 / 12 leakage / integrity tests passed
```

---

## Leakage-Safe Design

The project treats temporal leakage as a hard failure.

### Pre-match

For every target match:

```text
all contributing historical matches
must finish strictly before target kickoff
```

Rolling features are lagged before aggregation, and audit timestamps record the latest contributing source match.

### In-play

For a snapshot at time `t`:

```text
only events available at or before t
may contribute
```

No later event, final score, or eventual match statistic may enter the snapshot representation.

### Splitting and transforms

- splitting is chronological at **match level**;
- all snapshots from one match remain in one partition;
- imputers/scalers/calibrators are fit from training data only;
- validation/test labels never select the Berrar recency;
- bookmaker probabilities are a benchmark, not hidden predictive inputs;
- final test results are report-only after experimental decisions are frozen.

Perturbation tests explicitly modify target or future results and verify that earlier feature vectors remain unchanged.

---

## P1 — Berrar et al. (2024) Reimplementation

### What is reimplemented

P1 is treated as a **data-representation paper**, not as a requirement to reproduce the paper's ANN, k-NN, ordinal forest, or naive Bayes models.

The implemented representation includes:

```text
same-league longitudinal histories
        +
Super League construction across seasons
        +
total / home / away history views
        +
goals scored
goals conceded
normalized league success
        +
mean aggregation
        +
minimum six prior matches
        +
data-selected recency n
```

### Hybrid Super League

The real P1 history contains:

| Source | Seasons | Matches | Role |
|---|---:|---:|---|
| Football-Data EPL | 2000/01–2014/15 | **5,700** | Historical context only |
| StatsBomb EPL | 2015/16 | **380** | Target examples + causal current-season history |
| Combined Super League | 2000/01–2015/16 | **6,080** | Longitudinal history |

Historical same-league team histories continue across season boundaries. If a team has a gap in top-flight participation, its older EPL history is preserved; no lower-division matches are inserted.

### Paper-aligned feature sets

#### `total` — 6 features

For the home and away team:

```text
average goals scored
average goals conceded
normalized league rank
```

giving:

```text
2 teams × 3 metrics = 6 features
```

#### `homeaway` — 18 features

For both teams and each of:

```text
total
home
away
```

calculate:

```text
average goals scored
average goals conceded
normalized league rank
```

giving:

```text
2 teams × 3 views × 3 metrics = 18 features
```

### Minimum-history rule

The paper-aligned eligibility rule is:

```text
home prior same-league matches >= 6
AND
away prior same-league matches >= 6
```

A team does **not** need the full selected `n` matches. If fewer than `n` but at least six are available, all causally available matches up to `n` are used.

### Normalized league success

Normalized rank is:

\[
r = \frac{N-R}{N-1}
\]

where:

- `R` is the causal temporary rank;
- `N` is the number of teams in the current target season.

Historical performance can cross season boundaries, but the rank population is restricted to clubs participating in the target season.

### Pearson recency selection

The real implementation evaluates every:

```text
n = 9, 10, ..., 100
```

For each candidate:

\[
\Delta GD_n =
\overline{GD}_{home,n} -
\overline{GD}_{away,n}
\]

\[
\Delta Scr_n =
\overline{GoalsScored}_{home,n} -
\overline{GoalsScored}_{away,n}
\]

\[
Signal_n = \Delta GD_n + \Delta Scr_n
\]

The chosen `n` maximizes the positive Pearson correlation between `Signal_n` and observed home goal margin **using training labels only**.

The real run selected:

```text
n = 28
training Pearson correlation ≈ 0.3650
```

The selected value is now frozen.

### P1 real-run contracts

| Quantity | Result |
|---|---:|
| Historical seasons | **15** |
| Historical Football-Data matches | **5,700** |
| StatsBomb target matches | **380** |
| Combined history | **6,080** |
| P1-eligible target matches | **374** |
| Excluded cold-start matches | **6** |
| Common train | **251** |
| Common validation | **77** |
| Common test | **46** |
| Selected recency | **28** |
| Total feature contract | **6** |
| Homeaway feature contract | **18** |
| P1 readiness | **15 / 15 PASS** |
| P1 leakage/source tests | **13 / 13 PASS** |

---

## P1 Experiments

The P1 experiments use a fixed Random Forest **representation probe**. These scores are not the final Task C model benchmark; the purpose is to isolate representation effects while holding the learner constant.

### EXP01 — Representation Ablation

EXP01 compares:

```text
P1 total, fixed n=10
P1 total, selected n=28
P1 homeaway, selected n=28
MD1 baseline
MD1 + P1 total
MD1 + P1 homeaway
market baseline
```

All learned P1 comparisons use the same P1-common population:

```text
251 train / 77 validation / 46 test
```

#### RPS results

Lower is better.

| Representation | Features | Validation RPS | Test RPS |
|---|---:|---:|---:|
| Market | — | **0.19693** | **0.17485** |
| P1 Total Selected | 6 | 0.21293 | 0.22943 |
| P1 Home/Away Selected | 18 | 0.21313 | 0.22079 |
| MD1 + P1 Home/Away | 146 | 0.21755 | 0.20209 |
| MD1 + P1 Total | 134 | 0.21837 | 0.20022 |
| P1 Total Fixed | 6 | 0.22347 | 0.20831 |
| MD1 Baseline | 128 | 0.22363 | **0.19727** |

Validation selected:

```text
P1 shape: P1_TOTAL_SELECTED
project representation: MD1_PLUS_P1_HOMEAWAY
```

before the test set was opened.

The held-out test period did **not** show a robust improvement from adding the Berrar representation to the existing MD1 event-derived representation. The market remained the strongest probabilistic benchmark.

### EXP01B — Single Season vs Super League

EXP01B isolates only the effect of carrying history across seasons.

Both arms use:

```text
same 6 Berrar total features
same frozen n = 28
same minimum-six rule
same learner
same split
same metric implementation
```

The difference is only:

```text
Single Season:
StatsBomb EPL 2015/16 prior matches only
history resets at the season boundary

Super League:
Football-Data EPL 2000/01–2014/15
+ causal StatsBomb EPL 2015/16 history
```

#### Coverage

| Representation | Eligible | Coverage |
|---|---:|---:|
| Single Season | **320 / 380** | **84.21%** |
| Super League | **374 / 380** | **98.42%** |

The Super League therefore creates valid Berrar features for:

```text
54 additional matches
```

and substantially reduces the start-of-season cold-start problem.

#### Predictive RPS on the identical common population

| Representation | Validation RPS | Test RPS |
|---|---:|---:|
| Single Season | 0.21987 | **0.19897** |
| Super League | **0.21012** | 0.21565 |

Validation:

```text
ΔRPS = RPS(Super League) - RPS(Single Season)
     = -0.00975

95% paired-bootstrap CI:
[-0.02793, +0.00722]
```

Test:

```text
ΔRPS = +0.01668

95% paired-bootstrap CI:
[-0.00233, +0.03672]
```

Both confidence intervals include zero.

### P1 empirical conclusion

The Super League has a clear **data-representation / coverage benefit**:

> carrying histories across seasons increases P1 eligibility from **84.2% to 98.4%** and removes most early-season cold start.

However, the project did **not** find a robust held-out predictive improvement:

> validation favored the Super League, while the final test period favored single-season history, and the paired-bootstrap confidence interval crossed zero in both cases.

Therefore the main demonstrated value of the Berrar representation in this project is **better longitudinal coverage**, not a statistically established improvement in held-out H/D/A predictive accuracy.

This negative/mixed predictive result is retained rather than optimized away.

---

## Data-Stage Freeze

After EXP01B, the data stage is formally closed.

The frozen decisions are:

```text
P1 selected n = 28
P1 minimum prior matches = 6
P1 contracts = 6 total / 18 homeaway
outer split = existing chronological match-level split
market role = independent benchmark
```

The closure artifact records:

```text
DATA_STAGE_EXPERIMENTS_COMPLETE
```

From this point onward, later model results must **not** be used to change:

```text
P1 n
target split
feature definitions
eligibility rules
```

The next phase is prediction/modeling.

---

## Modeling Roadmap

### Task C — Pre-Match Outcome Classification

Target:

```text
P(Home), P(Draw), P(Away)
```

Primary metric:

```text
Ranked Probability Score (RPS)
```

Secondary metrics:

```text
Log-Loss
Brier Score
ECE
reliability diagrams
```

The full model-comparison phase will include the mandatory project baselines/model families and the P2 method, with hyperparameter selection and calibration confined to train/validation data.

The de-vigged bookmaker market remains the main external benchmark on the **same held-out match IDs**.

### Task R — Goal-Margin Regression

Target:

```text
home goals - away goals
```

clipped to:

```text
[-5, +5]
```

Metrics:

```text
MAE
RMSE
correlation
```

The same frozen pre-match data representation will be used.

### Task L — In-Play Forecasting

Each live example combines:

```text
frozen pre-match context
+
causal event-prefix features up to time t
```

Evaluation will track performance as a function of match minute while preserving match-level split boundaries.

If Berrar features are used in Task L, they remain constant across all snapshots of the same match; only the live event-prefix features evolve.

### P2 — Natural Gradient Boosting

The planned model-paper contribution is a from-scratch implementation of a Natural Gradient Boosting method based on:

> O'Malley, Sykulski, Lumpkin & Schuler (2023),  
> *Probabilistic Prediction of Oceanographic Velocities with Multivariate Gaussian Natural Gradient Boosting.*

The method will be adapted explicitly to the football classification/regression requirements rather than treated as already implemented.

---

## Pipeline Architecture

```text
                         RAW SOURCES
                             │
             ┌───────────────┴────────────────┐
             │                                │
        StatsBomb                        Football-Data
   matches/events/lineups              odds + approved
             │                         P1 history results
             │                                │
             └───────────────┬────────────────┘
                             ▼
                           BRONZE
                    immutable raw inputs
                             │
                             ▼
                           SILVER
             normalized relational football tables
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   MD1 pre-match       Berrar P1         Live event-prefix
      history          Super League          snapshots
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                            GOLD
              modeling-ready frozen tables
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
        Task C             Task R             Task L
        H/D/A              margin              live
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                       MODEL COMPARISON
                             │
                             ▼
              calibration / market / SHAP
```

---

## Repository Structure

The current data-pipeline branch is organized approximately as:

```text
.
├── assets/
│   └── github-banner.png
│
├── code/
│   └── data_pipeline/
│       ├── DATA_RESULTS_INDEX.md
│       ├── LATEST_DATA_RESULT_PATH.txt
│       ├── LATEST_DATA_RUN_SUMMARY.json
│       │
│       ├── Mid_Defence_1_Data_Work_Colab_Runner_ml_project.ipynb
│       ├── Mid_Defence_1_P1_Berrar_Hybrid_Colab_Runner.ipynb
│       ├── Mid_Defence_1_P1_Berrar_Hybrid_Colab_Runner_FOLDER.ipynb
│       │
│       ├── extracted/
│       │   └── md1_data_pipeline_code/
│       │       ├── README.md
│       │       ├── requirements_colab.txt
│       │       ├── SMOKE_TEST.md
│       │       ├── P1_IMPLEMENTATION_NOTES.md
│       │       ├── CHANGELOG_P1_HYBRID.md
│       │       │
│       │       ├── md1_data_pipeline/
│       │       │   ├── __init__.py
│       │       │   ├── core.py
│       │       │   ├── workflow.py
│       │       │   ├── reporting.py
│       │       │   ├── p1_berrar.py
│       │       │   └── p1_hybrid.py
│       │       │
│       │       └── colab_results/
│       │           └── EPL_2000_01_2015_16_Berrar_P1_HYBRID_REAL/
│       │
│       └── results/
│           ├── Premier_League_2015_16_DATA_ONLY_FULL_REAL/
│           ├── EXP01_P1_REPRESENTATION_TASK_C/
│           └── EXP01B_SUPERLEAGUE_ABLATION/
│
└── README.md
```

Generated Bronze/Silver/Gold datasets and large raw event files should normally be treated as reproducible artifacts rather than hand-edited source files.

---

## Key Output Artifacts

### Baseline

```text
results/
└── Premier_League_2015_16_DATA_ONLY_FULL_REAL/
    ├── audit/
    └── data/
        └── gold/
            ├── gold_prematch.parquet
            ├── gold_snapshots.parquet
            ├── feature_dictionary.csv
            └── split_manifest.csv
```

### P1

```text
extracted/md1_data_pipeline_code/colab_results/
└── EPL_2000_01_2015_16_Berrar_P1_HYBRID_REAL/
    ├── audit/
    │   ├── p1_eligibility.csv
    │   ├── p1_feature_dictionary.csv
    │   ├── p1_fidelity_table.csv
    │   ├── p1_history_coverage.csv
    │   ├── p1_leakage_tests.csv
    │   ├── p1_pearson_recency_search.csv
    │   ├── p1_readiness.csv
    │   ├── p1_selected_recency.json
    │   ├── p1_source_roles.csv
    │   └── p1_team_identity_audit.csv
    │
    └── data/
        ├── processed/
        │   └── p1_by_n/
        │       ├── n_009.parquet
        │       ├── ...
        │       └── n_100.parquet
        │
        └── gold/
            ├── p1_features_total_fixed.parquet
            ├── p1_features_total_selected.parquet
            ├── p1_features_homeaway_selected.parquet
            └── p1_split_manifest.csv
```

### EXP01B data-stage closure

```text
results/
└── EXP01B_SUPERLEAGUE_ABLATION/
    └── audit/
        ├── DATA_STAGE_CLOSURE.json
        ├── exp01b_leakage_and_method_checks.csv
        ├── exp01b_readiness.csv
        ├── experiment_summary.json
        └── FROZEN_BEFORE_TEST.json
```

---

## Reproducibility

### Environment

Use the provided Colab requirements file:

```bash
pip install -r code/data_pipeline/extracted/md1_data_pipeline_code/requirements_colab.txt
```

### Recommended execution order

```text
1. Baseline MD1 real pipeline
2. Berrar P1 hybrid real pipeline
3. EXP01 representation ablation
4. EXP01B Super-League ablation
5. Freeze data stage
6. Run Task C modeling
7. Run Task R modeling
8. Run Task L modeling
9. Run P2 / calibration / SHAP analyses
```

Do not rerun earlier data stages merely because a later predictive model performs poorly. Any change to frozen data decisions must be motivated by a verified data defect and documented as a new run.

---

## Scientific Interpretation Rules

This project deliberately separates:

```text
data representation quality
from
learner quality
```

The EXP01 and EXP01B Random Forest scores are **representation probes**, not final model-selection results.

A paper reimplementation is not considered unsuccessful simply because it does not beat the project baseline or bookmaker market. Negative and mixed results are retained when the methodology is faithful and leakage-safe.

The final predictive claims will be based on the dedicated modeling experiments, not on the P1 diagnostic learner.

---

## Features

- End-to-end StatsBomb data integration
- Bronze / Silver / Gold data architecture
- Match-, team-, player-, event-, and lineup-level relational processing
- Leakage-aware rolling pre-match feature engineering
- 128-feature verified pre-match representation
- 21 causal live snapshots per match
- 189-feature verified in-play representation
- Football-Data identity matching and de-vigged market probabilities
- Chronological match-level splitting
- Explicit target/future perturbation leakage tests
- Berrar Super League P1 reimplementation
- Full `n = 9...100` Pearson recency search
- Six-feature `total` and 18-feature `homeaway` representations
- Single-season vs multi-season cold-start ablation
- Reproducible audit manifests and readiness gates
- Train-only transformation discipline
- Planned calibrated Task C / R / L model suite
- Planned P2 Natural Gradient Boosting reimplementation
- Planned SHAP-based model interpretation

---

## References

### Data and representation paper — P1

> Berrar, D., Lopes, P., & Dubitzky, W. (2024).  
> **A data- and knowledge-driven framework for developing machine learning models to predict soccer match outcomes.**  
> *Machine Learning*, 113, 8165–8204.  
> https://doi.org/10.1007/s10994-024-06625-9

### Model paper — P2 direction

> O'Malley, M., Sykulski, A. M., Lumpkin, R., & Schuler, A. (2023).  
> **Probabilistic Prediction of Oceanographic Velocities with Multivariate Gaussian Natural Gradient Boosting.**  
> *Environmental Data Science*, 2, e10.  
> https://doi.org/10.1017/eds.2023.4

Open-access paper:

```text
https://www.cambridge.org/core/journals/environmental-data-science/article/probabilistic-prediction-of-oceanographic-velocities-with-multivariate-gaussian-natural-gradient-boosting/F26F2BD51213758208B0EBAE51D1A973
```

### Data sources

StatsBomb Open Data:

```text
https://github.com/statsbomb/open-data
```

Football-Data.co.uk:

```text
https://www.football-data.co.uk/
```

---

## Project Milestone

The repository has completed the **data integration + P1 representation stage**.

The frozen hand-off to the next phase is:

```text
DATA_STAGE_EXPERIMENTS_COMPLETE
        ↓
Task C prediction/model comparison
        ↓
Task R
        ↓
Task L
        ↓
P2 / NGBoost
        ↓
calibration + market comparison + SHAP
```

No later test/model result should be used to retrospectively change the frozen data split, Berrar recency, feature definitions, or eligibility rules.
