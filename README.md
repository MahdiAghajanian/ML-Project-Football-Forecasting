<h1 align="center">Forecasting Competitive Football</h1>

<p align="center">
  <img src="assets/github-banner.png" width="100%" alt="Forecasting Competitive Football">
</p>

<p align="center">
Probabilistic pre-match and in-play football forecasting with leakage-safe event pipelines, Berrar-style longitudinal representations, and Natural Gradient Boosting.
</p>

This repository contains the complete data-engineering, modeling, evaluation, and interpretation workflow for Premier League 2015/16. The final experiment uses a chronological holdout, validation-only model selection and calibration, an independent de-vigged bookmaker baseline, and a paper-aligned NGBoost implementation.

## Final Held-Out Results

The final test set contains 46 matches and 966 in-play snapshots. Model and representation choices were frozen using training and validation data before test evaluation. Lower is better for RPS, RMSE, and negative log-likelihood.

| Task | Model | Evaluation | Final result |
|---|---|---|---:|
| Pre-match outcome (Task C) | De-vigged market | RPS | **0.1748** |
| Pre-match outcome (Task C) | Random Forest, raw | RPS | **0.1958** |
| Pre-match outcome (Task C) | NGBoost, Platt calibrated | RPS | **0.2142** |
| Pre-match margin (Task R) | Kernel Ridge, Nyström | RMSE | **1.7878** |
| Pre-match margin (Task R) | NGBoost | RMSE | **1.8665** |
| In-play outcome (Task L) | NGBoost, raw | RPS | **0.1590** |
| In-play margin (Task L) | Gradient Boosting | RMSE | **1.4477** |
| In-play margin (Task L) | NGBoost | RMSE | **1.4667** |

The main findings are:

- the bookmaker market remained the strongest pre-match probabilistic benchmark;
- NGBoost produced the best aggregate in-play outcome RPS;
- live match information reduced NGBoost RPS from 0.2082 at minute 0 to 0.0698 at full time;
- validation-only scale calibration improved Task R NGBoost test NLL from 3.1862 to 2.1749;
- the multivariate Gaussian NGBoost experiment was strongly under-dispersed, so its raw uncertainty intervals are reported as a limitation rather than presented as calibrated forecasts.

These results describe one competition-season and a 46-match test period. Rankings therefore have substantial sampling uncertainty and should not be generalized to other leagues or seasons without additional evaluation.

## Project Scope

The system supports three forecasting settings:

| Task | Prediction time | Inputs | Outputs |
|---|---|---|---|
| Task C | Pre-match | Causal team history and frozen P1 representation | Away / Draw / Home probabilities |
| Task R | Pre-match | Same pre-match representation | Goal margin clipped to `[-5, 5]` |
| Task L | In-play | Pre-match context and event prefix available at time `t` | Updated outcome probabilities and final margin |

The modeling suite includes Dummy, SVM, Random Forest, Gradient Boosting, XGBoost, LightGBM, kernel methods, and NGBoost where applicable. It also includes probability calibration, class-imbalance experiments, uncertainty coverage, market comparison, kernel-scaling analysis, per-minute evaluation, feature-group ablation, SHAP explanations, runtime measurement, and failure-case analysis.

## Final Experiment Configuration

| Item | Final value |
|---|---:|
| StatsBomb target matches | 380 |
| P1-eligible matches | 374 |
| Train / validation / test matches | 251 / 77 / 46 |
| P1-eligible live snapshots | 7,854 |
| Live test snapshots | 966 |
| Pre-match features | 146 |
| In-play features | 207 |
| Selected Berrar recency | 28 matches |
| Hyperparameter candidates per family | 3 |
| Notebook execution | 31 / 31 code cells |
| Final completion checks | 25 / 25 passed |

The pre-match representation combines 128 MD1 features with the validation-selected 18-feature Berrar home/away representation. Those 18 pre-match values remain fixed across all snapshots of the same match; only leakage-safe event-prefix features evolve during play.

## Data and Representation Pipeline

StatsBomb Open Data is the primary modeling source. It supplies match metadata, labels, lineups, and 1,313,773 event rows for Premier League 2015/16. Football-Data.co.uk is used in two explicitly separated roles:

1. EPL 2015/16 bookmaker odds provide the independent de-vigged market benchmark.
2. Basic completed-match results from EPL 2000/01–2014/15 provide approved historical context for the Berrar Super League representation.

Historical Football-Data rows are not supervised target examples, and bookmaker probabilities never enter the predictive feature matrices.

The data architecture follows immutable Bronze, normalized Silver, and modeling-ready Gold stages:

```text
StatsBomb events, matches, lineups       Football-Data results and odds
                 │                                   │
                 └──────────────┬────────────────────┘
                                ▼
                         Bronze / Silver
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       MD1 pre-match       Berrar P1       Live event prefixes
              └─────────────────┼─────────────────┘
                                ▼
                              Gold
                                │
                    Task C / Task R / Task L
                                │
                                ▼
          model comparison, calibration, SHAP, uncertainty
```

## Paper Reimplementations

### Berrar et al. — longitudinal representation

The P1 stage reproduces the paper's same-league longitudinal history, cross-season Super League construction, six-feature `total` representation, 18-feature `homeaway` representation, minimum-six-history rule, mean aggregation, normalized league success, and Pearson recency selection.

The combined history contains 5,700 Football-Data matches and 380 StatsBomb target-season matches. Training-only Pearson selection froze the recency at `n = 28`. Carrying history across seasons increased P1 eligibility from 84.2% to 98.4%, although it did not establish a robust held-out predictive improvement.

### O'Malley et al. — Natural Gradient Boosting

The modeling stage uses NGBoost for categorical outcome distributions and Normal goal-margin distributions. It additionally adapts the supplied multivariate method to predict a bivariate Gaussian distribution over home and away goals, including conditional means, variances, and covariance.

The multivariate experiment reports joint NLL, Energy Score, marginal coverage, joint elliptical coverage, and covariance eigenvalue diagnostics. Its poor raw coverage is retained transparently as an empirical limitation of the Gaussian adaptation to football goal counts.

## Leakage and Evaluation Controls

- All data splits are chronological and match-level.
- Every snapshot from a match remains in the same partition.
- Historical matches must finish strictly before the target kickoff.
- Live features use only events available at or before each snapshot cutoff.
- Imputers, scalers, model selection, calibration, and uncertainty scaling use training or validation data only.
- Test labels do not select the P1 recency, representation, model configuration, or calibrator.
- Market probabilities are evaluated on the same held-out match IDs but remain outside all feature matrices.
- Perturbation and readiness tests fail if target or future information can affect an earlier feature vector.

## Final Artifacts

- [Source modeling notebook](code/modeling/Football_Forecasting_NGBoost_Colab.ipynb)
- [Executed full notebook](code/modeling/outputs/ngboost_p1_corrected_full/Football_Forecasting_NGBoost_executed_full.ipynb)
- [Final results narrative](code/modeling/outputs/ngboost_p1_corrected_full/automatic_results_narrative.md)
- [Completion checklist](code/modeling/outputs/ngboost_p1_corrected_full/completion_checklist.json)
- [Final evidence archive](code/modeling/outputs/ngboost_p1_corrected_full_evidence.zip)
- [Modeling instructions](code/modeling/README.md)

The retained final result directory contains evaluation tables, figures, fitted models, row-level predictions, runtime versions, environment metadata, and the fully executed notebook. Earlier exploratory and superseded model-output folders are intentionally excluded.

## Reproduction

The notebook is configured for the final three-candidate `full` run.

For Google Colab, open `code/modeling/Football_Forecasting_NGBoost_Colab.ipynb`, attach the repository from Google Drive, and choose **Runtime → Run all**.

For local CPU execution, follow [the modeling setup instructions](code/modeling/README.md). After the one-time environment installation, run:

```powershell
.\.venv\Scripts\python.exe code\modeling\run_notebook_local.py
```

Dependencies are reused on subsequent runs. Local environments and Python bytecode are excluded through `.gitignore`.

## Repository Structure

```text
.
├── assets/
│   └── github-banner.png
├── code/
│   ├── data_pipeline/
│   │   ├── extracted/md1_data_pipeline_code/
│   │   └── results/
│   │       ├── Premier_League_2015_16_DATA_ONLY_FULL_REAL/
│   │       ├── EXP01_P1_REPRESENTATION_TASK_C/
│   │       └── EXP01B_SUPERLEAGUE_ABLATION/
│   └── modeling/
│       ├── Football_Forecasting_NGBoost_Colab.ipynb
│       ├── requirements_colab_ngboost.txt
│       ├── run_notebook_local.py
│       └── outputs/
│           ├── ngboost_p1_corrected_full/
│           └── ngboost_p1_corrected_full_evidence.zip
└── README.md
```

## References

- Berrar, D., Lopes, P., & Dubitzky, W. (2024). *A data- and knowledge-driven framework for developing machine learning models to predict soccer match outcomes*. Machine Learning, 113, 8165–8204. [DOI](https://doi.org/10.1007/s10994-024-06625-9)
- O'Malley, M., Sykulski, A. M., Lumpkin, R., & Schuler, A. (2023). *Probabilistic Prediction of Oceanographic Velocities with Multivariate Gaussian Natural Gradient Boosting*. Environmental Data Science, 2, e10. [DOI](https://doi.org/10.1017/eds.2023.4)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [Football-Data.co.uk](https://www.football-data.co.uk/)
