# NGBoost modeling notebook

Open `Football_Forecasting_NGBoost_Colab.ipynb` in Google Colab and choose
**Runtime -> Run all**. The notebook auto-detects the repository at the Phase 1
Google Drive path documented by the data pipeline. If the folder was renamed,
set `PROJECT_ROOT_OVERRIDE` in Section 3.

The default `standard` mode runs the full required experiment matrix with two
validation candidates per family. Use `quick` for a plumbing run and `full` for
the final report. Results, figures, fitted models, row-level predictions, runtime
versions, and a ZIP evidence bundle are written beneath:

`code/modeling/outputs/ngboost_p1_corrected_<mode>/`

For example, a final `RUN_MODE = "full"` run writes to
`code/modeling/outputs/ngboost_p1_corrected_full/` and creates the neighboring
`ngboost_p1_corrected_full_evidence.zip`. This separate name prevents corrected
results from mixing with the earlier standard/full evidence folders.

The notebook consumes the frozen P1 representation experiment in
`code/data_pipeline/results/EXP01_P1_REPRESENTATION_TASK_C/`. Its primary
pre-match matrix contains 128 MD1 features plus the validation-selected 18
Berrar home/away features. Those 18 pre-match values are inherited unchanged by
each eligible match's live snapshots. The notebook verifies the upstream P1
decision and readiness artifacts before fitting any model.

For regression, point-model candidates are selected by validation RMSE, while
NGBoost candidates are selected by validation negative log-likelihood so that a
collapsed predictive scale cannot win on mean accuracy alone. Task R exports
raw and validation-scale-calibrated uncertainty diagnostics. The multivariate
P2 experiment exports marginal and joint coverage in addition to NLL and Energy
Score.
