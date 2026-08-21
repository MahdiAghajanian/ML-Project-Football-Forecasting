# NGBoost modeling notebook

Open `Football_Forecasting_NGBoost_Colab.ipynb` in Google Colab and choose
**Runtime -> Run all**. The notebook auto-detects the repository at the Phase 1
Google Drive path documented by the data pipeline. If the folder was renamed,
set `PROJECT_ROOT_OVERRIDE` in Section 3.

The default `standard` mode runs the full required experiment matrix with two
validation candidates per family. Use `quick` for a plumbing run and `full` for
the final report. Results, figures, fitted models, row-level predictions, runtime
versions, and a ZIP evidence bundle are written beneath:

`code/modeling/outputs/ngboost_p1_full_run/`

The notebook consumes the frozen P1 representation experiment in
`code/data_pipeline/results/EXP01_P1_REPRESENTATION_TASK_C/`. Its primary
pre-match matrix contains 128 MD1 features plus the validation-selected 18
Berrar home/away features. Those 18 pre-match values are inherited unchanged by
each eligible match's live snapshots. The notebook verifies the upstream P1
decision and readiness artifacts before fitting any model.
