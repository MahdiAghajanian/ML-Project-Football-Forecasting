# NGBoost modeling notebook

Open `Football_Forecasting_NGBoost_Colab.ipynb` in Google Colab and choose
**Runtime -> Run all**. The notebook auto-detects the repository at the Phase 1
Google Drive path documented by the data pipeline. If the folder was renamed,
set `PROJECT_ROOT_OVERRIDE` in Section 3.

The default `standard` mode runs the full required experiment matrix with two
validation candidates per family. Use `quick` for a plumbing run and `full` for
the final report. Results, figures, fitted models, row-level predictions, runtime
versions, and a ZIP evidence bundle are written beneath:

`code/modeling/outputs/ngboost_full_run/`
