# Mid Defence 1 Data Pipeline Code

This archive contains the implementation extracted from the original monolithic Colab notebook.

## Structure

- `md1_data_pipeline/core.py` — configuration, data download, cleaning, integration, leakage-safe feature construction, snapshots, audits, tests, and pipeline orchestration.
- `md1_data_pipeline/reporting.py` — descriptive profiles, plots, chronological/class summaries, readiness gate, and evidence ZIP/index exports.
- `md1_data_pipeline/workflow.py` — high-level configuration and one-call execution helpers.
- `requirements_colab.txt` — Colab-compatible dependencies.

## Google Drive placement

Upload `md1_data_pipeline_code.zip` to:

`My Drive/ml_project/code/data_pipeline/md1_data_pipeline_code.zip`

The companion Colab notebook mounts Drive, uses `My Drive/ml_project/code/data_pipeline` as the code workspace, extracts this archive into its `extracted` subfolder, adds the package to `sys.path`, installs dependencies, and executes the workflow.

## Run modes

- `synthetic_demo`: fast verification without downloading the full season.
- `real`: real StatsBomb/Football-Data processing. With `MAX_MATCHES = None`, it processes the complete selected season.
