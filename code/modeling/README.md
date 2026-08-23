# Modeling and P2 Evaluation

This directory now contains two explicitly different NGBoost-related workflows.

## Required P2 — project reimplementation

The required paper-reproduction path is:

```text
code/modeling/P2_Reimplementation_Colab.ipynb
code/modeling/prepare_p2_reimplementation.py
code/modeling/run_p2_reimplementation_final.py
code/modeling/p2_reimplementation/              # generated after verified preparation
code/modeling/requirements_colab_p2_reimplementation.txt
```

The P2 workflow reconstructs the committed project source under a non-conflicting local namespace, verifies the source archive and per-file SHA-256 manifest before patching, and does **not** install or import the external `ngboost` package. The exact integration patches are documented in `P2_REIMPLEMENTATION_PATCHES.md`.

Run in Colab by opening `P2_Reimplementation_Colab.ipynb` and choosing **Runtime → Run all**. For a local run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\modeling\requirements_colab_p2_reimplementation.txt
.\.venv\Scripts\python.exe code\modeling\prepare_p2_reimplementation.py
$env:PYTHONPATH = "code\modeling"
.\.venv\Scripts\python.exe -m pytest code\modeling\test_p2_reimplementation.py -q
.\.venv\Scripts\python.exe code\modeling\run_p2_reimplementation_final.py --mode full
```

A successful full run writes report/evidence material to:

```text
code/modeling/outputs/p2_reimplementation_full/
```

The P2 evaluator includes Task C raw/calibrated metrics, class diagnostics, reliability, confusion matrices, row predictions and worst cases; Task R point/distributional metrics, interval coverage and worst cases; Task L live-versus-frozen evaluation by minute and phase; multivariate validation selection and covariance diagnostics; sampled peak RSS and prediction latency; and a P2-versus-official-library comparison table.

### Feature and split contract

The reimplementation retains the chronological 251/77/46 P1-eligible match split. It reconstructs the 128 MD1 pre-match and 189 MD1 live feature sets from the **data-pipeline audit** (`feature_distribution_summary.csv`) and takes the 18 P1 fields from the frozen P1 parquet schema. It therefore does not depend on an old modeling `run_config.json` to define its input features.

## Historical official-library NGBoost baseline

`Football_Forecasting_NGBoost_Colab.ipynb`, `requirements_colab_ngboost.txt`, and:

```text
code/modeling/outputs/ngboost_p1_corrected_full/
```

are retained as the completed **official NGBoost library baseline/model-suite experiment**. That notebook imports PyPI `ngboost==0.5.11`. Its results are useful as a reference comparator, but they are **not** the required P2 project reimplementation and should be labelled accordingly in the report and unified result tables.

The P2 runner may read the historical metric CSVs at the end of a run solely to produce a side-by-side comparison. Those files are not used for feature selection, P2 tuning, model fitting, calibration or P2 predictions.

## Reporting rule

Do not copy historical NGBoost numbers into the P2 reimplementation rows. Report P2 numbers only after `P2_Reimplementation_Colab.ipynb` completes successfully. The generated `P2_REPORT_INSERT.md`, tables, figures, resource measurements and row-level predictions are the evidence source for the final P2 report sections.
