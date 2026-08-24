# Modeling and P2 Evaluation

This directory contains two explicitly different NGBoost-related workflows.

## Required P2 — project reimplementation

The required paper-reproduction path is:

```text
code/modeling/p2_source/                     # canonical implementation
code/modeling/ngboost/__init__.py            # local package facade
code/modeling/p2_source_manifest_sha256.json
code/modeling/prepare_p2_reimplementation.py # verifier only
code/modeling/P2_Reimplementation_Colab.ipynb
code/modeling/run_p2_reimplementation_final.py
code/modeling/requirements_colab_p2_reimplementation.txt
```

The P2 source is used directly from `p2_source/`. The local `ngboost` facade preserves the same imports used by the earlier workflow (`from ngboost import NGBClassifier, NGBRegressor`) while resolving submodules to the committed project implementation. The P2 requirements do **not** install the external PyPI `ngboost` package.

Before training, `prepare_p2_reimplementation.py` verifies all 28 Python source files against `p2_source_manifest_sha256.json`, compiles them, and confirms that the local package facade imports successfully. It does not unpack, copy, or rewrite an archive.

For Colab, open `P2_Reimplementation_Colab.ipynb` and choose **Runtime → Run all**. For local CPU execution:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\modeling\requirements_colab_p2_reimplementation.txt
$env:PYTHONPATH = "code\modeling"
.\.venv\Scripts\python.exe code\modeling\prepare_p2_reimplementation.py
.\.venv\Scripts\python.exe -m pytest code\modeling\test_p2_reimplementation.py -q
.\.venv\Scripts\python.exe code\modeling\run_p2_reimplementation_final.py --mode full
```

A successful full run writes:

```text
code/modeling/outputs/p2_reimplementation_full/
```

The evaluator covers Task C raw/calibrated metrics, class diagnostics, reliability, confusion matrices, row predictions and worst cases; Task R point/distributional metrics, predictive intervals and worst cases; Task L live-versus-frozen classification/regression by minute and phase; multivariate validation selection and covariance diagnostics; sampled peak RSS and prediction latency. The finalizer also creates phase reliability figures, a strict evidence-derived checklist, an expanded report insert, and a unified P2-versus-official-library comparison for C/R/L.

### Feature and split contract

The P2 evaluation retains the chronological 251/77/46 P1-eligible match split. It reconstructs 128 MD1 pre-match and 189 MD1 live feature sets from the data-pipeline audit and takes the 18 P1 fields from the frozen P1 parquet schema, yielding 146 pre-match and 207 live features. It does not depend on the historical modeling `run_config.json` to define inputs.

## Historical official-library NGBoost baseline

`Football_Forecasting_NGBoost_Colab.ipynb`, `requirements_colab_ngboost.txt`, and `outputs/ngboost_p1_corrected_full/` are retained as the completed official-library model-suite experiment. That workflow used PyPI `ngboost==0.5.11`; its results are reference baselines, not the required P2 reimplementation.

The P2 finalizer may read those historical metric CSVs only after the P2 run to create labelled side-by-side comparisons. Historical files are never used to select P2 features, tune models, calibrate probabilities, scale uncertainty, or generate P2 predictions.

## Reporting rule

Do not copy historical NGBoost numbers into P2 rows. Report P2 values only after `P2_Reimplementation_Colab.ipynb` completes successfully. The generated P2 output directory is the evidence source for the final P2 report sections.
