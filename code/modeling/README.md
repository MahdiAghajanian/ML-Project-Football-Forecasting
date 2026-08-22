# NGBoost Modeling and Evaluation

`Football_Forecasting_NGBoost_Colab.ipynb` is the final modeling notebook for pre-match outcome classification, pre-match goal-margin regression, and in-play forecasting. It is configured in `full` mode with three validation candidates per model family.

The retained final run is located at:

```text
code/modeling/outputs/ngboost_p1_corrected_full/
```

The neighboring `ngboost_p1_corrected_full_evidence.zip` archive contains the complete reproducibility bundle.

## Google Colab

1. Place the repository in Google Drive.
2. Open `Football_Forecasting_NGBoost_Colab.ipynb` in Colab.
3. If the repository folder was renamed, set `PROJECT_ROOT_OVERRIDE` in Section 3.
4. Choose **Runtime → Run all**.

The notebook installs and verifies its pinned environment, locates the frozen Phase 1 data, executes all required experiments, and writes a new full-result directory.

## Local CPU

From the repository root, prepare the isolated environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code\modeling\requirements_colab_ngboost.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name ngboost-local --display-name "Python (.venv NGBoost)"
```

Run the complete notebook with:

```powershell
.\.venv\Scripts\python.exe code\modeling\run_notebook_local.py
```

The environment is reused on later runs. The runner keeps the source notebook clean, saves a fully executed notebook in the result directory, and refreshes the evidence archive after successful completion.

## Data Contract

The notebook consumes the frozen Phase 1 experiment at:

```text
code/data_pipeline/results/EXP01_P1_REPRESENTATION_TASK_C/
```

The final representation contains 128 MD1 pre-match features plus 18 validation-selected Berrar home/away features. Those 18 values are inherited unchanged by every eligible live snapshot. The notebook verifies the upstream selection, readiness, leakage, feature, and split contracts before fitting any model.

Point-regression candidates are selected by validation RMSE. NGBoost regression candidates are selected by validation negative log-likelihood, and uncertainty scaling is estimated from validation residuals only. The multivariate experiment reports both marginal and joint coverage alongside NLL, Energy Score, and covariance diagnostics.
