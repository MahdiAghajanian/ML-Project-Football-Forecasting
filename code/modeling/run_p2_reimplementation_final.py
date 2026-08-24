#!/usr/bin/env python
"""Run P2 from the project-local p2_source implementation and finalize evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_p2_reimplementation_core as core


def arg(name, default=""):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def md_table(df, n=20):
    if df is None or df.empty:
        return "_No rows available._"
    x = df.head(n).copy()
    for c in x.columns:
        x[c] = x[c].map(lambda v: "" if pd.isna(v) else (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)))
    return "\n".join([
        "| " + " | ".join(x.columns) + " |",
        "| " + " | ".join(["---"] * len(x.columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in x.to_numpy()],
    ])


def pick(df, **conds):
    x = df
    for c, v in conds.items():
        if c not in x.columns:
            return None
        x = x[x[c].astype(str).str.casefold() == str(v).casefold()]
    return None if x.empty else x.iloc[0]


def official_comparison(root, tab, outdir):
    """Build C/R/L comparison and surface every baseline-read problem."""
    rows, errors = [], []
    old = root / "code/modeling/outputs/ngboost_p1_corrected_full/tables"
    c = pd.read_csv(tab / "p2_task_c_metrics.csv")
    r = pd.read_csv(tab / "p2_task_r_metrics.csv")
    lc = pd.read_csv(tab / "p2_task_l_outcome_metrics.csv")
    lr = pd.read_csv(tab / "p2_task_l_margin_metrics.csv")

    pc = pick(c, Calibration="Platt")
    plraw = pick(lc, Evaluation="Live Raw")
    plcal = pick(lc, Evaluation="Live Platt")
    plr = pick(lr, Evaluation="Live")
    if pc is not None: rows.append(["C", "Platt", "P2 project reimplementation", "RPS", float(pc.RPS)])
    rows.append(["R", "Point", "P2 project reimplementation", "RMSE", float(r.iloc[0].RMSE)])
    if plraw is not None: rows.append(["L outcome", "Raw", "P2 project reimplementation", "RPS", float(plraw.RPS)])
    if plcal is not None: rows.append(["L outcome", "Platt", "P2 project reimplementation", "RPS", float(plcal.RPS)])
    if plr is not None: rows.append(["L margin", "Point", "P2 project reimplementation", "RMSE", float(plr.RMSE)])

    def read_baseline(label, fn):
        try:
            fn()
        except Exception as exc:  # captured as explicit evidence, never silently ignored
            errors.append(f"{label}: {type(exc).__name__}: {exc}")

    if not old.exists():
        errors.append(f"Historical baseline table directory is missing: {old}")
    else:
        def add_c():
            oc = pd.read_csv(old / "task_c_all_metrics.csv")
            x = oc[(oc.Model == "NGBoost") & (oc.Calibration == "Platt")].iloc[0]
            rows.append(["C", "Platt", "Official NGBoost library baseline", "RPS", float(x.RPS)])
        read_baseline("Task C baseline", add_c)

        def add_r():
            orr = pd.read_csv(old / "task_r_all_metrics.csv")
            x = orr[orr.Model == "NGBoost"].iloc[0]
            rows.append(["R", "Point", "Official NGBoost library baseline", "RMSE", float(x.RMSE)])
        read_baseline("Task R baseline", add_r)

        def add_lc():
            olc = pd.read_csv(old / "task_l_classification_all_metrics.csv")
            added = 0
            for cal in ("Raw", "Platt"):
                x = olc[(olc.Model == "NGBoost") & (olc.Calibration == cal)]
                if not x.empty:
                    rows.append(["L outcome", cal, "Official NGBoost library baseline", "RPS", float(x.iloc[0].RPS)])
                    added += 1
            if added == 0:
                raise RuntimeError("No NGBoost Task L classification rows found")
        read_baseline("Task L outcome baseline", add_lc)

        def add_lr():
            olr = pd.read_csv(old / "task_l_regression_all_metrics.csv")
            x = olr[olr.Model == "NGBoost"]
            if x.empty:
                raise RuntimeError("No NGBoost Task L regression row found")
            rows.append(["L margin", "Point", "Official NGBoost library baseline", "RMSE", float(x.iloc[0].RMSE)])
        read_baseline("Task L margin baseline", add_lr)

    mv = tab / "p2_multivariate_metrics.csv"
    oldmv = old / "p2_multivariate_ngboost_metrics.csv"
    if mv.exists():
        x = pd.read_csv(mv).iloc[0]
        rows.append(["P2 multivariate", "Raw", "P2 project reimplementation", "JointNLL", float(x.JointNLL)])
        if oldmv.exists():
            try:
                y = pd.read_csv(oldmv).iloc[0]
                rows.append(["P2 multivariate", "Raw", "Official NGBoost library baseline", "JointNLL", float(y.JointNLL)])
            except Exception as exc:
                errors.append(f"Multivariate baseline: {type(exc).__name__}: {exc}")

    comp = pd.DataFrame(rows, columns=["Task", "Evaluation", "Implementation", "Metric", "Value"])
    comp.to_csv(tab / "p2_vs_official_library.csv", index=False)
    err_path = outdir / "official_baseline_comparison_errors.txt"
    if errors:
        err_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print("Official-baseline comparison warnings:\n" + "\n".join(errors), file=sys.stderr)
    elif err_path.exists():
        err_path.unlink()
    return comp, errors


def phase_figures(tab, figdir):
    path = tab / "p2_task_l_phase_reliability.csv"
    if not path.exists(): return []
    d = pd.read_csv(path); made = []
    for phase, g in d.groupby("Phase", sort=False):
        fig, ax = plt.subplots(figsize=(5.5, 4.5)); ax.plot([0,1],[0,1], "--", linewidth=1)
        for label, q in g.groupby("Evaluation"):
            q = q.dropna(subset=["mean_confidence", "accuracy"])
            ax.plot(q.mean_confidence, q.accuracy, marker="o", label=label)
        ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title=f"Task L reliability — {phase}", xlim=(0,1), ylim=(0,1)); ax.legend(fontsize=8)
        fig.tight_layout(); name = f"p2_task_l_reliability_{str(phase).replace('/','_')}.png"; fig.savefig(figdir / name, dpi=180); plt.close(fig); made.append(name)
    return made


def nonempty(path):
    return path.exists() and path.stat().st_size > 0


def finalize():
    root = core.root_of(arg("--project-root", "")); mode = arg("--mode", "full")
    out = root / f"code/modeling/outputs/p2_reimplementation_{mode}"; tab = out / "tables"; figdir = out / "figures"; models = out / "models"

    # Core creates legacy draft checklist/report files. Remove them before doing
    # any finalization so a failed finalizer can never leave optimistic evidence.
    for stale in (out / "p2_completion_checklist.json", out / "P2_REPORT_INSERT.md"):
        if stale.exists(): stale.unlink()

    comp, comparison_errors = official_comparison(root, tab, out)
    figs = phase_figures(tab, figdir)

    required_tables = [
        "p2_task_c_metrics.csv", "p2_task_c_predictions.csv", "p2_task_c_worst10.csv", "p2_task_c_reliability.csv",
        "p2_task_c_confusion_raw.csv", "p2_task_c_confusion_platt.csv",
        "p2_task_r_metrics.csv", "p2_task_r_predictions.csv", "p2_task_r_worst10.csv",
        "p2_task_l_outcome_metrics.csv", "p2_task_l_outcome_by_minute.csv", "p2_task_l_outcome_by_phase.csv",
        "p2_task_l_outcome_predictions.csv", "p2_task_l_outcome_worst10.csv", "p2_task_l_phase_reliability.csv",
        "p2_task_l_margin_metrics.csv", "p2_task_l_margin_by_minute.csv", "p2_task_l_margin_by_phase.csv",
        "p2_task_l_margin_predictions.csv", "p2_task_l_margin_worst10.csv",
        "p2_multivariate_metrics.csv", "p2_multivariate_predictions.csv", "p2_compute_and_peak_memory.csv",
        "p2_vs_official_library.csv",
    ]
    required_figures = [
        "p2_task_c_reliability.png", "p2_task_r_intervals.png",
        "p2_task_l_outcome_rps_vs_minute.png", "p2_task_l_margin_rmse_vs_minute.png",
        "p2_multivariate_predicted_correlation.png",
    ]
    required_models = [
        "p2_task_c_classifier.joblib", "p2_task_c_platt.joblib", "p2_task_r_regressor.joblib",
        "p2_task_l_classifier.joblib", "p2_task_l_platt.joblib", "p2_task_l_regressor.joblib",
        "p2_multivariate_goals.joblib",
    ]

    both = lambda t: set(comp.loc[comp.Task == t, "Implementation"]) >= {"P2 project reimplementation", "Official NGBoost library baseline"}
    run_cfg = out / "run_config.json"
    cfg = json.loads(run_cfg.read_text(encoding="utf-8")) if nonempty(run_cfg) else {}
    local_runtime = bool((root / "code/modeling/p2_source").is_dir() and (root / "code/modeling/ngboost/__init__.py").is_file())
    external_disabled = cfg.get("external_ngboost_runtime_dependency") is False

    checklist = {
        "direct_p2_source_runtime": local_runtime,
        "external_ngboost_not_runtime_dependency": external_disabled,
        "all_required_tables_nonempty": all(nonempty(tab / f) for f in required_tables),
        "all_required_figures_nonempty": all(nonempty(figdir / f) for f in required_figures) and len(figs) >= 4,
        "all_required_models_nonempty": all(nonempty(models / f) for f in required_models),
        "run_config_present": nonempty(run_cfg),
        "summary_present": nonempty(out / "p2_summary.json"),
        "official_library_comparison_C": both("C"),
        "official_library_comparison_R": both("R"),
        "official_library_comparison_L_outcome": both("L outcome"),
        "official_library_comparison_L_margin": both("L margin"),
        "official_comparison_has_no_errors": len(comparison_errors) == 0,
        "multivariate_evidence": nonempty(tab / "p2_multivariate_metrics.csv") and nonempty(models / "p2_multivariate_goals.joblib"),
        "peak_memory_and_latency_evidence": nonempty(tab / "p2_compute_and_peak_memory.csv"),
    }
    checklist["static_and_run_evidence_complete"] = all(checklist.values())
    (out / "p2_completion_checklist.json").write_text(json.dumps(checklist, indent=2), encoding="utf-8")

    c = pd.read_csv(tab / "p2_task_c_metrics.csv"); r = pd.read_csv(tab / "p2_task_r_metrics.csv")
    lc = pd.read_csv(tab / "p2_task_l_outcome_metrics.csv"); lr = pd.read_csv(tab / "p2_task_l_margin_metrics.csv")
    phasec = pd.read_csv(tab / "p2_task_l_outcome_by_phase.csv"); phaser = pd.read_csv(tab / "p2_task_l_margin_by_phase.csv")
    compute = pd.read_csv(tab / "p2_compute_and_peak_memory.csv")
    lines = [
        "# P2 report insert — project reimplementation", "",
        "## 4.2 Model-paper reimplementation", "",
        "The required P2 model is executed from the visible `code/modeling/p2_source/` implementation through a project-local `ngboost` facade. The external PyPI NGBoost package is not a runtime dependency of this P2 path. The historical library run is retained only as a labelled baseline. Hyperparameter selection, probability calibration, and uncertainty scaling use validation data only.", "",
        "## Unified P2 vs official-library comparison", "", md_table(comp), "",
        "## Task C — pre-match outcome", "", md_table(c), "",
        "Calibration is reported both raw and after validation-only Platt calibration. Reliability bins, confusion matrices, row-level probabilities and the ten worst held-out predictions are stored with the generated evidence.", "",
        "## Task R — pre-match margin", "", md_table(r), "",
        "Point accuracy is accompanied by raw/calibrated NLL and 50/80/95% predictive-interval coverage. Row-level means, scales, intervals and the ten largest errors are retained.", "",
        "## Task L — live outcome and margin", "", md_table(lc), "", md_table(lr), "",
        "The live models are compared at every snapshot with a frozen P2 pre-match baseline. Classification and regression metrics are written by minute and phase; phase reliability diagrams are generated for the live outcome model.", "",
        "### Task L outcome by phase", "", md_table(phasec), "", "### Task L margin by phase", "", md_table(phaser), "",
        "## Compute and memory", "", md_table(compute), "",
        "Training time uses wall-clock measurement while peak RSS is sampled repeatedly during fitting. Prediction latency reports repeated-batch p50/p95/p99 measurements.", "",
        "## Failure analysis and limitations", "",
        "Worst-case Task C, Task R, live-outcome and live-margin rows are saved explicitly rather than selected narratively. Results cover one league-season and a 46-match held-out set; live snapshots are repeated observations within matches, and Gaussian score/margin assumptions may remain imperfectly calibrated despite validation-only dispersion correction.", "",
        "## Appendix A", "", "See `code/modeling/P2_APPENDIX_A_DERIVATION.md` for the natural-gradient, Fisher-information, initialization, line-search and multivariate derivations.",
    ]
    (out / "P2_REPORT_INSERT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(checklist, indent=2))
    if not checklist["static_and_run_evidence_complete"]:
        print("WARNING: P2 final evidence checklist is not fully satisfied.", file=sys.stderr)


if __name__ == "__main__":
    core.main()
    finalize()
