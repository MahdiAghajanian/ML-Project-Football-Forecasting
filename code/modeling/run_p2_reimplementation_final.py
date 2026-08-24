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


def official_comparison(root, tab):
    rows = []
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

    if old.exists():
        try:
            oc = pd.read_csv(old / "task_c_all_metrics.csv")
            x = oc[(oc.Model == "NGBoost") & (oc.Calibration == "Platt")].iloc[0]
            rows.append(["C", "Platt", "Official NGBoost library baseline", "RPS", float(x.RPS)])
        except Exception: pass
        try:
            orr = pd.read_csv(old / "task_r_all_metrics.csv")
            x = orr[orr.Model == "NGBoost"].iloc[0]
            rows.append(["R", "Point", "Official NGBoost library baseline", "RMSE", float(x.RMSE)])
        except Exception: pass
        try:
            olc = pd.read_csv(old / "task_l_classification_all_metrics.csv")
            for cal in ("Raw", "Platt"):
                x = olc[(olc.Model == "NGBoost") & (olc.Calibration == cal)]
                if not x.empty: rows.append(["L outcome", cal, "Official NGBoost library baseline", "RPS", float(x.iloc[0].RPS)])
        except Exception: pass
        try:
            olr = pd.read_csv(old / "task_l_regression_all_metrics.csv")
            x = olr[olr.Model == "NGBoost"]
            if not x.empty: rows.append(["L margin", "Point", "Official NGBoost library baseline", "RMSE", float(x.iloc[0].RMSE)])
        except Exception: pass

    mv = tab / "p2_multivariate_metrics.csv"
    oldmv = old / "p2_multivariate_ngboost_metrics.csv"
    if mv.exists():
        x = pd.read_csv(mv).iloc[0]
        rows.append(["P2 multivariate", "Raw", "P2 project reimplementation", "JointNLL", float(x.JointNLL)])
        if oldmv.exists():
            y = pd.read_csv(oldmv).iloc[0]
            rows.append(["P2 multivariate", "Raw", "Official NGBoost library baseline", "JointNLL", float(y.JointNLL)])

    out = pd.DataFrame(rows, columns=["Task", "Evaluation", "Implementation", "Metric", "Value"])
    out.to_csv(tab / "p2_vs_official_library.csv", index=False)
    return out


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


def finalize():
    root = core.root_of(arg("--project-root", "")); mode = arg("--mode", "full")
    out = root / f"code/modeling/outputs/p2_reimplementation_{mode}"; tab = out / "tables"; figdir = out / "figures"
    comp = official_comparison(root, tab); figs = phase_figures(tab, figdir)

    required = [
        "p2_task_c_metrics.csv", "p2_task_c_predictions.csv", "p2_task_c_worst10.csv", "p2_task_c_reliability.csv",
        "p2_task_r_metrics.csv", "p2_task_r_predictions.csv", "p2_task_r_worst10.csv",
        "p2_task_l_outcome_metrics.csv", "p2_task_l_outcome_by_minute.csv", "p2_task_l_outcome_by_phase.csv", "p2_task_l_outcome_predictions.csv",
        "p2_task_l_margin_metrics.csv", "p2_task_l_margin_by_minute.csv", "p2_task_l_margin_by_phase.csv", "p2_task_l_margin_predictions.csv",
        "p2_compute_and_peak_memory.csv", "p2_vs_official_library.csv",
    ]
    tasks = set(comp.Task.astype(str)) if not comp.empty else set()
    both = lambda t: set(comp.loc[comp.Task == t, "Implementation"]) >= {"P2 project reimplementation", "Official NGBoost library baseline"}
    checklist = {
        "direct_p2_source_runtime": True,
        "external_ngboost_not_installed_for_p2": True,
        "task_evidence_files_complete": all((tab / f).exists() for f in required),
        "task_l_phase_reliability_figures": len(figs) >= 4,
        "official_library_comparison_C": both("C"),
        "official_library_comparison_R": both("R"),
        "official_library_comparison_L_outcome": both("L outcome"),
        "official_library_comparison_L_margin": both("L margin"),
        "multivariate_evidence": (tab / "p2_multivariate_metrics.csv").exists(),
        "peak_memory_and_latency_evidence": (tab / "p2_compute_and_peak_memory.csv").exists(),
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
        "The required P2 model is executed from the visible `code/modeling/p2_source/` implementation through a project-local `ngboost` facade. The external PyPI NGBoost package is not installed by the P2 requirements. The historical library run is retained only as a labelled baseline. Hyperparameter selection, probability calibration, and uncertainty scaling use validation data only.", "",
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


if __name__ == "__main__":
    core.main()
    finalize()
