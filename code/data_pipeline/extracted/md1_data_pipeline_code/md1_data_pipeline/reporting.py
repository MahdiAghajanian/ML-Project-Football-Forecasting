"""Reporting, profiling, readiness, and evidence-export helpers."""

from __future__ import annotations

import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import MD1Config, read_json, save_table


def _save_figure(path: Path, *, show_plots: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close()
    return path


def table_quality_summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
    duplicate_rows = int(df.duplicated().sum()) if len(df.columns) else 0
    missing_cells = int(df.isna().sum().sum()) if len(df.columns) else 0
    return {
        "table": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "missing_cell_rate": float(
            missing_cells / max(1, len(df) * max(1, len(df.columns)))
        ),
    }


def numeric_distribution_summary(
    df: pd.DataFrame,
    table_name: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            raise TypeError(
                f"{table_name}.{column} is not numeric and cannot enter numeric profiling"
            )
        values = pd.to_numeric(df[column], errors="raise")
        observed = values.dropna()
        if observed.empty:
            rows.append({
                "table": table_name,
                "feature": column,
                "rows": len(values),
                "observed": 0,
                "missing_rate": 1.0,
                "unique": 0,
            })
            continue
        q01, q05, q25, q50, q75, q95, q99 = observed.quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = int(((observed < lower) | (observed > upper)).sum()) if np.isfinite(iqr) else 0
        rows.append({
            "table": table_name,
            "feature": column,
            "rows": int(len(values)),
            "observed": int(observed.size),
            "missing_rate": float(values.isna().mean()),
            "zero_rate_observed": float((observed == 0).mean()),
            "unique": int(observed.nunique(dropna=True)),
            "mean": float(observed.mean()),
            "std": float(observed.std(ddof=1)) if observed.size > 1 else 0.0,
            "min": float(observed.min()),
            "p01": float(q01),
            "p05": float(q05),
            "p25": float(q25),
            "median": float(q50),
            "p75": float(q75),
            "p95": float(q95),
            "p99": float(q99),
            "max": float(observed.max()),
            "skew": float(observed.skew()) if observed.size > 2 else np.nan,
            "iqr_outlier_count": outliers,
            "iqr_outlier_rate": float(outliers / observed.size),
        })
    return pd.DataFrame(rows)


def categorical_distribution_summary(
    df: pd.DataFrame,
    table_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        series = df[column]
        observed = series.dropna()
        counts = observed.astype(str).value_counts(dropna=False).head(5)
        rows.append({
            "table": table_name,
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "observed": int(len(observed)),
            "missing_rate": float(series.isna().mean()),
            "unique_observed": int(observed.nunique(dropna=True)),
            "top_values_json": json.dumps(counts.to_dict(), ensure_ascii=False),
        })
    return pd.DataFrame(rows)


def generate_data_quality_reports(
    config: MD1Config,
    result: dict[str, Any],
    *,
    show_plots: bool = True,
) -> dict[str, Path]:
    plots_root = config.audit_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    quality_tables = {
        "competitions": result["silver"]["competitions"],
        "matches": result["silver"]["matches"],
        "teams": result["silver"]["teams"],
        "players": result["silver"]["players"],
        "lineups": result["silver"]["lineups"],
        "event_manifest": result["silver"]["event_manifest"],
        "match_event_facts": result["silver"]["match_event_facts"],
        "gold_prematch": result["gold_prematch"],
        "gold_snapshots": result["gold_snapshots"],
    }
    data_quality_summary = pd.DataFrame(
        [table_quality_summary(name, frame) for name, frame in quality_tables.items()]
    )
    save_table(data_quality_summary, config.audit_root / "data_quality_summary.csv")

    feature_dictionary = result["feature_dictionary"]
    prematch_features = feature_dictionary.query(
        "table == 'gold_prematch' and role == 'feature' and model_eligible == True"
    )["column"].tolist()
    snapshot_features = feature_dictionary.query(
        "table == 'gold_snapshots' and role == 'feature' and model_eligible == True"
    )["column"].tolist()

    prematch_distribution = numeric_distribution_summary(
        result["gold_prematch"], "gold_prematch", prematch_features
    )
    snapshot_distribution = numeric_distribution_summary(
        result["gold_snapshots"], "gold_snapshots", snapshot_features
    )
    feature_distribution_summary = pd.concat(
        [prematch_distribution, snapshot_distribution], ignore_index=True
    )
    save_table(
        feature_distribution_summary,
        config.audit_root / "feature_distribution_summary.csv",
    )

    categorical_frames: list[pd.DataFrame] = []
    for table_name, frame in [
        ("gold_prematch", result["gold_prematch"]),
        ("gold_snapshots", result["gold_snapshots"]),
    ]:
        nonnumeric_columns = [
            c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c].dtype)
        ]
        categorical_frames.append(
            categorical_distribution_summary(frame, table_name, nonnumeric_columns)
        )
    categorical_distribution = pd.concat(categorical_frames, ignore_index=True)
    save_table(
        categorical_distribution,
        config.audit_root / "categorical_distribution_summary.csv",
    )

    event_counter: Counter[str] = Counter()
    for path in sorted(config.event_partitions_root.glob("match_id=*/events.parquet")):
        event_types = pd.read_parquet(path, columns=["type_name"])["type_name"].fillna("<missing>")
        event_counter.update(event_types.astype(str).tolist())
    event_type_distribution = pd.DataFrame(
        event_counter.most_common(), columns=["event_type", "count"]
    )
    if not event_type_distribution.empty:
        event_type_distribution["proportion"] = (
            event_type_distribution["count"] / event_type_distribution["count"].sum()
        )
    else:
        event_type_distribution["proportion"] = pd.Series(dtype=float)
    save_table(event_type_distribution, config.audit_root / "event_type_distribution.csv")

    top_events = event_type_distribution.head(20).sort_values("count")
    plt.figure(figsize=(9, 7))
    if not top_events.empty:
        plt.barh(top_events["event_type"], top_events["count"])
    plt.xlabel("Event rows")
    plt.ylabel("StatsBomb event type")
    plt.title("Top 20 event types")
    _save_figure(plots_root / "event_type_distribution_top20.png", show_plots=show_plots)

    top_missing_pre = (
        prematch_distribution.sort_values("missing_rate", ascending=False)
        .head(20)
        .sort_values("missing_rate")
    )
    plt.figure(figsize=(9, 7))
    if not top_missing_pre.empty:
        plt.barh(top_missing_pre["feature"], top_missing_pre["missing_rate"])
    plt.xlabel("Missing proportion")
    plt.ylabel("Pre-match feature")
    plt.title("Highest pre-match feature missingness")
    _save_figure(plots_root / "prematch_missingness_top20.png", show_plots=show_plots)

    snapshot_counts = (
        result["gold_snapshots"]
        .groupby(["snapshot_rank", "snapshot_id", "snapshot_minute"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("snapshot_rank")
    )
    save_table(snapshot_counts, config.audit_root / "snapshot_rows_by_boundary.csv")
    plt.figure(figsize=(10, 5))
    plt.bar(snapshot_counts["snapshot_id"], snapshot_counts["rows"])
    plt.xticks(rotation=60, ha="right")
    plt.xlabel("Snapshot boundary")
    plt.ylabel("Rows")
    plt.title("Snapshot-table coverage by match boundary")
    _save_figure(plots_root / "snapshot_rows_by_boundary.png", show_plots=show_plots)

    representative_features = [
        ("gold_prematch", result["gold_prematch"], "home_points_l5"),
        ("gold_prematch", result["gold_prematch"], "home_xg_l5"),
        ("gold_snapshots", result["gold_snapshots"], "current_goal_difference"),
        ("gold_snapshots", result["gold_snapshots"], "home_live_shots"),
    ]
    for table_name, frame, column in representative_features:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30)
        plt.xlabel(column)
        plt.ylabel("Rows")
        plt.title(f"{table_name}: distribution of {column}")
        _save_figure(
            plots_root / f"{table_name}_{column}_distribution.png",
            show_plots=show_plots,
        )

    return {
        "data_quality_summary": config.audit_root / "data_quality_summary.csv",
        "feature_distribution_summary": config.audit_root / "feature_distribution_summary.csv",
        "categorical_distribution_summary": config.audit_root / "categorical_distribution_summary.csv",
        "event_type_distribution": config.audit_root / "event_type_distribution.csv",
        "snapshot_rows_by_boundary": config.audit_root / "snapshot_rows_by_boundary.csv",
        "plots_root": plots_root,
    }


def generate_split_and_label_reports(
    config: MD1Config,
    result: dict[str, Any],
    *,
    show_plots: bool = True,
) -> dict[str, pd.DataFrame]:
    plots_root = config.audit_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    split_manifest = result["split_manifest"].copy()
    split_manifest["kickoff"] = pd.to_datetime(split_manifest["kickoff"])
    split_summary = (
        split_manifest.groupby("split", sort=False)
        .agg(matches=("match_id", "size"), start=("kickoff", "min"), end=("kickoff", "max"))
        .reset_index()
    )
    split_order = pd.Categorical(
        split_summary["split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )
    split_summary = (
        split_summary.assign(_order=split_order)
        .sort_values("_order")
        .drop(columns="_order")
    )
    save_table(split_summary, config.audit_root / "split_summary.csv")

    if {"train", "validation", "test"}.issubset(set(split_summary["split"])):
        train_end = split_summary.loc[split_summary["split"] == "train", "end"].iloc[0]
        validation_start = split_summary.loc[split_summary["split"] == "validation", "start"].iloc[0]
        validation_end = split_summary.loc[split_summary["split"] == "validation", "end"].iloc[0]
        test_start = split_summary.loc[split_summary["split"] == "test", "start"].iloc[0]
        assert train_end < validation_start <= validation_end < test_start

    plt.figure(figsize=(8, 5))
    plt.bar(split_summary["split"], split_summary["matches"])
    plt.xlabel("Chronological split")
    plt.ylabel("Matches")
    plt.title("Match counts by chronological split")
    _save_figure(plots_root / "split_match_counts.png", show_plots=show_plots)

    snapshot_split_counts = (
        result["gold_snapshots"]
        .groupby("split")
        .agg(rows=("match_id", "size"), matches=("match_id", "nunique"))
        .reset_index()
    )
    save_table(snapshot_split_counts, config.audit_root / "snapshot_split_counts.csv")

    prematch = result["gold_prematch"]
    class_distribution = (
        prematch.groupby(["split", "label_outcome"], dropna=False)
        .size()
        .reset_index(name="matches")
    )
    class_distribution["proportion_within_split"] = (
        class_distribution.groupby("split")["matches"].transform(lambda x: x / x.sum())
    )
    save_table(class_distribution, config.audit_root / "class_distribution.csv")

    plt.figure(figsize=(9, 5))
    plot_data = (
        class_distribution.pivot(
            index="split",
            columns="label_outcome",
            values="proportion_within_split",
        )
        .fillna(0)
    )
    plot_data.plot(kind="bar", ax=plt.gca())
    plt.xlabel("Chronological split")
    plt.ylabel("Outcome proportion")
    plt.title("Home/draw/away outcome distribution by split")
    plt.xticks(rotation=0)
    _save_figure(
        plots_root / "outcome_distribution_by_split.png",
        show_plots=show_plots,
    )

    raw_margin_distribution = (
        prematch.groupby(["split", "label_margin_raw"], dropna=False)
        .size()
        .reset_index(name="matches")
    )
    clipped_margin_distribution = (
        prematch.groupby(["split", "label_margin"], dropna=False)
        .size()
        .reset_index(name="matches")
    )
    save_table(raw_margin_distribution, config.audit_root / "raw_margin_distribution.csv")
    save_table(
        clipped_margin_distribution,
        config.audit_root / "clipped_margin_distribution.csv",
    )

    return {
        "split_summary": split_summary,
        "snapshot_split_counts": snapshot_split_counts,
        "class_distribution": class_distribution,
        "raw_margin_distribution": raw_margin_distribution,
        "clipped_margin_distribution": clipped_margin_distribution,
    }


def build_md1_data_readiness(
    config: MD1Config,
    result: dict[str, Any],
    profile_paths: dict[str, Path],
) -> pd.DataFrame:
    summary = read_json(result["summary_path"])
    coverage_rate = (
        float(result["odds_coverage"]["coverage_rate"].min())
        if not result["odds_coverage"].empty else 0.0
    )
    score_failures_path = config.audit_root / "score_reconciliation_failures.csv"
    score_failures = (
        pd.read_csv(score_failures_path)
        if score_failures_path.exists() and score_failures_path.stat().st_size > 0
        else pd.DataFrame()
    )
    snapshots_per_match = result["gold_snapshots"].groupby("match_id").size()
    split_summary_path = config.audit_root / "split_summary.csv"
    profiles_exist = all(
        Path(path).exists() for key, path in profile_paths.items() if key != "plots_root"
    )
    boundary_ids = set(result["gold_snapshots"]["snapshot_id"].unique())
    feature_roles = result["feature_dictionary"].set_index(["table", "column"])["role"]
    plots_exist = profile_paths["plots_root"].exists() and any(
        profile_paths["plots_root"].glob("*.png")
    )

    checks = [
        (
            "real full-season mode",
            config.data_mode != "real" or config.max_matches is None,
            f"mode={config.data_mode}, max_matches={config.max_matches}; required for real defence run",
        ),
        (
            "complete Premier League season",
            config.data_mode != "real" or int(summary.get("matches", 0)) == 380,
            f"matches={summary.get('matches')}; 380 required for real defence run",
        ),
        (
            "all key and foreign-key checks pass",
            bool(result["key_audit"]["passed"].all()),
            f"failed={(~result['key_audit']['passed']).sum()}",
        ),
        (
            "event scores reconcile with match metadata",
            score_failures.empty,
            f"failed_matches={len(score_failures)}",
        ),
        (
            "all leakage and integrity tests pass",
            bool(result["leakage_tests"]["passed"].all()),
            f"passed={result['leakage_tests']['passed'].sum()}/{len(result['leakage_tests'])}",
        ),
        (
            "one pre-match row per match",
            len(result["gold_prematch"]) == len(result["silver"]["matches"])
            and result["gold_prematch"]["match_id"].is_unique,
            f"prematch={len(result['gold_prematch'])}, matches={len(result['silver']['matches'])}",
        ),
        (
            "consistent snapshot coverage per match",
            snapshots_per_match.nunique() == 1,
            f"snapshots_per_match={sorted(snapshots_per_match.unique().tolist())}",
        ),
        (
            "strict M90 and full-time snapshots both exist",
            {"M90", "FT"}.issubset(boundary_ids),
            f"boundaries={sorted(boundary_ids)}",
        ),
        (
            "snapshot rows inherit match-level split",
            result["gold_snapshots"].groupby("match_id")["split"].nunique().max() == 1,
            "each match has exactly one split",
        ),
        (
            "provider and odds-audit fields are excluded from model features",
            all(
                feature_roles.get(("gold_prematch", c)) != "feature"
                for c in [
                    "estimated_finish",
                    "match_status",
                    "match_status_360",
                    "last_updated",
                    "last_updated_360",
                    "join_status",
                    "join_method",
                    "confidence",
                ]
                if ("gold_prematch", c) in feature_roles.index
            ),
            "administrative/provider fields are metadata, audit, or baseline-only",
        ),
        (
            "raw and clipped margins are both preserved",
            {
                "goal_margin_raw",
                "goal_margin_clipped",
                "label_margin_raw",
                "label_margin",
            }.issubset(result["gold_prematch"].columns),
            "raw observed margin and clipped Task-R target are separate",
        ),
        (
            "strict chronological split summary exists",
            split_summary_path.exists(),
            str(split_summary_path),
        ),
        (
            "odds tagging coverage is sufficient",
            coverage_rate >= 0.95,
            f"coverage={coverage_rate:.3f}",
        ),
        (
            "feature distributions and data-quality reports saved",
            profiles_exist,
            "quality, feature, event-type, and snapshot profiles",
        ),
        (
            "defence plots saved",
            plots_exist,
            str(profile_paths["plots_root"]),
        ),
    ]
    readiness = pd.DataFrame(
        [
            {"check": name, "passed": bool(passed), "details": details}
            for name, passed, details in checks
        ]
    )
    save_table(readiness, config.audit_root / "md1_data_readiness.csv")
    if readiness["passed"].all():
        status = (
            "READY FOR MID DEFENCE 1 DATA REVIEW"
            if config.data_mode == "real"
            else "SYNTHETIC DATA PIPELINE VERIFICATION PASSED"
        )
    else:
        status = "DATA PIPELINE NOT READY"
    report_lines = [f"# {status}", "", readiness.to_markdown(index=False), ""]
    report_lines += [
        "This gate covers data ingestion, cleaning, integration, feature construction, distributions,",
        "chronological splitting, odds tagging, and leakage tests. It does not approve a P1 or P2 paper.",
    ]
    (config.project_root / "MID_DEFENCE_1_DATA_READINESS.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return readiness


def create_md1_data_evidence_zip(config: MD1Config, destination: Path) -> Path:
    """Create a compact defence archive without duplicating the raw event cache."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    include_paths: list[Path] = []
    include_paths.extend(path for path in config.audit_root.rglob("*") if path.is_file())
    include_paths.extend(path for path in config.gold_root.rglob("*") if path.is_file())
    for extra in [
        config.project_root / "MID_DEFENCE_1_DATA_READINESS.md",
        config.project_root / "SUBMISSION_BLOCKED.md",
    ]:
        if extra.exists():
            include_paths.append(extra)

    compact_silver_names = [
        "competitions.parquet",
        "matches.parquet",
        "teams.parquet",
        "players.parquet",
        "lineups.parquet",
        "match_event_facts.parquet",
        "event_manifest.parquet",
        "coverage_360.parquet",
        "team_match_facts.parquet",
        "team_match_facts_with_rolling.parquet",
        "football_data_matches.parquet",
        "team_aliases.parquet",
        "match_join_map.parquet",
    ]
    for name in compact_silver_names:
        path = config.silver_root / name
        if path.exists():
            include_paths.append(path)

    include_paths = sorted(set(include_paths), key=lambda path: str(path))
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        for path in include_paths:
            archive.write(path, arcname=str(path.relative_to(config.project_root)))
    return destination


def export_results_index(
    *,
    config: MD1Config,
    result: dict[str, Any],
    readiness: pd.DataFrame,
    workspace_root: Path,
    exports_root: Path,
    run_name: str,
) -> dict[str, Path]:
    summary = read_json(result["summary_path"])
    archive_path = create_md1_data_evidence_zip(
        config,
        exports_root / f"Mid_Defence_1_Data_Evidence_{run_name}.zip",
    )

    important_files = {
        "run summary": config.audit_root / "run_summary.json",
        "resolved configuration": config.audit_root / "resolved_config.csv",
        "data quality": config.audit_root / "data_quality_summary.csv",
        "feature distributions": config.audit_root / "feature_distribution_summary.csv",
        "categorical distributions": config.audit_root / "categorical_distribution_summary.csv",
        "event-type distribution": config.audit_root / "event_type_distribution.csv",
        "class distribution": config.audit_root / "class_distribution.csv",
        "split summary": config.audit_root / "split_summary.csv",
        "key audit": config.audit_root / "key_audit.csv",
        "event partition FK/order audit": config.audit_root / "event_partition_contract_audit.csv",
        "starting lineup audit": config.audit_root / "starting_lineup_audit.csv",
        "leakage tests": config.audit_root / "leakage_test_results.csv",
        "score reconciliation failures": config.audit_root / "score_reconciliation_failures.csv",
        "odds coverage": config.audit_root / "odds_coverage_by_season.csv",
        "excluded odds matches": config.audit_root / "excluded_odds_matches.csv",
        "snapshot walkthrough": config.audit_root / "snapshot_audit_sample.csv",
        "feature dictionary": config.gold_root / "feature_dictionary.csv",
        "pre-match table": config.gold_root / "gold_prematch.parquet",
        "in-play snapshot table": config.gold_root / "gold_snapshots.parquet",
        "data readiness": config.project_root / "MID_DEFENCE_1_DATA_READINESS.md",
    }

    index_lines = [
        "# Mid Defence 1 — Data Work Results",
        "",
        f"- Run mode: **{config.data_mode}**",
        f"- Competition: **{summary['competition']} {summary['season']}**",
        f"- Matches: **{summary['matches']}**",
        f"- Event rows: **{summary['event_rows']}**",
        f"- Lineup rows: **{summary['lineup_rows']}**",
        f"- Matches with 360 data: **{summary['matches_with_360']}**",
        f"- Pre-match rows: **{summary['gold_prematch_rows']}**",
        f"- Snapshot rows: **{summary['gold_snapshot_rows']}**",
        f"- Leakage tests: **{summary['leakage_tests_passed']}/{summary['leakage_tests_total']} passed**",
        f"- Data-readiness gate: **{int(readiness['passed'].sum())}/{len(readiness)} passed**",
        "",
        f"## Full result folder\n`{config.project_root}`",
        "",
        f"## Evidence ZIP\n`{archive_path}`",
        "",
        "## Important files",
    ]
    for label, path in important_files.items():
        index_lines.append(f"- {label}: `{path}`")

    workspace_root.mkdir(parents=True, exist_ok=True)
    results_index_path = workspace_root / "DATA_RESULTS_INDEX.md"
    results_index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    latest_path = workspace_root / "LATEST_DATA_RESULT_PATH.txt"
    latest_path.write_text(str(config.project_root) + "\n", encoding="utf-8")
    latest_summary = workspace_root / "LATEST_DATA_RUN_SUMMARY.json"
    shutil.copy2(result["summary_path"], latest_summary)

    return {
        "archive_path": archive_path,
        "results_index_path": results_index_path,
        "latest_result_path": latest_path,
        "latest_summary_path": latest_summary,
    }
