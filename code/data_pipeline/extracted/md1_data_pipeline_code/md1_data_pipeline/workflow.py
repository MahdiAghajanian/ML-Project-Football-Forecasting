"""High-level execution helpers used by the Colab runner."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .core import MD1Config, run_full_pipeline, save_table
from .p1_berrar import run_p1_berrar_pipeline
from .p1_hybrid import run_hybrid_p1_pipeline
from .reporting import (
    build_md1_data_readiness,
    export_results_index,
    generate_data_quality_reports,
    generate_split_and_label_reports,
)


def build_default_config(
    *,
    results_root: Path,
    run_mode: str = "real",
    max_matches: int | None = None,
    run_name: str | None = None,
) -> tuple[str, MD1Config]:
    run_mode = run_mode.strip()
    if run_mode not in {"real", "synthetic_demo"}:
        raise ValueError("run_mode must be 'real' or 'synthetic_demo'.")

    if run_mode == "real":
        effective_max_matches = max_matches
        default_name = (
            "Premier_League_2015_16_DATA_ONLY_FULL_REAL"
            if max_matches is None
            else f"Premier_League_2015_16_REAL_{max_matches}_MATCHES"
        )
    else:
        effective_max_matches = 16 if max_matches is None else max_matches
        default_name = "Synthetic_Data_Pipeline_Verification"

    resolved_run_name = run_name or default_name
    config = MD1Config(
        project_root=results_root / resolved_run_name,
        data_mode=run_mode,
        competition_id=2,
        season_id=27,
        competition_name="Premier League",
        season_name="2015/2016",
        football_data_code="1516",
        football_data_division="E0",
        max_matches=effective_max_matches,
        date_tolerance_days=1,
        train_fraction=0.65,
        validation_fraction=0.20,
        snapshot_minutes=tuple(range(0, 91, 5)),
        rolling_windows=(3, 5, 10),
        overwrite_downloads=False,
        download_360=True,
        download_workers=6,
        random_seed=42,
    )
    config.ensure_directories()
    return resolved_run_name, config



def build_p1_default_config(
    *,
    results_root: Path,
    run_mode: str = "real",
    run_name: str | None = None,
    full_recency_search: bool = True,
) -> tuple[str, MD1Config]:
    """Resolve the final TA-approved Berrar P1 configuration.

    Real mode uses complete Football-Data EPL results from 2000/01--2014/15 as
    historical context and StatsBomb EPL 2015/16 as the only supervised target
    season.  The existing MD1 baseline remains opt-in and unchanged.
    """
    run_mode = run_mode.strip()
    if run_mode not in {"real", "synthetic_demo"}:
        raise ValueError("run_mode must be 'real' or 'synthetic_demo'.")
    default_name = (
        "EPL_2000_01_2015_16_Berrar_P1_HYBRID_REAL"
        if run_mode == "real"
        else "Synthetic_Berrar_P1_Hybrid_Verification"
    )
    resolved_run_name = run_name or default_name
    if run_mode == "real":
        recency_min, recency_max, min_prior = 9, 100, 6
    else:
        # Production-shaped smoke grid: every candidate is eligible to be
        # meaningfully evaluated under the paper's minimum-six rule.
        recency_min, recency_max, min_prior = 6, 10, 6
    if not full_recency_search and run_mode == "real":
        # Explicit development shortcut only; never report as the paper run.
        recency_min, recency_max = 9, 20
    config = MD1Config(
        project_root=results_root / resolved_run_name,
        data_mode=run_mode,
        competition_id=2,
        season_id=27,
        competition_name="Premier League",
        season_name="2015/2016",
        football_data_code="1516",
        football_data_division="E0",
        max_matches=None,
        date_tolerance_days=1,
        train_fraction=0.65,
        validation_fraction=0.20,
        snapshot_minutes=tuple(range(0, 91, 5)),
        rolling_windows=(3, 5, 10),
        overwrite_downloads=False,
        download_360=True,
        download_workers=6,
        random_seed=42,
        p1_enabled=True,
        p1_history_provider="football_data",
        p1_competition_name="Premier League",
        p1_history_start_season="2000/2001",
        p1_history_end_season="2014/2015",
        p1_history_division="E0",
        p1_require_complete_seasons=True,
        p1_expected_team_count=20,
        p1_expected_matches_per_season=380,
        p1_target_from_statsbomb=True,
        p1_recency_min=recency_min,
        p1_recency_max=recency_max,
        p1_min_prior_matches=min_prior,
        p1_odd_n_policy="floor",
        p1_primary_recency_method="pearson",
        p1_feature_view="both",
        p1_fixed_baseline_n=10 if run_mode == "real" else 6,
        p1_cache_all_n=True,
        p1_run_knn_search=False,
        p1_knn_mode="project_temporal",
    )
    config.ensure_directories()
    return resolved_run_name, config

def execute_p1_workflow(*, config: MD1Config) -> dict[str, Any]:
    """Run the final TA-approved hybrid Berrar P1 branch."""
    if not config.p1_enabled:
        raise ValueError("P1 workflow requested but config.p1_enabled is False.")
    return run_hybrid_p1_pipeline(config)

def record_environment(config: MD1Config) -> pd.DataFrame:
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "requests",
        "urllib3",
        "Unidecode",
        "matplotlib",
        "tabulate",
    ]
    rows = [
        {"component": "python", "version": platform.python_version()},
        {"component": "platform", "version": platform.platform()},
    ]
    for package_name in packages:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        rows.append({"component": package_name, "version": version})
    frame = pd.DataFrame(rows)
    save_table(frame, config.audit_root / "environment_versions.csv")
    return frame


def execute_complete_workflow(
    *,
    config: MD1Config,
    workspace_root: Path,
    exports_root: Path,
    run_name: str,
    show_plots: bool = True,
    fail_on_readiness: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    result = run_full_pipeline(config)
    elapsed_minutes = (time.perf_counter() - started_at) / 60.0

    environment = record_environment(config)
    profile_paths = generate_data_quality_reports(
        config,
        result,
        show_plots=show_plots,
    )
    report_tables = generate_split_and_label_reports(
        config,
        result,
        show_plots=show_plots,
    )
    readiness = build_md1_data_readiness(config, result, profile_paths)
    if fail_on_readiness and not readiness["passed"].all():
        failed = readiness.loc[~readiness["passed"]]
        raise AssertionError(
            "Mid Defence 1 data-readiness gate failed:\n"
            + failed.to_string(index=False)
        )

    exports = export_results_index(
        config=config,
        result=result,
        readiness=readiness,
        workspace_root=workspace_root,
        exports_root=exports_root,
        run_name=run_name,
    )

    return {
        "result": result,
        "environment": environment,
        "profile_paths": profile_paths,
        "report_tables": report_tables,
        "readiness": readiness,
        "exports": exports,
        "elapsed_minutes": elapsed_minutes,
    }
