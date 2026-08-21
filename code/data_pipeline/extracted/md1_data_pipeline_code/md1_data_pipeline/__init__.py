"""Mid Defence 1 data pipeline package with optional Berrar P1 representation."""

from .core import MD1Config, run_full_pipeline
from .p1_berrar import (
    PAPER_HOMEAWAY_FEATURES,
    PAPER_TOTAL_FEATURES,
    build_p1_features_for_n,
    build_p1_super_league,
    merge_existing_prematch_with_p1,
    run_p1_berrar_pipeline,
    select_p1_recency_knn,
    select_p1_recency_pearson,
)
from .p1_hybrid import (
    configured_history_seasons,
    prepare_hybrid_p1_data,
    run_hybrid_p1_pipeline,
)
from .workflow import (
    build_default_config,
    build_p1_default_config,
    execute_complete_workflow,
    execute_p1_workflow,
    record_environment,
)
from .reporting import (
    build_md1_data_readiness,
    create_md1_data_evidence_zip,
    export_results_index,
    generate_data_quality_reports,
    generate_split_and_label_reports,
)

__all__ = [
    "MD1Config",
    "run_full_pipeline",
    "build_default_config",
    "execute_complete_workflow",
    "record_environment",
    "build_md1_data_readiness",
    "create_md1_data_evidence_zip",
    "export_results_index",
    "generate_data_quality_reports",
    "generate_split_and_label_reports",
    "PAPER_TOTAL_FEATURES",
    "PAPER_HOMEAWAY_FEATURES",
    "build_p1_super_league",
    "merge_existing_prematch_with_p1",
    "build_p1_features_for_n",
    "select_p1_recency_pearson",
    "select_p1_recency_knn",
    "run_p1_berrar_pipeline",
    "run_hybrid_p1_pipeline",
    "prepare_hybrid_p1_data",
    "configured_history_seasons",
    "build_p1_default_config",
    "execute_p1_workflow",
]
