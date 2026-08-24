"""Project-local NGBoost runtime facade backed by ``code/modeling/p2_source``.

The P2 implementation remains visible and unchanged under ``p2_source/``.  This
small package makes that source importable under the same ``ngboost`` package
name used by the historical modeling code, without installing the external
PyPI package.  Submodules are resolved directly from ``p2_source``.
"""
from __future__ import annotations

from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "p2_source"
if not _SOURCE.is_dir():
    raise ImportError(f"Project P2 source directory is missing: {_SOURCE}")

# Make ``ngboost.api``, ``ngboost.distns``, etc. resolve directly to p2_source.
__path__ = [str(_SOURCE)]

from .api import NGBClassifier as _SourceNGBClassifier, NGBRegressor, NGBSurvival
from .distns import Bernoulli
from .helpers import load_ngboost_model
from .learners import default_tree_learner
from .ngboost import NGBoost
from .scores import LogScore


def _project_is_fitted(self):
    """Modern sklearn fitted-state hook; does not alter model mathematics."""
    return self.init_params is not None and bool(self.base_models)


NGBoost.__sklearn_is_fitted__ = _project_is_fitted


class NGBClassifier(_SourceNGBClassifier):
    """Thin sklearn-clone-compatible facade over the project classifier.

    The project source's inherited ``get_params`` exposes
    ``validation_fraction`` and ``early_stopping_rounds``.  Adding those two
    constructor parameters here makes sklearn cloning/Pipelines consistent
    without changing the underlying boosting implementation.
    """

    def __init__(
        self,
        Dist=Bernoulli,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
    ):
        super().__init__(
            Dist=Dist,
            Score=Score,
            Base=Base,
            natural_gradient=natural_gradient,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=minibatch_frac,
            col_sample=col_sample,
            verbose=verbose,
            verbose_eval=verbose_eval,
            tol=tol,
            random_state=random_state,
        )
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds


__version__ = "project-p2-reimplementation"
__all__ = [
    "NGBClassifier",
    "NGBRegressor",
    "NGBSurvival",
    "NGBoost",
    "load_ngboost_model",
]
