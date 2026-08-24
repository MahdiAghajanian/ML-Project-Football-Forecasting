"""Compatibility name for the required P2 project reimplementation.

The canonical implementation lives in ``code/modeling/p2_source`` and is exposed
through the project-local ``ngboost`` facade so existing modeling code can use the
same import style as before without installing the external PyPI package.
"""
from __future__ import annotations

import sys

import ngboost as _project_ngboost
import ngboost.distns as _distns
import ngboost.scores as _scores

NGBClassifier = _project_ngboost.NGBClassifier
NGBRegressor = _project_ngboost.NGBRegressor
NGBSurvival = _project_ngboost.NGBSurvival
NGBoost = _project_ngboost.NGBoost
load_ngboost_model = _project_ngboost.load_ngboost_model

# Preserve imports used by the existing P2 evaluator without loading a second
# copy of distribution/score classes under a different module name.
sys.modules[__name__ + ".distns"] = _distns
sys.modules[__name__ + ".scores"] = _scores

__version__ = _project_ngboost.__version__
__all__ = ["NGBClassifier", "NGBRegressor", "NGBSurvival", "NGBoost", "load_ngboost_model"]
