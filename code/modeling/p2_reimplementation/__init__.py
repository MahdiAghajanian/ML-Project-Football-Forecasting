"""Compatibility name for the required P2 project reimplementation.

The canonical implementation lives in ``code/modeling/p2_source`` and is exposed
through the project-local ``ngboost`` facade so existing modeling code can use the
same import style as before without installing the external PyPI package.
"""
from __future__ import annotations

import importlib
import sys

import ngboost as _project_ngboost

NGBClassifier = _project_ngboost.NGBClassifier
NGBRegressor = _project_ngboost.NGBRegressor
NGBSurvival = _project_ngboost.NGBSurvival
NGBoost = _project_ngboost.NGBoost
load_ngboost_model = _project_ngboost.load_ngboost_model

# Preserve imports used by older P2 evaluator/test code without loading second
# copies of distribution, score, or manifold classes under another namespace.
for _name in ("distns", "scores", "manifold"):
    sys.modules[__name__ + "." + _name] = importlib.import_module("ngboost." + _name)

__version__ = _project_ngboost.__version__
__all__ = ["NGBClassifier", "NGBRegressor", "NGBSurvival", "NGBoost", "load_ngboost_model"]
