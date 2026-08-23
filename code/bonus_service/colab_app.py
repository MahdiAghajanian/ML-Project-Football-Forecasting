from __future__ import annotations

from compat import install_pickle_compat, patch_loaded_app

# Install the class expected by the training notebook's joblib artifacts before
# importing app.py, because app.py loads those artifacts at module import time.
install_pickle_compat()

import app as core  # noqa: E402

patch_loaded_app(core)

# Expose the FastAPI application and useful service state for Colab/uvicorn.
app = core.app
MATCH_GROUPS = core.MATCH_GROUPS
PRE_FEATURES = core.PRE_FEATURES
LIVE_FEATURES = core.LIVE_FEATURES
P1_FEATURES = core.P1_FEATURES
SNAPSHOTS = core.SNAPSHOTS
