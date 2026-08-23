from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN = ROOT / "code/modeling/outputs/ngboost_p1_corrected_full"
MODEL_DIR = FINAL_RUN / "models"
RUN_CONFIG = FINAL_RUN / "run_config.json"


def _find_one(filename: str) -> Path:
    matches = list((ROOT / "code/data_pipeline/results").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename} under code/data_pipeline/results")
    matches.sort(key=lambda p: ("Premier_League_2015_16_DATA_ONLY_FULL_REAL" not in str(p), len(str(p))))
    return matches[0]


SNAPSHOT_PATH = _find_one("gold_snapshots.parquet")
with RUN_CONFIG.open("r", encoding="utf-8") as fh:
    CONFIG = json.load(fh)
PRE_FEATURES: list[str] = CONFIG["pre_features"]
LIVE_FEATURES: list[str] = CONFIG["live_features"]

# Pre-kickoff Models 1/2 and live Model-3 variants are loaded once at startup.
PRE_OUTCOME_MODEL_PATH = MODEL_DIR / "task_c_lightgbm.joblib"
PRE_MARGIN_MODEL_PATH = MODEL_DIR / "task_r_kernel_ridge_nystroem.joblib"
LIVE_OUTCOME_MODEL_PATH = MODEL_DIR / "task_l_classifier_lightgbm.joblib"
LIVE_MARGIN_MODEL_PATH = MODEL_DIR / "task_l_regressor_gbm.joblib"
PRE_OUTCOME_MODEL = joblib.load(PRE_OUTCOME_MODEL_PATH)
PRE_MARGIN_MODEL = joblib.load(PRE_MARGIN_MODEL_PATH)
LIVE_OUTCOME_MODEL = joblib.load(LIVE_OUTCOME_MODEL_PATH)
LIVE_MARGIN_MODEL = joblib.load(LIVE_MARGIN_MODEL_PATH)

SNAPSHOTS = pd.read_parquet(SNAPSHOT_PATH)
missing_features = [c for c in set(PRE_FEATURES + LIVE_FEATURES) if c not in SNAPSHOTS.columns]
if missing_features:
    raise RuntimeError(f"Gold snapshot table is missing model features: {missing_features[:10]}")
SNAPSHOTS = SNAPSHOTS.sort_values(["match_id", "snapshot_rank"]).reset_index(drop=True)
MATCH_GROUPS = {int(mid): g.reset_index(drop=True) for mid, g in SNAPSHOTS.groupby("match_id", sort=False)}


def _class_mapping(model: Any, probabilities: np.ndarray) -> dict[str, float]:
    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classes = list(getattr(model.named_steps.get("model"), "classes_", []))
    if len(classes) != len(probabilities):
        classes = ["A", "D", "H"]
    aliases = {
        "A": "away", "Away": "away", "away": "away", "0": "away",
        "D": "draw", "Draw": "draw", "draw": "draw", "1": "draw",
        "H": "home", "Home": "home", "home": "home", "2": "home",
    }
    normalized = {"away": 0.0, "draw": 0.0, "home": 0.0}
    for cls, value in zip(classes, probabilities):
        key = str(cls)
        if key in aliases:
            normalized[aliases[key]] = float(value)
    if sum(normalized.values()) == 0:
        normalized = dict(zip(["away", "draw", "home"], map(float, probabilities)))
    return normalized


def _select_snapshot(match_id: int, minute: float | None, snapshot_id: str | None) -> pd.Series:
    group = MATCH_GROUPS.get(int(match_id))
    if group is None:
        raise HTTPException(status_code=404, detail=f"Unknown match_id {match_id}")
    if snapshot_id is not None:
        exact = group[group["snapshot_id"].astype(str) == snapshot_id]
        if exact.empty:
            raise HTTPException(status_code=404, detail=f"snapshot_id {snapshot_id!r} not available for match")
        return exact.iloc[-1]
    if minute is None:
        minute = 0.0
    eligible = group[group["snapshot_rank"].astype(float) <= float(minute)]
    return eligible.iloc[-1] if not eligible.empty else group.iloc[0]


def _frame(row: pd.Series, features: list[str]) -> pd.DataFrame:
    return row[features].to_frame().T


def _tree_shap(row: pd.Series, top_k: int = 8) -> list[dict[str, float | str]]:
    try:
        x = _frame(row, LIVE_FEATURES)
        if hasattr(LIVE_OUTCOME_MODEL, "named_steps"):
            prep = LIVE_OUTCOME_MODEL.named_steps.get("prep")
            tree = LIVE_OUTCOME_MODEL.named_steps.get("model")
            x_trans = prep.transform(x) if prep is not None else x.to_numpy()
            try:
                names = list(prep.get_feature_names_out(LIVE_FEATURES)) if prep is not None else LIVE_FEATURES
            except Exception:
                names = LIVE_FEATURES[: np.asarray(x_trans).shape[1]]
        else:
            tree = LIVE_OUTCOME_MODEL
            x_trans = x
            names = LIVE_FEATURES
        values = shap.TreeExplainer(tree).shap_values(x_trans)
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[1] == 1:
                arr = arr[:, 0, :]
            if arr.ndim == 2 and arr.shape[0] != len(names):
                arr = arr[np.argmax(np.abs(arr).sum(axis=1))]
        if arr.ndim == 2:
            arr = arr[0] if arr.shape[0] == 1 else arr[:, -1]
        arr = np.ravel(arr)
        n = min(len(arr), len(names))
        ranked = sorted(((names[i], float(arr[i])) for i in range(n)), key=lambda z: abs(z[1]), reverse=True)[:top_k]
        return [{"feature": str(name), "shap_value": value} for name, value in ranked]
    except Exception as exc:
        return [{"feature": "SHAP unavailable", "shap_value": 0.0, "detail": str(exc)[:180]}]


@lru_cache(maxsize=1024)
def _prematch_cached(match_id: int) -> dict[str, Any]:
    row = MATCH_GROUPS[int(match_id)].iloc[0]
    x = _frame(row, PRE_FEATURES)
    start = time.perf_counter()
    proba = np.asarray(PRE_OUTCOME_MODEL.predict_proba(x))[0]
    margin = float(np.asarray(PRE_MARGIN_MODEL.predict(x)).reshape(-1)[0])
    model_ms = (time.perf_counter() - start) * 1000.0
    return {
        "match_id": int(match_id),
        "home_team": str(row.get("home_team_name", "Home")),
        "away_team": str(row.get("away_team_name", "Away")),
        "outcome_probabilities": _class_mapping(PRE_OUTCOME_MODEL, proba),
        "expected_final_margin": margin,
        "model_inference_ms": model_ms,
        "feature_source": str(SNAPSHOT_PATH.relative_to(ROOT)),
        "outcome_model": PRE_OUTCOME_MODEL_PATH.name,
        "margin_model": PRE_MARGIN_MODEL_PATH.name,
    }


@lru_cache(maxsize=4096)
def _predict_cached(match_id: int, snapshot_id: str) -> dict[str, Any]:
    row = _select_snapshot(match_id, None, snapshot_id)
    x = _frame(row, LIVE_FEATURES)
    start = time.perf_counter()
    proba = np.asarray(LIVE_OUTCOME_MODEL.predict_proba(x))[0]
    margin = float(np.asarray(LIVE_MARGIN_MODEL.predict(x)).reshape(-1)[0])
    model_ms = (time.perf_counter() - start) * 1000.0
    return {
        "match_id": int(match_id),
        "snapshot_id": str(row["snapshot_id"]),
        "snapshot_minute": float(row.get("snapshot_minute", row["snapshot_rank"])),
        "snapshot_rank": float(row["snapshot_rank"]),
        "score": {"home": int(row.get("current_home_score", 0)), "away": int(row.get("current_away_score", 0))},
        "red_cards": {"home": int(row.get("home_live_red_cards", 0)), "away": int(row.get("away_live_red_cards", 0))},
        "outcome_probabilities": _class_mapping(LIVE_OUTCOME_MODEL, proba),
        "expected_final_margin": margin,
        "top_shap": _tree_shap(row),
        "model_inference_ms": model_ms,
        "feature_source": str(SNAPSHOT_PATH.relative_to(ROOT)),
        "outcome_model": LIVE_OUTCOME_MODEL_PATH.name,
        "margin_model": LIVE_MARGIN_MODEL_PATH.name,
    }


class PredictionRequest(BaseModel):
    match_id: int
    minute: float | None = None
    snapshot_id: str | None = None


app = FastAPI(
    title="Football Forecasting Bonus Service",
    version="1.0.0",
    description="Low-latency pre-match and in-play serving over the exact Gold representation used during training.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "matches_loaded": len(MATCH_GROUPS), "snapshots_loaded": len(SNAPSHOTS), "pre_features": len(PRE_FEATURES), "live_features": len(LIVE_FEATURES)}


@app.get("/matches")
def matches(split: str | None = Query(default="test")) -> list[dict[str, Any]]:
    frame = SNAPSHOTS
    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.lower() == split.lower()]
    first = frame.groupby("match_id", as_index=False).first()
    return [{
        "match_id": int(row["match_id"]),
        "home_team": str(row.get("home_team_name", "Home")),
        "away_team": str(row.get("away_team_name", "Away")),
        "split": str(row.get("split", "unknown")),
    } for _, row in first.iterrows()]


@app.get("/prematch/{match_id}")
def prematch(match_id: int) -> dict[str, Any]:
    if int(match_id) not in MATCH_GROUPS:
        raise HTTPException(status_code=404, detail=f"Unknown match_id {match_id}")
    return _prematch_cached(int(match_id))


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    row = _select_snapshot(request.match_id, request.minute, request.snapshot_id)
    return _predict_cached(int(request.match_id), str(row["snapshot_id"]))


@app.get("/predict/{match_id}")
def predict_get(match_id: int, minute: float = 0.0) -> dict[str, Any]:
    row = _select_snapshot(match_id, minute, None)
    return _predict_cached(int(match_id), str(row["snapshot_id"]))


@app.get("/replay/{match_id}")
def replay(match_id: int) -> list[dict[str, Any]]:
    group = MATCH_GROUPS.get(int(match_id))
    if group is None:
        raise HTTPException(status_code=404, detail=f"Unknown match_id {match_id}")
    return [_predict_cached(int(match_id), str(sid)) for sid in group["snapshot_id"].astype(str)]
