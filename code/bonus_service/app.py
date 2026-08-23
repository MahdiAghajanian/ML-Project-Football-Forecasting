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

from compat import CalibratedClassifierBundle, install_pickle_compat

ROOT = Path(__file__).resolve().parents[2]
FINAL_RUN = ROOT / "code/modeling/outputs/ngboost_p1_corrected_full"
MODEL_DIR = FINAL_RUN / "models"
RUN_CONFIG = FINAL_RUN / "run_config.json"
P1_DATA = ROOT / "code/data_pipeline/results/EXP01_P1_REPRESENTATION_TASK_C/data"
P1_AUGMENTED = P1_DATA / "prematch_md1_plus_p1_homeaway.parquet"


def _find_one(filename: str) -> Path:
    matches = list((ROOT / "code/data_pipeline/results").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not locate {filename} under code/data_pipeline/results")
    matches.sort(key=lambda p: ("Premier_League_2015_16_DATA_ONLY_FULL_REAL" not in str(p), len(str(p))))
    return matches[0]


SNAPSHOT_PATH = _find_one("gold_snapshots.parquet")
DATA_ROOT = SNAPSHOT_PATH.parent.parent
STATSBOMB_MATCHES_ROOT = DATA_ROOT / "bronze/statsbomb/matches"

with RUN_CONFIG.open("r", encoding="utf-8") as fh:
    CONFIG = json.load(fh)
PRE_FEATURES: list[str] = CONFIG["pre_features"]
LIVE_FEATURES: list[str] = CONFIG["live_features"]
P1_FEATURES: list[str] = CONFIG["p1_features"]

# The training notebook serialized its custom Platt calibrator from __main__.
# Register a byte-compatible compatibility class before loading any artifact.
install_pickle_compat()

PRE_OUTCOME_MODEL_PATH = MODEL_DIR / "task_c_lightgbm.joblib"
PRE_MARGIN_MODEL_PATH = MODEL_DIR / "task_r_kernel_ridge_nystroem.joblib"
LIVE_OUTCOME_MODEL_PATH = MODEL_DIR / "task_l_classifier_lightgbm.joblib"
LIVE_MARGIN_MODEL_PATH = MODEL_DIR / "task_l_regressor_gbm.joblib"


def _load_classifier(path: Path):
    payload = joblib.load(path)
    return CalibratedClassifierBundle(payload) if isinstance(payload, dict) else payload


PRE_OUTCOME_MODEL = _load_classifier(PRE_OUTCOME_MODEL_PATH)
PRE_MARGIN_MODEL = joblib.load(PRE_MARGIN_MODEL_PATH)
LIVE_OUTCOME_MODEL = _load_classifier(LIVE_OUTCOME_MODEL_PATH)
LIVE_MARGIN_MODEL = joblib.load(LIVE_MARGIN_MODEL_PATH)


def _load_match_metadata() -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for path in sorted(STATSBOMB_MATCHES_ROOT.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for match in payload:
            try:
                match_id = int(match["match_id"])
            except Exception:
                continue
            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            metadata[match_id] = {
                "home_team": str(home.get("home_team_name", "Home")),
                "away_team": str(away.get("away_team_name", "Away")),
                "match_date": str(match.get("match_date", "")),
                "kick_off": str(match.get("kick_off", "")),
                "final_home_score": int(match.get("home_score", 0)),
                "final_away_score": int(match.get("away_score", 0)),
            }
    return metadata


MATCH_META = _load_match_metadata()

# Recreate the exact frozen MD1 + P1 representation used by the modeling notebook:
# the raw Gold snapshot table carries the 189 MD1 live features, while the selected
# 18 P1 pre-match values are joined many-to-one to every snapshot of a match.
SNAPSHOTS = pd.read_parquet(SNAPSHOT_PATH)
if not P1_AUGMENTED.exists():
    raise FileNotFoundError(f"Missing frozen P1 representation: {P1_AUGMENTED}")
p1_augmented = pd.read_parquet(P1_AUGMENTED)
missing_p1 = [c for c in ["match_id", *P1_FEATURES] if c not in p1_augmented.columns]
if missing_p1:
    raise RuntimeError(f"Frozen P1 table is missing required columns: {missing_p1[:10]}")
p1_payload = p1_augmented[["match_id", *P1_FEATURES]].copy()
if p1_payload["match_id"].duplicated().any():
    raise RuntimeError("Frozen P1 table must contain one row per match_id")
SNAPSHOTS = SNAPSHOTS.merge(
    p1_payload,
    on="match_id",
    how="inner",
    validate="many_to_one",
)

missing_features = [c for c in set(PRE_FEATURES + LIVE_FEATURES) if c not in SNAPSHOTS.columns]
if missing_features:
    raise RuntimeError(f"Enriched Gold snapshot table is missing model features: {missing_features[:10]}")
SNAPSHOTS = SNAPSHOTS.sort_values(["match_id", "snapshot_rank"]).reset_index(drop=True)
MATCH_GROUPS = {int(mid): g.reset_index(drop=True) for mid, g in SNAPSHOTS.groupby("match_id", sort=False)}


def _match_meta(match_id: int) -> dict[str, Any]:
    return MATCH_META.get(
        int(match_id),
        {
            "home_team": "Home",
            "away_team": "Away",
            "match_date": "",
            "kick_off": "",
            "final_home_score": 0,
            "final_away_score": 0,
        },
    )


def _class_mapping(probabilities: np.ndarray) -> dict[str, float]:
    # Training labels were encoded 0=Away, 1=Draw, 2=Home.
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(values) != 3:
        raise RuntimeError(f"Expected three outcome probabilities, got {len(values)}")
    values = values / values.sum()
    return {"away": float(values[0]), "draw": float(values[1]), "home": float(values[2])}


def _select_snapshot(match_id: int, minute: float | None, snapshot_id: str | None) -> pd.Series:
    group = MATCH_GROUPS.get(int(match_id))
    if group is None:
        raise HTTPException(status_code=404, detail=f"Unknown match_id {match_id}")
    if snapshot_id is not None:
        exact = group[group["snapshot_id"].astype(str) == snapshot_id]
        if exact.empty:
            raise HTTPException(status_code=404, detail=f"snapshot_id {snapshot_id!r} not available for match")
        return exact.iloc[-1]
    requested = 0.0 if minute is None else float(minute)
    time_column = "snapshot_minute" if "snapshot_minute" in group.columns else "snapshot_rank"
    eligible = group[group[time_column].astype(float) <= requested]
    return eligible.iloc[-1] if not eligible.empty else group.iloc[0]


def _frame(row: pd.Series, features: list[str]) -> pd.DataFrame:
    return row[features].to_frame().T


def _tree_shap(row: pd.Series, top_k: int = 8) -> list[dict[str, Any]]:
    try:
        x = _frame(row, LIVE_FEATURES)
        underlying = getattr(LIVE_OUTCOME_MODEL, "model", LIVE_OUTCOME_MODEL)
        if hasattr(underlying, "named_steps"):
            prep = underlying.named_steps.get("prep")
            tree = underlying.named_steps.get("model")
            x_trans = prep.transform(x) if prep is not None else x.to_numpy()
            try:
                names = list(prep.get_feature_names_out(LIVE_FEATURES)) if prep is not None else LIVE_FEATURES
            except Exception:
                names = LIVE_FEATURES[: np.asarray(x_trans).shape[1]]
        else:
            tree = underlying
            x_trans = x
            names = LIVE_FEATURES
        values = shap.TreeExplainer(tree).shap_values(x_trans)
        arr = np.asarray(values)
        if arr.ndim == 3:
            # LightGBM multiclass SHAP can be sample x feature x class.
            if arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[1] == 3:
                class_id = int(np.argmax(LIVE_OUTCOME_MODEL.predict_proba(x)[0]))
                arr = arr[:, class_id]
        if arr.ndim == 2:
            arr = arr[0] if arr.shape[0] == 1 else arr[:, -1]
        arr = np.ravel(arr)
        n = min(len(arr), len(names))
        ranked = sorted(
            ((names[i], float(arr[i])) for i in range(n)),
            key=lambda z: abs(z[1]),
            reverse=True,
        )[:top_k]
        return [{"feature": str(name), "shap_value": value} for name, value in ranked]
    except Exception as exc:
        return [{"feature": "SHAP unavailable", "shap_value": 0.0, "detail": str(exc)[:180]}]


@lru_cache(maxsize=1024)
def _prematch_cached(match_id: int) -> dict[str, Any]:
    row = MATCH_GROUPS[int(match_id)].iloc[0]
    meta = _match_meta(match_id)
    x = _frame(row, PRE_FEATURES)
    start = time.perf_counter()
    proba = np.asarray(PRE_OUTCOME_MODEL.predict_proba(x))[0]
    margin = float(np.asarray(PRE_MARGIN_MODEL.predict(x)).reshape(-1)[0])
    model_ms = (time.perf_counter() - start) * 1000.0
    return {
        "match_id": int(match_id),
        "home_team": meta["home_team"],
        "away_team": meta["away_team"],
        "match_date": meta["match_date"],
        "outcome_probabilities": _class_mapping(proba),
        "expected_final_margin": margin,
        "model_inference_ms": model_ms,
        "feature_source": "MD1 gold_snapshots + frozen EXP01 P1 home/away join",
        "outcome_model": PRE_OUTCOME_MODEL_PATH.name,
        "margin_model": PRE_MARGIN_MODEL_PATH.name,
    }


@lru_cache(maxsize=4096)
def _predict_cached(match_id: int, snapshot_id: str) -> dict[str, Any]:
    row = _select_snapshot(match_id, None, snapshot_id)
    meta = _match_meta(match_id)
    x = _frame(row, LIVE_FEATURES)
    start = time.perf_counter()
    proba = np.asarray(LIVE_OUTCOME_MODEL.predict_proba(x))[0]
    margin = float(np.asarray(LIVE_MARGIN_MODEL.predict(x)).reshape(-1)[0])
    model_ms = (time.perf_counter() - start) * 1000.0
    return {
        "match_id": int(match_id),
        "home_team": meta["home_team"],
        "away_team": meta["away_team"],
        "snapshot_id": str(row["snapshot_id"]),
        "snapshot_minute": float(row.get("snapshot_minute", row["snapshot_rank"])),
        "snapshot_rank": float(row["snapshot_rank"]),
        "score": {"home": int(row.get("current_home_score", 0)), "away": int(row.get("current_away_score", 0))},
        "red_cards": {"home": int(row.get("home_live_red_cards", 0)), "away": int(row.get("away_live_red_cards", 0))},
        "outcome_probabilities": _class_mapping(proba),
        "expected_final_margin": margin,
        "top_shap": _tree_shap(row),
        "model_inference_ms": model_ms,
        "feature_source": "MD1 gold_snapshots + frozen EXP01 P1 home/away join",
        "outcome_model": LIVE_OUTCOME_MODEL_PATH.name,
        "margin_model": LIVE_MARGIN_MODEL_PATH.name,
    }


class PredictionRequest(BaseModel):
    match_id: int
    minute: float | None = None
    snapshot_id: str | None = None


app = FastAPI(
    title="Football Forecasting Bonus Service",
    version="1.2.0",
    description="Low-latency pre-match and in-play serving over the frozen MD1 + P1 representation used during training.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "matches_loaded": len(MATCH_GROUPS),
        "matches_named": sum(mid in MATCH_META for mid in MATCH_GROUPS),
        "snapshots_loaded": len(SNAPSHOTS),
        "pre_features": len(PRE_FEATURES),
        "live_features": len(LIVE_FEATURES),
        "p1_features": len(P1_FEATURES),
    }


@app.get("/matches")
def matches(split: str | None = Query(default="test")) -> list[dict[str, Any]]:
    frame = SNAPSHOTS
    if split and "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.lower() == split.lower()]
    first = frame.groupby("match_id", as_index=False).first()
    result = []
    for _, row in first.iterrows():
        match_id = int(row["match_id"])
        meta = _match_meta(match_id)
        result.append({
            "match_id": match_id,
            "home_team": meta["home_team"],
            "away_team": meta["away_team"],
            "match_date": meta["match_date"],
            "split": str(row.get("split", "unknown")),
        })
    return result


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
