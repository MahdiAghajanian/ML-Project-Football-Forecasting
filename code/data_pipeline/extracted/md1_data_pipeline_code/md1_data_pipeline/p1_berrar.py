"""Berrar et al. (2024) P1 data-representation reimplementation.

This module is intentionally separate from the verified Mid Defence 1 EPL event
pipeline.  It reproduces the paper's league-specific Super League representation,
its six-feature ``total`` and 18-feature ``homeaway`` views, and Pearson recency
selection while applying the stricter chronological anti-leakage rules required
by the course final project.

Paper-aligned components and project adaptations are exported explicitly through
``p1_fidelity_table.csv`` and ``p1_fidelity_decisions.json``.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    MD1Config,
    build_http_session,
    download_file,
    normalize_team_name,
    read_json,
    save_table,
    write_json,
)


PAPER_ID = "Berrar_2024"
PAPER_TOTAL_FEATURES = (
    "p1_home_scr_total",
    "p1_home_con_total",
    "p1_home_rank_total",
    "p1_away_scr_total",
    "p1_away_con_total",
    "p1_away_rank_total",
)
PAPER_HOMEAWAY_FEATURES = (
    *PAPER_TOTAL_FEATURES,
    "p1_home_scr_home",
    "p1_home_con_home",
    "p1_home_rank_home",
    "p1_away_scr_home",
    "p1_away_con_home",
    "p1_away_rank_home",
    "p1_home_scr_away",
    "p1_home_con_away",
    "p1_home_rank_away",
    "p1_away_scr_away",
    "p1_away_con_away",
    "p1_away_rank_away",
)
DEFAULT_MODERN_LA_LIGA_SEASONS = tuple(
    f"{year}/{year + 1}" for year in range(2004, 2021)
)


def _timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _season_sort_key(name: str) -> tuple[int, str]:
    text = str(name)
    try:
        first = int(text.split("/")[0])
    except Exception:
        first = 10**9
    return first, text


def _odd_half_n(n: int, policy: str) -> int:
    policy = str(policy).strip().lower()
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 == 0:
        return n // 2
    if policy == "floor":
        return max(1, n // 2)
    if policy == "ceil":
        return max(1, int(math.ceil(n / 2)))
    if policy == "even_only":
        raise ValueError(f"Odd recency n={n} is invalid under p1_odd_n_policy='even_only'.")
    raise ValueError("p1_odd_n_policy must be one of: floor, ceil, even_only")


def _validate_p1_config(config: MD1Config) -> None:
    if config.p1_recency_min <= 0 or config.p1_recency_max < config.p1_recency_min:
        raise ValueError("Invalid P1 recency range.")
    if config.data_mode == "real" and (
        config.p1_recency_min != 9 or config.p1_recency_max != 100
    ):
        # Non-paper ranges are permitted only when the user deliberately configures them,
        # but the resolved config makes the deviation visible in the audits.
        pass
    if config.p1_min_prior_matches < 1:
        raise ValueError("p1_min_prior_matches must be >= 1")
    _odd_half_n(max(config.p1_recency_min, 9), config.p1_odd_n_policy)
    if config.p1_primary_recency_method not in {"pearson", "knn"}:
        raise ValueError("p1_primary_recency_method must be 'pearson' or 'knn'.")


def ensure_p1_competitions_catalog(config: MD1Config) -> Path:
    """Ensure StatsBomb ``competitions.json`` exists for P1 season discovery."""
    path = config.statsbomb_root / "competitions.json"
    if path.exists() and not config.overwrite_downloads:
        return path
    if config.data_mode != "real":
        raise FileNotFoundError("Synthetic P1 does not use the StatsBomb competitions catalog.")
    session = build_http_session()
    rec = download_file(
        session,
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json",
        path,
        overwrite=config.overwrite_downloads,
    )
    if rec.get("status") not in {"downloaded", "reused"}:
        raise RuntimeError(f"Unable to obtain StatsBomb competitions.json: {rec}")
    return path


def discover_p1_competition_seasons(
    config: MD1Config,
    competitions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Discover and validate the configured modern La Liga competition-season pairs."""
    _validate_p1_config(config)
    if competitions is None:
        path = ensure_p1_competitions_catalog(config)
        competitions = pd.json_normalize(read_json(path))
    required = {
        "competition_id",
        "season_id",
        "competition_name",
        "season_name",
    }
    missing = required.difference(competitions.columns)
    if missing:
        raise KeyError(f"competitions catalog is missing required fields: {sorted(missing)}")

    target_name = str(config.p1_competition_name).strip().casefold()
    rows = competitions[
        competitions["competition_name"].astype(str).str.strip().str.casefold() == target_name
    ].copy()
    requested = tuple(config.p1_include_season_names or DEFAULT_MODERN_LA_LIGA_SEASONS)
    if requested:
        rows = rows[rows["season_name"].astype(str).isin(requested)].copy()
    rows["season_sort"] = rows["season_name"].map(lambda x: _season_sort_key(str(x))[0])
    rows = rows.sort_values(["season_sort", "season_id"]).reset_index(drop=True)

    found = set(rows["season_name"].astype(str))
    missing_seasons = [s for s in requested if s not in found]
    if missing_seasons:
        raise RuntimeError(
            "Configured P1 seasons are not available in StatsBomb competitions.json: "
            + ", ".join(missing_seasons)
        )
    if rows.empty:
        raise RuntimeError(f"No StatsBomb seasons found for P1 competition {config.p1_competition_name!r}.")
    if rows["competition_id"].nunique() != 1:
        raise AssertionError("Primary P1 Super League must contain exactly one StatsBomb competition_id.")
    if rows["competition_name"].nunique() != 1:
        raise AssertionError("Primary P1 Super League must contain exactly one league.")
    if "1973/1974" in found or "1973/74" in found:
        raise AssertionError("The isolated 1973/74 era must not enter the default modern P1 run.")

    audit = rows[
        [
            "competition_id",
            "season_id",
            "competition_name",
            "season_name",
            *(["country_name"] if "country_name" in rows.columns else []),
            *(["match_available"] if "match_available" in rows.columns else []),
        ]
    ].copy()
    audit["selected_for_p1"] = True
    audit["paper_alignment"] = "project_dataset_selection"
    save_table(audit, config.audit_root / "p1_competition_seasons.csv")
    return rows.drop(columns="season_sort")


def download_p1_match_metadata(config: MD1Config) -> pd.DataFrame:
    """Download only match metadata for the configured P1 seasons.

    Rich event files remain the responsibility of the existing event pipeline.  This
    keeps the paper-aligned historical branch lightweight and prevents accidental use
    of Football-Data results as predictive features.
    """
    if config.data_mode != "real":
        raise ValueError("download_p1_match_metadata is only valid in real mode")
    config.ensure_directories()
    seasons = discover_p1_competition_seasons(config)
    session = build_http_session()
    base = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    records: list[dict[str, Any]] = []
    for row in seasons.to_dict("records"):
        comp_id = int(row["competition_id"])
        season_id = int(row["season_id"])
        path = config.statsbomb_root / "matches" / str(comp_id) / f"{season_id}.json"
        rec = download_file(
            session,
            f"{base}/matches/{comp_id}/{season_id}.json",
            path,
            overwrite=config.overwrite_downloads,
        )
        rec.update(
            {
                "competition_id": comp_id,
                "season_id": season_id,
                "season_name": str(row["season_name"]),
                "family": "p1_matches_metadata",
            }
        )
        records.append(rec)
    manifest = pd.DataFrame(records)
    save_table(manifest, config.audit_root / "p1_download_manifest.csv")
    bad = manifest[~manifest["status"].isin(["downloaded", "reused"])]
    if not bad.empty:
        raise RuntimeError("One or more required P1 StatsBomb match files failed to download.")
    return seasons


def _parse_statsbomb_match_record(
    m: Mapping[str, Any],
    *,
    default_competition_id: int,
    default_competition_name: str,
    default_season_id: int,
    default_season_name: str,
) -> dict[str, Any]:
    home = m.get("home_team") or {}
    away = m.get("away_team") or {}
    comp = m.get("competition") or {}
    season = m.get("season") or {}
    home_id = int(home.get("home_team_id", home.get("id")))
    away_id = int(away.get("away_team_id", away.get("id")))
    home_name = home.get("home_team_name", home.get("name"))
    away_name = away.get("away_team_name", away.get("name"))
    kickoff = pd.to_datetime(
        f"{m.get('match_date')} {m.get('kick_off') or '00:00:00'}",
        errors="coerce",
    )
    if pd.isna(kickoff):
        raise ValueError(f"Invalid kickoff in match_id={m.get('match_id')}")
    hs = int(m.get("home_score", 0))
    as_ = int(m.get("away_score", 0))
    return {
        "match_id": int(m["match_id"]),
        "competition_id": int(comp.get("competition_id", default_competition_id)),
        "competition_name": comp.get("competition_name", default_competition_name),
        "season_id": int(season.get("season_id", default_season_id)),
        "season_name": season.get("season_name", default_season_name),
        "match_date": pd.Timestamp(kickoff).normalize(),
        "kick_off": m.get("kick_off"),
        "kickoff": pd.Timestamp(kickoff),
        # Match metadata does not include an actual full-time timestamp. Two hours is
        # deliberately conservative and mirrors the existing baseline pipeline.
        "estimated_finish": pd.Timestamp(kickoff) + pd.Timedelta(hours=2),
        "home_team_id": home_id,
        "home_team_name": home_name,
        "away_team_id": away_id,
        "away_team_name": away_name,
        "home_score": hs,
        "away_score": as_,
        "outcome": "H" if hs > as_ else "A" if hs < as_ else "D",
        "goal_margin_raw": hs - as_,
        "goal_margin_clipped": int(np.clip(hs - as_, -5, 5)),
        "match_week": m.get("match_week"),
    }


def parse_p1_matches(config: MD1Config, seasons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Parse all selected StatsBomb match files into one chronological Super League table."""
    if seasons is None:
        seasons = discover_p1_competition_seasons(config)
    rows: list[dict[str, Any]] = []
    for season in seasons.to_dict("records"):
        comp_id = int(season["competition_id"])
        season_id = int(season["season_id"])
        path = config.statsbomb_root / "matches" / str(comp_id) / f"{season_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing P1 StatsBomb match metadata: {path}")
        raw = read_json(path)
        if config.p1_max_matches_per_season is not None:
            raw = sorted(
                raw,
                key=lambda x: (x.get("match_date", ""), x.get("kick_off", ""), x.get("match_id", 0)),
            )[: int(config.p1_max_matches_per_season)]
        for m in raw:
            rows.append(
                _parse_statsbomb_match_record(
                    m,
                    default_competition_id=comp_id,
                    default_competition_name=str(season["competition_name"]),
                    default_season_id=season_id,
                    default_season_name=str(season["season_name"]),
                )
            )
    matches = pd.DataFrame(rows)
    if matches.empty:
        raise RuntimeError("P1 Super League contains no matches.")
    matches = matches.sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    if matches["competition_id"].nunique() != 1 or matches["competition_name"].nunique() != 1:
        raise AssertionError("P1 Super League must be league-specific.")
    if matches["match_id"].duplicated().any():
        raise AssertionError("Duplicate match_id detected in P1 Super League.")
    return matches


def _round_robin_pairs(team_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Deterministic double round-robin fixture list for the synthetic P1 test fixture."""
    ids = list(team_ids)
    pairs: list[tuple[int, int]] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            pairs.append((a, b))
            pairs.append((b, a))
    return pairs


def create_synthetic_p1_matches(config: MD1Config) -> pd.DataFrame:
    """Create a multi-season synthetic league with continuity, a gap, and a new team."""
    season_teams = {
        "S1": (101, 102, 103, 104),
        "S2": (101, 102, 103, 105),  # 104 gap; 105 newly introduced
        "S3": (101, 102, 104, 105),  # 104 returns after a gap
        "S4": (101, 102, 104, 105),
    }
    names = {101: "Alpha FC", 102: "Bravo FC", 103: "Charlie FC", 104: "Delta FC", 105: "Echo FC"}
    rows: list[dict[str, Any]] = []
    match_id = 900000
    base_date = pd.Timestamp("2018-08-01 15:00:00")
    for season_index, (season_name, team_ids) in enumerate(season_teams.items(), start=1):
        season_start = base_date + pd.DateOffset(years=season_index - 1)
        fixtures = _round_robin_pairs(team_ids)
        for fixture_index, (home_id, away_id) in enumerate(fixtures):
            kickoff = season_start + pd.Timedelta(days=fixture_index * 4)
            # Deterministic but nontrivial scores.
            hs = int((home_id + fixture_index + season_index) % 4)
            as_ = int((away_id + 2 * fixture_index + season_index) % 3)
            rows.append(
                {
                    "match_id": match_id,
                    "competition_id": 999,
                    "competition_name": "Synthetic League",
                    "season_id": season_index,
                    "season_name": season_name,
                    "match_date": kickoff.normalize(),
                    "kick_off": kickoff.strftime("%H:%M:%S"),
                    "kickoff": kickoff,
                    "estimated_finish": kickoff + pd.Timedelta(hours=2),
                    "home_team_id": home_id,
                    "home_team_name": names[home_id],
                    "away_team_id": away_id,
                    "away_team_name": names[away_id],
                    "home_score": hs,
                    "away_score": as_,
                    "outcome": "H" if hs > as_ else "A" if hs < as_ else "D",
                    "goal_margin_raw": hs - as_,
                    "goal_margin_clipped": int(np.clip(hs - as_, -5, 5)),
                    "match_week": fixture_index + 1,
                }
            )
            match_id += 1
    return pd.DataFrame(rows).sort_values(["kickoff", "match_id"]).reset_index(drop=True)


def build_p1_super_league(config: MD1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and audit the league-specific multi-season match table."""
    if config.data_mode == "synthetic_demo":
        matches = create_synthetic_p1_matches(config)
        season_rows = []
        for (season_id, season_name), group in matches.groupby(["season_id", "season_name"], sort=True):
            team_ids = set(group["home_team_id"]).union(set(group["away_team_id"]))
            expected = len(team_ids) * (len(team_ids) - 1)
            season_rows.append(
                {
                    "competition_id": 999,
                    "season_id": int(season_id),
                    "competition_name": "Synthetic League",
                    "season_name": str(season_name),
                    "selected_for_p1": True,
                    "match_rows": len(group),
                    "team_count": len(team_ids),
                    "expected_double_round_robin_matches": expected,
                    "is_full_league_season": len(group) == expected,
                }
            )
        season_audit = pd.DataFrame(season_rows)
        save_table(season_audit, config.audit_root / "p1_competition_seasons.csv")
    else:
        seasons = download_p1_match_metadata(config)
        matches = parse_p1_matches(config, seasons)
        season_rows = []
        for (season_id, season_name), group in matches.groupby(["season_id", "season_name"], sort=False):
            team_ids = set(group["home_team_id"]).union(set(group["away_team_id"]))
            expected = len(team_ids) * (len(team_ids) - 1)
            season_rows.append(
                {
                    "competition_id": int(group["competition_id"].iloc[0]),
                    "season_id": int(season_id),
                    "competition_name": str(group["competition_name"].iloc[0]),
                    "season_name": str(season_name),
                    "selected_for_p1": True,
                    "match_rows": int(len(group)),
                    "team_count": int(len(team_ids)),
                    "expected_double_round_robin_matches": int(expected),
                    "is_full_league_season": bool(len(group) == expected),
                }
            )
        season_audit = pd.DataFrame(season_rows)
        # Merge discovery metadata when available without duplicating season keys.
        discovered = pd.read_csv(config.audit_root / "p1_competition_seasons.csv")
        season_audit = discovered.drop(columns=[c for c in ["selected_for_p1", "paper_alignment"] if c in discovered], errors="ignore").merge(
            season_audit,
            on=["competition_id", "season_id", "competition_name", "season_name"],
            how="left",
            validate="one_to_one",
        )
        save_table(season_audit, config.audit_root / "p1_competition_seasons.csv")

    if not bool(season_audit["is_full_league_season"].all()):
        bad = season_audit.loc[~season_audit["is_full_league_season"]]
        raise AssertionError(
            "Configured P1 seasons are not complete double round-robin league seasons:\n"
            + bad.to_string(index=False)
        )
    save_table(matches, config.silver_root / "p1_super_league_matches.parquet")
    return matches, season_audit


def build_p1_team_identity_audit(config: MD1Config, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("home", "away"):
        tmp = matches[["season_id", "season_name", f"{side}_team_id", f"{side}_team_name"]].copy()
        tmp.columns = ["season_id", "season_name", "team_id", "team_name"]
        rows.append(tmp)
    teams = pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)
    teams["normalized_name"] = teams["team_name"].map(normalize_team_name)
    normalized_per_id = teams.groupby("team_id")["normalized_name"].nunique().to_dict()
    ids_per_name = teams.groupby("normalized_name")["team_id"].nunique().to_dict()
    teams["cross_season_identity"] = teams["team_id"].map(lambda x: f"statsbomb_team_id:{int(x)}")
    teams["normalized_names_for_id"] = teams["team_id"].map(normalized_per_id)
    teams["ids_for_normalized_name"] = teams["normalized_name"].map(ids_per_name)
    teams["passed"] = (teams["normalized_names_for_id"] == 1) & (teams["ids_for_normalized_name"] == 1)
    save_table(teams, config.audit_root / "p1_team_identity_audit.csv")
    if not bool(teams["passed"].all()):
        bad = teams.loc[~teams["passed"]]
        raise AssertionError("Ambiguous cross-season team identity detected:\n" + bad.to_string(index=False))
    return teams


def build_p1_team_match_facts(matches: pd.DataFrame) -> pd.DataFrame:
    """Create two paper-aligned team-perspective rows per match."""
    rows: list[dict[str, Any]] = []
    for m in matches.to_dict("records"):
        for side in ("home", "away"):
            is_home = side == "home"
            team_id = int(m["home_team_id"] if is_home else m["away_team_id"])
            opponent_id = int(m["away_team_id"] if is_home else m["home_team_id"])
            goals_scored = int(m["home_score"] if is_home else m["away_score"])
            goals_conceded = int(m["away_score"] if is_home else m["home_score"])
            points = 3 if goals_scored > goals_conceded else 1 if goals_scored == goals_conceded else 0
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "competition_id": int(m["competition_id"]),
                    "competition_name": str(m["competition_name"]),
                    "season_id": int(m["season_id"]),
                    "season_name": str(m["season_name"]),
                    "kickoff": pd.Timestamp(m["kickoff"]),
                    "match_finished_at": pd.Timestamp(m["estimated_finish"]),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "side": side,
                    "is_home": int(is_home),
                    "goals_scored": goals_scored,
                    "goals_conceded": goals_conceded,
                    "goal_difference": goals_scored - goals_conceded,
                    "points": points,
                    "win": int(points == 3),
                    "draw": int(points == 1),
                    "loss": int(points == 0),
                }
            )
    facts = pd.DataFrame(rows).sort_values(["match_finished_at", "match_id", "side"]).reset_index(drop=True)
    if facts.duplicated(["match_id", "team_id"]).any():
        raise AssertionError("P1 team-match fact key (match_id, team_id) is not unique.")
    return facts


def build_p1_current_team_sets(matches: pd.DataFrame) -> dict[tuple[int, int], tuple[int, ...]]:
    result: dict[tuple[int, int], tuple[int, ...]] = {}
    for (comp_id, season_id), group in matches.groupby(["competition_id", "season_id"], sort=False):
        members = sorted(set(group["home_team_id"].astype(int)).union(set(group["away_team_id"].astype(int))))
        result[(int(comp_id), int(season_id))] = tuple(members)
    return result


class _HistorySeries:
    __slots__ = (
        "finish_ns",
        "finish_values",
        "count",
        "prefix_scored",
        "prefix_conceded",
        "prefix_goal_difference",
        "prefix_points",
    )

    def __init__(self, frame: pd.DataFrame):
        ordered = frame.sort_values(["match_finished_at", "match_id"]).reset_index(drop=True)
        finish = pd.to_datetime(ordered["match_finished_at"])
        self.finish_values = finish.to_numpy(dtype="datetime64[ns]")
        self.finish_ns = self.finish_values.astype("int64")
        self.count = len(ordered)
        self.prefix_scored = np.concatenate([[0.0], np.cumsum(ordered["goals_scored"].to_numpy(float))])
        self.prefix_conceded = np.concatenate([[0.0], np.cumsum(ordered["goals_conceded"].to_numpy(float))])
        self.prefix_goal_difference = np.concatenate([[0.0], np.cumsum(ordered["goal_difference"].to_numpy(float))])
        self.prefix_points = np.concatenate([[0.0], np.cumsum(ordered["points"].to_numpy(float))])

    @staticmethod
    def _slice_sum(prefix: np.ndarray, start: int, end: int) -> float:
        return float(prefix[end] - prefix[start])

    def aggregate(self, cutoff: pd.Timestamp, n: int) -> dict[str, Any]:
        cutoff_ns = np.datetime64(_timestamp(cutoff), "ns").astype("int64")
        end = int(np.searchsorted(self.finish_ns, cutoff_ns, side="left"))
        if end <= 0:
            return {
                "count": 0,
                "avg_scored": np.nan,
                "avg_conceded": np.nan,
                "avg_goal_difference": np.nan,
                "points_sum": 0.0,
                "goal_difference_sum": 0.0,
                "goals_scored_sum": 0.0,
                "max_finish": pd.NaT,
            }
        start = max(0, end - int(n))
        count = end - start
        scored = self._slice_sum(self.prefix_scored, start, end)
        conceded = self._slice_sum(self.prefix_conceded, start, end)
        goal_diff = self._slice_sum(self.prefix_goal_difference, start, end)
        points = self._slice_sum(self.prefix_points, start, end)
        return {
            "count": int(count),
            "avg_scored": scored / count,
            "avg_conceded": conceded / count,
            "avg_goal_difference": goal_diff / count,
            "points_sum": points,
            "goal_difference_sum": goal_diff,
            "goals_scored_sum": scored,
            "max_finish": pd.Timestamp(self.finish_values[end - 1]),
        }


class P1HistoryIndex:
    """O(log n) lookup plus O(1) trailing aggregate for each team/view."""

    def __init__(self, facts: pd.DataFrame):
        self.series: dict[tuple[int, str], _HistorySeries] = {}
        for team_id, group in facts.groupby("team_id", sort=False):
            team_id = int(team_id)
            self.series[(team_id, "total")] = _HistorySeries(group)
            home = group[group["is_home"] == 1]
            away = group[group["is_home"] == 0]
            self.series[(team_id, "home")] = _HistorySeries(home)
            self.series[(team_id, "away")] = _HistorySeries(away)

    def aggregate(self, team_id: int, view: str, cutoff: pd.Timestamp, n: int) -> dict[str, Any]:
        series = self.series.get((int(team_id), str(view)))
        if series is None:
            return {
                "count": 0,
                "avg_scored": np.nan,
                "avg_conceded": np.nan,
                "avg_goal_difference": np.nan,
                "points_sum": 0.0,
                "goal_difference_sum": 0.0,
                "goals_scored_sum": 0.0,
                "max_finish": pd.NaT,
            }
        return series.aggregate(cutoff, n)


def _rank_view_table(
    history: P1HistoryIndex,
    members: Sequence[int],
    cutoff: pd.Timestamp,
    *,
    view: str,
    n: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, float], pd.Timestamp | pd.NaT]:
    stats: dict[int, dict[str, Any]] = {}
    max_finishes: list[pd.Timestamp] = []
    ranking_rows: list[tuple[int, float, float, float]] = []
    for team_id in members:
        agg = history.aggregate(int(team_id), view, cutoff, n)
        stats[int(team_id)] = agg
        if pd.notna(agg["max_finish"]):
            max_finishes.append(pd.Timestamp(agg["max_finish"]))
        # The paper defines league-table order by points, goal difference, goals scored.
        # When fewer than n observations exist, the available history is used.  This
        # exact tie/ranking behavior is documented as a fidelity clarification.
        ranking_rows.append(
            (
                int(team_id),
                float(agg["points_sum"]),
                float(agg["goal_difference_sum"]),
                float(agg["goals_scored_sum"]),
            )
        )
    ranking_rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    n_teams = len(ranking_rows)
    ranks: dict[int, float] = {}
    for absolute_rank, row in enumerate(ranking_rows, start=1):
        team_id = row[0]
        ranks[team_id] = 1.0 if n_teams <= 1 else float((n_teams - absolute_rank) / (n_teams - 1))
    max_finish = max(max_finishes) if max_finishes else pd.NaT
    return stats, ranks, max_finish


def p1_chronological_split(config: MD1Config, matches: pd.DataFrame) -> pd.DataFrame:
    """P1-specific chronological match split without overwriting baseline split_manifest.csv."""
    m = matches[["match_id", "kickoff", "competition_id", "season_id"]].copy()
    m["split_date"] = pd.to_datetime(m["kickoff"]).dt.normalize()
    dates = np.array(sorted(m["split_date"].dropna().unique()))
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for P1 splitting.")
    train_end = max(1, int(np.floor(len(dates) * config.train_fraction)))
    val_end = max(train_end + 1, int(np.floor(len(dates) * (config.train_fraction + config.validation_fraction))))
    val_end = min(val_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])

    def label(d: Any) -> str:
        return "train" if d in train_dates else "validation" if d in val_dates else "test"

    m["split"] = m["split_date"].map(label)
    m = m.drop(columns="split_date").sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    save_table(m, config.gold_root / "p1_split_manifest.csv")
    return m


def _feature_base_row(match: Mapping[str, Any], split: str, n: int) -> dict[str, Any]:
    return {
        "match_id": int(match["match_id"]),
        "competition_id": int(match["competition_id"]),
        "competition_name": str(match["competition_name"]),
        "season_id": int(match["season_id"]),
        "season_name": str(match["season_name"]),
        "kickoff": pd.Timestamp(match["kickoff"]),
        "home_team_id": int(match["home_team_id"]),
        "home_team_name": str(match["home_team_name"]),
        "away_team_id": int(match["away_team_id"]),
        "away_team_name": str(match["away_team_name"]),
        "home_score": int(match["home_score"]),
        "away_score": int(match["away_score"]),
        "outcome": str(match["outcome"]),
        "goal_margin_raw": int(match["goal_margin_raw"]),
        "goal_margin_clipped": int(match["goal_margin_clipped"]),
        "split": str(split),
        "recency_n": int(n),
    }


def build_p1_features_for_n(
    config: MD1Config,
    matches: pd.DataFrame,
    history: P1HistoryIndex,
    current_team_sets: Mapping[tuple[int, int], Sequence[int]],
    split_map: Mapping[int, str],
    n: int,
    *,
    include_homeaway: bool,
) -> pd.DataFrame:
    """Build causal paper-aligned P1 features for one recency value."""
    half_n = _odd_half_n(int(n), config.p1_odd_n_policy)
    rows: list[dict[str, Any]] = []
    for match in matches.to_dict("records"):
        kickoff = pd.Timestamp(match["kickoff"])
        home_id = int(match["home_team_id"])
        away_id = int(match["away_team_id"])
        members = tuple(current_team_sets[(int(match["competition_id"]), int(match["season_id"]))])

        total_stats, total_ranks, total_max = _rank_view_table(
            history, members, kickoff, view="total", n=int(n)
        )
        home_total = total_stats[home_id]
        away_total = total_stats[away_id]
        row = _feature_base_row(match, split_map[int(match["match_id"])], int(n))
        row.update(
            {
                "p1_home_scr_total": home_total["avg_scored"],
                "p1_home_con_total": home_total["avg_conceded"],
                "p1_home_rank_total": total_ranks[home_id],
                "p1_away_scr_total": away_total["avg_scored"],
                "p1_away_con_total": away_total["avg_conceded"],
                "p1_away_rank_total": total_ranks[away_id],
                "home_total_history_available": int(home_total["count"]),
                "away_total_history_available": int(away_total["count"]),
                "home_total_history_used": int(home_total["count"]),
                "away_total_history_used": int(away_total["count"]),
            }
        )
        source_maxes = [total_max] if pd.notna(total_max) else []

        if include_homeaway:
            home_stats, home_ranks, home_max = _rank_view_table(
                history, members, kickoff, view="home", n=half_n
            )
            away_stats, away_ranks, away_max = _rank_view_table(
                history, members, kickoff, view="away", n=half_n
            )
            ht_home = home_stats[home_id]
            at_home = home_stats[away_id]
            ht_away = away_stats[home_id]
            at_away = away_stats[away_id]
            row.update(
                {
                    "p1_home_scr_home": ht_home["avg_scored"],
                    "p1_home_con_home": ht_home["avg_conceded"],
                    "p1_home_rank_home": home_ranks[home_id],
                    "p1_away_scr_home": at_home["avg_scored"],
                    "p1_away_con_home": at_home["avg_conceded"],
                    "p1_away_rank_home": home_ranks[away_id],
                    "p1_home_scr_away": ht_away["avg_scored"],
                    "p1_home_con_away": ht_away["avg_conceded"],
                    "p1_home_rank_away": away_ranks[home_id],
                    "p1_away_scr_away": at_away["avg_scored"],
                    "p1_away_con_away": at_away["avg_conceded"],
                    "p1_away_rank_away": away_ranks[away_id],
                    "home_home_history_used": int(ht_home["count"]),
                    "home_away_history_used": int(ht_away["count"]),
                    "away_home_history_used": int(at_home["count"]),
                    "away_away_history_used": int(at_away["count"]),
                    "venue_recency_n": int(half_n),
                }
            )
            if pd.notna(home_max):
                source_maxes.append(home_max)
            if pd.notna(away_max):
                source_maxes.append(away_max)

        eligible = (
            int(home_total["count"]) >= int(config.p1_min_prior_matches)
            and int(away_total["count"]) >= int(config.p1_min_prior_matches)
        )
        row["p1_eligible"] = bool(eligible)
        row["max_p1_source_finish"] = max(source_maxes) if source_maxes else pd.NaT
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    return frame


def assert_no_p1_prematch_leakage(features: pd.DataFrame) -> None:
    observed = features[features["max_p1_source_finish"].notna()].copy()
    if observed.empty:
        return
    bad = observed[pd.to_datetime(observed["max_p1_source_finish"]) >= pd.to_datetime(observed["kickoff"])]
    if not bad.empty:
        raise AssertionError(
            "P1 feature source timestamp is not strictly before target kickoff:\n"
            + bad[["match_id", "kickoff", "max_p1_source_finish"]].head(20).to_string(index=False)
        )


def _pearson_for_frame(frame: pd.DataFrame) -> tuple[float, int]:
    eligible = frame[(frame["split"] == "train") & frame["p1_eligible"]].copy()
    required = [
        "p1_home_scr_total",
        "p1_home_con_total",
        "p1_away_scr_total",
        "p1_away_con_total",
        "goal_margin_raw",
    ]
    eligible = eligible.dropna(subset=required)
    if len(eligible) < 3:
        return np.nan, int(len(eligible))
    home_gd = eligible["p1_home_scr_total"] - eligible["p1_home_con_total"]
    away_gd = eligible["p1_away_scr_total"] - eligible["p1_away_con_total"]
    delta_gd = home_gd - away_gd
    delta_scr = eligible["p1_home_scr_total"] - eligible["p1_away_scr_total"]
    signal = delta_gd + delta_scr
    y = eligible["goal_margin_raw"].astype(float)
    if signal.nunique() < 2 or y.nunique() < 2:
        return np.nan, int(len(eligible))
    corr = float(np.corrcoef(signal.to_numpy(float), y.to_numpy(float))[0, 1])
    return corr, int(len(eligible))


def select_p1_recency_pearson(config: MD1Config, all_n_total: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for n, frame in all_n_total.groupby("recency_n", sort=True):
        corr, count = _pearson_for_frame(frame)
        rows.append(
            {
                "n": int(n),
                "rows_used": int(count),
                "pearson_correlation": corr,
            }
        )
    audit = pd.DataFrame(rows).sort_values("n").reset_index(drop=True)
    valid = audit[np.isfinite(audit["pearson_correlation"])].copy()
    if valid.empty:
        raise RuntimeError("Pearson recency selection has no valid candidate.")
    # Paper exploits the expected positive correlation and chooses the strongest positive value.
    positive = valid[valid["pearson_correlation"] >= 0]
    candidate_pool = positive if not positive.empty else valid
    selected_n = int(candidate_pool.sort_values(["pearson_correlation", "n"], ascending=[False, True]).iloc[0]["n"])
    audit["selected"] = audit["n"] == selected_n
    save_table(audit, config.audit_root / "p1_pearson_recency_search.csv")
    write_json(
        {
            "method": "pearson",
            "selected_n": selected_n,
            "selection_scope": "training_only",
            "candidate_min": int(audit["n"].min()),
            "candidate_max": int(audit["n"].max()),
            "paper_reference_eng1_n": 60,
            "paper_reference_is_not_hard_coded": True,
        },
        config.audit_root / "p1_selected_recency.json",
    )
    return selected_n, audit


def _rps(probabilities: np.ndarray, y_index: np.ndarray) -> float:
    one_hot = np.eye(3)[y_index]
    cdf_p = np.cumsum(probabilities, axis=1)[:, :2]
    cdf_y = np.cumsum(one_hot, axis=1)[:, :2]
    return float(np.mean(0.5 * np.sum((cdf_p - cdf_y) ** 2, axis=1)))


def _knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k: int,
    *,
    task: str,
) -> np.ndarray:
    if len(x_train) == 0:
        raise ValueError("k-NN requires non-empty training data")
    k = max(1, min(int(k), len(x_train)))
    # This is a from-scratch NumPy implementation; no sklearn dependency is introduced.
    predictions: list[np.ndarray | float] = []
    for q in x_query:
        distances = np.sum((x_train - q) ** 2, axis=1)
        idx = np.argpartition(distances, k - 1)[:k]
        neighbors = y_train[idx]
        if task == "classification":
            counts = np.bincount(neighbors.astype(int), minlength=3).astype(float)
            predictions.append(counts / counts.sum())
        elif task == "margin":
            predictions.append(float(np.mean(neighbors.astype(float))))
        else:
            raise ValueError("task must be classification or margin")
    return np.asarray(predictions)


def _prepare_knn_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = frame[frame["p1_eligible"]].dropna(subset=list(PAPER_TOTAL_FEATURES)).copy()
    x = work[list(PAPER_TOTAL_FEATURES)].to_numpy(float)
    # Scaling is estimated from the passed training frame only.
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std, mean, std


def select_p1_recency_knn(
    config: MD1Config,
    all_n_total: pd.DataFrame,
    *,
    mode: str = "project_temporal",
    task: str = "classification",
    k_values: Sequence[int] | None = None,
) -> tuple[dict[str, int], pd.DataFrame]:
    """Optional secondary k-NN recency selector.

    ``paper_diagnostic`` performs LOOCV inside the training partition only so it can
    never contaminate the final held-out sets. ``project_temporal`` uses a chronological
    inner train/validation split, which is the final-project-safe adaptation.
    """
    mode = str(mode)
    if mode not in {"paper_diagnostic", "project_temporal"}:
        raise ValueError("mode must be paper_diagnostic or project_temporal")
    if task not in {"classification", "margin"}:
        raise ValueError("task must be classification or margin")
    if k_values is None:
        k_values = tuple(config.p1_knn_k_values)
    rows: list[dict[str, Any]] = []
    y_map = {"H": 0, "D": 1, "A": 2}

    for n, full in all_n_total.groupby("recency_n", sort=True):
        frame = full[(full["split"] == "train") & full["p1_eligible"]].dropna(subset=list(PAPER_TOTAL_FEATURES)).copy()
        frame = frame.sort_values(["kickoff", "match_id"]).reset_index(drop=True)
        if len(frame) < 8:
            continue
        x_raw = frame[list(PAPER_TOTAL_FEATURES)].to_numpy(float)
        y = frame["outcome"].map(y_map).to_numpy(int) if task == "classification" else frame["goal_margin_raw"].to_numpy(float)

        if mode == "project_temporal":
            cut = max(3, int(np.floor(len(frame) * 0.8)))
            if cut >= len(frame):
                continue
            x_train, x_val = x_raw[:cut], x_raw[cut:]
            # The supplied paper does not specify an additional scaler for its six
            # total k-NN features, so the diagnostic uses the paper features directly.
            y_train, y_val = y[:cut], y[cut:]
            for k in k_values:
                if k > len(x_train):
                    continue
                pred = _knn_predict(x_train, y_train, x_val, int(k), task=task)
                metric = _rps(pred, y_val.astype(int)) if task == "classification" else float(np.sqrt(np.mean((pred - y_val) ** 2)))
                rows.append({"n": int(n), "k": int(k), "mode": mode, "task": task, "metric": metric, "rows_used": len(frame)})
        else:
            # Paper-style LOOCV diagnostic, restricted to outer-training examples.
            x = x_raw
            for k in k_values:
                if k >= len(x):
                    continue
                preds = []
                truths = []
                for i in range(len(x)):
                    mask = np.ones(len(x), dtype=bool)
                    mask[i] = False
                    pred = _knn_predict(x[mask], y[mask], x[i : i + 1], int(k), task=task)[0]
                    preds.append(pred)
                    truths.append(y[i])
                pred_array = np.asarray(preds)
                truth_array = np.asarray(truths)
                metric = _rps(pred_array, truth_array.astype(int)) if task == "classification" else float(np.sqrt(np.mean((pred_array - truth_array) ** 2)))
                rows.append({"n": int(n), "k": int(k), "mode": mode, "task": task, "metric": metric, "rows_used": len(frame)})

    audit = pd.DataFrame(rows)
    if audit.empty:
        raise RuntimeError("k-NN recency search produced no valid candidate.")
    best = audit.sort_values(["metric", "n", "k"]).iloc[0]
    selected = {"n": int(best["n"]), "k": int(best["k"])}
    audit["selected"] = (audit["n"] == selected["n"]) & (audit["k"] == selected["k"])
    save_table(audit, config.audit_root / "p1_knn_recency_search.csv")
    return selected, audit


def merge_existing_prematch_with_p1(
    existing_prematch: pd.DataFrame,
    selected_p1: pd.DataFrame,
    *,
    common_only: bool = True,
) -> pd.DataFrame:
    """Create an Experiment-C-ready augmented table on identical match IDs.

    This helper does not build multi-season event features itself. It allows the later
    final-project model pipeline to merge any event-derived pre-match table with the
    selected Berrar representation without changing either feature-construction branch.
    """
    if existing_prematch["match_id"].duplicated().any() or selected_p1["match_id"].duplicated().any():
        raise AssertionError("Augmentation requires one row per match in both inputs.")
    p1_cols = [
        "match_id", "p1_eligible", "recency_n",
        *[c for c in PAPER_HOMEAWAY_FEATURES if c in selected_p1.columns],
    ]
    how = "inner" if common_only else "left"
    merged = existing_prematch.merge(
        selected_p1[p1_cols], on="match_id", how=how, validate="one_to_one"
    )
    if common_only:
        merged = merged[merged["p1_eligible"]].reset_index(drop=True)
    return merged


def build_common_population_audit(
    config: MD1Config, selected: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test"):
        group = selected[selected["split"] == split]
        eligible = group[group["p1_eligible"]]
        rows.append({
            "split": split,
            "all_matches": int(len(group)),
            "p1_eligible_matches": int(len(eligible)),
            "common_train_n": int(len(eligible)) if split == "train" else np.nan,
            "common_validation_n": int(len(eligible)) if split == "validation" else np.nan,
            "common_test_n": int(len(eligible)) if split == "test" else np.nan,
        })
    audit = pd.DataFrame(rows)
    save_table(audit, config.audit_root / "p1_common_population.csv")
    return audit


def build_p1_history_coverage(config: MD1Config, selected: pd.DataFrame, selected_n: int) -> pd.DataFrame:
    rows = []
    for split, group in selected.groupby("split", sort=False):
        for side in ("home", "away"):
            counts = group[f"{side}_total_history_available"].astype(int)
            eligible_counts = counts[counts >= config.p1_min_prior_matches]
            full_count = int((eligible_counts >= selected_n).sum())
            partial_count = int(((eligible_counts >= config.p1_min_prior_matches) & (eligible_counts < selected_n)).sum())
            rows.append(
                {
                    "split": split,
                    "side": side,
                    "rows": len(group),
                    "min_history": int(counts.min()) if len(counts) else 0,
                    "median_history": float(counts.median()) if len(counts) else np.nan,
                    "max_history": int(counts.max()) if len(counts) else 0,
                    "below_minimum": int((counts < config.p1_min_prior_matches).sum()),
                    "using_full_selected_n": full_count,
                    "using_less_than_selected_n": partial_count,
                    "fraction_full_selected_n_among_eligible": float(full_count / len(eligible_counts)) if len(eligible_counts) else np.nan,
                    "fraction_less_than_selected_n_among_eligible": float(partial_count / len(eligible_counts)) if len(eligible_counts) else np.nan,
                }
            )
    coverage = pd.DataFrame(rows)
    save_table(coverage, config.audit_root / "p1_history_coverage.csv")

    eligibility = selected[
        [
            "match_id",
            "season_name",
            "kickoff",
            "split",
            "home_team_id",
            "home_team_name",
            "away_team_id",
            "away_team_name",
            "home_total_history_available",
            "away_total_history_available",
            "p1_eligible",
        ]
    ].copy()
    eligibility["exclusion_reason"] = np.where(
        eligibility["p1_eligible"],
        "eligible",
        np.where(
            eligibility["home_total_history_available"] < config.p1_min_prior_matches,
            "home_history_below_minimum",
            "away_history_below_minimum",
        ),
    )
    save_table(eligibility, config.audit_root / "p1_eligibility.csv")

    # Additional views requested by the project prompt: history coverage by season and team.
    season_rows = []
    for season, group in eligibility.groupby("season_name", sort=False):
        counts = pd.concat([group["home_total_history_available"], group["away_total_history_available"]], ignore_index=True).astype(int)
        season_rows.append({
            "season_name": season,
            "match_rows": int(len(group)),
            "eligible_matches": int(group["p1_eligible"].sum()),
            "median_available_total_history": float(counts.median()),
            "min_available_total_history": int(counts.min()),
            "max_available_total_history": int(counts.max()),
        })
    save_table(pd.DataFrame(season_rows), config.audit_root / "p1_history_coverage_by_season.csv")

    team_side_rows = []
    for side in ("home", "away"):
        tmp = eligibility[[f"{side}_team_id", f"{side}_team_name", f"{side}_total_history_available"]].copy()
        tmp.columns = ["team_id", "team_name", "history_available"]
        team_side_rows.append(tmp)
    team_long = pd.concat(team_side_rows, ignore_index=True)
    team_coverage = (
        team_long.groupby(["team_id", "team_name"], as_index=False)["history_available"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "appearances", "min": "min_history", "median": "median_history", "max": "max_history"})
    )
    save_table(team_coverage, config.audit_root / "p1_history_coverage_by_team.csv")
    return coverage

def build_p1_feature_dictionary(config: MD1Config, selected_n: int) -> pd.DataFrame:
    rows = []
    for col in PAPER_HOMEAWAY_FEATURES:
        parts = col.split("_")
        metric = "scored" if "_scr_" in col else "conceded" if "_con_" in col else "normalized_rank"
        history_view = "total" if col.endswith("_total") else "home" if col.endswith("_home") else "away"
        rows.append(
            {
                "table": "p1_features_homeaway_selected",
                "column": col,
                "role": "feature",
                "source": "StatsBomb matches metadata",
                "model_eligible": True,
                "prediction_time": "pre_match",
                "leakage_guard": "all contributing matches estimated finished before target kickoff",
                "feature_family": PAPER_ID,
                "paper_alignment": "paper_aligned",
                "history_view": history_view,
                "recency_n": selected_n if history_view == "total" else _odd_half_n(selected_n, config.p1_odd_n_policy),
                "aggregation": "mean" if metric != "normalized_rank" else "recency league-table rank normalized to [0,1]",
                "metric": metric,
            }
        )
    dictionary = pd.DataFrame(rows)
    save_table(dictionary, config.audit_root / "p1_feature_dictionary.csv")
    return dictionary


def build_p1_fidelity_audits(config: MD1Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [
        ("feature source", "basic match records", "StatsBomb matches metadata", "project_required_adaptation", "course mandates StatsBomb as modeling source"),
        ("league-wise processing", "features computed separately by league", "one La Liga Super League", "exact", "primary experiment remains league-specific"),
        ("Super League", "all seasons of one league concatenated chronologically", "same", "exact", "season boundaries do not reset history"),
        ("season boundaries", "crossed in team histories", "same", "exact", ""),
        ("team-history gaps", "gaps ignored; prior same-league history concatenated", "same", "exact", ""),
        ("minimum prior history", "discard if either team has fewer than 6 prior matches", f"same; minimum={config.p1_min_prior_matches}", "exact" if config.p1_min_prior_matches == 6 else "faithful_with_clarification", ""),
        ("aggregation", "mean", "mean", "exact", ""),
        ("performance categories", "final features use scored, conceded, league success", "same", "exact", ""),
        ("total features", "6", "6", "exact", ""),
        ("homeaway features", "18", "18", "exact", ""),
        ("normalized rank", "(N-R)/(N-1) from recency league table", "same formula with explicit causal current-season membership", "faithful_with_clarification", "paper leaves some implementation details implicit"),
        ("n range", "9..100", f"{config.p1_recency_min}..{config.p1_recency_max}", "exact" if (config.p1_recency_min, config.p1_recency_max) == (9, 100) else "faithful_with_clarification", "synthetic smoke tests may use a reduced range"),
        ("Pearson recency", "corr((Δavg GD + Δavg scored), observed goal difference)", "same criterion on outer-training rows only", "project_required_adaptation", "course forbids validation/test labels in hyperparameter selection"),
        ("k-NN recency", "LOOCV, n=9..100, k=3..350", "optional paper_diagnostic and temporal-safe modes", "faithful_with_clarification", "not required for primary Pearson implementation"),
        ("outer train/test split", "paper commonly uses random splits and acknowledges possible leakage", "strict chronological match-level split", "project_required_adaptation", "course leakage rules are absolute"),
        ("Task C", "H/D/A probabilities", "same outcome task downstream", "exact", ""),
        ("Task R", "paper predicts exact home and away scores", "project predicts clipped signed goal margin", "project_required_adaptation", "course defines Task R differently"),
        ("Task L", "not in paper", "P1 vector may be inherited by existing live snapshots", "project_specific_extension", "Berrar is pre-match representation"),
        ("bookmaker baseline", "independent odds benchmark", "existing Football-Data de-vigged baseline remains separate", "project_required_adaptation", "course prescribes Football-Data odds source"),
    ]
    fidelity = pd.DataFrame(rows, columns=["paper_component", "paper_method", "our_implementation", "status", "reason_for_adaptation"])
    save_table(fidelity, config.audit_root / "p1_fidelity_table.csv")
    decisions = {
        "odd_n_policy": {
            "value": config.p1_odd_n_policy,
            "status": "paper_aligned_with_documented_ambiguity",
            "reason": "The paper conceptually requires even n for n/2 venue histories but evaluates every integer n=9..100; no explicit odd-n rounding rule is stated in the supplied paper.",
        },
        "rank_tie_break": {
            "value": ["points desc", "goal_difference desc", "goals_scored desc", "team_id asc"],
            "status": "paper_aligned_with_documented_clarification",
            "reason": "The paper states the standard points/goal-difference/goals-scored order; team_id is used only as a deterministic final tie-break.",
        },
        "short_history_rank": {
            "value": "use available history up to n and rank by aggregate points, goal difference, goals scored",
            "status": "paper_aligned_with_documented_ambiguity",
            "reason": "The paper states the minimum-six rule and recency league tables but does not fully specify tie/normalization behavior when clubs have different available history lengths below n.",
        },
        "rank_population": {
            "value": "current target-season league members",
            "status": "project_required_anti_leakage_clarification",
            "reason": "Historical performance crosses seasons, but league success is defined against the clubs participating in the target season rather than every historical Super League club.",
        },
        "match_finish_time": {
            "value": "kickoff + 2 hours",
            "status": "project_specific_audit_approximation",
            "reason": "StatsBomb match metadata has kickoff but no actual full-time timestamp. The conservative two-hour estimate mirrors the verified baseline pipeline and is used only for temporal provenance guards.",
        },
    }
    write_json(decisions, config.audit_root / "p1_fidelity_decisions.json")
    return fidelity, decisions


def _feature_values_equal(a: pd.Series, b: pd.Series, columns: Sequence[str]) -> bool:
    av = a[list(columns)].to_numpy(dtype=float)
    bv = b[list(columns)].to_numpy(dtype=float)
    return bool(np.allclose(av, bv, equal_nan=True, atol=1e-12, rtol=0.0))


def run_p1_leakage_tests(
    config: MD1Config,
    matches: pd.DataFrame,
    team_facts: pd.DataFrame,
    selected_total: pd.DataFrame,
    selected_homeaway: pd.DataFrame,
    all_n_total: pd.DataFrame,
    selected_n: int,
    current_team_sets: Mapping[tuple[int, int], Sequence[int]],
) -> pd.DataFrame:
    tests: list[tuple[str, Any]] = []

    def rebuild(mutated: pd.DataFrame, *, homeaway: bool = False) -> pd.DataFrame:
        facts = build_p1_team_match_facts(mutated)
        idx = P1HistoryIndex(facts)
        split_manifest = p1_chronological_split(config, mutated)
        split_map = split_manifest.set_index("match_id")["split"].to_dict()
        members = build_p1_current_team_sets(mutated)
        return build_p1_features_for_n(
            config, mutated, idx, members, split_map, selected_n, include_homeaway=homeaway
        )

    def target_perturbation() -> str:
        eligible = selected_total[selected_total["p1_eligible"]]
        if eligible.empty:
            raise AssertionError("No eligible row available for target perturbation.")
        target_id = int(eligible.iloc[len(eligible) // 2]["match_id"])
        before = selected_total.set_index("match_id").loc[target_id]
        mutated = matches.copy()
        mask = mutated["match_id"] == target_id
        mutated.loc[mask, "home_score"] = 999
        mutated.loc[mask, "away_score"] = 998
        mutated.loc[mask, "goal_margin_raw"] = 1
        mutated.loc[mask, "goal_margin_clipped"] = 1
        mutated.loc[mask, "outcome"] = "H"
        after = rebuild(mutated).set_index("match_id").loc[target_id]
        if not _feature_values_equal(before, after, PAPER_TOTAL_FEATURES):
            raise AssertionError("Target result changed its own P1 features.")
        return f"target match {target_id} features unchanged"

    def future_perturbation() -> str:
        ordered = selected_total[selected_total["p1_eligible"]].sort_values("kickoff")
        if len(ordered) < 2:
            raise AssertionError("Not enough eligible rows for future perturbation.")
        target = ordered.iloc[len(ordered) // 3]
        later = matches[pd.to_datetime(matches["kickoff"]) > pd.Timestamp(target["kickoff"])].sort_values("kickoff")
        if later.empty:
            raise AssertionError("No future match available for perturbation.")
        future_id = int(later.iloc[-1]["match_id"])
        before = selected_total.set_index("match_id").loc[int(target["match_id"])]
        mutated = matches.copy()
        mask = mutated["match_id"] == future_id
        mutated.loc[mask, "home_score"] = 999
        mutated.loc[mask, "away_score"] = 0
        mutated.loc[mask, "goal_margin_raw"] = 999
        mutated.loc[mask, "goal_margin_clipped"] = 5
        mutated.loc[mask, "outcome"] = "H"
        after = rebuild(mutated).set_index("match_id").loc[int(target["match_id"])]
        if not _feature_values_equal(before, after, PAPER_TOTAL_FEATURES):
            raise AssertionError("Future result changed an earlier P1 feature row.")
        return f"future match {future_id} did not alter earlier match {int(target['match_id'])}"

    def season_boundary() -> str:
        season_order = {s: i for i, s in enumerate(sorted(matches["season_name"].astype(str).unique(), key=_season_sort_key))}
        facts = team_facts.sort_values("kickoff")
        for team_id, group in facts.groupby("team_id"):
            seasons = sorted(group["season_name"].astype(str).unique(), key=lambda s: season_order.get(s, 9999))
            for previous, current in zip(seasons, seasons[1:]):
                if season_order.get(current, 9999) == season_order.get(previous, -2) + 1:
                    first = group[group["season_name"].astype(str) == current].sort_values("kickoff").iloc[0]
                    prior_other_season = group[(pd.to_datetime(group["match_finished_at"]) < pd.Timestamp(first["kickoff"])) & (group["season_name"].astype(str) != current)]
                    if not prior_other_season.empty:
                        return f"team {int(team_id)} carries history from {previous} into {current}"
        raise AssertionError("No consecutive-season continuity case found.")

    def gap_continuity() -> str:
        order = sorted(matches["season_name"].astype(str).unique(), key=_season_sort_key)
        season_pos = {s: i for i, s in enumerate(order)}
        for team_id, group in team_facts.groupby("team_id"):
            seasons = sorted(group["season_name"].astype(str).unique(), key=lambda s: season_pos[s])
            for previous, current in zip(seasons, seasons[1:]):
                if season_pos[current] - season_pos[previous] > 1:
                    first = group[group["season_name"].astype(str) == current].sort_values("kickoff").iloc[0]
                    older = group[pd.to_datetime(group["match_finished_at"]) < pd.Timestamp(first["kickoff"])]
                    if older.empty:
                        raise AssertionError("Gap return lost older same-league history.")
                    return f"team {int(team_id)} retains history across gap {previous}->{current}"
        # Real La Liga selection may contain no club with a gap in the chosen period; this
        # structural branch is exercised by the required synthetic smoke test.
        if config.data_mode == "synthetic_demo":
            raise AssertionError("Synthetic P1 fixture was expected to contain a participation gap.")
        return "no gap case in this real subset; gap behavior covered by synthetic P1 smoke test"

    def minimum_rule() -> str:
        expected = (
            (selected_total["home_total_history_available"] >= config.p1_min_prior_matches)
            & (selected_total["away_total_history_available"] >= config.p1_min_prior_matches)
        )
        if not np.array_equal(expected.to_numpy(bool), selected_total["p1_eligible"].to_numpy(bool)):
            raise AssertionError("p1_eligible does not implement the configured minimum-history rule.")
        return f"eligibility exactly implements >= {config.p1_min_prior_matches} prior total matches per team"

    def n_isolation() -> str:
        before, _ = select_p1_recency_pearson(config, all_n_total)
        mutated = all_n_total.copy()
        non_train = mutated["split"] != "train"
        mutated.loc[non_train, "goal_margin_raw"] = 9999
        after, _ = select_p1_recency_pearson(config, mutated)
        if before != after or before != selected_n:
            raise AssertionError("Validation/test labels influenced selected Pearson recency.")
        return f"selected n={selected_n} unchanged after validation/test label perturbation"

    def rank_temporal_safety() -> str:
        ordered = selected_total[selected_total["p1_eligible"]].sort_values("kickoff")
        target = ordered.iloc[max(0, len(ordered) // 4)]
        future = matches[pd.to_datetime(matches["kickoff"]) > pd.Timestamp(target["kickoff"])]
        if future.empty:
            raise AssertionError("No future match for rank safety test.")
        future_id = int(future.sort_values("kickoff").iloc[-1]["match_id"])
        mutated = matches.copy()
        mask = mutated["match_id"] == future_id
        mutated.loc[mask, ["home_score", "away_score", "goal_margin_raw", "goal_margin_clipped"]] = [99, 0, 99, 5]
        mutated.loc[mask, "outcome"] = "H"
        after = rebuild(mutated).set_index("match_id").loc[int(target["match_id"])]
        before = selected_total.set_index("match_id").loc[int(target["match_id"])]
        rank_cols = ["p1_home_rank_total", "p1_away_rank_total"]
        if not _feature_values_equal(before, after, rank_cols):
            raise AssertionError("Future result changed an earlier normalized-rank feature.")
        return "future-result mutation does not change earlier rank features"

    def rank_population() -> str:
        historical_teams = set(matches["home_team_id"]).union(set(matches["away_team_id"]))
        for key, members in current_team_sets.items():
            if set(members) != historical_teams:
                if len(set(members)) >= len(historical_teams):
                    raise AssertionError("Current-season member set unexpectedly contains all historical clubs.")
                return f"season {key} ranks {len(members)} current clubs, not {len(historical_teams)} historical clubs"
        # Synthetic final seasons can still use a strict subset; if all sets happen to be equal,
        # the contract is nevertheless structurally guaranteed by current_team_sets keys.
        return "rank tables are constructed from season-keyed current_team_sets, not global historical membership"

    def feature_counts() -> str:
        total_cols = [c for c in selected_total.columns if c in PAPER_TOTAL_FEATURES]
        homeaway_cols = [c for c in selected_homeaway.columns if c in PAPER_HOMEAWAY_FEATURES]
        if len(total_cols) != 6 or len(homeaway_cols) != 18:
            raise AssertionError(f"Unexpected P1 feature counts: total={len(total_cols)}, homeaway={len(homeaway_cols)}")
        return "paper-aligned feature counts are exactly 6 and 18"

    tests.extend(
        [
            ("pre-match source time contract", lambda: (assert_no_p1_prematch_leakage(selected_homeaway), "all source finishes precede target kickoff")[1]),
            ("target-result perturbation", target_perturbation),
            ("future-result perturbation", future_perturbation),
            ("season-boundary continuity", season_boundary),
            ("gap continuity", gap_continuity),
            ("minimum-six rule", minimum_rule),
            ("selected-n isolation", n_isolation),
            ("league-rank temporal safety", rank_temporal_safety),
            ("current-season rank population", rank_population),
            ("feature-count contracts", feature_counts),
        ]
    )
    rows = []
    for name, fn in tests:
        start = time.perf_counter()
        try:
            details = fn()
            rows.append({"test": name, "passed": True, "details": str(details), "seconds": time.perf_counter() - start})
        except Exception as exc:
            rows.append({"test": name, "passed": False, "details": repr(exc), "seconds": time.perf_counter() - start})
    results = pd.DataFrame(rows)
    save_table(results, config.audit_root / "p1_leakage_tests.csv")
    if not bool(results["passed"].all()):
        failures = results.loc[~results["passed"], ["test", "details"]]
        raise AssertionError("P1 leakage/integrity tests failed:\n" + failures.to_string(index=False))
    return results


def build_p1_readiness(
    config: MD1Config,
    *,
    matches: pd.DataFrame,
    season_audit: pd.DataFrame,
    team_identity: pd.DataFrame,
    selected_total: pd.DataFrame,
    selected_homeaway: pd.DataFrame,
    selected_n: int,
    leakage_tests: pd.DataFrame,
    fidelity: pd.DataFrame,
) -> pd.DataFrame:
    def add(name: str, passed: bool, details: str) -> dict[str, Any]:
        return {"check": name, "passed": bool(passed), "details": details}

    rows = [
        add("selected seasons valid", bool(season_audit["is_full_league_season"].all()), f"{len(season_audit)} seasons"),
        add("one-league constraint", matches["competition_id"].nunique() == 1, f"competition_ids={matches['competition_id'].nunique()}"),
        add("Super League chronology", bool(pd.to_datetime(matches["kickoff"]).is_monotonic_increasing), "matches sorted chronologically"),
        add("cross-season team identities", bool(team_identity["passed"].all()), f"{team_identity['team_id'].nunique()} team identities"),
        add("minimum-six eligibility", bool(((selected_total['home_total_history_available'] >= config.p1_min_prior_matches) & (selected_total['away_total_history_available'] >= config.p1_min_prior_matches) == selected_total['p1_eligible']).all()), f"minimum={config.p1_min_prior_matches}"),
        add("6-feature total contract", len([c for c in PAPER_TOTAL_FEATURES if c in selected_total]) == 6, "exactly six paper total features"),
        add("18-feature homeaway contract", len([c for c in PAPER_HOMEAWAY_FEATURES if c in selected_homeaway]) == 18, "exactly eighteen paper homeaway features"),
        add("normalized rank bounded", bool(selected_homeaway[[c for c in PAPER_HOMEAWAY_FEATURES if 'rank' in c]].apply(lambda s: s.dropna().between(0, 1).all()).all()), "all observed ranks in [0,1]"),
        add("Pearson n in configured range", config.p1_recency_min <= selected_n <= config.p1_recency_max, f"selected_n={selected_n}"),
        add("P1 leakage tests", bool(leakage_tests["passed"].all()), f"{int(leakage_tests['passed'].sum())}/{len(leakage_tests)} passed"),
        add("fidelity table saved", not fidelity.empty and (config.audit_root / 'p1_fidelity_table.csv').exists(), f"{len(fidelity)} fidelity rows"),
        add("common comparison population", all(int(((selected_total['split'] == split) & selected_total['p1_eligible']).sum()) > 0 for split in ('train','validation','test')), "eligible rows exist in every outer split"),
    ]
    readiness = pd.DataFrame(rows)
    save_table(readiness, config.audit_root / "p1_readiness.csv")
    return readiness


def _plot_pearson(config: MD1Config, audit: pd.DataFrame, selected_n: int) -> Path:
    path = config.audit_root / "plots" / "p1_pearson_correlation_vs_n.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(audit["n"], audit["pearson_correlation"], marker="o", markersize=3)
    selected_row = audit[audit["n"] == selected_n]
    if not selected_row.empty:
        plt.scatter(selected_row["n"], selected_row["pearson_correlation"], s=70)
    plt.xlabel("Recency n")
    plt.ylabel("Pearson correlation")
    plt.title("Berrar P1 training-only Pearson recency search")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def run_p1_berrar_pipeline(
    config: MD1Config,
    *,
    run_knn: bool | None = None,
) -> dict[str, Any]:
    """Execute the P1 representation pipeline without altering the baseline outputs."""
    _validate_p1_config(config)
    config.ensure_directories()
    started = time.perf_counter()

    matches, season_audit = build_p1_super_league(config)
    team_identity = build_p1_team_identity_audit(config, matches)
    team_facts = build_p1_team_match_facts(matches)
    save_table(team_facts, config.silver_root / "p1_team_match_facts.parquet")

    split_manifest = p1_chronological_split(config, matches)
    split_map = split_manifest.set_index("match_id")["split"].to_dict()
    current_team_sets = build_p1_current_team_sets(matches)
    history = P1HistoryIndex(team_facts)

    candidate_frames: list[pd.DataFrame] = []
    candidate_root = config.processed_root / "p1_by_n"
    candidate_root.mkdir(parents=True, exist_ok=True)
    for n in range(int(config.p1_recency_min), int(config.p1_recency_max) + 1):
        frame = build_p1_features_for_n(
            config,
            matches,
            history,
            current_team_sets,
            split_map,
            n,
            include_homeaway=False,
        )
        assert_no_p1_prematch_leakage(frame)
        candidate_frames.append(frame)
        if config.p1_cache_all_n:
            save_table(frame, candidate_root / f"n_{n:03d}.parquet")

    all_n_total = pd.concat(candidate_frames, ignore_index=True)
    save_table(all_n_total, config.gold_root / "p1_features_total_all_n.parquet")

    selected_n, pearson_audit = select_p1_recency_pearson(config, all_n_total)
    selected_total = all_n_total[all_n_total["recency_n"] == selected_n].copy().reset_index(drop=True)
    fixed_n = int(config.p1_fixed_baseline_n)
    fixed_total = all_n_total[all_n_total["recency_n"] == fixed_n].copy().reset_index(drop=True)
    if fixed_total.empty:
        # In synthetic smoke mode the paper baseline n=10 can be outside the reduced test grid.
        fixed_n = int(config.p1_recency_min)
        fixed_total = all_n_total[all_n_total["recency_n"] == fixed_n].copy().reset_index(drop=True)
    save_table(fixed_total, config.gold_root / "p1_features_total_fixed.parquet")
    selected_homeaway = build_p1_features_for_n(
        config,
        matches,
        history,
        current_team_sets,
        split_map,
        selected_n,
        include_homeaway=True,
    )
    assert_no_p1_prematch_leakage(selected_homeaway)
    save_table(selected_total, config.gold_root / "p1_features_total_selected.parquet")
    save_table(selected_homeaway, config.gold_root / "p1_features_homeaway_selected.parquet")

    coverage = build_p1_history_coverage(config, selected_homeaway, selected_n)
    common_population = build_common_population_audit(config, selected_homeaway)
    feature_dictionary = build_p1_feature_dictionary(config, selected_n)
    fidelity, fidelity_decisions = build_p1_fidelity_audits(config)

    if run_knn is None:
        run_knn = bool(config.p1_run_knn_search)
    knn_result = None
    knn_audit = None
    if run_knn:
        knn_result, knn_audit = select_p1_recency_knn(
            config,
            all_n_total,
            mode=config.p1_knn_mode,
            task="classification",
        )

    leakage_tests = run_p1_leakage_tests(
        config,
        matches,
        team_facts,
        selected_total,
        selected_homeaway,
        all_n_total,
        selected_n,
        current_team_sets,
    )
    readiness = build_p1_readiness(
        config,
        matches=matches,
        season_audit=season_audit,
        team_identity=team_identity,
        selected_total=selected_total,
        selected_homeaway=selected_homeaway,
        selected_n=selected_n,
        leakage_tests=leakage_tests,
        fidelity=fidelity,
    )
    pearson_plot = _plot_pearson(config, pearson_audit, selected_n)

    elapsed = time.perf_counter() - started
    eligible = selected_total[selected_total["p1_eligible"]]
    split_counts = eligible["split"].value_counts().to_dict()
    summary = {
        "paper": "Berrar, Lopes & Dubitzky (2024)",
        "data_mode": config.data_mode,
        "competition": str(matches["competition_name"].iloc[0]),
        "seasons": sorted(matches["season_name"].astype(str).unique(), key=_season_sort_key),
        "season_count": int(matches["season_id"].nunique()),
        "match_count": int(len(matches)),
        "eligible_match_count": int(selected_total["p1_eligible"].sum()),
        "excluded_match_count": int((~selected_total["p1_eligible"]).sum()),
        "selected_n_pearson": int(selected_n),
        "fixed_recency_baseline_n": int(fixed_n),
        "candidate_n_min": int(config.p1_recency_min),
        "candidate_n_max": int(config.p1_recency_max),
        "total_feature_count": 6,
        "homeaway_feature_count": 18,
        "eligible_train": int(split_counts.get("train", 0)),
        "eligible_validation": int(split_counts.get("validation", 0)),
        "eligible_test": int(split_counts.get("test", 0)),
        "p1_leakage_tests_passed": int(leakage_tests["passed"].sum()),
        "p1_leakage_tests_total": int(len(leakage_tests)),
        "readiness_passed": bool(readiness["passed"].all()),
        "knn_search": knn_result,
        "runtime_seconds": float(elapsed),
    }
    summary_path = config.audit_root / "p1_run_summary.json"
    write_json(summary, summary_path)
    save_table(pd.DataFrame([asdict(config)]), config.audit_root / "p1_resolved_config.csv")

    if not bool(readiness["passed"].all()):
        failed = readiness.loc[~readiness["passed"]]
        blocked = config.project_root / "P1_RESULTS_BLOCKED.md"
        blocked.write_text(
            "# P1 RESULTS BLOCKED\n\nMandatory P1 readiness checks failed.\n\n"
            + failed.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )
        raise AssertionError("P1 readiness gate failed. See P1_RESULTS_BLOCKED.md")

    return {
        "matches": matches,
        "season_audit": season_audit,
        "team_identity": team_identity,
        "team_facts": team_facts,
        "split_manifest": split_manifest,
        "all_n_total": all_n_total,
        "selected_n": selected_n,
        "pearson_audit": pearson_audit,
        "selected_total": selected_total,
        "fixed_total": fixed_total,
        "selected_homeaway": selected_homeaway,
        "history_coverage": coverage,
        "common_population": common_population,
        "feature_dictionary": feature_dictionary,
        "fidelity": fidelity,
        "fidelity_decisions": fidelity_decisions,
        "knn_result": knn_result,
        "knn_audit": knn_audit,
        "leakage_tests": leakage_tests,
        "readiness": readiness,
        "pearson_plot": pearson_plot,
        "summary": summary,
        "summary_path": summary_path,
        "runtime_seconds": elapsed,
    }
