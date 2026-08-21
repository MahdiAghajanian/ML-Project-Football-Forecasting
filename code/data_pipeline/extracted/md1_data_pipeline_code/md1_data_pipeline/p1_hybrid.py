"""TA-approved hybrid Berrar P1 pipeline.

Source roles are deliberately narrow and auditable:

* StatsBomb EPL 2015/16 is the target/modeling season.  Its match IDs, labels,
  current-season completed results, event pipeline, and live snapshots remain the
  primary project data.
* Football-Data EPL 2000/01--2014/15 supplies only basic completed-match history
  (date, teams, full-time goals) for the long Berrar Super League context.
* Football-Data EPL 2015/16 odds remain an independent market benchmark and are
  never ingested by this historical P1 branch.

The Berrar feature engine itself is reused from :mod:`p1_berrar`: six ``total``
features, 18 ``homeaway`` features, mean aggregation, minimum six prior matches,
and recency n selected over 9..100 by the paper's Pearson criterion.  The outer
split and recency selection are chronological/training-only as required by the
course specification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    MD1Config,
    REAL_EPL_ALIAS_PAIRS,
    build_http_session,
    chronological_match_level_split,
    download_file,
    normalize_team_name,
    read_json,
    save_table,
    write_json,
)
from .p1_berrar import (
    PAPER_HOMEAWAY_FEATURES,
    PAPER_ID,
    PAPER_TOTAL_FEATURES,
    P1HistoryIndex,
    _feature_values_equal,
    _odd_half_n,
    _parse_statsbomb_match_record,
    _plot_pearson,
    _round_robin_pairs,
    assert_no_p1_prematch_leakage,
    build_common_population_audit,
    build_p1_current_team_sets,
    build_p1_features_for_n,
    build_p1_history_coverage,
    build_p1_team_match_facts,
    select_p1_recency_knn,
    select_p1_recency_pearson,
)

HISTORICAL_ALLOWED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
FORBIDDEN_HISTORY_COLUMN_TOKENS = (
    "odds", "b365", "psh", "psd", "psa", "avgh", "avgd", "avga",
    "shot", "corner", "card", "foul", "referee", "half", "ht",
)


def _season_label(start_year: int) -> str:
    return f"{start_year}/{start_year + 1}"


def _football_data_code(season_name: str) -> str:
    start = int(str(season_name).split("/")[0])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def configured_history_seasons(config: MD1Config) -> tuple[str, ...]:
    """Resolve the inclusive Football-Data history interval."""
    start = int(str(config.p1_history_start_season).split("/")[0])
    end = int(str(config.p1_history_end_season).split("/")[0])
    if end < start:
        raise ValueError("p1_history_end_season precedes p1_history_start_season")
    return tuple(_season_label(y) for y in range(start, end + 1))


def _canonical_fd_name(name: Any) -> str:
    """Canonical provider spelling used to bridge Football-Data and StatsBomb."""
    norm = normalize_team_name(name)
    # REAL_EPL_ALIAS_PAIRS is keyed by StatsBomb-style normalized names.  Applying
    # it here too makes the bridge robust if a Football-Data historical file uses
    # a long rather than abbreviated club name.
    return REAL_EPL_ALIAS_PAIRS.get(norm, norm)


def download_historical_football_data(config: MD1Config) -> pd.DataFrame:
    """Download complete pre-target EPL season CSVs used only for P1 history."""
    if config.data_mode != "real":
        raise ValueError("Historical Football-Data download is only valid in real mode")
    root = config.football_data_root / "p1_history"
    root.mkdir(parents=True, exist_ok=True)
    session = build_http_session()
    records: list[dict[str, Any]] = []
    for season_name in configured_history_seasons(config):
        code = _football_data_code(season_name)
        path = root / f"{config.p1_history_division}_{code}.csv"
        url = f"https://www.football-data.co.uk/mmz4281/{code}/{config.p1_history_division}.csv"
        rec = download_file(session, url, path, overwrite=config.overwrite_downloads)
        rec.update({
            "season_name": season_name,
            "season_code": code,
            "division": config.p1_history_division,
            "family": "p1_history_results",
            "permitted_use": "Berrar historical context only",
        })
        records.append(rec)
    manifest = pd.DataFrame(records)
    save_table(manifest, config.audit_root / "p1_history_download_manifest.csv")
    bad = manifest[~manifest["status"].isin(["downloaded", "reused"])]
    if not bad.empty:
        raise RuntimeError("One or more Football-Data historical seasons failed to download:\n" + bad.to_string(index=False))
    return manifest


def download_target_statsbomb_matches(config: MD1Config) -> Path:
    """Download only EPL 2015/16 StatsBomb match metadata for the P1 target rows."""
    if config.data_mode != "real":
        raise ValueError("StatsBomb target download is only valid in real mode")
    path = config.statsbomb_root / "matches" / str(config.competition_id) / f"{config.season_id}.json"
    if path.exists() and not config.overwrite_downloads:
        return path
    session = build_http_session()
    url = (
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/"
        f"{config.competition_id}/{config.season_id}.json"
    )
    rec = download_file(session, url, path, overwrite=config.overwrite_downloads)
    save_table(pd.DataFrame([{**rec, "family": "p1_target_statsbomb_matches"}]), config.audit_root / "p1_target_download_manifest.csv")
    if rec.get("status") not in {"downloaded", "reused"}:
        raise RuntimeError(f"Unable to obtain StatsBomb target match metadata: {rec}")
    return path


def parse_target_statsbomb_matches(config: MD1Config) -> pd.DataFrame:
    path = download_target_statsbomb_matches(config)
    raw = read_json(path)
    rows = [
        _parse_statsbomb_match_record(
            m,
            default_competition_id=config.competition_id,
            default_competition_name=config.competition_name,
            default_season_id=config.season_id,
            default_season_name=config.season_name,
        )
        for m in raw
    ]
    frame = pd.DataFrame(rows).sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("StatsBomb P1 target season is empty")
    if frame["match_id"].duplicated().any():
        raise AssertionError("Duplicate StatsBomb target match_id detected")
    frame["source_provider"] = "statsbomb_target"
    frame["source_role"] = "primary_modeling_target_and_current_season_history"
    return frame


def _read_fd_results_only(path: Path, season_name: str) -> pd.DataFrame:
    """Read only the five TA-approved Football-Data result fields.

    Historical Football-Data files are not schema-stable across eras.  Some old
    CSVs contain rows with more trailing market/statistics fields than the header
    declares.  Reading the entire file therefore makes pandas' C parser reject an
    otherwise valid match row (for example, ``Expected 57 fields ... saw 72``).

    P1 is permitted to use only the stable match-result identity fields anyway, so
    we validate the header first and then parse *only* those columns with
    ``usecols``.  This keeps every match row while preventing any odds/statistics
    columns from entering the historical representation.  We deliberately do not
    use ``on_bad_lines='skip'`` because silently dropping matches would invalidate
    the season-completeness contract.
    """
    encoding: str | None = None
    header: pd.DataFrame | None = None
    last_error: Exception | None = None

    # Old Football-Data archives can be Windows-1252; current files are often UTF-8.
    for candidate in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            header = pd.read_csv(path, nrows=0, encoding=candidate)
            encoding = candidate
            break
        except UnicodeDecodeError as exc:
            last_error = exc

    if header is None or encoding is None:
        raise UnicodeError(f"Unable to decode Football-Data CSV {path}: {last_error}")

    missing = [c for c in HISTORICAL_ALLOWED_COLUMNS if c not in header.columns]
    if missing:
        raise KeyError(f"{path.name} is missing required Football-Data fields: {missing}")

    # Critical: usecols prevents variable-width trailing legacy fields from
    # triggering a ParserError while preserving the five approved columns.
    raw = pd.read_csv(
        path,
        usecols=list(HISTORICAL_ALLOWED_COLUMNS),
        encoding=encoding,
        low_memory=False,
    )
    raw = raw.dropna(how="all").reset_index(drop=True)

    # Do not carry any other columns into the historical canonical table.
    frame = raw[list(HISTORICAL_ALLOWED_COLUMNS)].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, format="mixed", errors="coerce")
    frame["FTHG"] = pd.to_numeric(frame["FTHG"], errors="coerce")
    frame["FTAG"] = pd.to_numeric(frame["FTAG"], errors="coerce")
    frame["season_name"] = season_name
    frame["source_file"] = path.name
    return frame


def parse_historical_football_data(config: MD1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse and hard-audit all configured pre-target EPL seasons."""
    root = config.football_data_root / "p1_history"
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for season_name in configured_history_seasons(config):
        code = _football_data_code(season_name)
        path = root / f"{config.p1_history_division}_{code}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing P1 historical Football-Data CSV: {path}")
        frame = _read_fd_results_only(path, season_name)
        frames.append(frame)
        audits.append(_audit_complete_season(frame, season_name, provider="football_data_history", config=config))
    history = pd.concat(frames, ignore_index=True)
    audit = pd.DataFrame(audits).sort_values("season_name").reset_index(drop=True)
    save_table(audit, config.audit_root / "p1_season_completeness.csv")
    if config.p1_require_complete_seasons and not bool(audit["passed"].all()):
        raise AssertionError("One or more Football-Data P1 history seasons are incomplete:\n" + audit.loc[~audit["passed"]].to_string(index=False))
    return history, audit


def _audit_complete_season(frame: pd.DataFrame, season_name: str, *, provider: str, config: MD1Config) -> dict[str, Any]:
    clean = frame.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    home = clean["HomeTeam"].map(_canonical_fd_name)
    away = clean["AwayTeam"].map(_canonical_fd_name)
    teams = sorted(set(home).union(set(away)))
    directed = pd.DataFrame({"home": home, "away": away})
    duplicate_directed = int(directed.duplicated(["home", "away"]).sum())
    per_team = pd.concat([home, away]).value_counts()
    unordered = directed.apply(lambda r: tuple(sorted((r["home"], r["away"]))), axis=1).value_counts()
    expected_teams = int(config.p1_expected_team_count)
    expected_matches = int(config.p1_expected_matches_per_season)
    expected_team_matches = 2 * (expected_teams - 1)
    passed = (
        len(frame) == expected_matches
        and len(clean) == expected_matches
        and len(teams) == expected_teams
        and duplicate_directed == 0
        and len(per_team) == expected_teams
        and bool((per_team == expected_team_matches).all())
        and len(unordered) == expected_teams * (expected_teams - 1) // 2
        and bool((unordered == 2).all())
    )
    return {
        "season_name": season_name,
        "provider": provider,
        "rows": int(len(frame)),
        "complete_rows": int(len(clean)),
        "team_count": int(len(teams)),
        "expected_team_count": expected_teams,
        "expected_matches": expected_matches,
        "duplicate_directed_fixtures": duplicate_directed,
        "min_matches_per_team": int(per_team.min()) if len(per_team) else 0,
        "max_matches_per_team": int(per_team.max()) if len(per_team) else 0,
        "unordered_pair_count": int(len(unordered)),
        "pairs_with_wrong_count": int((unordered != 2).sum()) if len(unordered) else 0,
        "missing_core_values": int(len(frame) - len(clean)),
        "passed": bool(passed),
    }


def audit_target_statsbomb_completeness(config: MD1Config, target: pd.DataFrame) -> pd.DataFrame:
    fd_shape = pd.DataFrame({
        "Date": pd.to_datetime(target["kickoff"]).dt.normalize(),
        "HomeTeam": target["home_team_name"],
        "AwayTeam": target["away_team_name"],
        "FTHG": target["home_score"],
        "FTAG": target["away_score"],
    })
    row = _audit_complete_season(fd_shape, config.season_name, provider="statsbomb_target", config=config)
    out = pd.DataFrame([row])
    save_table(out, config.audit_root / "p1_target_season_completeness.csv")
    if config.p1_require_complete_seasons and not bool(out["passed"].all()):
        raise AssertionError("StatsBomb EPL target season failed completeness audit:\n" + out.to_string(index=False))
    return out


def build_cross_source_team_mapping(
    config: MD1Config,
    historical_raw: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Resolve provider names to stable longitudinal team IDs without fuzzy matching."""
    target_rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        tmp = target[[f"{side}_team_id", f"{side}_team_name"]].drop_duplicates().copy()
        tmp.columns = ["statsbomb_team_id", "statsbomb_team_name"]
        target_rows.extend(tmp.to_dict("records"))
    target_teams = pd.DataFrame(target_rows).drop_duplicates("statsbomb_team_id")
    target_teams["canonical_team_key"] = target_teams["statsbomb_team_name"].map(_canonical_fd_name)
    if target_teams["canonical_team_key"].duplicated().any():
        raise AssertionError("Two StatsBomb target teams collapse to the same canonical Football-Data key")
    canonical_to_id = {
        str(r["canonical_team_key"]): int(r["statsbomb_team_id"])
        for r in target_teams.to_dict("records")
    }

    fd_names = sorted(set(historical_raw["HomeTeam"].astype(str)).union(set(historical_raw["AwayTeam"].astype(str))))
    fd_keys = sorted({_canonical_fd_name(n) for n in fd_names})
    next_negative = -1
    for key in fd_keys:
        if key not in canonical_to_id:
            canonical_to_id[key] = next_negative
            next_negative -= 1

    rows: list[dict[str, Any]] = []
    target_by_key = target_teams.set_index("canonical_team_key").to_dict("index")
    for name in fd_names:
        key = _canonical_fd_name(name)
        target_info = target_by_key.get(key, {})
        rows.append({
            "provider": "football_data_history",
            "provider_team_name": name,
            "provider_normalized": normalize_team_name(name),
            "canonical_team_key": key,
            "longitudinal_team_id": int(canonical_to_id[key]),
            "statsbomb_team_id": target_info.get("statsbomb_team_id", np.nan),
            "statsbomb_team_name": target_info.get("statsbomb_team_name", ""),
            "mapping_method": "explicit_or_normalized_alias" if key != normalize_team_name(name) else "normalized_exact",
            "is_target_season_team": bool(key in target_by_key),
            "passed": bool(key),
        })
    for r in target_teams.to_dict("records"):
        key = str(r["canonical_team_key"])
        rows.append({
            "provider": "statsbomb_target",
            "provider_team_name": r["statsbomb_team_name"],
            "provider_normalized": normalize_team_name(r["statsbomb_team_name"]),
            "canonical_team_key": key,
            "longitudinal_team_id": int(r["statsbomb_team_id"]),
            "statsbomb_team_id": int(r["statsbomb_team_id"]),
            "statsbomb_team_name": r["statsbomb_team_name"],
            "mapping_method": "statsbomb_id_plus_explicit_alias" if key != normalize_team_name(r["statsbomb_team_name"]) else "statsbomb_id_normalized_exact",
            "is_target_season_team": True,
            "passed": True,
        })
    audit = pd.DataFrame(rows).sort_values(["provider", "canonical_team_key", "provider_team_name"]).reset_index(drop=True)
    # Every target key must map to its actual StatsBomb team ID; no target key may be ambiguous.
    target_fd = audit[(audit["provider"] == "football_data_history") & audit["is_target_season_team"]]
    if target_fd.groupby("canonical_team_key")["longitudinal_team_id"].nunique().max() > 1:
        raise AssertionError("Ambiguous Football-Data -> StatsBomb team mapping detected")
    save_table(audit, config.audit_root / "p1_team_identity_audit.csv")
    return audit, canonical_to_id


def canonicalize_historical_matches(
    config: MD1Config,
    historical_raw: pd.DataFrame,
    canonical_to_id: Mapping[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    season_order = {s: i for i, s in enumerate(configured_history_seasons(config), start=1)}
    for season_name, group in historical_raw.groupby("season_name", sort=False):
        ordered = group.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
        for row_idx, r in enumerate(ordered.to_dict("records"), start=1):
            date = pd.Timestamp(r["Date"])
            home_key = _canonical_fd_name(r["HomeTeam"])
            away_key = _canonical_fd_name(r["AwayTeam"])
            hs, as_ = int(r["FTHG"]), int(r["FTAG"])
            synthetic_match_id = -(season_order[str(season_name)] * 1000 + row_idx)
            rows.append({
                "match_id": synthetic_match_id,
                "competition_id": int(config.competition_id),
                "competition_name": str(config.competition_name),
                "season_id": int(str(season_name).split("/")[0]),
                "season_name": str(season_name),
                "match_date": date.normalize(),
                "kick_off": None,
                # Football-Data historical files provide date but not reliable kickoff
                # time for these old seasons.  End-of-day finish makes same-day history
                # conservative.  These rows predate the target season and are never
                # supervised examples in the final project.
                "kickoff": date.normalize(),
                "estimated_finish": date.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59),
                "home_team_id": int(canonical_to_id[home_key]),
                "home_team_name": str(r["HomeTeam"]),
                "away_team_id": int(canonical_to_id[away_key]),
                "away_team_name": str(r["AwayTeam"]),
                "home_score": hs,
                "away_score": as_,
                "outcome": "H" if hs > as_ else "A" if hs < as_ else "D",
                "goal_margin_raw": hs - as_,
                "goal_margin_clipped": int(np.clip(hs - as_, -5, 5)),
                "match_week": np.nan,
                "source_provider": "football_data_history",
                "source_role": "p1_historical_context_only",
                "source_file": str(r.get("source_file", "")),
                "home_team_key": home_key,
                "away_team_key": away_key,
            })
    out = pd.DataFrame(rows).sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    return out


def canonicalize_target_matches(
    target: pd.DataFrame,
) -> pd.DataFrame:
    out = target.copy()
    out["home_team_key"] = out["home_team_name"].map(_canonical_fd_name)
    out["away_team_key"] = out["away_team_name"].map(_canonical_fd_name)
    out["source_provider"] = "statsbomb_target"
    out["source_role"] = "primary_modeling_target_and_current_season_history"
    out["source_file"] = "StatsBomb matches JSON"
    return out


def create_synthetic_hybrid_data(config: MD1Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Production-shaped synthetic hybrid fixture for full smoke testing."""
    names = {
        101: "Alpha FC", 102: "Bravo FC", 103: "Charlie FC",
        104: "Delta FC", 105: "Echo FC", 106: "Foxtrot FC", 107: "Golf FC",
    }
    historical_members = {
        "2012/2013": (101, 102, 103, 104, 105, 106),
        "2013/2014": (101, 102, 103, 104, 105, 107),  # 106 gap
        "2014/2015": (101, 102, 103, 104, 105, 106),  # 106 returns
    }
    target_members = (101, 102, 103, 104, 105, 106)
    hist_rows: list[dict[str, Any]] = []
    for season_index, (season_name, members) in enumerate(historical_members.items()):
        start = pd.Timestamp(f"{2012 + season_index}-08-01")
        for fixture_index, (h, a) in enumerate(_round_robin_pairs(members)):
            d = start + pd.Timedelta(days=fixture_index * 3)
            hs = int((h + fixture_index + season_index) % 4)
            as_ = int((a + 2 * fixture_index + season_index) % 3)
            hist_rows.append({
                "Date": d,
                "HomeTeam": names[h],
                "AwayTeam": names[a],
                "FTHG": hs,
                "FTAG": as_,
                "season_name": season_name,
                "source_file": f"synthetic_{season_name}.csv",
            })
    historical_raw = pd.DataFrame(hist_rows)

    target_rows: list[dict[str, Any]] = []
    start = pd.Timestamp("2015-08-08 15:00:00")
    match_id = 800000
    for fixture_index, (h, a) in enumerate(_round_robin_pairs(target_members)):
        kickoff = start + pd.Timedelta(days=fixture_index * 3)
        hs = int((h + fixture_index + 4) % 4)
        as_ = int((a + 3 * fixture_index + 2) % 3)
        target_rows.append({
            "match_id": match_id,
            "competition_id": config.competition_id,
            "competition_name": config.competition_name,
            "season_id": config.season_id,
            "season_name": config.season_name,
            "match_date": kickoff.normalize(),
            "kick_off": kickoff.strftime("%H:%M:%S"),
            "kickoff": kickoff,
            "estimated_finish": kickoff + pd.Timedelta(hours=2),
            "home_team_id": h,
            "home_team_name": names[h],
            "away_team_id": a,
            "away_team_name": names[a],
            "home_score": hs,
            "away_score": as_,
            "outcome": "H" if hs > as_ else "A" if hs < as_ else "D",
            "goal_margin_raw": hs - as_,
            "goal_margin_clipped": int(np.clip(hs - as_, -5, 5)),
            "match_week": fixture_index + 1,
            "source_provider": "statsbomb_target",
            "source_role": "primary_modeling_target_and_current_season_history",
        })
        match_id += 1
    target = pd.DataFrame(target_rows)

    # Synthetic uses its actual 6-team round-robin expectations, not real EPL 20/380.
    hist_audits = []
    for season_name, group in historical_raw.groupby("season_name", sort=False):
        fake = config.p1_expected_team_count, config.p1_expected_matches_per_season
        config.p1_expected_team_count, config.p1_expected_matches_per_season = 6, 30
        try:
            hist_audits.append(_audit_complete_season(group, season_name, provider="football_data_history", config=config))
        finally:
            config.p1_expected_team_count, config.p1_expected_matches_per_season = fake
    history_audit = pd.DataFrame(hist_audits)
    fake = config.p1_expected_team_count, config.p1_expected_matches_per_season
    config.p1_expected_team_count, config.p1_expected_matches_per_season = 6, 30
    try:
        target_shape = pd.DataFrame({
            "Date": target["match_date"], "HomeTeam": target["home_team_name"], "AwayTeam": target["away_team_name"],
            "FTHG": target["home_score"], "FTAG": target["away_score"],
        })
        target_audit = pd.DataFrame([_audit_complete_season(target_shape, config.season_name, provider="statsbomb_target", config=config)])
    finally:
        config.p1_expected_team_count, config.p1_expected_matches_per_season = fake
    return historical_raw, target, history_audit, target_audit


def build_source_role_audit(config: MD1Config) -> pd.DataFrame:
    rows = [
        {
            "source": "StatsBomb EPL 2015/16",
            "role": "primary modeling/target season",
            "admitted_fields": "match metadata, labels, events/lineups in baseline; completed target-season results as causal P1 history",
            "model_examples": True,
            "p1_historical_context": True,
            "market_baseline": False,
            "approval": "project-required primary source",
        },
        {
            "source": "Football-Data EPL prior seasons",
            "role": "Berrar historical context only",
            "admitted_fields": ", ".join(HISTORICAL_ALLOWED_COLUMNS),
            "model_examples": False,
            "p1_historical_context": True,
            "market_baseline": False,
            "approval": config.p1_ta_approval_note,
        },
        {
            "source": "Football-Data EPL 2015/16 odds",
            "role": "independent market benchmark",
            "admitted_fields": "date, teams, supported 1X2 odds",
            "model_examples": False,
            "p1_historical_context": False,
            "market_baseline": True,
            "approval": "project-required second source",
        },
    ]
    audit = pd.DataFrame(rows)
    save_table(audit, config.audit_root / "p1_source_roles.csv")
    write_json({"ta_approval_note": config.p1_ta_approval_note, "roles": rows}, config.audit_root / "p1_source_roles.json")
    return audit


def prepare_hybrid_p1_data(config: MD1Config) -> dict[str, Any]:
    """Prepare complete historical context plus StatsBomb target rows."""
    config.ensure_directories()
    if config.data_mode == "synthetic_demo":
        historical_raw, target, history_audit, target_audit = create_synthetic_hybrid_data(config)
    else:
        download_historical_football_data(config)
        historical_raw, history_audit = parse_historical_football_data(config)
        target = parse_target_statsbomb_matches(config)
        target_audit = audit_target_statsbomb_completeness(config, target)

    team_identity, canonical_to_id = build_cross_source_team_mapping(config, historical_raw, target)
    historical = canonicalize_historical_matches(config, historical_raw, canonical_to_id)
    target = canonicalize_target_matches(target)

    target_start = pd.to_datetime(target["kickoff"]).min()
    if pd.to_datetime(historical["estimated_finish"]).max() >= target_start:
        raise AssertionError("Football-Data historical context overlaps the StatsBomb target season")
    if str(config.season_name) in set(historical["season_name"].astype(str)):
        raise AssertionError("Target-season Football-Data results entered the Berrar historical branch")

    combined = pd.concat([historical, target], ignore_index=True, sort=False)
    combined = combined.sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    if combined["match_id"].duplicated().any():
        raise AssertionError("Combined Berrar history has duplicate match IDs")

    save_table(historical, config.silver_root / "p1_historical_epl_matches.parquet")
    save_table(target, config.silver_root / "p1_target_statsbomb_matches.parquet")
    save_table(combined, config.silver_root / "p1_super_league_matches.parquet")
    save_table(history_audit, config.audit_root / "p1_season_completeness.csv")
    save_table(target_audit, config.audit_root / "p1_target_season_completeness.csv")
    source_roles = build_source_role_audit(config)
    return {
        "historical_raw": historical_raw,
        "historical_matches": historical,
        "target_matches": target,
        "combined_matches": combined,
        "history_season_audit": history_audit,
        "target_season_audit": target_audit,
        "team_identity": team_identity,
        "source_roles": source_roles,
    }


def build_hybrid_feature_dictionary(config: MD1Config, selected_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in PAPER_HOMEAWAY_FEATURES:
        metric = "scored" if "_scr_" in col else "conceded" if "_con_" in col else "normalized_rank"
        view = "total" if col.endswith("_total") else "home" if col.endswith("_home") else "away"
        rows.append({
            "table": "p1_features_homeaway_selected",
            "column": col,
            "role": "feature",
            "source": "Football-Data pre-2015/16 results + prior completed StatsBomb EPL 2015/16 matches",
            "model_eligible": True,
            "prediction_time": "strictly pre-match",
            "leakage_guard": "all contributing matches finished strictly before target kickoff; target/future perturbation tests",
            "feature_family": PAPER_ID,
            "paper_alignment": "paper_aligned_representation_with_TA_approved_source_adaptation",
            "history_view": view,
            "recency_n": int(selected_n if view == "total" else _odd_half_n(selected_n, config.p1_odd_n_policy)),
            "aggregation": "mean" if metric != "normalized_rank" else "causal recency league-table rank normalized to [0,1]",
            "metric": metric,
        })
    out = pd.DataFrame(rows)
    save_table(out, config.audit_root / "p1_feature_dictionary.csv")
    return out


def build_hybrid_fidelity_audits(config: MD1Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = [
        ("feature source", "basic complete match records", "Football-Data complete EPL results before 2015/16 + StatsBomb completed target-season matches", "project_required_adaptation", "TA explicitly approved Football-Data historical results for Berrar history"),
        ("super league", "all seasons of one league concatenated chronologically", "EPL 2000/01--2014/15 history + causal EPL 2015/16 StatsBomb continuation", "exact", "same-league continuity preserved"),
        ("modeling examples", "paper uses generated league datasets", "final supervised examples remain StatsBomb EPL 2015/16 only", "project_required_adaptation", "keeps StatsBomb as project primary modeling source"),
        ("season boundaries", "team histories continue", "same", "exact", ""),
        ("team-history gaps", "same-league gaps ignored", "same; no lower-division matches inserted", "exact", ""),
        ("minimum prior history", "6 prior matches per team", str(config.p1_min_prior_matches), "exact" if config.p1_min_prior_matches == 6 else "faithful_with_clarification", ""),
        ("aggregation", "mean", "mean", "exact", ""),
        ("performance categories", "goals scored, conceded, league success", "same", "exact", ""),
        ("total feature set", "6", "6", "exact", ""),
        ("homeaway feature set", "18", "18", "exact", ""),
        ("normalized rank", "(N-R)/(N-1)", "same, with target-season current membership and causal recent histories", "faithful_with_clarification", "paper leaves some operational details implicit"),
        ("n range", "9..100", f"{config.p1_recency_min}..{config.p1_recency_max}", "exact" if (config.p1_recency_min, config.p1_recency_max) == (9, 100) else "faithful_with_clarification", "synthetic tests deliberately use a smaller range"),
        ("Pearson recency", "corr(Δavg GD + Δavg scored, observed goal difference)", "same criterion on StatsBomb training rows only", "project_required_adaptation", "validation/test labels cannot select n"),
        ("outer split", "paper commonly used random model splits and acknowledged leakage risk", "same chronological match split as MD1 baseline", "project_required_adaptation", "course leakage rule"),
        ("Task R", "exact home/away score", "signed clipped goal margin downstream", "project_required_adaptation", "course task definition"),
        ("Task L", "not defined", "frozen Berrar pre-match vector can be inherited by live snapshots", "project_specific_extension", ""),
        ("bookmaker odds", "independent reference", "Football-Data odds remain separate and never enter P1 history", "project_required_adaptation", "course market-baseline requirement"),
    ]
    out = pd.DataFrame(rows, columns=["paper_component", "paper_method", "our_implementation", "status", "reason_for_adaptation"])
    save_table(out, config.audit_root / "p1_fidelity_table.csv")
    decisions = {
        "data_source_exception": {
            "status": "TA_approved",
            "decision": "Use complete Football-Data EPL results only for Berrar historical context; keep StatsBomb EPL 2015/16 as target/modeling season.",
            "approval_note": config.p1_ta_approval_note,
        },
        "historical_interval": {
            "value": f"{config.p1_history_start_season} through {config.p1_history_end_season}",
            "reason": "Complete consecutive pre-target EPL seasons, sufficient for n<=100 and fixed before final evaluation.",
        },
        "odd_n_policy": {
            "value": config.p1_odd_n_policy,
            "status": "paper_aligned_with_documented_ambiguity",
            "reason": "Paper conceptually uses n/2 venue histories while evaluating every integer n=9..100.",
        },
        "historical_match_time": {
            "value": "Football-Data old results treated as complete at end-of-day",
            "status": "conservative_temporal_rule",
            "reason": "Old CSV history is used only before the 2015/16 target season; no target feature can use a same-day historical row.",
        },
    }
    write_json(decisions, config.audit_root / "p1_fidelity_decisions.json")
    return out, decisions


def run_hybrid_p1_leakage_tests(
    config: MD1Config,
    *,
    historical_matches: pd.DataFrame,
    target_matches: pd.DataFrame,
    combined_matches: pd.DataFrame,
    team_facts: pd.DataFrame,
    selected_total: pd.DataFrame,
    selected_homeaway: pd.DataFrame,
    all_n_total: pd.DataFrame,
    selected_n: int,
    current_team_sets: Mapping[tuple[int, int], Sequence[int]],
    split_map: Mapping[int, str],
) -> pd.DataFrame:
    tests: list[tuple[str, Any]] = []

    def rebuild(mut_combined: pd.DataFrame, mut_target: pd.DataFrame | None = None, *, homeaway: bool = False) -> pd.DataFrame:
        tgt = target_matches if mut_target is None else mut_target
        facts = build_p1_team_match_facts(mut_combined)
        idx = P1HistoryIndex(facts)
        members = build_p1_current_team_sets(tgt)
        return build_p1_features_for_n(config, tgt, idx, members, split_map, selected_n, include_homeaway=homeaway)

    def target_perturbation() -> str:
        target_id = int(selected_total[selected_total["p1_eligible"]].iloc[len(selected_total[selected_total["p1_eligible"]]) // 2]["match_id"])
        before = selected_total.set_index("match_id").loc[target_id]
        mut_t = target_matches.copy()
        mut_c = combined_matches.copy()
        for df in (mut_t, mut_c):
            mask = df["match_id"] == target_id
            df.loc[mask, ["home_score", "away_score", "goal_margin_raw", "goal_margin_clipped"]] = [999, 998, 1, 1]
            df.loc[mask, "outcome"] = "H"
        after = rebuild(mut_c, mut_t).set_index("match_id").loc[target_id]
        if not _feature_values_equal(before, after, PAPER_TOTAL_FEATURES):
            raise AssertionError("Target result changed its own Berrar features")
        return f"target {target_id} unchanged"

    def future_perturbation() -> str:
        eligible = selected_total[selected_total["p1_eligible"]].sort_values("kickoff")
        target = eligible.iloc[len(eligible) // 3]
        later = target_matches[pd.to_datetime(target_matches["kickoff"]) > pd.Timestamp(target["kickoff"])].sort_values("kickoff")
        future_id = int(later.iloc[-1]["match_id"])
        mut_t = target_matches.copy(); mut_c = combined_matches.copy()
        for df in (mut_t, mut_c):
            mask = df["match_id"] == future_id
            df.loc[mask, ["home_score", "away_score", "goal_margin_raw", "goal_margin_clipped"]] = [99, 0, 99, 5]
            df.loc[mask, "outcome"] = "H"
        before = selected_total.set_index("match_id").loc[int(target["match_id"])]
        after = rebuild(mut_c, mut_t).set_index("match_id").loc[int(target["match_id"])]
        if not _feature_values_equal(before, after, PAPER_TOTAL_FEATURES):
            raise AssertionError("Future target result changed earlier features")
        return f"future {future_id} did not alter earlier {int(target['match_id'])}"

    def historical_positive_control() -> str:
        # Find an early target row with a recent Football-Data source match for one of its teams.
        facts = team_facts[team_facts["match_id"] < 0].copy()
        for target in target_matches.sort_values("kickoff").to_dict("records"):
            for side in ("home", "away"):
                tid = int(target[f"{side}_team_id"])
                prior = facts[(facts["team_id"] == tid) & (pd.to_datetime(facts["match_finished_at"]) < pd.Timestamp(target["kickoff"]))].sort_values("match_finished_at")
                if prior.empty:
                    continue
                hist_match_id = int(prior.iloc[-1]["match_id"])
                before = selected_total.set_index("match_id").loc[int(target["match_id"])]
                mut_c = combined_matches.copy()
                mask = mut_c["match_id"] == hist_match_id
                mut_c.loc[mask, "home_score"] = mut_c.loc[mask, "home_score"].astype(int) + 7
                mut_c.loc[mask, "goal_margin_raw"] = mut_c.loc[mask, "home_score"].astype(int) - mut_c.loc[mask, "away_score"].astype(int)
                mut_c.loc[mask, "goal_margin_clipped"] = mut_c.loc[mask, "goal_margin_raw"].clip(-5, 5)
                mut_c.loc[mask, "outcome"] = np.where(mut_c.loc[mask, "goal_margin_raw"] > 0, "H", np.where(mut_c.loc[mask, "goal_margin_raw"] < 0, "A", "D"))
                after = rebuild(mut_c).set_index("match_id").loc[int(target["match_id"])]
                if _feature_values_equal(before, after, PAPER_TOTAL_FEATURES):
                    continue
                return f"allowed old history match {hist_match_id} changes later target {int(target['match_id'])} (positive control)"
        raise AssertionError("Could not demonstrate that approved historical results contribute to a target feature")

    def source_separation() -> str:
        if str(config.season_name) in set(historical_matches["season_name"].astype(str)):
            raise AssertionError("Football-Data target-season results entered P1 history")
        if set(target_matches["source_provider"].astype(str)) != {"statsbomb_target"}:
            raise AssertionError("Non-StatsBomb rows entered target supervised examples")
        if set(historical_matches["source_provider"].astype(str)) != {"football_data_history"}:
            raise AssertionError("Historical context contains an unexpected provider")
        return "pre-target Football-Data history and StatsBomb target rows are isolated"

    def forbidden_history_fields() -> str:
        cols = [str(c).lower() for c in historical_matches.columns]
        bad = [c for c in cols if any(tok in c for tok in FORBIDDEN_HISTORY_COLUMN_TOKENS)]
        if bad:
            raise AssertionError(f"Forbidden Football-Data statistics/odds leaked into canonical history columns: {bad}")
        return "canonical P1 history carries no odds/shots/cards/corners fields"

    def season_boundary() -> str:
        order = sorted(combined_matches["season_name"].astype(str).unique(), key=lambda s: int(str(s).split("/")[0]))
        pos = {s: i for i, s in enumerate(order)}
        for tid, group in team_facts.groupby("team_id"):
            seasons = sorted(group["season_name"].astype(str).unique(), key=lambda s: pos[s])
            for a, b in zip(seasons, seasons[1:]):
                if pos[b] == pos[a] + 1:
                    first = group[group["season_name"].astype(str) == b].sort_values("kickoff").iloc[0]
                    older = group[(pd.to_datetime(group["match_finished_at"]) < pd.Timestamp(first["kickoff"])) & (group["season_name"].astype(str) != b)]
                    if not older.empty:
                        return f"team {int(tid)} carries history {a}->{b}"
        raise AssertionError("No cross-season continuity case found")

    def gap_continuity() -> str:
        order = sorted(combined_matches["season_name"].astype(str).unique(), key=lambda s: int(str(s).split("/")[0]))
        pos = {s: i for i, s in enumerate(order)}
        for tid, group in team_facts.groupby("team_id"):
            seasons = sorted(group["season_name"].astype(str).unique(), key=lambda s: pos[s])
            for a, b in zip(seasons, seasons[1:]):
                if pos[b] - pos[a] > 1:
                    first = group[group["season_name"].astype(str) == b].sort_values("kickoff").iloc[0]
                    older = group[pd.to_datetime(group["match_finished_at"]) < pd.Timestamp(first["kickoff"])]
                    if not older.empty:
                        return f"team {int(tid)} retains same-league history across gap {a}->{b}"
        if config.data_mode == "real":
            return "no qualifying target-team gap found in real interval; structural gap behavior passed synthetic smoke test"
        raise AssertionError("Synthetic hybrid fixture must contain a participation gap")

    def minimum_rule() -> str:
        expected = (
            (selected_total["home_total_history_available"] >= config.p1_min_prior_matches)
            & (selected_total["away_total_history_available"] >= config.p1_min_prior_matches)
        )
        if not np.array_equal(expected.to_numpy(bool), selected_total["p1_eligible"].to_numpy(bool)):
            raise AssertionError("P1 eligibility does not implement the minimum-history rule")
        return f">={config.p1_min_prior_matches} prior total matches per team"

    def n_isolation() -> str:
        before, _ = select_p1_recency_pearson(config, all_n_total)
        mutated = all_n_total.copy()
        mutated.loc[mutated["split"] != "train", "goal_margin_raw"] = 9999
        after, _ = select_p1_recency_pearson(config, mutated)
        if before != after or before != selected_n:
            raise AssertionError("Validation/test labels influenced selected n")
        return f"selected n={selected_n} unchanged after non-train label mutation"

    def rank_temporal_safety() -> str:
        eligible = selected_total[selected_total["p1_eligible"]].sort_values("kickoff")
        target = eligible.iloc[max(0, len(eligible) // 4)]
        later = target_matches[pd.to_datetime(target_matches["kickoff"]) > pd.Timestamp(target["kickoff"])].sort_values("kickoff")
        fid = int(later.iloc[-1]["match_id"])
        mut_t = target_matches.copy(); mut_c = combined_matches.copy()
        for df in (mut_t, mut_c):
            mask = df["match_id"] == fid
            df.loc[mask, ["home_score", "away_score", "goal_margin_raw", "goal_margin_clipped"]] = [99, 0, 99, 5]
            df.loc[mask, "outcome"] = "H"
        before = selected_total.set_index("match_id").loc[int(target["match_id"])]
        after = rebuild(mut_c, mut_t).set_index("match_id").loc[int(target["match_id"])]
        if not _feature_values_equal(before, after, ["p1_home_rank_total", "p1_away_rank_total"]):
            raise AssertionError("Future result changed earlier normalized rank")
        return "future target mutation leaves earlier ranks unchanged"

    def rank_population() -> str:
        historical_ids = set(combined_matches["home_team_id"]).union(set(combined_matches["away_team_id"]))
        target_ids = set(target_matches["home_team_id"]).union(set(target_matches["away_team_id"]))
        members = next(iter(current_team_sets.values()))
        if set(members) != target_ids:
            raise AssertionError("Rank population is not the target-season membership")
        if len(historical_ids) > len(target_ids) and set(members) == historical_ids:
            raise AssertionError("Rank population incorrectly contains all historical clubs")
        return f"rank population={len(members)} target-season clubs; combined history has {len(historical_ids)} clubs"

    def feature_counts() -> str:
        if len([c for c in PAPER_TOTAL_FEATURES if c in selected_total]) != 6:
            raise AssertionError("Total feature contract != 6")
        if len([c for c in PAPER_HOMEAWAY_FEATURES if c in selected_homeaway]) != 18:
            raise AssertionError("Homeaway feature contract != 18")
        return "exact 6/18 paper feature contracts"

    tests.extend([
        ("pre-match source time contract", lambda: (assert_no_p1_prematch_leakage(selected_homeaway), "all source finishes precede target kickoff")[1]),
        ("target-result perturbation", target_perturbation),
        ("future-result perturbation", future_perturbation),
        ("historical positive control", historical_positive_control),
        ("source-role separation", source_separation),
        ("forbidden historical fields", forbidden_history_fields),
        ("season-boundary continuity", season_boundary),
        ("gap continuity", gap_continuity),
        ("minimum-six rule", minimum_rule),
        ("selected-n isolation", n_isolation),
        ("league-rank temporal safety", rank_temporal_safety),
        ("current-season rank population", rank_population),
        ("feature-count contracts", feature_counts),
    ])
    rows: list[dict[str, Any]] = []
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            details = fn()
            rows.append({"test": name, "passed": True, "details": str(details), "seconds": time.perf_counter() - t0})
        except Exception as exc:
            rows.append({"test": name, "passed": False, "details": repr(exc), "seconds": time.perf_counter() - t0})
    out = pd.DataFrame(rows)
    save_table(out, config.audit_root / "p1_leakage_tests.csv")
    if not bool(out["passed"].all()):
        raise AssertionError("Hybrid P1 leakage/integrity tests failed:\n" + out.loc[~out["passed"], ["test", "details"]].to_string(index=False))
    return out


def build_hybrid_readiness(
    config: MD1Config,
    *,
    data: Mapping[str, Any],
    selected_total: pd.DataFrame,
    selected_homeaway: pd.DataFrame,
    selected_n: int,
    leakage_tests: pd.DataFrame,
    fidelity: pd.DataFrame,
) -> pd.DataFrame:
    hist_audit = data["history_season_audit"]
    target_audit = data["target_season_audit"]
    target = data["target_matches"]
    history = data["historical_matches"]
    team_identity = data["team_identity"]
    def add(name: str, passed: bool, details: str) -> dict[str, Any]:
        return {"check": name, "passed": bool(passed), "details": details}
    rows = [
        add("TA-approved source design", config.p1_history_provider == "football_data" and bool(config.p1_ta_approval_note), config.p1_ta_approval_note),
        add("historical season completeness", bool(hist_audit["passed"].all()), f"{int(hist_audit['passed'].sum())}/{len(hist_audit)} historical seasons complete"),
        add("StatsBomb target completeness", bool(target_audit["passed"].all()), f"target rows={len(target)}"),
        add("target examples are StatsBomb only", set(target["source_provider"].astype(str)) == {"statsbomb_target"}, "all supervised P1 target rows are StatsBomb"),
        add("Football-Data target results excluded", str(config.season_name) not in set(history["season_name"].astype(str)), f"history ends {config.p1_history_end_season}"),
        add("cross-source target identities", bool(team_identity["passed"].all()), f"{int(team_identity['is_target_season_team'].sum())} mapping rows refer to target clubs"),
        add("chronological target split", bool(pd.to_datetime(target["kickoff"]).is_monotonic_increasing), "StatsBomb target matches sorted chronologically"),
        add("6-feature total contract", len([c for c in PAPER_TOTAL_FEATURES if c in selected_total]) == 6, "exactly six"),
        add("18-feature homeaway contract", len([c for c in PAPER_HOMEAWAY_FEATURES if c in selected_homeaway]) == 18, "exactly eighteen"),
        add("normalized rank bounded", bool(selected_homeaway[[c for c in PAPER_HOMEAWAY_FEATURES if "rank" in c]].apply(lambda s: s.dropna().between(0,1).all()).all()), "all observed ranks in [0,1]"),
        add("selected n valid", config.p1_recency_min <= selected_n <= config.p1_recency_max, f"selected_n={selected_n}"),
        add("minimum-history eligibility", bool((((selected_total['home_total_history_available'] >= config.p1_min_prior_matches) & (selected_total['away_total_history_available'] >= config.p1_min_prior_matches)) == selected_total['p1_eligible']).all()), f"minimum={config.p1_min_prior_matches}"),
        add("leakage/source tests", bool(leakage_tests["passed"].all()), f"{int(leakage_tests['passed'].sum())}/{len(leakage_tests)} passed"),
        add("fidelity audit", not fidelity.empty and (config.audit_root / "p1_fidelity_table.csv").exists(), f"{len(fidelity)} rows"),
        add("common comparison population", all(int(((selected_total["split"] == s) & selected_total["p1_eligible"]).sum()) > 0 for s in ("train","validation","test")), "eligible rows in train/validation/test"),
    ]
    out = pd.DataFrame(rows)
    save_table(out, config.audit_root / "p1_readiness.csv")
    return out


def run_hybrid_p1_pipeline(config: MD1Config, *, run_knn: bool | None = None) -> dict[str, Any]:
    """Execute the final TA-approved Berrar P1 design."""
    if not config.p1_enabled:
        raise ValueError("P1 workflow requested but p1_enabled=False")
    if config.p1_history_provider != "football_data":
        raise ValueError("Final approved P1 workflow requires p1_history_provider='football_data'")
    config.ensure_directories()
    t0 = time.perf_counter()
    data = prepare_hybrid_p1_data(config)
    target = data["target_matches"]
    combined = data["combined_matches"]

    team_facts = build_p1_team_match_facts(combined)
    # Preserve provider role on fact rows for audits/positive controls.
    provider_by_match = combined.set_index("match_id")["source_provider"].to_dict()
    team_facts["source_provider"] = team_facts["match_id"].map(provider_by_match)
    save_table(team_facts, config.silver_root / "p1_team_match_facts.parquet")

    # Use the exact same split function as the verified MD1 baseline, but only on
    # the StatsBomb target season.  This is the strongest split-alignment contract.
    split_manifest = chronological_match_level_split(config, target)
    save_table(split_manifest, config.gold_root / "p1_split_manifest.csv")
    split_map = split_manifest.set_index("match_id")["split"].to_dict()
    current_team_sets = build_p1_current_team_sets(target)
    history_index = P1HistoryIndex(team_facts)

    candidate_frames: list[pd.DataFrame] = []
    candidate_root = config.processed_root / "p1_by_n"
    candidate_root.mkdir(parents=True, exist_ok=True)
    for n in range(int(config.p1_recency_min), int(config.p1_recency_max) + 1):
        frame = build_p1_features_for_n(
            config, target, history_index, current_team_sets, split_map, n, include_homeaway=False
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
        fixed_n = int(config.p1_recency_min)
        fixed_total = all_n_total[all_n_total["recency_n"] == fixed_n].copy().reset_index(drop=True)
    selected_homeaway = build_p1_features_for_n(
        config, target, history_index, current_team_sets, split_map, selected_n, include_homeaway=True
    )
    assert_no_p1_prematch_leakage(selected_homeaway)
    save_table(fixed_total, config.gold_root / "p1_features_total_fixed.parquet")
    save_table(selected_total, config.gold_root / "p1_features_total_selected.parquet")
    save_table(selected_homeaway, config.gold_root / "p1_features_homeaway_selected.parquet")

    coverage = build_p1_history_coverage(config, selected_homeaway, selected_n)
    common_population = build_common_population_audit(config, selected_homeaway)
    feature_dictionary = build_hybrid_feature_dictionary(config, selected_n)
    fidelity, fidelity_decisions = build_hybrid_fidelity_audits(config)

    if run_knn is None:
        run_knn = bool(config.p1_run_knn_search)
    knn_result = knn_audit = None
    if run_knn:
        knn_result, knn_audit = select_p1_recency_knn(config, all_n_total, mode=config.p1_knn_mode, task="classification")

    leakage_tests = run_hybrid_p1_leakage_tests(
        config,
        historical_matches=data["historical_matches"],
        target_matches=target,
        combined_matches=combined,
        team_facts=team_facts,
        selected_total=selected_total,
        selected_homeaway=selected_homeaway,
        all_n_total=all_n_total,
        selected_n=selected_n,
        current_team_sets=current_team_sets,
        split_map=split_map,
    )
    readiness = build_hybrid_readiness(
        config,
        data=data,
        selected_total=selected_total,
        selected_homeaway=selected_homeaway,
        selected_n=selected_n,
        leakage_tests=leakage_tests,
        fidelity=fidelity,
    )
    pearson_plot = _plot_pearson(config, pearson_audit, selected_n)

    eligible = selected_total[selected_total["p1_eligible"]]
    split_counts = eligible["split"].value_counts().to_dict()
    summary = {
        "paper": "Berrar, Lopes & Dubitzky (2024)",
        "design": "TA-approved Football-Data history + StatsBomb EPL 2015/16 target",
        "data_mode": config.data_mode,
        "history_provider": "football_data",
        "history_seasons": configured_history_seasons(config) if config.data_mode == "real" else sorted(data["historical_matches"]["season_name"].unique()),
        "history_season_count": int(data["historical_matches"]["season_name"].nunique()),
        "history_match_count": int(len(data["historical_matches"])),
        "target_provider": "statsbomb",
        "target_season": str(config.season_name),
        "target_match_count": int(len(target)),
        "combined_history_match_count": int(len(combined)),
        "eligible_target_matches": int(selected_total["p1_eligible"].sum()),
        "excluded_target_matches": int((~selected_total["p1_eligible"]).sum()),
        "selected_n_pearson": int(selected_n),
        "fixed_recency_baseline_n": int(fixed_n),
        "candidate_n_min": int(config.p1_recency_min),
        "candidate_n_max": int(config.p1_recency_max),
        "candidate_n_count": int(config.p1_recency_max - config.p1_recency_min + 1),
        "total_feature_count": 6,
        "homeaway_feature_count": 18,
        "eligible_train": int(split_counts.get("train", 0)),
        "eligible_validation": int(split_counts.get("validation", 0)),
        "eligible_test": int(split_counts.get("test", 0)),
        "leakage_tests_passed": int(leakage_tests["passed"].sum()),
        "leakage_tests_total": int(len(leakage_tests)),
        "readiness_passed": bool(readiness["passed"].all()),
        "knn_search": knn_result,
        "runtime_seconds": float(time.perf_counter() - t0),
        "ta_approval_note": config.p1_ta_approval_note,
    }
    write_json(summary, config.audit_root / "p1_run_summary.json")

    if not bool(readiness["passed"].all()):
        raise AssertionError("P1 RESULTS BLOCKED by readiness gate:\n" + readiness.loc[~readiness["passed"]].to_string(index=False))

    return {
        **data,
        "team_facts": team_facts,
        "split_manifest": split_manifest,
        "all_n_total": all_n_total,
        "selected_total": selected_total,
        "fixed_total": fixed_total,
        "selected_homeaway": selected_homeaway,
        "selected_n": selected_n,
        "pearson_audit": pearson_audit,
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
    }
