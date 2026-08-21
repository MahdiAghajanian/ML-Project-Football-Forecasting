"""Core Mid Defence 1 data pipeline implementation.

Extracted from the original Colab notebook without changing the pipeline algorithms.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import platform
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from unidecode import unidecode

@dataclass
class MD1Config:
    project_root: Path = Path("mid_defence1_output")
    data_mode: str = "synthetic_demo"  # synthetic_demo | real
    competition_id: int = 2
    season_id: int = 27
    competition_name: str = "Premier League"
    season_name: str = "2015/2016"
    football_data_code: str = "1516"
    football_data_division: str = "E0"
    max_matches: int | None = None
    date_tolerance_days: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    snapshot_minutes: tuple[int, ...] = tuple(range(0, 91, 5))
    rolling_windows: tuple[int, ...] = (3, 5, 10)
    overwrite_downloads: bool = False
    download_360: bool = True
    download_workers: int = 6
    random_seed: int = 42

    # Optional P1 (Berrar et al. 2024) branch. Defaults preserve the verified
    # Mid Defence 1 baseline because P1 is opt-in.
    p1_enabled: bool = False
    # P1 source design approved by the TA: StatsBomb EPL 2015/16 remains the
    # target/modeling season; complete Football-Data EPL results from prior
    # seasons provide only the long causal history required by Berrar.
    p1_history_provider: str = "football_data"
    p1_competition_name: str = "Premier League"
    p1_include_season_names: tuple[str, ...] = tuple(
        f"{year}/{year + 1}" for year in range(2000, 2015)
    )
    p1_history_start_season: str = "2000/2001"
    p1_history_end_season: str = "2014/2015"
    p1_history_division: str = "E0"
    p1_require_complete_seasons: bool = True
    p1_expected_team_count: int = 20
    p1_expected_matches_per_season: int = 380
    p1_target_from_statsbomb: bool = True
    p1_ta_approval_note: str = (
        "TA approved Football-Data historical Premier League results for constructing "
        "Berrar historical features; StatsBomb EPL 2015/16 remains the primary target/event dataset."
    )
    p1_recency_min: int = 9
    p1_recency_max: int = 100
    p1_min_prior_matches: int = 6
    p1_odd_n_policy: str = "floor"
    p1_primary_recency_method: str = "pearson"
    p1_feature_view: str = "both"
    p1_fixed_baseline_n: int = 10
    p1_cache_all_n: bool = True
    p1_run_knn_search: bool = False
    p1_knn_mode: str = "project_temporal"
    p1_knn_k_values: tuple[int, ...] = tuple(range(3, 351))
    p1_max_matches_per_season: int | None = None

    @property
    def bronze_root(self) -> Path:
        return self.project_root / "data" / "bronze"

    @property
    def statsbomb_root(self) -> Path:
        return self.bronze_root / "statsbomb"

    @property
    def football_data_root(self) -> Path:
        return self.bronze_root / "football_data"

    @property
    def silver_root(self) -> Path:
        return self.project_root / "data" / "silver"

    @property
    def gold_root(self) -> Path:
        return self.project_root / "data" / "gold"

    @property
    def audit_root(self) -> Path:
        return self.project_root / "audit"

    @property
    def processed_root(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def event_partitions_root(self) -> Path:
        return self.silver_root / "events"

    @property
    def frames_partitions_root(self) -> Path:
        return self.silver_root / "frames_360"

    def ensure_directories(self) -> None:
        for p in [
            self.project_root,
            self.bronze_root,
            self.statsbomb_root,
            self.football_data_root,
            self.silver_root,
            self.gold_root,
            self.processed_root,
            self.audit_root,
            self.event_partitions_root,
            self.frames_partitions_root,
        ]:
            p.mkdir(parents=True, exist_ok=True)

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Write parquet when requested and CSV otherwise; always create parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported table extension: {path.suffix}")
    return path


def nested_get(obj: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def parse_timestamp_seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value)
    try:
        hh, mm, ss = text.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)
    except Exception:
        return 0.0


def normalize_team_name(name: Any) -> str:
    if pd.isna(name):
        return ""
    text = unidecode(str(name).lower())
    text = text.replace("&", " and ")
    text = re.sub(r"\b(fc|afc|cf|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def de_vig_1x2(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    odds = np.array([home_odds, draw_odds, away_odds], dtype=float)
    if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
        return {"market_p_home": np.nan, "market_p_draw": np.nan, "market_p_away": np.nan}
    raw = 1.0 / odds
    fair = raw / raw.sum()
    return {
        "market_p_home": float(fair[0]),
        "market_p_draw": float(fair[1]),
        "market_p_away": float(fair[2]),
    }


def build_http_session() -> requests.Session:
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "MD1-football-forecasting-course-project/1.0"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


_download_thread_local = threading.local()


def get_thread_http_session() -> requests.Session:
    """Return one retry-configured requests session per download worker."""
    if not hasattr(_download_thread_local, "session"):
        _download_thread_local.session = build_http_session()
    return _download_thread_local.session


def download_file(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    optional: bool = False,
    overwrite: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return {
            "url": url,
            "local_path": str(path),
            "status": "reused",
            "http_status": 200,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=timeout) as response:
            if optional and response.status_code == 404:
                return {
                    "url": url,
                    "local_path": str(path),
                    "status": "optional_missing",
                    "http_status": 404,
                    "bytes": 0,
                    "sha256": "",
                }
            response.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        tmp.replace(path)
        return {
            "url": url,
            "local_path": str(path),
            "status": "downloaded",
            "http_status": 200,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        if optional:
            return {
                "url": url,
                "local_path": str(path),
                "status": "optional_error",
                "http_status": None,
                "bytes": 0,
                "sha256": "",
                "error": repr(exc),
            }
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 8)
# -----------------------------------------------------------------------------
def _event(
    event_id: str,
    index: int,
    period: int,
    minute: int,
    second: int,
    team_id: int,
    team_name: str,
    type_name: str,
    *,
    player_id: int | None = None,
    player_name: str | None = None,
    location: list[float] | None = None,
    possession: int | None = None,
    possession_team_id: int | None = None,
    possession_team_name: str | None = None,
    pass_data: dict[str, Any] | None = None,
    shot_data: dict[str, Any] | None = None,
    foul_committed: dict[str, Any] | None = None,
    bad_behaviour: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp_minute = minute if period == 1 else max(0, minute - 45)
    row: dict[str, Any] = {
        "id": event_id,
        "index": index,
        "period": period,
        "timestamp": f"00:{timestamp_minute:02d}:{second:02d}.000",
        "minute": minute,
        "second": second,
        "type": {"id": 0, "name": type_name},
        "team": {"id": team_id, "name": team_name},
        "possession": possession or index,
        "possession_team": {
            "id": possession_team_id or team_id,
            "name": possession_team_name or team_name,
        },
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "location": location or [60.0, 40.0],
    }
    if player_id is not None:
        row["player"] = {"id": player_id, "name": player_name or f"Player {player_id}"}
    if pass_data is not None:
        row["pass"] = pass_data
    if shot_data is not None:
        row["shot"] = shot_data
    if foul_committed is not None:
        row["foul_committed"] = foul_committed
    if bad_behaviour is not None:
        row["bad_behaviour"] = bad_behaviour
    return row


def create_synthetic_bronze(config: MD1Config, n_matches: int = 16) -> None:
    """Create a realistic, tiny StatsBomb-shaped fixture for offline execution/tests."""
    if config.project_root.exists():
        shutil.rmtree(config.project_root)
    config.ensure_directories()

    teams = [
        (1, "Arsenal"),
        (2, "Chelsea"),
        (3, "Manchester United"),
        (4, "Leicester City"),
    ]
    competitions = [
        {
            "competition_id": config.competition_id,
            "season_id": config.season_id,
            "country_name": "England",
            "competition_name": config.competition_name,
            "competition_gender": "male",
            "competition_youth": False,
            "competition_international": False,
            "season_name": config.season_name,
            "match_available_360": "synthetic",
            "match_available": "synthetic",
        }
    ]
    write_json(competitions, config.statsbomb_root / "competitions.json")

    fixtures: list[dict[str, Any]] = []
    base = pd.Timestamp("2015-08-08 15:00:00")
    pairings = [
        (0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2),
        (1, 0), (3, 2), (2, 0), (3, 1), (3, 0), (2, 1),
        (0, 1), (2, 3), (1, 2), (3, 0),
    ]
    score_patterns = [(2, 1), (1, 1), (0, 1), (3, 0), (1, 2), (0, 0), (2, 2), (1, 0)]

    for i in range(n_matches):
        home_idx, away_idx = pairings[i % len(pairings)]
        home_id, home_name = teams[home_idx]
        away_id, away_name = teams[away_idx]
        hs, as_ = score_patterns[i % len(score_patterns)]
        kickoff = base + pd.Timedelta(days=7 * i)
        match_id = 1001 + i
        fixtures.append(
            {
                "match_id": match_id,
                "match_date": kickoff.strftime("%Y-%m-%d"),
                "kick_off": kickoff.strftime("%H:%M:%S.000"),
                "competition": {
                    "competition_id": config.competition_id,
                    "country_name": "England",
                    "competition_name": config.competition_name,
                },
                "season": {"season_id": config.season_id, "season_name": config.season_name},
                "home_team": {"home_team_id": home_id, "home_team_name": home_name},
                "away_team": {"away_team_id": away_id, "away_team_name": away_name},
                "home_score": hs,
                "away_score": as_,
                "match_week": i + 1,
                "competition_stage": {"id": 1, "name": "Regular Season"},
                "stadium": {"id": 100 + home_id, "name": f"{home_name} Stadium"},
                "referee": {"id": 500 + i, "name": f"Referee {i+1}"},
                "match_status": "available",
                "match_status_360": "available" if i < 3 else "unscheduled",
                "last_updated": "synthetic",
                "last_updated_360": "synthetic" if i < 3 else None,
                "metadata": {"data_version": "1.1.0"},
            }
        )

        # Lineups: 11 players per team.
        lineup_rows = []
        for tid, tname in [(home_id, home_name), (away_id, away_name)]:
            lineup = []
            for p in range(1, 12):
                pid = tid * 100 + p
                lineup.append(
                    {
                        "player_id": pid,
                        "player_name": f"{tname} Player {p}",
                        "player_nickname": None,
                        "jersey_number": p,
                        "country": {"id": 68, "name": "England"},
                        "positions": [
                            {
                                "position_id": p,
                                "position": "Goalkeeper" if p == 1 else "Outfield",
                                "from": "00:00",
                                "to": None,
                                "from_period": 1,
                                "to_period": None,
                                "start_reason": "Starting XI",
                                "end_reason": None,
                            }
                        ],
                    }
                )
            lineup_rows.append({"team_id": tid, "team_name": tname, "lineup": lineup})
        write_json(lineup_rows, config.statsbomb_root / "lineups" / f"{match_id}.json")

        events: list[dict[str, Any]] = []
        idx = 1
        # Match start and recurring actions.
        events.append(_event(f"{match_id}-{idx}", idx, 1, 0, 0, home_id, home_name, "Starting XI")); idx += 1
        events.append(_event(f"{match_id}-{idx}", idx, 1, 0, 1, away_id, away_name, "Starting XI")); idx += 1
        for minute in [3, 8, 13, 18, 23, 28, 33, 38, 43, 45, 47, 52, 57, 62, 67, 72, 77, 82, 87]:
            period = 1 if minute <= 45 else 2
            tid, tname = (home_id, home_name) if minute % 2 else (away_id, away_name)
            events.append(
                _event(
                    f"{match_id}-{idx}", idx, period, minute, (minute * 7) % 60,
                    tid, tname, "Pass", player_id=tid * 100 + 2,
                    location=[50.0 + (minute % 20), 30.0],
                    pass_data={"end_location": [85.0 if minute % 3 == 0 else 70.0, 35.0], "height": {"name": "Ground Pass"}},
                )
            ); idx += 1
            if minute % 10 in (3, 7):
                events.append(
                    _event(
                        f"{match_id}-{idx}", idx, period, minute, ((minute * 7) + 5) % 60,
                        tid, tname, "Pressure", player_id=tid * 100 + 4,
                        location=[75.0, 40.0],
                    )
                ); idx += 1

        # Include a first-half stoppage event and a period-2 event sharing minute 45.
        events.append(_event(f"{match_id}-{idx}", idx, 1, 46, 10, home_id, home_name, "Pass",
                             pass_data={"end_location": [82.0, 40.0]})); idx += 1
        events.append(_event(f"{match_id}-{idx}", idx, 2, 45, 10, away_id, away_name, "Pass",
                             pass_data={"end_location": [70.0, 40.0]})); idx += 1

        goal_minutes_home = [12, 64, 78][:hs]
        goal_minutes_away = [31, 71, 84][:as_]
        for minute, tid, tname in [
            *[(m, home_id, home_name) for m in goal_minutes_home],
            *[(m, away_id, away_name) for m in goal_minutes_away],
        ]:
            period = 1 if minute < 45 else 2
            events.append(
                _event(
                    f"{match_id}-{idx}", idx, period, minute, 20,
                    tid, tname, "Shot", player_id=tid * 100 + 9,
                    location=[105.0, 40.0],
                    shot_data={
                        "outcome": {"id": 97, "name": "Goal"},
                        "statsbomb_xg": 0.25,
                        "end_location": [120.0, 40.0, 1.0],
                        "body_part": {"name": "Right Foot"},
                    },
                )
            ); idx += 1

        # Non-goal shots for both teams.
        for minute, tid, tname, outcome in [
            (9, home_id, home_name, "Saved"),
            (27, away_id, away_name, "Off T"),
            (54, home_id, home_name, "Blocked"),
            (73, away_id, away_name, "Saved"),
        ]:
            period = 1 if minute < 45 else 2
            events.append(
                _event(
                    f"{match_id}-{idx}", idx, period, minute, 40,
                    tid, tname, "Shot", player_id=tid * 100 + 10,
                    location=[100.0, 35.0],
                    shot_data={
                        "outcome": {"id": 96, "name": outcome},
                        "statsbomb_xg": 0.08,
                        "end_location": [118.0, 40.0, 1.0],
                    },
                )
            ); idx += 1

        if i % 5 == 0:
            events.append(
                _event(
                    f"{match_id}-{idx}", idx, 2, 69, 5,
                    away_id, away_name, "Foul Committed", player_id=away_id * 100 + 6,
                    foul_committed={"card": {"id": 5, "name": "Red Card"}},
                )
            ); idx += 1

        events.append(_event(f"{match_id}-{idx}", idx, 2, 90, 30, home_id, home_name, "Half End")); idx += 1
        write_json(events, config.statsbomb_root / "events" / f"{match_id}.json")

        if i < 3:
            frames = [
                {
                    "event_uuid": events[min(5, len(events)-1)]["id"],
                    "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0],
                    "freeze_frame": [
                        {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
                        {"teammate": False, "actor": False, "keeper": False, "location": [65.0, 42.0]},
                    ],
                }
            ]
            write_json(frames, config.statsbomb_root / "three-sixty" / f"{match_id}.json")

    write_json(fixtures, config.statsbomb_root / "matches" / str(config.competition_id) / f"{config.season_id}.json")

    # Football-Data-shaped CSV. Deliberately use provider aliases.
    fd_alias = {
        "Manchester United": "Man United",
        "Leicester City": "Leicester",
    }
    fd_rows = []
    for i, m in enumerate(fixtures):
        fd_rows.append(
            {
                "Div": config.football_data_division,
                "Date": pd.Timestamp(m["match_date"]).strftime("%d/%m/%y"),
                "HomeTeam": fd_alias.get(m["home_team"]["home_team_name"], m["home_team"]["home_team_name"]),
                "AwayTeam": fd_alias.get(m["away_team"]["away_team_name"], m["away_team"]["away_team_name"]),
                "FTHG": m["home_score"],
                "FTAG": m["away_score"],
                "FTR": "H" if m["home_score"] > m["away_score"] else "A" if m["home_score"] < m["away_score"] else "D",
                "HS": 10 + i,
                "AS": 8 + i,
                "B365H": round(1.8 + 0.03 * (i % 5), 2),
                "B365D": round(3.3 + 0.02 * (i % 4), 2),
                "B365A": round(4.0 - 0.04 * (i % 5), 2),
            }
        )
    pd.DataFrame(fd_rows).to_csv(
        config.football_data_root / f"{config.football_data_division}_{config.football_data_code}.csv",
        index=False,
    )

    manifest = []
    for path in sorted(config.bronze_root.rglob("*")):
        if path.is_file():
            manifest.append(
                {
                    "url": "synthetic://generated",
                    "local_path": str(path),
                    "status": "generated",
                    "http_status": None,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    save_table(pd.DataFrame(manifest), config.audit_root / "raw_download_manifest.csv")


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 10)
# -----------------------------------------------------------------------------
def download_real_bronze(config: MD1Config) -> pd.DataFrame:
    """Download the complete selected season to Drive with caching and resume support."""
    config.ensure_directories()
    session = build_http_session()
    base = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
    records: list[dict[str, Any]] = []

    # Small index files are downloaded first so the complete match list is known.
    records.append(download_file(
        session,
        f"{base}/competitions.json",
        config.statsbomb_root / "competitions.json",
        overwrite=config.overwrite_downloads,
    ))
    matches_path = config.statsbomb_root / "matches" / str(config.competition_id) / f"{config.season_id}.json"
    records.append(download_file(
        session,
        f"{base}/matches/{config.competition_id}/{config.season_id}.json",
        matches_path,
        overwrite=config.overwrite_downloads,
    ))

    raw_matches = read_json(matches_path)
    raw_matches = sorted(
        raw_matches,
        key=lambda x: (x.get("match_date", ""), x.get("kick_off", ""), x.get("match_id", 0)),
    )
    if config.max_matches is not None:
        raw_matches = raw_matches[: config.max_matches]

    # Build one job per required file. 360 is optional because StatsBomb only publishes it for selected matches.
    jobs: list[dict[str, Any]] = []
    families = [("events", False), ("lineups", False)]
    if config.download_360:
        families.append(("three-sixty", True))
    for match in raw_matches:
        match_id = int(match["match_id"])
        for family, optional in families:
            jobs.append({
                "url": f"{base}/{family}/{match_id}.json",
                "path": config.statsbomb_root / family / f"{match_id}.json",
                "optional": optional,
                "match_id": match_id,
                "family": family,
            })

    print(
        f"Downloading/reusing {len(jobs):,} match files for {len(raw_matches):,} matches "
        f"with {max(1, config.download_workers)} worker(s)."
    )

    def run_job(job: Mapping[str, Any]) -> dict[str, Any]:
        rec = download_file(
            get_thread_http_session(),
            str(job["url"]),
            Path(job["path"]),
            optional=bool(job["optional"]),
            overwrite=config.overwrite_downloads,
            timeout=120,
        )
        rec.update({"match_id": job["match_id"], "family": job["family"]})
        return rec

    fatal_errors: list[str] = []
    workers = max(1, int(config.download_workers))
    if workers == 1:
        for i, job in enumerate(jobs, start=1):
            try:
                records.append(run_job(job))
            except Exception as exc:
                fatal_errors.append(f"{job['family']}/{job['match_id']}: {exc}")
            if i % 100 == 0 or i == len(jobs):
                print(f"Match files completed: {i:,}/{len(jobs):,}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(run_job, job): job for job in jobs}
            for i, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    fatal_errors.append(f"{job['family']}/{job['match_id']}: {exc}")
                if i % 100 == 0 or i == len(jobs):
                    print(f"Match files completed: {i:,}/{len(jobs):,}")

    # Bookmaker odds for the overlapping league season.
    fd_url = (
        f"https://www.football-data.co.uk/mmz4281/"
        f"{config.football_data_code}/{config.football_data_division}.csv"
    )
    records.append(download_file(
        session,
        fd_url,
        config.football_data_root / f"{config.football_data_division}_{config.football_data_code}.csv",
        overwrite=config.overwrite_downloads,
        timeout=120,
    ))

    manifest = pd.DataFrame(records)
    save_table(manifest, config.audit_root / "raw_download_manifest.csv")
    if not manifest.empty:
        download_summary = (
            manifest.groupby(["family" if "family" in manifest.columns else "status", "status"], dropna=False)
            .agg(files=("local_path", "count"), bytes=("bytes", "sum"))
            .reset_index()
        )
        save_table(download_summary, config.audit_root / "download_summary.csv")

    if fatal_errors:
        error_path = config.audit_root / "fatal_download_errors.txt"
        error_path.write_text("\n".join(fatal_errors), encoding="utf-8")
        raise RuntimeError(
            f"{len(fatal_errors)} required downloads failed. Re-run the notebook to resume. "
            f"Details: {error_path}"
        )

    required = manifest[~manifest["status"].isin(["optional_missing", "optional_error"])]
    bad_required = required[~required["status"].isin(["downloaded", "reused"])]
    if not bad_required.empty:
        raise RuntimeError("Some required files were not downloaded successfully.")

    print("Dataset download/cache stage completed successfully.")
    return manifest


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 12)
# -----------------------------------------------------------------------------
def load_competitions(config: MD1Config) -> pd.DataFrame:
    return pd.json_normalize(read_json(config.statsbomb_root / "competitions.json"))


def parse_matches(config: MD1Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = config.statsbomb_root / "matches" / str(config.competition_id) / f"{config.season_id}.json"
    rows = read_json(path)
    rows = sorted(rows, key=lambda x: (x.get("match_date", ""), x.get("kick_off", ""), x.get("match_id", 0)))
    if config.max_matches is not None:
        rows = rows[: config.max_matches]

    match_rows: list[dict[str, Any]] = []
    team_rows: dict[int, dict[str, Any]] = {}
    for m in rows:
        home_id = int(nested_get(m, "home_team", "home_team_id"))
        away_id = int(nested_get(m, "away_team", "away_team_id"))
        home_name = nested_get(m, "home_team", "home_team_name")
        away_name = nested_get(m, "away_team", "away_team_name")
        kickoff_text = f"{m.get('match_date')} {m.get('kick_off') or '00:00:00'}"
        kickoff = pd.to_datetime(kickoff_text, errors="coerce")
        hs = int(m.get("home_score", 0))
        as_ = int(m.get("away_score", 0))
        outcome = "H" if hs > as_ else "A" if hs < as_ else "D"
        match_rows.append(
            {
                "match_id": int(m["match_id"]),
                "competition_id": int(nested_get(m, "competition", "competition_id", default=config.competition_id)),
                "competition_name": nested_get(m, "competition", "competition_name", default=config.competition_name),
                "season_id": int(nested_get(m, "season", "season_id", default=config.season_id)),
                "season_name": nested_get(m, "season", "season_name", default=config.season_name),
                "match_date": pd.to_datetime(m.get("match_date"), errors="coerce").normalize(),
                "kick_off": m.get("kick_off"),
                "kickoff": kickoff,
                "estimated_finish": kickoff + pd.Timedelta(hours=2),
                "home_team_id": home_id,
                "home_team_name": home_name,
                "away_team_id": away_id,
                "away_team_name": away_name,
                "home_score": hs,
                "away_score": as_,
                "outcome": outcome,
                "goal_margin_raw": int(hs - as_),
                "goal_margin_clipped": int(np.clip(hs - as_, -5, 5)),
                "match_week": m.get("match_week"),
                "competition_stage": nested_get(m, "competition_stage", "name"),
                "stadium_id": nested_get(m, "stadium", "id"),
                "stadium_name": nested_get(m, "stadium", "name"),
                "referee_id": nested_get(m, "referee", "id"),
                "referee_name": nested_get(m, "referee", "name"),
                "match_status": m.get("match_status"),
                "match_status_360": m.get("match_status_360"),
                "last_updated": m.get("last_updated") or m.get("match_updated"),
                "last_updated_360": m.get("last_updated_360") or m.get("match_updated_360"),
            }
        )
        team_rows[home_id] = {"team_id": home_id, "team_name": home_name, "provider": "statsbomb"}
        team_rows[away_id] = {"team_id": away_id, "team_name": away_name, "provider": "statsbomb"}
    matches = pd.DataFrame(match_rows).sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    teams = pd.DataFrame(team_rows.values()).sort_values("team_id").reset_index(drop=True)
    return matches, teams


def _position_starts_at_kickoff(position: Mapping[str, Any]) -> tuple[bool, str]:
    """Detect a starting position robustly across StatsBomb lineup variants."""
    if position.get("start_reason") == "Starting XI":
        return True, "start_reason_starting_xi"
    from_period = position.get("from_period")
    if from_period not in (None, 1):
        return False, "not_starting"
    value = position.get("from")
    if value is None:
        return False, "not_starting"
    text = str(value).strip()
    parts = text.split(":")
    try:
        if len(parts) == 2:
            seconds = float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            seconds = float(text)
    except (TypeError, ValueError):
        return False, "not_starting"
    if abs(seconds) <= 1e-9:
        return True, "position_segment_from_00_00"
    return False, "not_starting"


def parse_lineups_for_match(config: MD1Config, match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = config.statsbomb_root / "lineups" / f"{match_id}.json"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = read_json(path)
    lineup_rows: list[dict[str, Any]] = []
    player_rows: dict[int, dict[str, Any]] = {}
    for team in raw:
        team_id = team.get("team_id")
        team_name = team.get("team_name")
        for player in team.get("lineup", []):
            player_id = player.get("player_id")
            if player_id is None:
                continue
            positions = player.get("positions") or []
            if not positions:
                positions = [{}]
            for position_index, pos in enumerate(positions):
                is_starting, starter_method = _position_starts_at_kickoff(pos)
                lineup_rows.append(
                    {
                        "match_id": match_id,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player.get("player_name"),
                        "jersey_number": player.get("jersey_number"),
                        "position_index": position_index,
                        "position_id": pos.get("position_id"),
                        "position_name": pos.get("position"),
                        "from_time": pos.get("from"),
                        "to_time": pos.get("to"),
                        "from_period": pos.get("from_period"),
                        "to_period": pos.get("to_period"),
                        "start_reason": pos.get("start_reason"),
                        "end_reason": pos.get("end_reason"),
                        "is_starting": is_starting,
                        "starter_detection_method": starter_method,
                    }
                )
            player_rows[int(player_id)] = {
                "player_id": int(player_id),
                "player_name": player.get("player_name"),
                "player_nickname": player.get("player_nickname"),
                "country_id": nested_get(player, "country", "id"),
                "country_name": nested_get(player, "country", "name"),
            }
    return pd.DataFrame(lineup_rows), pd.DataFrame(player_rows.values())


def canonical_goal_assignment(
    type_name: str | None,
    shot_outcome: str | None,
    team_id: Any,
    period: int,
) -> tuple[int | None, str | None]:
    """Return the scoring team for a canonical match-score event.

    StatsBomb supplies two views of an own goal. `Own Goal For` is the
    scoring-team view used for the score; `Own Goal Against` is retained only
    for auditing so the same incident is not counted twice.
    """
    if period == 5:
        return None, "penalty_shootout_excluded"
    if type_name == "Shot" and shot_outcome == "Goal":
        return int(team_id), "shot_goal"
    if type_name == "Own Goal For":
        return int(team_id), "own_goal_for"
    if type_name == "Own Goal Against":
        return None, "own_goal_against_audit_only"
    return None, None


def parse_events_for_match(config: MD1Config, match_row: Mapping[str, Any]) -> pd.DataFrame:
    match_id = int(match_row["match_id"])
    path = config.statsbomb_root / "events" / f"{match_id}.json"
    raw = read_json(path)
    home_id = int(match_row["home_team_id"])
    away_id = int(match_row["away_team_id"])
    rows: list[dict[str, Any]] = []
    for e in raw:
        location = e.get("location") or [np.nan, np.nan]
        pass_end = nested_get(e, "pass", "end_location", default=[]) or []
        shot_end = nested_get(e, "shot", "end_location", default=[]) or []
        carry_end = nested_get(e, "carry", "end_location", default=[]) or []
        type_name = nested_get(e, "type", "name")
        team_id = nested_get(e, "team", "id")
        shot_outcome = nested_get(e, "shot", "outcome", "name")
        foul_card = nested_get(e, "foul_committed", "card", "name")
        bad_card = nested_get(e, "bad_behaviour", "card", "name")
        card_name = foul_card or bad_card
        event_period = int(e.get("period", 0) or 0)
        goal_team_id, goal_source = canonical_goal_assignment(
            type_name, shot_outcome, team_id, event_period
        )

        minute = int(e.get("minute", 0) or 0)
        second = int(e.get("second", 0) or 0)
        rows.append(
            {
                "event_id": e.get("id"),
                "match_id": match_id,
                "raw_index": int(e.get("index", 0) or 0),
                "period": event_period,
                "timestamp": e.get("timestamp"),
                "timestamp_seconds": parse_timestamp_seconds(e.get("timestamp")),
                "minute": minute,
                "second": second,
                "official_second": minute * 60 + second,
                "possession": e.get("possession"),
                "possession_team_id": nested_get(e, "possession_team", "id"),
                "possession_team_name": nested_get(e, "possession_team", "name"),
                "duration": e.get("duration"),
                "type_id": nested_get(e, "type", "id"),
                "type_name": type_name,
                "team_id": team_id,
                "team_name": nested_get(e, "team", "name"),
                "player_id": nested_get(e, "player", "id"),
                "player_name": nested_get(e, "player", "name"),
                "position_id": nested_get(e, "position", "id"),
                "position_name": nested_get(e, "position", "name"),
                "play_pattern_id": nested_get(e, "play_pattern", "id"),
                "play_pattern_name": nested_get(e, "play_pattern", "name"),
                "location_x": location[0] if len(location) > 0 else np.nan,
                "location_y": location[1] if len(location) > 1 else np.nan,
                "pass_end_x": pass_end[0] if len(pass_end) > 0 else np.nan,
                "pass_end_y": pass_end[1] if len(pass_end) > 1 else np.nan,
                "carry_end_x": carry_end[0] if len(carry_end) > 0 else np.nan,
                "carry_end_y": carry_end[1] if len(carry_end) > 1 else np.nan,
                "pass_outcome_name": nested_get(e, "pass", "outcome", "name"),
                "pass_type_name": nested_get(e, "pass", "type", "name"),
                "shot_end_x": shot_end[0] if len(shot_end) > 0 else np.nan,
                "shot_end_y": shot_end[1] if len(shot_end) > 1 else np.nan,
                "shot_outcome_name": shot_outcome,
                "shot_statsbomb_xg": nested_get(e, "shot", "statsbomb_xg", default=0.0) or 0.0,
                "card_name": card_name,
                "under_pressure": bool(e.get("under_pressure", False)),
                "counterpress": bool(e.get("counterpress", False)),
                "goal_team_id": goal_team_id,
                "goal_source": goal_source,
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values(
        ["period", "timestamp_seconds", "minute", "second", "raw_index", "event_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    events["event_order"] = np.arange(len(events), dtype=np.int64)
    return events


def parse_frames_360_for_match(config: MD1Config, match_id: int) -> pd.DataFrame:
    path = config.statsbomb_root / "three-sixty" / f"{match_id}.json"
    if not path.exists():
        return pd.DataFrame()
    raw = read_json(path)
    rows: list[dict[str, Any]] = []
    for frame in raw:
        for frame_index, player in enumerate(frame.get("freeze_frame") or []):
            location = player.get("location") or [np.nan, np.nan]
            rows.append(
                {
                    "match_id": match_id,
                    "event_uuid": frame.get("event_uuid"),
                    "frame_index": frame_index,
                    "teammate": player.get("teammate"),
                    "actor": player.get("actor"),
                    "keeper": player.get("keeper"),
                    "location_x": location[0] if len(location) else np.nan,
                    "location_y": location[1] if len(location) > 1 else np.nan,
                    "visible_area_json": json.dumps(frame.get("visible_area")),
                }
            )
    return pd.DataFrame(rows)


def _team_event_summary(events: pd.DataFrame, team_id: int) -> dict[str, float]:
    own = events[events["team_id"] == team_id]
    shots = own[own["type_name"] == "Shot"]
    passes = own[own["type_name"] == "Pass"]
    completed = passes["pass_outcome_name"].isna().sum() if not passes.empty else 0
    pass_completion = float(completed / len(passes)) if len(passes) else np.nan

    start_x = pd.to_numeric(own["location_x"], errors="coerce")
    destination_x = pd.Series(np.nan, index=own.index, dtype=float)
    pass_mask = own["type_name"] == "Pass"
    carry_mask = own["type_name"] == "Carry"
    destination_x.loc[pass_mask] = pd.to_numeric(own.loc[pass_mask, "pass_end_x"], errors="coerce")
    destination_x.loc[carry_mask] = pd.to_numeric(own.loc[carry_mask, "carry_end_x"], errors="coerce")
    final_third_entries = int(((pass_mask | carry_mask) & (start_x < 80) & (destination_x >= 80)).sum())

    red_cards = int(own["card_name"].isin(["Red Card", "Second Yellow"]).sum())
    possession_events = events["possession_team_id"].notna().sum()
    possession_share = (
        float((events["possession_team_id"] == team_id).sum() / possession_events)
        if possession_events
        else np.nan
    )
    return {
        "shots": int(len(shots)),
        "shots_on_target": int(shots["shot_outcome_name"].isin(["Goal", "Saved"]).sum()),
        "xg": float(pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0).sum()),
        "passes": int(len(passes)),
        "completed_passes": int(completed),
        "pass_completion": pass_completion,
        "pressures": int((own["type_name"] == "Pressure").sum()),
        "carries": int((own["type_name"] == "Carry").sum()),
        "fouls": int((own["type_name"] == "Foul Committed").sum()),
        "corners": int((own["pass_type_name"] == "Corner").sum()),
        "final_third_entries": final_third_entries,
        "red_cards": red_cards,
        "possession_share": possession_share,
    }


def build_silver_tables(config: MD1Config) -> dict[str, pd.DataFrame]:
    config.ensure_directories()
    competitions = load_competitions(config)
    matches, teams = parse_matches(config)
    selected_ids = set(matches["match_id"].astype(int))

    if config.event_partitions_root.exists():
        shutil.rmtree(config.event_partitions_root)
    if config.frames_partitions_root.exists():
        shutil.rmtree(config.frames_partitions_root)
    config.event_partitions_root.mkdir(parents=True, exist_ok=True)
    config.frames_partitions_root.mkdir(parents=True, exist_ok=True)

    all_lineups: list[pd.DataFrame] = []
    players_by_id: dict[int, dict[str, Any]] = {}
    event_fact_rows: list[dict[str, Any]] = []
    event_manifest_rows: list[dict[str, Any]] = []
    coverage_360_rows: list[dict[str, Any]] = []
    goal_audit_rows: list[dict[str, Any]] = []

    for i, match in enumerate(matches.to_dict("records"), start=1):
        match_id = int(match["match_id"])
        events = parse_events_for_match(config, match)
        event_path = config.event_partitions_root / f"match_id={match_id}" / "events.parquet"
        save_table(events, event_path)

        score_from_events_home = int((events["goal_team_id"] == int(match["home_team_id"])).sum()) if not events.empty else 0
        score_from_events_away = int((events["goal_team_id"] == int(match["away_team_id"])).sum()) if not events.empty else 0

        if not events.empty:
            goal_mask = events["goal_source"].notna()
            goal_cols = [
                "event_id", "event_order", "raw_index", "period", "minute", "second",
                "type_name", "team_id", "team_name", "shot_outcome_name",
                "goal_team_id", "goal_source",
            ]
            for goal_row in events.loc[goal_mask, goal_cols].to_dict("records"):
                goal_audit_rows.append({
                    "match_id": match_id,
                    "home_team_id": int(match["home_team_id"]),
                    "away_team_id": int(match["away_team_id"]),
                    "metadata_home_score": int(match["home_score"]),
                    "metadata_away_score": int(match["away_score"]),
                    **goal_row,
                })

        event_manifest_rows.append(
            {
                "match_id": match_id,
                "event_rows": len(events),
                "event_id_unique": bool(events["event_id"].is_unique) if not events.empty else True,
                "raw_index_unique": bool(events["raw_index"].is_unique) if not events.empty else True,
                "min_event_order": int(events["event_order"].min()) if not events.empty else -1,
                "max_event_order": int(events["event_order"].max()) if not events.empty else -1,
                "metadata_home_score": int(match["home_score"]),
                "metadata_away_score": int(match["away_score"]),
                "event_home_score": score_from_events_home,
                "event_away_score": score_from_events_away,
                "score_reconciles": (
                    score_from_events_home == int(match["home_score"])
                    and score_from_events_away == int(match["away_score"])
                ),
                "partition_path": str(event_path),
            }
        )

        for side, team_id in [("home", int(match["home_team_id"])), ("away", int(match["away_team_id"]))]:
            summary = _team_event_summary(events, team_id) if not events.empty else {}
            event_fact_rows.append({"match_id": match_id, "side": side, "team_id": team_id, **summary})

        lineups, players = parse_lineups_for_match(config, match_id)
        if not lineups.empty:
            all_lineups.append(lineups)
        if not players.empty:
            for row in players.to_dict("records"):
                players_by_id[int(row["player_id"])] = row

        frames = parse_frames_360_for_match(config, match_id)
        has_360 = not frames.empty
        if has_360:
            frame_path = config.frames_partitions_root / f"match_id={match_id}" / "frames_360.parquet"
            save_table(frames, frame_path)
        coverage_360_rows.append(
            {
                "match_id": match_id,
                "has_360": has_360,
                "frame_rows": len(frames),
                "three_sixty_file_exists": (config.statsbomb_root / "three-sixty" / f"{match_id}.json").exists(),
            }
        )
        if i % 50 == 0:
            print(f"Normalized {i}/{len(matches)} matches")

    lineups = pd.concat(all_lineups, ignore_index=True) if all_lineups else pd.DataFrame()
    players = pd.DataFrame(players_by_id.values()).sort_values("player_id").reset_index(drop=True) if players_by_id else pd.DataFrame()
    event_facts = pd.DataFrame(event_fact_rows)
    event_manifest = pd.DataFrame(event_manifest_rows)
    coverage_360 = pd.DataFrame(coverage_360_rows)
    goal_event_audit = pd.DataFrame(goal_audit_rows)
    score_failures = event_manifest.loc[~event_manifest["score_reconciles"]].copy()
    save_table(score_failures, config.audit_root / "score_reconciliation_failures.csv")
    save_table(goal_event_audit, config.audit_root / "goal_event_audit.csv")

    # Only keep selected competition-season in competitions table for this package.
    competitions_selected = competitions[
        (competitions["competition_id"] == config.competition_id)
        & (competitions["season_id"] == config.season_id)
    ].copy()

    tables = {
        "competitions": competitions_selected,
        "matches": matches,
        "teams": teams,
        "players": players,
        "lineups": lineups,
        "match_event_facts": event_facts,
        "event_manifest": event_manifest,
        "coverage_360": coverage_360,
    }
    for name, df in tables.items():
        save_table(df, config.silver_root / f"{name}.parquet")
    save_table(coverage_360, config.audit_root / "missing_360_report.csv")
    return tables


def load_event_partition(config: MD1Config, match_id: int) -> pd.DataFrame:
    path = config.event_partitions_root / f"match_id={int(match_id)}" / "events.parquet"
    return pd.read_parquet(path)


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 14)
# -----------------------------------------------------------------------------
def chronological_match_level_split(config: MD1Config, matches: pd.DataFrame) -> pd.DataFrame:
    m = matches[["match_id", "kickoff", "competition_id", "season_id"]].copy()
    m["split_date"] = pd.to_datetime(m["kickoff"]).dt.normalize()
    dates = np.array(sorted(m["split_date"].dropna().unique()))
    if len(dates) < 3:
        raise ValueError("At least three distinct dates are required for train/validation/test splitting.")
    train_end = max(1, int(np.floor(len(dates) * config.train_fraction)))
    val_end = max(train_end + 1, int(np.floor(len(dates) * (config.train_fraction + config.validation_fraction))))
    val_end = min(val_end, len(dates) - 1)
    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])

    def label(d: Any) -> str:
        if d in train_dates:
            return "train"
        if d in val_dates:
            return "validation"
        return "test"

    m["split"] = m["split_date"].map(label)
    split_manifest = m.drop(columns="split_date").sort_values(["kickoff", "match_id"]).reset_index(drop=True)
    save_table(split_manifest, config.gold_root / "split_manifest.csv")
    return split_manifest


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 16)
# -----------------------------------------------------------------------------
def build_team_match_facts(matches: pd.DataFrame, event_facts: pd.DataFrame) -> pd.DataFrame:
    fact_lookup = event_facts.set_index(["match_id", "team_id"]).to_dict("index")
    rows: list[dict[str, Any]] = []
    for m in matches.to_dict("records"):
        hs = int(m["home_score"])
        as_ = int(m["away_score"])
        for side in ["home", "away"]:
            team_id = int(m[f"{side}_team_id"])
            opponent_id = int(m["away_team_id"] if side == "home" else m["home_team_id"])
            goals_for = hs if side == "home" else as_
            goals_against = as_ if side == "home" else hs
            points = 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
            own = fact_lookup.get((int(m["match_id"]), team_id), {})
            opp = fact_lookup.get((int(m["match_id"]), opponent_id), {})
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "side": side,
                    "is_home": int(side == "home"),
                    "kickoff": pd.Timestamp(m["kickoff"]),
                    "match_finished_at": pd.Timestamp(m["estimated_finish"]),
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_difference": goals_for - goals_against,
                    "points": points,
                    "shots_for": own.get("shots", np.nan),
                    "shots_against": opp.get("shots", np.nan),
                    "shots_on_target_for": own.get("shots_on_target", np.nan),
                    "shots_on_target_against": opp.get("shots_on_target", np.nan),
                    "xg_for": own.get("xg", np.nan),
                    "xg_against": opp.get("xg", np.nan),
                    "passes_for": own.get("passes", np.nan),
                    "pass_completion_for": own.get("pass_completion", np.nan),
                    "pressures_for": own.get("pressures", np.nan),
                    "final_third_entries_for": own.get("final_third_entries", np.nan),
                    "red_cards_for": own.get("red_cards", np.nan),
                    "possession_share": own.get("possession_share", np.nan),
                }
            )
    facts = pd.DataFrame(rows).sort_values(["team_id", "kickoff", "match_id"]).reset_index(drop=True)
    facts["history_matches_before"] = facts.groupby("team_id").cumcount()
    prev_finish = facts.groupby("team_id")["match_finished_at"].shift(1)
    facts["rest_days"] = (facts["kickoff"] - prev_finish).dt.total_seconds() / 86400.0
    return facts


def add_leakage_safe_rolling_features(
    team_facts: pd.DataFrame,
    windows: Sequence[int] = (3, 5, 10),
) -> pd.DataFrame:
    df = team_facts.sort_values(["team_id", "kickoff", "match_id"]).reset_index(drop=True).copy()
    numeric_cols = [
        "points", "goals_for", "goals_against", "goal_difference",
        "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
        "xg_for", "xg_against", "passes_for", "pass_completion_for", "pressures_for",
        "final_third_entries_for", "red_cards_for", "possession_share",
    ]
    group = df.groupby("team_id", sort=False)
    for w in windows:
        for col in numeric_cols:
            df[f"{col}_l{w}"] = group[col].transform(
                lambda s, window=w: s.shift(1).rolling(window, min_periods=1).mean()
            )
        # Datetime rolling max via float nanoseconds avoids pandas datetime rolling aggregation errors.
        finish_ns = df["match_finished_at"].astype("int64").astype("float64")
        finish_ns[df["match_finished_at"].isna()] = np.nan
        rolling_ns = finish_ns.groupby(df["team_id"], sort=False).transform(
            lambda s, window=w: s.shift(1).rolling(window, min_periods=1).max()
        )
        df[f"max_source_finish_l{w}"] = pd.to_datetime(rolling_ns, unit="ns", errors="coerce")
        df[f"missing_history_l{w}"] = (df["history_matches_before"] < w).astype(int)
    return df


def assert_no_prematch_leakage(rolling_facts: pd.DataFrame, windows: Sequence[int]) -> None:
    for w in windows:
        source = rolling_facts[f"max_source_finish_l{w}"]
        ok = source.isna() | (source < rolling_facts["kickoff"])
        if not bool(ok.all()):
            bad = rolling_facts.loc[~ok, ["match_id", "team_id", "kickoff", f"max_source_finish_l{w}"]]
            raise AssertionError(f"Pre-match leakage for window {w}:\n{bad.head()}")


def build_gold_prematch(
    config: MD1Config,
    matches: pd.DataFrame,
    rolling_facts: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    audit_cols = [f"max_source_finish_l{w}" for w in config.rolling_windows]
    base_cols = ["match_id", "side", "rest_days", "history_matches_before"]
    feature_cols = [
        c for c in rolling_facts.columns
        if (
            (any(c.endswith(f"_l{w}") for w in config.rolling_windows) and not c.startswith("max_source_finish_"))
            or c.startswith("missing_history_l")
        )
    ]
    selected = rolling_facts[base_cols + feature_cols + audit_cols].copy()
    home = selected[selected["side"] == "home"].drop(columns="side").add_prefix("home_")
    away = selected[selected["side"] == "away"].drop(columns="side").add_prefix("away_")
    home = home.rename(columns={"home_match_id": "match_id"})
    away = away.rename(columns={"away_match_id": "match_id"})
    gold = matches.merge(home, on="match_id", how="left", validate="one_to_one")
    gold = gold.merge(away, on="match_id", how="left", validate="one_to_one")
    gold = gold.merge(split_manifest[["match_id", "split"]], on="match_id", how="left", validate="one_to_one")

    # Difference features are often more stable than raw home/away levels.
    for w in config.rolling_windows:
        for base in ["points", "goal_difference", "shots_for", "xg_for", "pass_completion_for", "pressures_for", "possession_share"]:
            h = f"home_{base}_l{w}"
            a = f"away_{base}_l{w}"
            if h in gold.columns and a in gold.columns:
                gold[f"diff_{base}_l{w}"] = gold[h] - gold[a]
    gold["rest_days_difference"] = gold["home_rest_days"] - gold["away_rest_days"]
    gold["label_outcome"] = gold["outcome"]
    gold["label_margin_raw"] = gold["goal_margin_raw"].astype(int)
    gold["label_margin"] = gold["goal_margin_clipped"].astype(int)
    save_table(gold, config.gold_root / "gold_prematch.parquet")
    return gold


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 18)
# -----------------------------------------------------------------------------
REAL_EPL_ALIAS_PAIRS = {
    "afc bournemouth": "bournemouth",
    "bournemouth": "bournemouth",
    "leicester city": "leicester",
    "manchester city": "man city",
    "manchester united": "man united",
    "newcastle united": "newcastle",
    "norwich city": "norwich",
    "stoke city": "stoke",
    "swansea city": "swansea",
    "tottenham hotspur": "tottenham",
    "west bromwich albion": "west brom",
    "west ham united": "west ham",
    "wolverhampton wanderers": "wolves",
    "brighton and hove albion": "brighton",
    "nottingham forest": "nott m forest",
    "sheffield united": "sheffield united",
}


def build_alias_table(teams: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team in teams.to_dict("records"):
        sb_norm = normalize_team_name(team["team_name"])
        fd_norm = REAL_EPL_ALIAS_PAIRS.get(sb_norm, sb_norm)
        rows.append(
            {
                "team_id": int(team["team_id"]),
                "statsbomb_team_name": team["team_name"],
                "statsbomb_normalized": sb_norm,
                "football_data_normalized": fd_norm,
                "mapping_method": "explicit_alias" if fd_norm != sb_norm else "normalized_exact",
                "manual_review": False,
            }
        )
    return pd.DataFrame(rows)


def load_football_data(config: MD1Config) -> pd.DataFrame:
    path = config.football_data_root / f"{config.football_data_division}_{config.football_data_code}.csv"
    fd = pd.read_csv(path)
    fd = fd.dropna(how="all").reset_index(drop=True)
    fd.insert(0, "fd_row_id", np.arange(len(fd), dtype=np.int64))
    fd["fd_date"] = pd.to_datetime(fd["Date"], dayfirst=True, format="mixed", errors="coerce")
    fd["home_normalized"] = fd["HomeTeam"].map(normalize_team_name)
    fd["away_normalized"] = fd["AwayTeam"].map(normalize_team_name)
    return fd


def _select_odds_columns(fd: pd.DataFrame) -> tuple[str, str, str, str]:
    candidates = [
        ("PSH", "PSD", "PSA", "Pinnacle/PS"),
        ("B365H", "B365D", "B365A", "Bet365"),
        ("AvgH", "AvgD", "AvgA", "Average"),
    ]
    for h, d, a, source in candidates:
        if {h, d, a}.issubset(fd.columns):
            return h, d, a, source
    raise ValueError("No supported 1X2 odds triplet found in Football-Data CSV.")


def build_match_join_map(
    config: MD1Config,
    matches: pd.DataFrame,
    fd: pd.DataFrame,
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    # This whitelist is the join contract. Score/stat columns are intentionally absent.
    identity_columns = {
        "fd_row_id", "fd_date", "HomeTeam", "AwayTeam", "home_normalized", "away_normalized"
    }
    forbidden = {"FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC"}
    if not identity_columns.isdisjoint(forbidden):
        raise AssertionError("Forbidden outcome/stat columns entered the odds join identity contract.")

    alias_by_id = aliases.set_index("team_id")["football_data_normalized"].to_dict()
    rows = []
    tol = pd.Timedelta(days=config.date_tolerance_days)
    for m in matches.to_dict("records"):
        target_date = pd.Timestamp(m["match_date"])
        expected_home = alias_by_id[int(m["home_team_id"])]
        expected_away = alias_by_id[int(m["away_team_id"])]
        date_candidates = fd[(fd["fd_date"] >= target_date - tol) & (fd["fd_date"] <= target_date + tol)]
        candidates = date_candidates[
            (date_candidates["home_normalized"] == expected_home)
            & (date_candidates["away_normalized"] == expected_away)
        ]
        if len(candidates) == 1:
            c = candidates.iloc[0]
            date_delta = abs((pd.Timestamp(c["fd_date"]) - target_date).days)
            alias_used = (
                normalize_team_name(m["home_team_name"]) != expected_home
                or normalize_team_name(m["away_team_name"]) != expected_away
            )
            method = "alias+date" if alias_used else "normalized_exact+date"
            if date_delta:
                method += f"_tolerance_{date_delta}d"
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "fd_row_id": int(c["fd_row_id"]),
                    "join_status": "accepted",
                    "join_method": method,
                    "confidence": 1.0 if date_delta == 0 else 0.95,
                    "candidate_count": 1,
                    "date_delta_days": date_delta,
                    "expected_home_normalized": expected_home,
                    "expected_away_normalized": expected_away,
                    "reason": "unique pre-match identity match",
                    "manual_review_needed": False,
                }
            )
        elif len(candidates) == 0:
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "fd_row_id": pd.NA,
                    "join_status": "unmatched",
                    "join_method": "none",
                    "confidence": 0.0,
                    "candidate_count": 0,
                    "date_delta_days": pd.NA,
                    "expected_home_normalized": expected_home,
                    "expected_away_normalized": expected_away,
                    "reason": "no unique date/home/away identity candidate",
                    "manual_review_needed": True,
                }
            )
        else:
            rows.append(
                {
                    "match_id": int(m["match_id"]),
                    "fd_row_id": pd.NA,
                    "join_status": "ambiguous",
                    "join_method": "multiple_candidates",
                    "confidence": 0.0,
                    "candidate_count": len(candidates),
                    "date_delta_days": pd.NA,
                    "expected_home_normalized": expected_home,
                    "expected_away_normalized": expected_away,
                    "reason": "multiple pre-match identity candidates",
                    "manual_review_needed": True,
                }
            )
    return pd.DataFrame(rows)


def integrate_odds(
    config: MD1Config,
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    gold_prematch: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fd = load_football_data(config)
    aliases = build_alias_table(teams)
    join_map = build_match_join_map(config, matches, fd, aliases)

    # Select the first complete odds triplet per row; older CSVs have patchy bookmaker columns.
    odds_candidates = [
        ("PSH", "PSD", "PSA", "Pinnacle/PS"),
        ("B365H", "B365D", "B365A", "Bet365"),
        ("AvgH", "AvgD", "AvgA", "Average"),
    ]
    if not any({h, d, a}.issubset(fd.columns) for h, d, a, _ in odds_candidates):
        raise ValueError("No supported 1X2 odds triplet found in Football-Data CSV.")
    odds_rows = []
    for row in fd.to_dict("records"):
        selected = None
        for h_col, d_col, a_col, source in odds_candidates:
            if not {h_col, d_col, a_col}.issubset(fd.columns):
                continue
            vals = [row.get(h_col), row.get(d_col), row.get(a_col)]
            if all(pd.notna(v) and float(v) > 1.0 for v in vals):
                selected = (float(vals[0]), float(vals[1]), float(vals[2]), source)
                break
        if selected is None:
            home_odds = draw_odds = away_odds = np.nan
            odds_source = "missing"
        else:
            home_odds, draw_odds, away_odds, odds_source = selected
        fair = de_vig_1x2(home_odds, draw_odds, away_odds)
        odds_rows.append(
            {
                "fd_row_id": int(row["fd_row_id"]),
                "odds_source": odds_source,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                **fair,
            }
        )
    odds = pd.DataFrame(odds_rows)
    accepted = join_map[join_map["join_status"] == "accepted"].copy()
    accepted["fd_row_id"] = accepted["fd_row_id"].astype(int)
    accepted = accepted.merge(odds, on="fd_row_id", how="left", validate="one_to_one")

    enriched = gold_prematch.merge(
        accepted[[
            "match_id", "fd_row_id", "join_status", "join_method", "confidence", "odds_source",
            "home_odds", "draw_odds", "away_odds", "market_p_home", "market_p_draw", "market_p_away",
        ]],
        on="match_id",
        how="left",
        validate="one_to_one",
    )
    enriched["odds_tagged"] = enriched["fd_row_id"].notna().astype(int)

    coverage = (
        enriched.groupby(["competition_id", "season_id", "season_name"], dropna=False)
        .agg(total_matches=("match_id", "size"), tagged_matches=("odds_tagged", "sum"))
        .reset_index()
    )
    coverage["coverage_rate"] = coverage["tagged_matches"] / coverage["total_matches"]
    excluded = join_map[join_map["join_status"] != "accepted"].copy()

    save_table(fd, config.silver_root / "football_data_matches.parquet")
    save_table(aliases, config.silver_root / "team_aliases.parquet")
    save_table(join_map, config.silver_root / "match_join_map.parquet")
    save_table(aliases, config.audit_root / "team_alias_map.csv")
    save_table(join_map, config.audit_root / "match_join_map.csv")
    save_table(coverage, config.audit_root / "odds_coverage_by_season.csv")
    save_table(excluded, config.audit_root / "excluded_odds_matches.csv")
    save_table(enriched, config.gold_root / "gold_prematch.parquet")
    return enriched, join_map, coverage, excluded


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 20)
# -----------------------------------------------------------------------------
def build_snapshot_cutoffs(events: pd.DataFrame, snapshot_minutes: Sequence[int]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    events = events.sort_values("event_order").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    specs: list[tuple[str, str, int | None, float]] = []
    for minute in snapshot_minutes:
        if minute == 45:
            specs.append(("M45", "strict_minute", 45, 45.0))
            specs.append(("HT", "half_time_boundary", None, 45.5))
        elif minute == 90:
            specs.append(("M90", "strict_minute", 90, 90.0))
            specs.append(("FT", "full_time_boundary", None, 90.5))
        else:
            specs.append((f"M{minute:02d}", "strict_minute", minute, float(minute)))

    for label, kind, minute, rank in specs:
        if kind == "half_time_boundary":
            mask = events["period"] == 1
            reference_second = float(events.loc[mask, "official_second"].max()) if mask.any() else 45 * 60.0
        elif kind == "full_time_boundary":
            mask = events["period"].isin([1, 2])
            reference_second = float(events.loc[mask, "official_second"].max()) if mask.any() else 90 * 60.0
        elif minute is not None and minute <= 45:
            # Prevent a period-2 event also labelled minute 45 from entering the first-half M45 snapshot.
            mask = (events["period"] == 1) & (events["official_second"] <= minute * 60)
            reference_second = float(minute * 60)
        else:
            # Strict regulation-minute boundary: second-half stoppage events after 90:00 are excluded.
            mask = events["period"].isin([1, 2]) & (events["official_second"] <= int(minute) * 60)
            reference_second = float(int(minute) * 60)

        used = events.loc[mask]
        cutoff = int(used["event_order"].max()) if not used.empty else -1
        next_orders = events.loc[events["event_order"] > cutoff, "event_order"]
        next_excluded = int(next_orders.min()) if not next_orders.empty else pd.NA
        rows.append(
            {
                "snapshot_id": label,
                "snapshot_kind": kind,
                "snapshot_minute": minute if minute is not None else (45 if kind == "half_time_boundary" else 90),
                "snapshot_rank": rank,
                "reference_second": reference_second,
                "snapshot_cutoff_event_order": cutoff,
                "next_excluded_event_order": next_excluded,
            }
        )
    return pd.DataFrame(rows).sort_values(["snapshot_rank", "snapshot_id"]).reset_index(drop=True)


def _prefix_team_features(prefix: pd.DataFrame, team_id: int, reference_second: float, current_period: int) -> dict[str, float]:
    own = prefix[prefix["team_id"] == team_id]

    def count_type(df: pd.DataFrame, type_name: str) -> int:
        return int((df["type_name"] == type_name).sum())

    def final_third_entry_count(df: pd.DataFrame) -> int:
        start_x = pd.to_numeric(df["location_x"], errors="coerce")
        destination_x = pd.Series(np.nan, index=df.index, dtype=float)
        pass_mask = df["type_name"] == "Pass"
        carry_mask = df["type_name"] == "Carry"
        destination_x.loc[pass_mask] = pd.to_numeric(df.loc[pass_mask, "pass_end_x"], errors="coerce")
        destination_x.loc[carry_mask] = pd.to_numeric(df.loc[carry_mask, "carry_end_x"], errors="coerce")
        return int(((pass_mask | carry_mask) & (start_x < 80) & (destination_x >= 80)).sum())

    shots = own[own["type_name"] == "Shot"]
    passes = own[own["type_name"] == "Pass"]
    completed = passes["pass_outcome_name"].isna().sum() if not passes.empty else 0
    possession_den = prefix["possession_team_id"].notna().sum()
    possession_share = float((prefix["possession_team_id"] == team_id).sum() / possession_den) if possession_den else np.nan

    out: dict[str, float] = {
        "shots": int(len(shots)),
        "shots_on_target": int(shots["shot_outcome_name"].isin(["Goal", "Saved"]).sum()),
        "xg": float(pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0).sum()),
        "passes": int(len(passes)),
        "completed_passes": int(completed),
        "pass_completion": float(completed / len(passes)) if len(passes) else np.nan,
        "pressures": count_type(own, "Pressure"),
        "carries": count_type(own, "Carry"),
        "fouls": count_type(own, "Foul Committed"),
        "corners": int((own["pass_type_name"] == "Corner").sum()),
        "final_third_entries": final_third_entry_count(own),
        "red_cards": int(own["card_name"].isin(["Red Card", "Second Yellow"]).sum()),
        "possession_share": possession_share,
    }

    for window_minutes in (5, 10):
        lower = reference_second - window_minutes * 60
        # Do not let a recent window straddle half-time: after HT, use period-2 events only.
        recent = own[
            (own["period"] == current_period)
            & (own["official_second"] > lower)
            & (own["official_second"] <= reference_second)
        ]
        recent_shots = recent[recent["type_name"] == "Shot"]
        out[f"shots_last{window_minutes}"] = int(len(recent_shots))
        out[f"xg_last{window_minutes}"] = float(pd.to_numeric(recent_shots["shot_statsbomb_xg"], errors="coerce").fillna(0).sum())
        out[f"pressures_last{window_minutes}"] = int((recent["type_name"] == "Pressure").sum())
        out[f"final_third_entries_last{window_minutes}"] = final_third_entry_count(recent)
    return out


def build_snapshots_for_match(
    config: MD1Config,
    match: Mapping[str, Any],
    events: pd.DataFrame,
) -> pd.DataFrame:
    cutoffs = build_snapshot_cutoffs(events, config.snapshot_minutes)
    rows: list[dict[str, Any]] = []
    home_id = int(match["home_team_id"])
    away_id = int(match["away_team_id"])

    for c in cutoffs.to_dict("records"):
        cutoff = int(c["snapshot_cutoff_event_order"])
        prefix = events[events["event_order"] <= cutoff] if cutoff >= 0 else events.iloc[0:0]
        home_score = int((prefix["goal_team_id"] == home_id).sum()) if not prefix.empty else 0
        away_score = int((prefix["goal_team_id"] == away_id).sum()) if not prefix.empty else 0
        current_period = 1 if c["snapshot_kind"] == "half_time_boundary" or float(c["snapshot_rank"]) <= 45 else 2
        home = _prefix_team_features(prefix, home_id, float(c["reference_second"]), current_period)
        away = _prefix_team_features(prefix, away_id, float(c["reference_second"]), current_period)
        row: dict[str, Any] = {
            "match_id": int(match["match_id"]),
            "snapshot_id": c["snapshot_id"],
            "snapshot_kind": c["snapshot_kind"],
            "snapshot_minute": c["snapshot_minute"],
            "snapshot_rank": c["snapshot_rank"],
            "snapshot_period": current_period,
            "reference_second": c["reference_second"],
            "snapshot_cutoff_event_order": cutoff,
            "max_used_event_order": int(prefix["event_order"].max()) if not prefix.empty else -1,
            "next_excluded_event_order": c["next_excluded_event_order"],
            "max_used_period": int(prefix["period"].max()) if not prefix.empty else 0,
            "max_used_minute": int(prefix.iloc[-1]["minute"]) if not prefix.empty else 0,
            "max_used_second": int(prefix.iloc[-1]["second"]) if not prefix.empty else 0,
            "n_events_so_far": int(len(prefix)),
            "current_home_score": home_score,
            "current_away_score": away_score,
            "current_goal_difference": home_score - away_score,
            "time_remaining_minutes": 0.0 if c["snapshot_kind"] == "full_time_boundary" else max(0.0, 90.0 - float(c["snapshot_rank"])),
            "final_home_score": int(match["home_score"]),
            "final_away_score": int(match["away_score"]),
            "label_outcome": match["outcome"],
            "label_margin_raw": int(match["goal_margin_raw"]),
            "label_margin": int(match["goal_margin_clipped"]),
        }
        for key, value in home.items():
            row[f"home_live_{key}"] = value
        for key, value in away.items():
            row[f"away_live_{key}"] = value
        row["man_advantage_home"] = int(away["red_cards"] - home["red_cards"])
        for key in ["shots", "shots_on_target", "xg", "pressures", "final_third_entries", "possession_share", "shots_last5", "xg_last5", "pressures_last5"]:
            row[f"live_diff_{key}"] = home[key] - away[key]
        rows.append(row)
    return pd.DataFrame(rows)


def build_gold_snapshots(
    config: MD1Config,
    matches: pd.DataFrame,
    gold_prematch: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> pd.DataFrame:
    snapshot_tables: list[pd.DataFrame] = []
    prematch_exclude = {
        "home_score", "away_score", "outcome", "goal_margin_raw", "goal_margin_clipped",
        "label_outcome", "label_margin_raw", "label_margin",
        "split", "match_date", "kick_off", "estimated_finish",
    }
    rolling_tokens = tuple(f"_l{w}" for w in config.rolling_windows)
    prematch_cols = []
    for c in gold_prematch.columns:
        if c == "match_id":
            prematch_cols.append(c)
        elif c.startswith("diff_") or c == "rest_days_difference":
            prematch_cols.append(c)
        elif c.startswith(("home_", "away_")) and (
            any(token in c for token in rolling_tokens)
            or c.endswith("rest_days")
            or c.endswith("history_matches_before")
            or "missing_history" in c
            or "max_source_finish" in c
        ):
            prematch_cols.append(c)
        elif c == "odds_tagged" or c.startswith("market_p_"):
            # Kept only as a reference/baseline column, never as a model feature.
            prematch_cols.append(c)
    prematch_base = gold_prematch[prematch_cols].copy()

    for i, match in enumerate(matches.to_dict("records"), start=1):
        events = load_event_partition(config, int(match["match_id"]))
        snap = build_snapshots_for_match(config, match, events)
        snapshot_tables.append(snap)
        if i % 50 == 0:
            print(f"Built snapshots for {i}/{len(matches)} matches")
    snapshots = pd.concat(snapshot_tables, ignore_index=True)
    snapshots = snapshots.merge(prematch_base, on="match_id", how="left", validate="many_to_one")
    snapshots = snapshots.merge(split_manifest[["match_id", "split"]], on="match_id", how="left", validate="many_to_one")
    save_table(snapshots, config.gold_root / "gold_snapshots.parquet")
    return snapshots


def assert_snapshot_contracts(snapshots: pd.DataFrame) -> None:
    if not bool((snapshots["max_used_event_order"] <= snapshots["snapshot_cutoff_event_order"]).all()):
        raise AssertionError("A snapshot used an event after its cutoff order.")
    split_counts = snapshots.groupby("match_id")["split"].nunique(dropna=False)
    if int(split_counts.max()) != 1:
        raise AssertionError("Snapshots of the same match appear in multiple splits.")
    ht = snapshots[snapshots["snapshot_id"] == "HT"]
    if not ht.empty and not bool((ht["max_used_period"] <= 1).all()):
        raise AssertionError("Half-time snapshot includes a period-2 event.")
    m90 = snapshots[snapshots["snapshot_id"] == "M90"].set_index("match_id")
    ft = snapshots[snapshots["snapshot_id"] == "FT"].set_index("match_id")
    if set(m90.index) != set(ft.index):
        raise AssertionError("Every match must have both M90 and FT snapshots.")
    if not bool((m90["snapshot_cutoff_event_order"] <= ft["snapshot_cutoff_event_order"]).all()):
        raise AssertionError("A strict M90 cutoff occurs after full time.")
    if not bool((ft["current_home_score"] == ft["final_home_score"]).all()):
        raise AssertionError("FT home score does not reconcile with match metadata.")
    if not bool((ft["current_away_score"] == ft["final_away_score"]).all()):
        raise AssertionError("FT away score does not reconcile with match metadata.")


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 22)
# -----------------------------------------------------------------------------
def build_feature_dictionary(config: MD1Config, prematch: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_cols = {
        "label_outcome", "label_margin", "label_margin_raw", "outcome",
        "goal_margin_raw", "goal_margin_clipped", "home_score", "away_score",
        "final_home_score", "final_away_score",
    }
    baseline_cols = {
        "market_p_home", "market_p_draw", "market_p_away",
        "home_odds", "draw_odds", "away_odds", "odds_tagged", "odds_source",
    }
    audit_cols = {
        "estimated_finish", "match_status", "match_status_360", "last_updated", "last_updated_360",
        "fd_row_id", "join_status", "join_method", "confidence", "candidate_count",
        "date_delta_days", "reason", "manual_review_needed",
    }
    metadata_cols = {
        "competition_id", "competition_name", "season_id", "season_name", "match_date",
        "kickoff", "kick_off", "match_week", "competition_stage", "stadium_id", "stadium_name",
        "referee_id", "referee_name", "snapshot_kind",
    }
    identifier_cols = {"match_id", "snapshot_id", "split", "home_team_id", "away_team_id"}
    audit_tokens = ("max_source", "cutoff", "max_used", "next_excluded")

    for table_name, df in [("gold_prematch", prematch), ("gold_snapshots", snapshots)]:
        for col in df.columns:
            if col in label_cols:
                role = "label"
            elif col in baseline_cols:
                role = "baseline_only"
            elif col in audit_cols or any(token in col for token in audit_tokens):
                role = "audit"
            elif col in identifier_cols or col.endswith("_team_id"):
                role = "identifier"
            elif col in metadata_cols or col.endswith("_name"):
                role = "metadata"
            elif not pd.api.types.is_numeric_dtype(df[col].dtype):
                # Unrecognized strings/categories are context, never silently promoted to model features.
                role = "metadata"
            else:
                role = "feature"

            if role == "label":
                prediction_time = "observed after full time; target only"
                source = "final match result from match metadata"
                guard = "excluded from every feature matrix"
            elif role == "baseline_only":
                prediction_time = "available before kickoff"
                source = "Football-Data bookmaker odds"
                guard = "market reference only; excluded from event-derived model features"
            elif role == "audit":
                prediction_time = "audit/provenance only"
                source = "provider metadata, join diagnostics, or leakage assertions"
                guard = "explicitly excluded from model features"
            elif role in {"identifier", "metadata"}:
                prediction_time = "identity/context only"
                source = "match or snapshot metadata"
                guard = "excluded from model feature matrix"
            elif table_name == "gold_prematch":
                prediction_time = "strictly before target kickoff"
                source = "prior completed matches"
                guard = "shift(1) before rolling; max_source_finish < target kickoff"
            else:
                prediction_time = "at snapshot boundary t"
                source = "pre-match vector + current-match event prefix"
                guard = "event_order <= snapshot cutoff; match-level split inheritance"

            rows.append(
                {
                    "table": table_name,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "role": role,
                    "model_eligible": role == "feature",
                    "prediction_time": prediction_time,
                    "source": source,
                    "fit_scope": "transformers fit on training split only" if role == "feature" else "not applicable",
                    "leakage_guard": guard,
                }
            )
    dictionary = pd.DataFrame(rows)
    save_table(dictionary, config.gold_root / "feature_dictionary.csv")
    return dictionary


def build_table_row_counts(config: MD1Config, tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        rows.append({"table": name, "rows": len(df), "columns": len(df.columns)})
    event_rows = 0
    event_partitions = 0
    for path in config.event_partitions_root.glob("match_id=*/events.parquet"):
        event_rows += len(pd.read_parquet(path, columns=["match_id"]))
        event_partitions += 1
    rows.append({"table": "events_partitioned", "rows": event_rows, "columns": np.nan, "partitions": event_partitions})
    result = pd.DataFrame(rows)
    save_table(result, config.audit_root / "table_row_counts.csv")
    return result


def build_event_partition_contract_audit(
    config: MD1Config,
    matches: pd.DataFrame,
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    match_lookup = matches.set_index("match_id")[["home_team_id", "away_team_id"]].to_dict("index")
    lineup_players = (
        lineups.groupby("match_id")["player_id"].apply(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))
        .to_dict() if not lineups.empty else {}
    )
    rows: list[dict[str, Any]] = []
    columns = [
        "match_id", "event_id", "event_order", "raw_index", "period", "timestamp_seconds",
        "minute", "second", "team_id", "possession_team_id", "player_id",
    ]
    for path in sorted(config.event_partitions_root.glob("match_id=*/events.parquet")):
        partition_match_id = int(path.parent.name.split("=", 1)[1])
        events = pd.read_parquet(path, columns=columns)
        allowed_teams = {
            int(match_lookup[partition_match_id]["home_team_id"]),
            int(match_lookup[partition_match_id]["away_team_id"]),
        }
        event_match_ids = set(pd.to_numeric(events["match_id"], errors="coerce").dropna().astype(int))
        event_teams = set(pd.to_numeric(events["team_id"], errors="coerce").dropna().astype(int))
        possession_teams = set(pd.to_numeric(events["possession_team_id"], errors="coerce").dropna().astype(int))
        event_players = set(pd.to_numeric(events["player_id"], errors="coerce").dropna().astype(int))
        expected_order = np.arange(len(events), dtype=np.int64)
        actual_order = pd.to_numeric(events["event_order"], errors="coerce").to_numpy()
        order_contiguous = bool(np.array_equal(actual_order, expected_order))
        sort_cols = ["period", "timestamp_seconds", "minute", "second", "raw_index", "event_id"]
        sorted_index = events.sort_values(sort_cols, kind="mergesort").index.to_numpy()
        chronological_order = bool(np.array_equal(sorted_index, np.arange(len(events))))
        rows.append({
            "match_id": partition_match_id,
            "partition_match_fk_valid": event_match_ids == {partition_match_id},
            "event_team_fk_valid": event_teams.issubset(allowed_teams),
            "possession_team_fk_valid": possession_teams.issubset(allowed_teams),
            "event_player_fk_valid": event_players.issubset(lineup_players.get(partition_match_id, set())),
            "event_order_contiguous": order_contiguous,
            "event_chronological_order_valid": chronological_order,
            "event_rows": len(events),
            "invalid_event_teams": len(event_teams - allowed_teams),
            "invalid_possession_teams": len(possession_teams - allowed_teams),
            "invalid_event_players": len(event_players - lineup_players.get(partition_match_id, set())),
        })
    audit = pd.DataFrame(rows)
    save_table(audit, config.audit_root / "event_partition_contract_audit.csv")
    return audit


def build_key_audit(
    config: MD1Config,
    silver: Mapping[str, pd.DataFrame],
    prematch: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    matches = silver["matches"]
    teams = silver["teams"]
    players = silver["players"]
    event_manifest = silver["event_manifest"]
    event_facts = silver["match_event_facts"]
    lineups = silver["lineups"]
    team_ids = set(pd.to_numeric(teams["team_id"], errors="coerce").dropna().astype(int))
    player_ids = set(pd.to_numeric(players["player_id"], errors="coerce").dropna().astype(int))
    match_ids = set(pd.to_numeric(matches["match_id"], errors="coerce").dropna().astype(int))

    match_team_fk_valid = set(matches["home_team_id"].astype(int)).union(set(matches["away_team_id"].astype(int))).issubset(team_ids)
    lineup_match_fk_valid = bool(lineups.empty or set(lineups["match_id"].astype(int)).issubset(match_ids))
    lineup_team_fk_valid = bool(lineups.empty or set(lineups["team_id"].dropna().astype(int)).issubset(team_ids))
    lineup_player_fk_valid = bool(lineups.empty or set(lineups["player_id"].dropna().astype(int)).issubset(player_ids))
    event_fact_match_fk_valid = set(event_facts["match_id"].astype(int)).issubset(match_ids)
    event_fact_team_fk_valid = set(event_facts["team_id"].astype(int)).issubset(team_ids)

    event_contract = build_event_partition_contract_audit(config, matches, lineups)
    contract_columns = [
        "partition_match_fk_valid", "event_team_fk_valid", "possession_team_fk_valid",
        "event_player_fk_valid", "event_order_contiguous", "event_chronological_order_valid",
    ]
    event_contract_all = bool(not event_contract.empty and event_contract[contract_columns].all().all())

    starter_counts = (
        lineups.loc[lineups["is_starting"].fillna(False)]
        .groupby(["match_id", "team_id"])["player_id"].nunique()
        if not lineups.empty else pd.Series(dtype=int)
    )
    expected_team_match_rows = 2 * len(matches)
    starters_valid = bool(len(starter_counts) == expected_team_match_rows and (starter_counts == 11).all())
    starter_audit = starter_counts.rename("unique_starters").reset_index()
    save_table(starter_audit, config.audit_root / "starting_lineup_audit.csv")

    checks = [
        {"check": "matches.match_id unique", "passed": bool(matches["match_id"].is_unique), "details": f"duplicates={matches['match_id'].duplicated().sum()}"},
        {"check": "match team foreign keys valid", "passed": match_team_fk_valid, "details": f"teams={len(team_ids)}"},
        {"check": "one event partition per match", "passed": int(event_manifest["match_id"].nunique()) == len(matches), "details": f"partitions={event_manifest['match_id'].nunique()}, matches={len(matches)}"},
        {"check": "event IDs unique within each match", "passed": bool(event_manifest["event_id_unique"].all()), "details": f"failures={(~event_manifest['event_id_unique']).sum()}"},
        {"check": "event indices unique within each match", "passed": bool(event_manifest["raw_index_unique"].all()), "details": f"failures={(~event_manifest['raw_index_unique']).sum()}"},
        {"check": "event partitions, foreign keys, and chronology valid", "passed": event_contract_all, "details": f"failed_matches={0 if event_contract.empty else (~event_contract[contract_columns].all(axis=1)).sum()}"},
        {"check": "event-derived score reconciles with match metadata", "passed": bool(event_manifest["score_reconciles"].all()), "details": f"failures={(~event_manifest['score_reconciles']).sum()}"},
        {"check": "lineup match foreign keys valid", "passed": lineup_match_fk_valid, "details": f"invalid={0 if lineups.empty else (~lineups['match_id'].isin(matches['match_id'])).sum()}"},
        {"check": "lineup team foreign keys valid", "passed": lineup_team_fk_valid, "details": f"teams={lineups['team_id'].nunique() if not lineups.empty else 0}"},
        {"check": "lineup player foreign keys valid", "passed": lineup_player_fk_valid, "details": f"players={lineups['player_id'].nunique() if not lineups.empty else 0}"},
        {"check": "event-fact foreign keys valid", "passed": event_fact_match_fk_valid and event_fact_team_fk_valid, "details": f"rows={len(event_facts)}"},
        {"check": "11 unique starters per team-match", "passed": starters_valid, "details": f"team_matches={len(starter_counts)}/{expected_team_match_rows}; min={starter_counts.min() if len(starter_counts) else 'NA'}; max={starter_counts.max() if len(starter_counts) else 'NA'}"},
        {"check": "gold_prematch one row per match", "passed": bool(prematch["match_id"].is_unique and len(prematch) == len(matches)), "details": f"gold={len(prematch)}, matches={len(matches)}"},
        {"check": "snapshot match foreign keys valid", "passed": bool(snapshots["match_id"].isin(matches["match_id"]).all()), "details": f"invalid={(~snapshots['match_id'].isin(matches['match_id'])).sum()}"},
    ]
    audit = pd.DataFrame(checks)
    save_table(audit, config.audit_root / "key_audit.csv")
    return audit


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 24)
# -----------------------------------------------------------------------------
def _test_chronological_split(split_manifest: pd.DataFrame) -> str:
    groups = {s: g["kickoff"] for s, g in split_manifest.groupby("split")}
    if {"train", "validation", "test"}.issubset(groups):
        assert groups["train"].max() < groups["validation"].min()
        assert groups["validation"].max() < groups["test"].min()
    return "strict train < validation < test chronology"


def _test_target_match_perturbation(config: MD1Config, team_facts: pd.DataFrame) -> str:
    rolled = add_leakage_safe_rolling_features(team_facts, config.rolling_windows)
    candidates = rolled[rolled["history_matches_before"] >= min(config.rolling_windows)].reset_index(drop=True)
    assert not candidates.empty
    sample_size = min(8, len(candidates))
    sample_indices = np.linspace(0, len(candidates) - 1, sample_size, dtype=int)
    selected = candidates.iloc[sample_indices][["match_id", "team_id"]].drop_duplicates()
    compare_cols = [c for c in rolled.columns if any(c.endswith(f"_l{w}") for w in config.rolling_windows)]

    for target in selected.to_dict("records"):
        # Mutate one target row at a time so another selected target cannot enter this row's history.
        mutated = team_facts.copy()
        mask = (mutated["match_id"] == target["match_id"]) & (mutated["team_id"] == target["team_id"])
        for col in ["points", "goals_for", "goals_against", "shots_for", "xg_for"]:
            mutated.loc[mask, col] = 999.0
        rolled_mut = add_leakage_safe_rolling_features(mutated, config.rolling_windows)
        mask_before = (rolled["match_id"] == target["match_id"]) & (rolled["team_id"] == target["team_id"])
        mask_after = (rolled_mut["match_id"] == target["match_id"]) & (rolled_mut["team_id"] == target["team_id"])
        before = rolled.loc[mask_before, compare_cols].reset_index(drop=True)
        after = rolled_mut.loc[mask_after, compare_cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after, check_dtype=False)
    return f"{len(selected)} target team-match rows unchanged after each target outcome was replaced by 999"


def _test_future_event_perturbation(config: MD1Config, matches: pd.DataFrame) -> str:
    positions = sorted(set([0, len(matches) // 2, len(matches) - 1]))
    checked = 0
    for position in positions:
        match = matches.iloc[position].to_dict()
        events = load_event_partition(config, int(match["match_id"]))
        base = build_snapshots_for_match(config, match, events)
        target_snapshots = [s for s in ("M30", "M60") if s in set(base["snapshot_id"])]
        future = events.iloc[-1].copy()
        future["event_id"] = f"injected-future-event-{match['match_id']}"
        future["period"] = 2
        future["minute"] = 85
        future["second"] = 0
        future["official_second"] = 85 * 60
        future["timestamp_seconds"] = 40 * 60
        future["raw_index"] = int(events["raw_index"].max()) + 100
        future["type_name"] = "Shot"
        future["shot_outcome_name"] = "Goal"
        future["shot_statsbomb_xg"] = 0.99
        future["team_id"] = int(match["home_team_id"])
        future["goal_team_id"] = int(match["home_team_id"])
        mutated = pd.concat([events, pd.DataFrame([future])], ignore_index=True)
        mutated = mutated.sort_values(
            ["period", "timestamp_seconds", "minute", "second", "raw_index", "event_id"], kind="mergesort"
        ).reset_index(drop=True)
        mutated["event_order"] = np.arange(len(mutated))
        after = build_snapshots_for_match(config, match, mutated)
        for snapshot_id in target_snapshots:
            base_row = base[base["snapshot_id"] == snapshot_id].reset_index(drop=True)
            after_row = after[after["snapshot_id"] == snapshot_id].reset_index(drop=True)
            compare = [c for c in base_row.columns if c not in {"next_excluded_event_order"}]
            pd.testing.assert_frame_equal(base_row[compare], after_row[compare], check_dtype=False)
            checked += 1
    return f"{checked} early snapshots across {len(positions)} matches unchanged after injecting 85th-minute goals"


def _test_half_time_boundary(config: MD1Config) -> str:
    events = pd.DataFrame(
        [
            {"event_id": "a", "period": 1, "timestamp_seconds": 46 * 60, "minute": 46, "second": 0, "official_second": 46 * 60, "raw_index": 1},
            {"event_id": "b", "period": 2, "timestamp_seconds": 10, "minute": 45, "second": 10, "official_second": 45 * 60 + 10, "raw_index": 2},
        ]
    )
    events = events.sort_values(["period", "timestamp_seconds", "minute", "second", "raw_index", "event_id"]).reset_index(drop=True)
    events["event_order"] = np.arange(len(events))
    cutoffs = build_snapshot_cutoffs(events, config.snapshot_minutes)
    ht_cutoff = int(cutoffs.loc[cutoffs["snapshot_id"] == "HT", "snapshot_cutoff_event_order"].iloc[0])
    used = events[events["event_order"] <= ht_cutoff]
    assert set(used["event_id"]) == {"a"}
    return "HT includes period-1 stoppage and excludes period-2 minute-45 event"




def _test_m90_and_full_time_boundaries(config: MD1Config) -> str:
    events = pd.DataFrame([
        {"event_id": "before", "period": 2, "timestamp_seconds": 44 * 60 + 59, "minute": 89, "second": 59, "official_second": 89 * 60 + 59, "raw_index": 1},
        {"event_id": "stoppage", "period": 2, "timestamp_seconds": 45 * 60 + 30, "minute": 90, "second": 30, "official_second": 90 * 60 + 30, "raw_index": 2},
    ])
    events = events.sort_values(["period", "timestamp_seconds", "minute", "second", "raw_index", "event_id"]).reset_index(drop=True)
    events["event_order"] = np.arange(len(events))
    cutoffs = build_snapshot_cutoffs(events, config.snapshot_minutes)
    m90_cutoff = int(cutoffs.loc[cutoffs["snapshot_id"] == "M90", "snapshot_cutoff_event_order"].iloc[0])
    ft_cutoff = int(cutoffs.loc[cutoffs["snapshot_id"] == "FT", "snapshot_cutoff_event_order"].iloc[0])
    assert set(events.loc[events["event_order"] <= m90_cutoff, "event_id"]) == {"before"}
    assert set(events.loc[events["event_order"] <= ft_cutoff, "event_id"]) == {"before", "stoppage"}
    return "M90 excludes a 90:30 stoppage event; FT includes it"


def _test_goal_margin_contract(matches: pd.DataFrame, prematch: pd.DataFrame, snapshots: pd.DataFrame) -> str:
    expected_raw = matches["home_score"].astype(int) - matches["away_score"].astype(int)
    assert np.array_equal(matches["goal_margin_raw"].astype(int).to_numpy(), expected_raw.to_numpy())
    assert np.array_equal(matches["goal_margin_clipped"].astype(int).to_numpy(), expected_raw.clip(-5, 5).to_numpy())
    prematch_map = prematch.set_index("match_id")
    match_map = matches.set_index("match_id")
    assert np.array_equal(prematch_map.loc[match_map.index, "label_margin_raw"].astype(int), match_map["goal_margin_raw"].astype(int))
    assert np.array_equal(prematch_map.loc[match_map.index, "label_margin"].astype(int), match_map["goal_margin_clipped"].astype(int))
    snapshot_check = snapshots.groupby("match_id")[["label_margin_raw", "label_margin"]].nunique()
    assert bool((snapshot_check == 1).all().all())
    return "raw margins preserve observed scores; modeling margins are clipped only in label_margin"


def _test_full_foreign_key_and_order_contract(config: MD1Config, key_audit: pd.DataFrame) -> str:
    if not bool(key_audit["passed"].all()):
        failed = key_audit.loc[~key_audit["passed"], ["check", "details"]]
        raise AssertionError(f"Key/order checks failed:\n{failed.to_string(index=False)}")
    event_audit = pd.read_csv(config.audit_root / "event_partition_contract_audit.csv")
    contract_columns = [
        "partition_match_fk_valid", "event_team_fk_valid", "possession_team_fk_valid",
        "event_player_fk_valid", "event_order_contiguous", "event_chronological_order_valid",
    ]
    assert bool(event_audit[contract_columns].all().all())
    return f"all foreign-key and event-order contracts pass for {len(event_audit)} event partitions"


def _test_own_goal_is_counted_once() -> str:
    scoring_team = 10
    conceding_team = 20
    regular_goal, regular_source = canonical_goal_assignment("Shot", "Goal", scoring_team, 1)
    own_for, own_for_source = canonical_goal_assignment("Own Goal For", None, scoring_team, 2)
    own_against, own_against_source = canonical_goal_assignment("Own Goal Against", None, conceding_team, 2)
    shootout, shootout_source = canonical_goal_assignment("Shot", "Goal", scoring_team, 5)

    assert regular_goal == scoring_team and regular_source == "shot_goal"
    assert own_for == scoring_team and own_for_source == "own_goal_for"
    assert own_against is None and own_against_source == "own_goal_against_audit_only"
    assert shootout is None and shootout_source == "penalty_shootout_excluded"
    return "Shot Goal and Own Goal For count; paired Own Goal Against and shoot-out goals do not"


def run_leakage_tests(
    config: MD1Config,
    silver: Mapping[str, pd.DataFrame],
    team_facts: pd.DataFrame,
    rolling_facts: pd.DataFrame,
    prematch: pd.DataFrame,
    snapshots: pd.DataFrame,
    split_manifest: pd.DataFrame,
    join_map: pd.DataFrame,
    key_audit: pd.DataFrame,
) -> pd.DataFrame:
    tests: list[tuple[str, Any]] = []
    tests.append(("pre-match source time contract", lambda: (assert_no_prematch_leakage(rolling_facts, config.rolling_windows), "all max_source_finish values are before target kickoff")[1]))
    tests.append(("snapshot cutoff contract", lambda: (assert_snapshot_contracts(snapshots), "all used event orders are at or before cutoff; split inheritance holds")[1]))
    tests.append(("chronological split contract", lambda: _test_chronological_split(split_manifest)))
    tests.append(("target-match perturbation test", lambda: _test_target_match_perturbation(config, team_facts)))

    tests.append(("multi-match future-event perturbation test", lambda: _test_future_event_perturbation(config, silver["matches"])))
    tests.append(("half-time boundary test", lambda: _test_half_time_boundary(config)))
    tests.append(("strict M90 versus full-time boundary test", lambda: _test_m90_and_full_time_boundaries(config)))
    tests.append(("raw and clipped goal-margin contract", lambda: _test_goal_margin_contract(silver["matches"], prematch, snapshots)))
    tests.append(("full foreign-key and event-order contract", lambda: _test_full_foreign_key_and_order_contract(config, key_audit)))
    tests.append(("canonical own-goal counting", _test_own_goal_is_counted_once))
    tests.append(("odds join uniqueness", lambda: (
        (lambda accepted: (
            (_ for _ in ()).throw(AssertionError("Accepted odds joins are not one-to-one"))
            if (accepted["match_id"].duplicated().any() or accepted["fd_row_id"].duplicated().any())
            else f"{len(accepted)} accepted one-to-one joins"
        ))(join_map[join_map["join_status"] == "accepted"])
    )))
    tests.append(("de-vig probability sum", lambda: (
        (lambda tagged: (
            (_ for _ in ()).throw(AssertionError("Market probabilities do not sum to one"))
            if not np.allclose(tagged.sum(axis=1), 1.0, atol=1e-10)
            else f"{len(tagged)} tagged rows sum to one"
        ))(prematch.loc[prematch["odds_tagged"] == 1, ["market_p_home", "market_p_draw", "market_p_away"]])
    )))

    rows = []
    for name, fn in tests:
        started = time.perf_counter()
        try:
            details = fn()
            rows.append({"test": name, "passed": True, "details": details, "seconds": time.perf_counter() - started})
        except Exception as exc:
            rows.append({"test": name, "passed": False, "details": repr(exc), "seconds": time.perf_counter() - started})
    results = pd.DataFrame(rows)
    save_table(results, config.audit_root / "leakage_test_results.csv")
    if not bool(results["passed"].all()):
        failures = results.loc[~results["passed"], ["test", "details"]]
        raise AssertionError(f"Leakage tests failed:\n{failures.to_string(index=False)}")
    return results


# -----------------------------------------------------------------------------
# Original notebook implementation section (cell 26)
# -----------------------------------------------------------------------------
def export_snapshot_audit_sample(config: MD1Config, snapshots: pd.DataFrame, match_id: int | None = None) -> pd.DataFrame:
    if match_id is None:
        test_rows = snapshots[snapshots["split"] == "test"]
        match_id = int(test_rows["match_id"].iloc[0] if not test_rows.empty else snapshots["match_id"].iloc[0])
    cols = [
        "match_id", "snapshot_id", "snapshot_kind", "snapshot_minute", "split",
        "snapshot_cutoff_event_order", "max_used_event_order", "next_excluded_event_order",
        "max_used_period", "max_used_minute", "max_used_second", "n_events_so_far",
        "current_home_score", "current_away_score", "current_goal_difference",
        "final_home_score", "final_away_score", "label_margin_raw", "label_margin",
        "home_live_red_cards", "away_live_red_cards", "man_advantage_home",
        "home_live_shots", "away_live_shots", "home_live_xg", "away_live_xg",
    ]
    sample = snapshots[snapshots["match_id"] == match_id][cols].sort_values(["snapshot_rank" if "snapshot_rank" in cols else "snapshot_minute", "snapshot_id"])
    # snapshot_rank is omitted from presentation columns, so sort before select if needed.
    sample = snapshots[snapshots["match_id"] == match_id].sort_values(["snapshot_rank", "snapshot_id"])[cols]
    save_table(sample, config.audit_root / "snapshot_audit_sample.csv")
    return sample


def write_run_summary(
    config: MD1Config,
    silver: Mapping[str, pd.DataFrame],
    prematch: pd.DataFrame,
    snapshots: pd.DataFrame,
    coverage: pd.DataFrame,
    tests: pd.DataFrame,
) -> Path:
    summary = {
        "data_mode": config.data_mode,
        "competition": config.competition_name,
        "season": config.season_name,
        "competition_id": config.competition_id,
        "season_id": config.season_id,
        "matches": len(silver["matches"]),
        "event_rows": int(silver["event_manifest"]["event_rows"].sum()),
        "lineup_rows": len(silver["lineups"]),
        "matches_with_360": int(silver["coverage_360"]["has_360"].sum()),
        "gold_prematch_rows": len(prematch),
        "gold_snapshot_rows": len(snapshots),
        "odds_coverage": coverage.to_dict("records"),
        "leakage_tests_passed": int(tests["passed"].sum()),
        "leakage_tests_total": len(tests),
        "snapshots_per_match": int(snapshots.groupby("match_id").size().iloc[0]) if not snapshots.empty else 0,
        "snapshot_boundaries": snapshots[["snapshot_id", "snapshot_kind", "snapshot_rank"]].drop_duplicates().sort_values("snapshot_rank").to_dict("records") if not snapshots.empty else [],
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    }
    path = config.audit_root / "run_summary.json"
    write_json(summary, path)
    return path


def run_full_pipeline(config: MD1Config) -> dict[str, Any]:
    np.random.seed(config.random_seed)
    config.ensure_directories()
    if config.data_mode == "synthetic_demo":
        create_synthetic_bronze(config)
    elif config.data_mode == "real":
        download_real_bronze(config)
    else:
        raise ValueError("data_mode must be 'synthetic_demo' or 'real'")

    silver = build_silver_tables(config)
    split_manifest = chronological_match_level_split(config, silver["matches"])
    team_facts = build_team_match_facts(silver["matches"], silver["match_event_facts"])
    rolling_facts = add_leakage_safe_rolling_features(team_facts, config.rolling_windows)
    assert_no_prematch_leakage(rolling_facts, config.rolling_windows)
    save_table(team_facts, config.silver_root / "team_match_facts.parquet")
    save_table(rolling_facts, config.silver_root / "team_match_facts_with_rolling.parquet")

    prematch = build_gold_prematch(config, silver["matches"], rolling_facts, split_manifest)
    prematch, join_map, coverage, excluded = integrate_odds(config, silver["matches"], silver["teams"], prematch)
    snapshots = build_gold_snapshots(config, silver["matches"], prematch, split_manifest)
    assert_snapshot_contracts(snapshots)

    feature_dictionary = build_feature_dictionary(config, prematch, snapshots)
    key_audit = build_key_audit(config, silver, prematch, snapshots)
    row_counts = build_table_row_counts(
        config,
        {
            **silver,
            "team_match_facts": team_facts,
            "gold_prematch": prematch,
            "gold_snapshots": snapshots,
            "split_manifest": split_manifest,
            "feature_dictionary": feature_dictionary,
        },
    )
    tests = run_leakage_tests(
        config, silver, team_facts, rolling_facts, prematch, snapshots, split_manifest, join_map, key_audit
    )
    snapshot_sample = export_snapshot_audit_sample(config, snapshots)
    summary_path = write_run_summary(config, silver, prematch, snapshots, coverage, tests)

    save_table(pd.DataFrame([asdict(config)]), config.audit_root / "resolved_config.csv")

    # Hard gate: invalid relational or leakage evidence must never flow into the defence archive.
    failed_key_checks = key_audit.loc[~key_audit["passed"].astype(bool)].copy()
    failed_leakage_tests = tests.loc[~tests["passed"].astype(bool)].copy()
    if not failed_key_checks.empty or not failed_leakage_tests.empty:
        blocked_lines = [
            "# SUBMISSION BLOCKED",
            "",
            "The football pipeline completed, but one or more mandatory checks failed.",
            "",
            "## Failed key/integrity checks",
            failed_key_checks.to_markdown(index=False) if not failed_key_checks.empty else "None",
            "",
            "## Failed leakage tests",
            failed_leakage_tests.to_markdown(index=False) if not failed_leakage_tests.empty else "None",
            "",
            "Inspect the audit files, correct the pipeline, and rerun all cells.",
        ]
        (config.project_root / "SUBMISSION_BLOCKED.md").write_text(
            "\n".join(blocked_lines) + "\n", encoding="utf-8"
        )
        raise AssertionError(
            "Mandatory integrity/leakage gate failed. "
            f"Key failures={len(failed_key_checks)}, leakage failures={len(failed_leakage_tests)}. "
            f"See {config.project_root / 'SUBMISSION_BLOCKED.md'}"
        )

    return {
        "config": config,
        "silver": silver,
        "split_manifest": split_manifest,
        "team_facts": team_facts,
        "rolling_facts": rolling_facts,
        "gold_prematch": prematch,
        "gold_snapshots": snapshots,
        "join_map": join_map,
        "odds_coverage": coverage,
        "excluded_odds_matches": excluded,
        "feature_dictionary": feature_dictionary,
        "key_audit": key_audit,
        "row_counts": row_counts,
        "leakage_tests": tests,
        "snapshot_audit_sample": snapshot_sample,
        "summary_path": summary_path,
    }
