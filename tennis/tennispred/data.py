"""Loading match history.

The expected format is Jeff Sackmann's public match files
(https://github.com/JeffSackmann/tennis_atp and tennis_wta, CC BY-NC-SA 4.0):
one CSV per year with winner_*/loser_* columns and per-player serve stats
(w_svpt, w_1stWon, w_2ndWon, ...), plus a players file with hand, height and
date of birth.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

RAW_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_{tour}/master"

ROUND_ORDER = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4, "RR": 5, "R128": 6, "R64": 7,
               "R32": 8, "R16": 9, "QF": 10, "SF": 11, "BR": 12, "F": 13}

MATCH_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_age", "winner_rank",
    "loser_id", "loser_name", "loser_hand", "loser_ht", "loser_age", "loser_rank",
    "score", "best_of", "round",
    "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon",
]


def download(data_dir: Path, tour: str = "atp", start_year: int = 1991, end_year: int | None = None,
             include_challengers: bool = False, force_recent: int = 2) -> None:
    """Fetch yearly match files and the players file into data_dir.

    Files already on disk are kept, except the most recent `force_recent`
    years which are refreshed because the upstream repo keeps appending to them.
    """
    end_year = end_year or pd.Timestamp.today().year
    data_dir.mkdir(parents=True, exist_ok=True)
    base = RAW_BASE.format(tour=tour)
    names = [f"{tour}_players.csv"]
    for y in range(start_year, end_year + 1):
        names.append(f"{tour}_matches_{y}.csv")
        if include_challengers and tour == "atp":
            names.append(f"{tour}_matches_qual_chall_{y}.csv")
    for name in names:
        dest = data_dir / name
        year = _year_of(name)
        fresh = year is None or year > end_year - force_recent
        if dest.exists() and not fresh:
            continue
        resp = _get(f"{base}/{name}")
        if resp is None:
            log.warning("missing upstream file %s", name)
            continue
        dest.write_bytes(resp.content)
        log.info("downloaded %s (%d bytes)", name, len(resp.content))


def _year_of(name: str) -> int | None:
    stem = Path(name).stem.split("_")[-1]
    return int(stem) if stem.isdigit() else None


def _get(url: str, retries: int = 4) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            log.warning("retrying %s: %s", url, exc)
            time.sleep(2 ** (attempt + 1))
    return None


def load_matches(data_dir: Path, tour: str = "atp", start_year: int = 1968) -> pd.DataFrame:
    """All yearly match files in data_dir, chronologically sorted."""
    files = sorted(p for p in data_dir.glob(f"{tour}_matches_*.csv")
                   if (_year_of(p.name) or 0) >= start_year)
    if not files:
        raise FileNotFoundError(f"no {tour}_matches_*.csv files in {data_dir}; run `download` first")
    frames = [pd.read_csv(p, low_memory=False) for p in files]
    df = pd.concat(frames, ignore_index=True)
    for col in MATCH_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[MATCH_COLUMNS].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date", "winner_id", "loser_id"])
    df["winner_id"] = df["winner_id"].astype(int)
    df["loser_id"] = df["loser_id"].astype(int)
    df["best_of"] = pd.to_numeric(df["best_of"], errors="coerce").fillna(3).astype(int)
    df["surface"] = df["surface"].fillna("Hard").replace({"": "Hard"})
    df["score"] = df["score"].fillna("").astype(str)
    df["round_order"] = df["round"].map(ROUND_ORDER).fillna(6)
    df = df.sort_values(["tourney_date", "tourney_id", "round_order", "match_num"], kind="stable")
    return df.reset_index(drop=True)


def load_players(data_dir: Path, tour: str = "atp") -> pd.DataFrame:
    path = data_dir / f"{tour}_players.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "name_first", "name_last", "hand", "dob", "height"])
    df = pd.read_csv(path, low_memory=False)
    df["dob"] = pd.to_datetime(df["dob"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    df["height"] = pd.to_numeric(df.get("height"), errors="coerce")
    return df


def is_completed(score: str) -> bool:
    """False for walkovers, retirements and defaults."""
    s = score.upper()
    return bool(s) and not any(tag in s for tag in ("W/O", "RET", "DEF", "WALKOVER", "ABD", "UNFINISHED"))
