"""Where the day's matches come from.

Two sources are provided:

* ``CsvFixtureSource``: a CSV/JSON you (or another job) drop in place, with
  columns date, player1, player2 and optionally tournament, surface, best_of,
  level, round.
* ``ApiTennisSource``: api-tennis.com's ``get_fixtures`` endpoint (needs an API
  key in TENNIS_API_KEY). Its field names follow api-tennis's public docs;
  check them against a live response before relying on it.

Anything that yields ``Fixture`` objects can be plugged in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

SLAM_KEYWORDS = ("australian open", "roland garros", "french open", "wimbledon", "us open")


@dataclass
class Fixture:
    date: pd.Timestamp
    player1: str
    player2: str
    tournament: str = ""
    surface: str | None = None
    best_of: int | None = None
    level: str | None = None
    round: str = ""
    start_time: str = ""


def infer_context(fx: Fixture, tour: str, known_surface: dict[str, str],
                  known_best_of: dict[str, int], known_level: dict[str, str]) -> Fixture:
    """Fill in surface / best_of / level from the tournament name when missing."""
    name = fx.tournament.lower()
    match = next((k for k in known_surface if k.lower() == name), None)
    if match is None:
        match = next((k for k in known_surface if k.lower() in name or name in k.lower()), None) if name else None
    slam = any(s in name for s in SLAM_KEYWORDS)
    if fx.surface is None:
        if match:
            fx.surface = known_surface[match]
        elif "wimbledon" in name:
            fx.surface = "Grass"
        elif "roland garros" in name or "french open" in name:
            fx.surface = "Clay"
        else:
            fx.surface = "Hard"
    if fx.level is None:
        fx.level = "G" if slam else (known_level.get(match, "A") if match else "A")
    if fx.best_of is None:
        if tour == "wta":
            fx.best_of = 3
        elif slam:
            fx.best_of = 5
        else:
            fx.best_of = known_best_of.get(match, 3) if match else 3
    return fx


class CsvFixtureSource:
    def __init__(self, path: Path):
        self.path = Path(path)

    def fixtures(self, day: pd.Timestamp) -> list[Fixture]:
        df = pd.read_json(self.path) if self.path.suffix == ".json" else pd.read_csv(self.path)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.normalize() == day.normalize()]
        out = []
        for r in df.to_dict("records"):
            out.append(Fixture(
                date=r["date"], player1=str(r["player1"]), player2=str(r["player2"]),
                tournament=str(r.get("tournament") or ""),
                surface=r.get("surface") if isinstance(r.get("surface"), str) else None,
                best_of=int(r["best_of"]) if pd.notna(r.get("best_of", None)) else None,
                level=r.get("level") if isinstance(r.get("level"), str) else None,
                round=str(r.get("round") or ""),
            ))
        return out


class ApiTennisSource:
    URL = "https://api.api-tennis.com/tennis/"

    def __init__(self, api_key: str | None = None, tour: str = "atp"):
        self.api_key = api_key or os.environ.get("TENNIS_API_KEY")
        if not self.api_key:
            raise RuntimeError("set TENNIS_API_KEY to use the api-tennis.com fixture source")
        self.event_type = {"atp": "atp singles", "wta": "wta singles"}[tour]

    def fixtures(self, day: pd.Timestamp) -> list[Fixture]:
        d = day.strftime("%Y-%m-%d")
        resp = requests.get(self.URL, params={"method": "get_fixtures", "APIkey": self.api_key,
                                              "date_start": d, "date_stop": d}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success", 1):
            raise RuntimeError(f"api-tennis error: {payload}")
        out = []
        for ev in payload.get("result", []) or []:
            if str(ev.get("event_type_type", "")).lower() != self.event_type:
                continue
            if str(ev.get("event_status", "")).lower() in ("finished", "retired", "walk over", "cancelled"):
                continue
            p1, p2 = ev.get("event_first_player"), ev.get("event_second_player")
            if not p1 or not p2 or "/" in p1 or "/" in p2:   # skip doubles pairs
                continue
            round_name = str(ev.get("tournament_round") or "")
            if "qualif" in round_name.lower():
                continue
            out.append(Fixture(date=pd.Timestamp(ev.get("event_date", d)), player1=p1, player2=p2,
                               tournament=str(ev.get("tournament_name") or ""), round=round_name,
                               start_time=str(ev.get("event_time") or "")))
        log.info("api-tennis: %d %s fixtures on %s", len(out), self.event_type, d)
        return out
