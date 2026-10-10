"""Synthetic match history in Sackmann's format, with known ground truth.

Used by the tests and the offline demo: every match is played point by point
with the random walk in ``markov.py``, using serve probabilities that depend on
hidden player skills plus a few planted effects (lefty advantage, height on
grass, ageing). The fitted model should recover the signs of those effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .markov import simulate_match

BASE_SPW = 0.63
TRUE_EFFECTS = {
    "lefty_srv_vs_righty": 0.12,   # serve log-odds bonus, lefty serving to a righty
    "height_grass": 0.08,          # per 10 cm above 185, on grass
    "surface": {"Hard": 0.0, "Clay": -0.12, "Grass": 0.15},
}


FIRST_NAMES = ["Alex", "Ben", "Carlos", "Daniil", "Emil", "Felix", "Grigor", "Holger", "Ivan", "Jannik"]


def _letters(i: int) -> str:
    """0 -> 'a', 25 -> 'z', 26 -> 'ba', ... (names must be alphabetic to be matchable)."""
    out = ""
    while True:
        out = chr(ord("a") + i % 26) + out
        i //= 26
        if i == 0:
            return out


def _season_surface(week: int) -> str:
    if 14 <= week < 23:
        return "Clay"
    if 23 <= week < 28:
        return "Grass"
    return "Hard"


def _score_string(sim, winner_is_a: bool) -> str:
    parts = []
    for (ga, gb), tb in zip(sim.sets, sim.tiebreaks):
        w, l = (ga, gb) if winner_is_a else (gb, ga)
        s = f"{w}-{l}"
        if tb is not None:
            s += f"({min(tb)})"
        parts.append(s)
    return " ".join(parts)


def generate(n_players: int = 160, start_year: int = 2016, n_years: int = 6, draw: int = 32,
             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pid = 100000 + np.arange(n_players)
    serve = rng.normal(0, 0.22, n_players)
    ret = rng.normal(0, 0.18, n_players)
    hand = np.where(rng.random(n_players) < 0.13, "L", "R")
    height = np.round(rng.normal(186, 7, n_players))
    birth_year = rng.integers(start_year - 34, start_year - 17, n_players)
    dob = pd.to_datetime([f"{y}0615" for y in birth_year], format="%Y%m%d")
    first = [FIRST_NAMES[i % len(FIRST_NAMES)] for i in range(n_players)]
    last = [f"Synth{_letters(i)}" for i in range(n_players)]
    players = pd.DataFrame({"player_id": pid, "name_first": first, "name_last": last, "hand": hand,
                            "dob": dob.strftime("%Y%m%d").astype(int), "ioc": "SYN", "height": height})

    def logit_spw(s: int, r: int, surface: str, date: pd.Timestamp) -> float:
        age = (date - dob[s]).days / 365.25
        age_eff = -0.004 * (age - 26) ** 2
        z = np.log(BASE_SPW / (1 - BASE_SPW)) + serve[s] - ret[r] + age_eff
        z += TRUE_EFFECTS["surface"][surface]
        if hand[s] == "L" and hand[r] == "R":
            z += TRUE_EFFECTS["lefty_srv_vs_righty"]
        if surface == "Grass":
            z += TRUE_EFFECTS["height_grass"] * (height[s] - 185) / 10
        return z

    rows = []
    rounds = {32: ["R32", "R16", "QF", "SF", "F"], 16: ["R16", "QF", "SF", "F"]}[draw]
    for year in range(start_year, start_year + n_years):
        for week in range(2, 46):
            date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(weeks=week)
            date -= pd.Timedelta(days=date.weekday())
            surface = _season_surface(week)
            slam = week in (3, 21, 26, 35)
            best_of = 5 if slam else 3
            level = "G" if slam else ("M" if week % 4 == 0 else "A")
            # Slowly drifting skills
            serve += rng.normal(0, 0.01, n_players)
            ret += rng.normal(0, 0.01, n_players)
            strength = serve + ret
            rank = np.empty(n_players)
            rank[np.argsort(-strength)] = np.arange(1, n_players + 1)
            entrants = rng.choice(n_players, draw, replace=False,
                                  p=np.exp(2 * strength) / np.exp(2 * strength).sum())
            alive = list(entrants)
            for rnd in rounds:
                nxt = []
                for j in range(0, len(alive), 2):
                    a, b = alive[j], alive[j + 1]
                    pa = 1 / (1 + np.exp(-logit_spw(a, b, surface, date)))
                    pb = 1 / (1 + np.exp(-logit_spw(b, a, surface, date)))
                    sim = simulate_match(pa, pb, best_of, rng)
                    w, l = (a, b) if sim.a_won else (b, a)
                    sp = sim.serve_points
                    ws, wk = (sp["a_svpt"], sp["a_svpt_won"]) if sim.a_won else (sp["b_svpt"], sp["b_svpt_won"])
                    ls, lk = (sp["b_svpt"], sp["b_svpt_won"]) if sim.a_won else (sp["a_svpt"], sp["a_svpt_won"])
                    first_won_w = int(round(wk * 0.7))
                    first_won_l = int(round(lk * 0.7))
                    rows.append({
                        "tourney_id": f"{year}-{week:02d}", "tourney_name": f"Synth Open {week}",
                        "surface": surface, "draw_size": draw, "tourney_level": level,
                        "tourney_date": int(date.strftime("%Y%m%d")), "match_num": len(rows),
                        "winner_id": pid[w], "winner_name": f"{first[w]} {last[w]}", "winner_hand": hand[w],
                        "winner_ht": height[w], "winner_age": round((date - dob[w]).days / 365.25, 1),
                        "winner_rank": rank[w],
                        "loser_id": pid[l], "loser_name": f"{first[l]} {last[l]}", "loser_hand": hand[l],
                        "loser_ht": height[l], "loser_age": round((date - dob[l]).days / 365.25, 1),
                        "loser_rank": rank[l],
                        "score": _score_string(sim, sim.a_won), "best_of": best_of, "round": rnd,
                        "w_svpt": ws, "w_1stWon": first_won_w, "w_2ndWon": wk - first_won_w,
                        "l_svpt": ls, "l_1stWon": first_won_l, "l_2ndWon": lk - first_won_l,
                    })
                    nxt.append(w)
                alive = nxt
    return pd.DataFrame(rows), players


def write(data_dir: Path, tour: str = "atp", **kwargs) -> None:
    matches, players = generate(**kwargs)
    data_dir.mkdir(parents=True, exist_ok=True)
    matches["year"] = matches["tourney_date"] // 10000
    for year, g in matches.groupby("year"):
        g.drop(columns="year").to_csv(data_dir / f"{tour}_matches_{year}.csv", index=False)
    players.to_csv(data_dir / f"{tour}_players.csv", index=False)
