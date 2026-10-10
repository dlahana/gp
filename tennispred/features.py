"""Point-in-time player state and serve-point features.

The history is replayed in date order. Before each match, a feature vector is
built for each *serving direction* (A serving to B, and B serving to A) from
state that existed before the match. The match result then updates the state.
That ordering prevents leakage, so a walk-forward backtest is honest.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import is_completed

SURFACES = ("Hard", "Clay", "Grass", "Carpet")

FEATURES = [
    "elo_diff", "surf_elo_diff",
    "s_serve", "r_return", "s_return", "r_serve",
    "form_diff", "log_rank_ratio",
    "s_left", "r_left", "lefty_srv_vs_righty", "lefty_ret_vs_righty",
    "s_height", "r_height", "s_height_missing", "r_height_missing",
    "s_height_grass", "s_height_clay",
    "s_age", "r_age", "s_age2", "r_age2",
    "s_fatigue", "r_fatigue", "s_rust", "r_rust",
    "s_exp", "r_exp", "h2h",
    "surf_grass", "surf_clay", "surf_carpet",
    "best_of_5", "lvl_slam", "lvl_masters", "lvl_finals", "lvl_davis", "lvl_challenger",
]

# Tunables
SERVE_DECAY = 0.95          # per-match decay of serve/return point tallies
SERVE_PRIOR_POINTS = 150.0  # shrinkage of a player's serve rate towards tour average
FORM_WINDOW = 10
FATIGUE_DAYS = 14
DEFAULT_RANK = 500
MEAN_HEIGHT = 185.0


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def elo_k(n_matches: int) -> float:
    """FiveThirtyEight-style decaying K factor."""
    return 250.0 / (n_matches + 5) ** 0.4


def elo_expect(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


@dataclass
class PlayerState:
    name: str = ""
    hand: str = "U"
    height: float = float("nan")
    dob: pd.Timestamp | None = None
    age_ref: tuple[float, pd.Timestamp] | None = None  # (age, date) when dob unknown
    elo: float = 1500.0
    surf_elo: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    n: int = 0
    n_surf: dict = field(default_factory=lambda: defaultdict(int))
    sv_won: float = 0.0
    sv_tot: float = 0.0
    rt_won: float = 0.0
    rt_tot: float = 0.0
    results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    dates: deque = field(default_factory=lambda: deque(maxlen=30))
    rank: float = float("nan")

    def age_at(self, date: pd.Timestamp) -> float:
        if self.dob is not None and not pd.isna(self.dob):
            return (date - self.dob).days / 365.25
        if self.age_ref is not None:
            age, ref = self.age_ref
            return age + (date - ref).days / 365.25
        return float("nan")


@dataclass
class Context:
    date: pd.Timestamp
    surface: str = "Hard"
    best_of: int = 3
    level: str = "A"


class History:
    """Replays matches and serves point-in-time features."""

    def __init__(self, players: pd.DataFrame | None = None):
        self.players: dict[int, PlayerState] = {}
        self.h2h: dict[tuple[int, int], int] = defaultdict(int)
        self.tour_sv_won = 0.0
        self.tour_sv_tot = 0.0
        self.last_date: pd.Timestamp | None = None
        self.tourney_surface: dict[str, str] = {}
        self.tourney_best_of: dict[str, int] = {}
        self.tourney_level: dict[str, str] = {}
        if players is not None:
            for row in players.itertuples(index=False):
                st = self.get(int(row.player_id))
                st.name = f"{getattr(row, 'name_first', '') or ''} {getattr(row, 'name_last', '') or ''}".strip()
                st.hand = row.hand if isinstance(row.hand, str) else "U"
                if not pd.isna(row.height):
                    st.height = float(row.height)
                if not pd.isna(row.dob):
                    st.dob = row.dob

    def get(self, pid: int) -> PlayerState:
        st = self.players.get(pid)
        if st is None:
            st = self.players[pid] = PlayerState()
        return st

    # ---------------------------------------------------------------- features

    @property
    def tour_spw(self) -> float:
        return self.tour_sv_won / self.tour_sv_tot if self.tour_sv_tot > 0 else 0.63

    def _serve_rate(self, st: PlayerState) -> float:
        m = SERVE_PRIOR_POINTS
        return (st.sv_won + m * self.tour_spw) / (st.sv_tot + m)

    def _return_rate(self, st: PlayerState) -> float:
        m = SERVE_PRIOR_POINTS
        return (st.rt_won + m * (1 - self.tour_spw)) / (st.rt_tot + m)

    @staticmethod
    def _form(st: PlayerState) -> float:
        return (sum(st.results) + 2.5) / (len(st.results) + 5)

    @staticmethod
    def _fatigue(st: PlayerState, date: pd.Timestamp) -> float:
        return float(sum(1 for d in st.dates if (date - d).days <= FATIGUE_DAYS))

    @staticmethod
    def _rust(st: PlayerState, date: pd.Timestamp) -> float:
        if not st.dates:
            return math.log1p(60)
        return math.log1p(max((date - st.dates[-1]).days, 0))

    def features(self, server: int, returner: int, ctx: Context) -> np.ndarray:
        s, r = self.get(server), self.get(returner)
        surf = ctx.surface if ctx.surface in SURFACES else "Hard"
        tour = self.tour_spw
        s_left = 1.0 if s.hand == "L" else 0.0
        r_left = 1.0 if r.hand == "L" else 0.0
        s_h = 0.0 if math.isnan(s.height) else (s.height - MEAN_HEIGHT) / 10
        r_h = 0.0 if math.isnan(r.height) else (r.height - MEAN_HEIGHT) / 10
        s_age, r_age = s.age_at(ctx.date), r.age_at(ctx.date)
        s_a = 0.0 if math.isnan(s_age) else (s_age - 26) / 5
        r_a = 0.0 if math.isnan(r_age) else (r_age - 26) / 5
        s_rank = DEFAULT_RANK if math.isnan(s.rank) else s.rank
        r_rank = DEFAULT_RANK if math.isnan(r.rank) else r.rank
        w_sr, w_rs = self.h2h[(server, returner)], self.h2h[(returner, server)]
        lvl = ctx.level
        f = {
            "elo_diff": (s.elo - r.elo) / 400,
            "surf_elo_diff": (s.surf_elo[surf] - r.surf_elo[surf]) / 400,
            "s_serve": _logit(self._serve_rate(s)) - _logit(tour),
            "r_return": _logit(self._return_rate(r)) - _logit(1 - tour),
            "s_return": _logit(self._return_rate(s)) - _logit(1 - tour),
            "r_serve": _logit(self._serve_rate(r)) - _logit(tour),
            "form_diff": self._form(s) - self._form(r),
            "log_rank_ratio": math.log(r_rank) - math.log(s_rank),
            "s_left": s_left,
            "r_left": r_left,
            "lefty_srv_vs_righty": s_left * (1 - r_left),
            "lefty_ret_vs_righty": r_left * (1 - s_left),
            "s_height": s_h,
            "r_height": r_h,
            "s_height_missing": float(math.isnan(s.height)),
            "r_height_missing": float(math.isnan(r.height)),
            "s_height_grass": s_h * (surf == "Grass"),
            "s_height_clay": s_h * (surf == "Clay"),
            "s_age": s_a,
            "r_age": r_a,
            "s_age2": s_a * s_a,
            "r_age2": r_a * r_a,
            "s_fatigue": self._fatigue(s, ctx.date),
            "r_fatigue": self._fatigue(r, ctx.date),
            "s_rust": self._rust(s, ctx.date),
            "r_rust": self._rust(r, ctx.date),
            "s_exp": math.log1p(s.n),
            "r_exp": math.log1p(r.n),
            "h2h": (w_sr - w_rs) / (w_sr + w_rs + 2),
            "surf_grass": float(surf == "Grass"),
            "surf_clay": float(surf == "Clay"),
            "surf_carpet": float(surf == "Carpet"),
            "best_of_5": float(ctx.best_of == 5),
            "lvl_slam": float(lvl == "G"),
            "lvl_masters": float(lvl in ("M", "PM", "P")),
            "lvl_finals": float(lvl == "F"),
            "lvl_davis": float(lvl in ("D", "O")),
            "lvl_challenger": float(lvl in ("C", "S", "15", "25", "60", "80", "100", "125")),
        }
        return np.array([f[k] for k in FEATURES], dtype=float)

    def elo_prob(self, a: int, b: int, surface: str, surface_weight: float = 0.5) -> float:
        sa, sb = self.get(a), self.get(b)
        ra = (1 - surface_weight) * sa.elo + surface_weight * sa.surf_elo[surface]
        rb = (1 - surface_weight) * sb.elo + surface_weight * sb.surf_elo[surface]
        return elo_expect(ra, rb)

    # ------------------------------------------------------------------ update

    def _absorb_row_info(self, pid: int, row, prefix: str, date: pd.Timestamp) -> None:
        st = self.get(pid)
        name = getattr(row, f"{prefix}_name")
        if isinstance(name, str) and name:
            st.name = name
        hand = getattr(row, f"{prefix}_hand")
        if st.hand == "U" and isinstance(hand, str) and hand in ("L", "R"):
            st.hand = hand
        ht = getattr(row, f"{prefix}_ht")
        if math.isnan(st.height) and not pd.isna(ht):
            st.height = float(ht)
        age = getattr(row, f"{prefix}_age")
        if st.dob is None and not pd.isna(age):
            st.age_ref = (float(age), date)
        rank = getattr(row, f"{prefix}_rank")
        if not pd.isna(rank):
            st.rank = float(rank)

    def update(self, row) -> None:
        date = row.tourney_date
        w, l = int(row.winner_id), int(row.loser_id)
        sw, sl = self.get(w), self.get(l)
        surf = row.surface if row.surface in SURFACES else "Hard"
        self.last_date = date
        if isinstance(row.tourney_name, str):
            self.tourney_surface[row.tourney_name] = surf
            self.tourney_best_of[row.tourney_name] = int(row.best_of)
            self.tourney_level[row.tourney_name] = str(row.tourney_level)

        # Serve / return point tallies (available even for most retirements).
        if _has_stats(row):
            w_won = float(row.w_1stWon) + float(row.w_2ndWon)
            l_won = float(row.l_1stWon) + float(row.l_2ndWon)
            w_sv, l_sv = float(row.w_svpt), float(row.l_svpt)
            for st in (sw, sl):
                for attr in ("sv_won", "sv_tot", "rt_won", "rt_tot"):
                    setattr(st, attr, getattr(st, attr) * SERVE_DECAY)
            sw.sv_won += w_won; sw.sv_tot += w_sv
            sl.sv_won += l_won; sl.sv_tot += l_sv
            sw.rt_won += l_sv - l_won; sw.rt_tot += l_sv
            sl.rt_won += w_sv - w_won; sl.rt_tot += w_sv
            self.tour_sv_won = 0.9999 * self.tour_sv_won + w_won + l_won
            self.tour_sv_tot = 0.9999 * self.tour_sv_tot + w_sv + l_sv

        self._absorb_row_info(w, row, "winner", date)
        self._absorb_row_info(l, row, "loser", date)

        if "W/O" in row.score.upper() or not row.score:
            return  # walkover: no tennis was played
        sw.dates.append(date)
        sl.dates.append(date)
        if not is_completed(row.score):
            return  # retirement: counts for fatigue, not for ratings

        # Elo (overall + surface)
        e = elo_expect(sw.elo, sl.elo)
        kw, kl = elo_k(sw.n), elo_k(sl.n)
        sw.elo += kw * (1 - e)
        sl.elo -= kl * (1 - e)
        es = elo_expect(sw.surf_elo[surf], sl.surf_elo[surf])
        ksw, ksl = elo_k(sw.n_surf[surf]), elo_k(sl.n_surf[surf])
        sw.surf_elo[surf] += ksw * (1 - es)
        sl.surf_elo[surf] -= ksl * (1 - es)
        sw.n += 1; sl.n += 1
        sw.n_surf[surf] += 1; sl.n_surf[surf] += 1
        sw.results.append(1); sl.results.append(0)
        self.h2h[(w, l)] += 1


def _has_stats(row) -> bool:
    try:
        return (float(row.w_svpt) >= 10 and float(row.l_svpt) >= 10
                and not pd.isna(row.w_1stWon) and not pd.isna(row.l_1stWon)
                and not pd.isna(row.w_2ndWon) and not pd.isna(row.l_2ndWon))
    except (TypeError, ValueError):
        return False


@dataclass
class FeatureTable:
    """One row per match; X_w = winner serving to loser, X_l = loser serving."""

    matches: pd.DataFrame
    X_w: np.ndarray
    X_l: np.ndarray
    k_w: np.ndarray   # serve points won by winner
    n_w: np.ndarray   # serve points played by winner
    k_l: np.ndarray
    n_l: np.ndarray
    elo_p: np.ndarray  # pre-match Elo P(winner wins), as a baseline

    def serve_observations(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        has = mask & (self.n_w > 0) & (self.n_l > 0)
        X = np.vstack([self.X_w[has], self.X_l[has]])
        k = np.concatenate([self.k_w[has], self.k_l[has]])
        n = np.concatenate([self.n_w[has], self.n_l[has]])
        return X, k, n


def build(matches: pd.DataFrame, players: pd.DataFrame | None = None) -> tuple[History, FeatureTable]:
    hist = History(players)
    m = len(matches)
    X_w = np.zeros((m, len(FEATURES)))
    X_l = np.zeros((m, len(FEATURES)))
    k_w, n_w, k_l, n_l = (np.zeros(m) for _ in range(4))
    elo_p = np.zeros(m)
    for i, row in enumerate(matches.itertuples(index=False)):
        w, l = int(row.winner_id), int(row.loser_id)
        # Absorb static info (hand/height/age/rank) first: it is known pre-match.
        hist._absorb_row_info(w, row, "winner", row.tourney_date)
        hist._absorb_row_info(l, row, "loser", row.tourney_date)
        ctx = Context(row.tourney_date, row.surface, int(row.best_of), str(row.tourney_level))
        X_w[i] = hist.features(w, l, ctx)
        X_l[i] = hist.features(l, w, ctx)
        elo_p[i] = hist.elo_prob(w, l, ctx.surface if ctx.surface in SURFACES else "Hard")
        if _has_stats(row):
            k_w[i] = float(row.w_1stWon) + float(row.w_2ndWon)
            n_w[i] = float(row.w_svpt)
            k_l[i] = float(row.l_1stWon) + float(row.l_2ndWon)
            n_l[i] = float(row.l_svpt)
        hist.update(row)
    matches = matches.reset_index(drop=True).copy()
    matches["completed"] = matches["score"].map(is_completed)
    table = FeatureTable(matches, X_w, X_l, k_w, n_w, k_l, n_l, elo_p)
    return hist, table
