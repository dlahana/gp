"""Tennis as a hierarchy of random walks.

A point is a Bernoulli trial whose success probability depends only on who is
serving. Games, tiebreaks, sets and matches are then absorbing random walks over
the score, and their win probabilities follow exactly from the two serve-point
probabilities:

    p_a = P(A wins a point on A's serve)
    p_b = P(B wins a point on B's serve)

Everything here is exact dynamic programming, and every function accepts either
floats or equal-shaped numpy arrays (the recursion only does arithmetic), so a
whole season of matches can be evaluated in one call. ``simulate_match`` is a Monte
Carlo version of the same walk, used for testing and for synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------

def p_game(p: float) -> float:
    """P(server holds) when the server wins each point with probability p."""
    q = 1.0 - p
    # Win to 0, 15 or 30, or reach deuce (3-3) and win the deuce random walk.
    before_deuce = p**4 * (1 + 4 * q + 10 * q**2)
    reach_deuce = 20 * p**3 * q**3
    # 1 - 2pq >= 1/2, so this is safe for any p (scalar or array).
    win_from_deuce = p**2 / (1 - 2 * p * q)
    return before_deuce + reach_deuce * win_from_deuce


# --------------------------------------------------------------------------
# Tiebreak
# --------------------------------------------------------------------------

def _tb_server_is_a(n_points: int) -> bool:
    """A serves point 0, then B serves 1-2, A serves 3-4, ..."""
    return ((n_points + 1) // 2) % 2 == 0


def p_tiebreak(p_a: float, p_b: float, target: int = 7) -> float:
    """P(A wins a first-to-`target`, win-by-two tiebreak; A serves first)."""

    @lru_cache(maxsize=None)
    def win(a: int, b: int) -> float:
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a == b and a >= target - 1:
            # Tied late: the walk repeats every two points (one serve each,
            # because the server changes on odd point counts).
            n = a + b
            if _tb_server_is_a(n):
                w, l = p_a * (1 - p_b), (1 - p_a) * p_b
            else:
                w, l = p_b * (1 - p_a), (1 - p_b) * p_a
            return w / (w + l)
        pa_point = p_a if _tb_server_is_a(a + b) else 1 - p_b
        return pa_point * win(a + 1, b) + (1 - pa_point) * win(a, b + 1)

    return win(0, 0)


# --------------------------------------------------------------------------
# Set
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SetOutcome:
    """Joint distribution of (set winner, parity of games played).

    Parity matters because it decides who serves first in the next set.
    A tiebreak counts as one game for this purpose.
    """

    a_even: float
    a_odd: float
    b_even: float
    b_odd: float

    @property
    def p_a(self) -> float:
        return self.a_even + self.a_odd


def set_outcome(p_a: float, p_b: float, a_serves_first: bool,
                tiebreak_at: int | None = 6, tb_target: int = 7) -> SetOutcome:
    """Distribution over set results.

    tiebreak_at: games-all score at which a tiebreak is played; None means an
    advantage set (play on until someone leads by two games).
    """
    hold_a, hold_b = p_game(p_a), p_game(p_b)
    tb_a_first = p_tiebreak(p_a, p_b, tb_target)
    tb_b_first = 1 - p_tiebreak(p_b, p_a, tb_target)

    @lru_cache(maxsize=None)
    def walk(ga: int, gb: int) -> tuple[float, float, float, float]:
        n = ga + gb
        if (ga >= 6 and ga - gb >= 2) or (tiebreak_at is not None and ga == tiebreak_at + 1 and gb == tiebreak_at):
            return (1.0, 0.0, 0.0, 0.0) if n % 2 == 0 else (0.0, 1.0, 0.0, 0.0)
        if (gb >= 6 and gb - ga >= 2) or (tiebreak_at is not None and gb == tiebreak_at + 1 and ga == tiebreak_at):
            return (0.0, 0.0, 1.0, 0.0) if n % 2 == 0 else (0.0, 0.0, 0.0, 1.0)
        a_serving = (n % 2 == 0) == a_serves_first
        if tiebreak_at is not None and ga == gb == tiebreak_at:
            pw = tb_a_first if a_serving else tb_b_first
            # Tiebreak ends the set at 7-6: 2*tiebreak_at + 1 games (odd).
            return (0.0, pw, 0.0, 1 - pw) if (n + 1) % 2 == 1 else (pw, 0.0, 1 - pw, 0.0)
        if tiebreak_at is None and ga == gb and ga >= 5:
            # Advantage set at 5-5, 6-6, ...: two games (one serve each) decide
            # whether someone breaks clear. Solve the repeating walk exactly.
            h1 = hold_a if a_serving else hold_b
            h2 = hold_b if a_serving else hold_a
            # A wins both games: A holds+breaks in either order.
            if a_serving:
                w, l = h1 * (1 - h2), (1 - h1) * h2
            else:
                w, l = (1 - h1) * h2, h1 * (1 - h2)
            pw = w / (w + l)
            # Ending score is (k+2, k): even total when n is even.
            par_even = (n % 2 == 0)
            return (pw, 0.0, 1 - pw, 0.0) if par_even else (0.0, pw, 0.0, 1 - pw)
        pg = hold_a if a_serving else 1 - hold_b
        win_next = walk(ga + 1, gb)
        lose_next = walk(ga, gb + 1)
        return tuple(pg * x + (1 - pg) * y for x, y in zip(win_next, lose_next))

    return SetOutcome(*walk(0, 0))


# --------------------------------------------------------------------------
# Match
# --------------------------------------------------------------------------

@dataclass
class MatchProbabilities:
    p_a: float                       # P(A wins the match)
    set_scores: dict[str, float] = field(default_factory=dict)  # "2-1" from A's view
    p_hold_a: float = 0.0
    p_hold_b: float = 0.0
    p_set_a: float = 0.0             # P(A wins a set when serving first), averaged

    def most_likely_score(self) -> tuple[str, float]:
        return max(self.set_scores.items(), key=lambda kv: kv[1])


def match_probabilities(p_a: float, p_b: float, best_of: int = 3,
                        final_set_tb_target: int = 7,
                        final_set_tiebreak_at: int | None = 6) -> MatchProbabilities:
    """Exact match-win probability and set-score distribution.

    The first server is decided by a coin toss, so both cases are averaged.
    final_set_tb_target=10 models the Grand Slam final-set rule since 2022.
    """
    p_a = np.clip(p_a, 1e-6, 1 - 1e-6)
    p_b = np.clip(p_b, 1e-6, 1 - 1e-6)
    sets_to_win = best_of // 2 + 1

    regular = {first: set_outcome(p_a, p_b, first) for first in (True, False)}
    final = {first: set_outcome(p_a, p_b, first, final_set_tiebreak_at, final_set_tb_target)
             for first in (True, False)}

    scores: dict[str, float] = {}

    def walk(sa: int, sb: int, a_first: bool, prob: float) -> None:
        if sa == sets_to_win or sb == sets_to_win:
            key = f"{sa}-{sb}"
            scores[key] = scores.get(key, 0.0) + prob
            return
        is_final = sa == sb == sets_to_win - 1
        o = (final if is_final else regular)[a_first]
        # Even number of games: same player serves first next set.
        walk(sa + 1, sb, a_first, prob * o.a_even)
        walk(sa + 1, sb, not a_first, prob * o.a_odd)
        walk(sa, sb + 1, a_first, prob * o.b_even)
        walk(sa, sb + 1, not a_first, prob * o.b_odd)

    walk(0, 0, True, 0.5 * np.ones_like(p_a))
    walk(0, 0, False, 0.5 * np.ones_like(p_a))

    p_win = sum(v for k, v in scores.items() if int(k.split("-")[0]) == sets_to_win)
    if np.ndim(p_win) == 0:
        p_win = float(p_win)
        scores = {k: float(v) for k, v in scores.items()}
    return MatchProbabilities(
        p_a=p_win,
        set_scores=dict(sorted(scores.items())),
        p_hold_a=p_game(p_a),
        p_hold_b=p_game(p_b),
        p_set_a=0.5 * (regular[True].p_a + regular[False].p_a),
    )


def p_match(p_a, p_b, best_of: int = 3, final_set_tb_target: int = 7):
    """P(A wins the match). Accepts scalars or numpy arrays (vectorised)."""
    return match_probabilities(p_a, p_b, best_of, final_set_tb_target).p_a


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

@dataclass
class SimulatedMatch:
    a_won: bool
    sets: list[tuple[int, int]]           # games per set, (A, B)
    tiebreaks: list[tuple[int, int] | None]
    serve_points: dict[str, int]          # svpt / svpt_won for each player
    longest_game: str = ""                # point sequence of the longest game, e.g. "SRSRSS..."
    longest_game_server_is_a: bool = True  # (S = server won the point, R = returner won)

    @property
    def total_points(self) -> int:
        return self.serve_points["a_svpt"] + self.serve_points["b_svpt"]


def simulate_match(p_a: float, p_b: float, best_of: int = 3, rng: np.random.Generator | None = None,
                   final_set_tb_target: int = 7) -> SimulatedMatch:
    """Play one match point by point."""
    rng = rng or np.random.default_rng()
    stats = {"a_svpt": 0, "a_won": 0, "b_svpt": 0, "b_won": 0}

    def point(a_serving: bool) -> bool:
        """Returns True if A wins the point."""
        if a_serving:
            stats["a_svpt"] += 1
            w = rng.random() < p_a
            stats["a_won"] += w
            return w
        stats["b_svpt"] += 1
        w = rng.random() < p_b
        stats["b_won"] += w
        return not w

    longest = {"seq": "", "a_serving": True}

    def game(a_serving: bool) -> bool:
        a = b = 0
        seq = []
        while True:
            won = point(a_serving)
            seq.append("S" if won == a_serving else "R")
            if won:
                a += 1
            else:
                b += 1
            if (a >= 4 or b >= 4) and abs(a - b) >= 2:
                if len(seq) > len(longest["seq"]):
                    longest["seq"], longest["a_serving"] = "".join(seq), a_serving
                return a > b

    def tiebreak(a_first: bool, target: int) -> tuple[bool, tuple[int, int]]:
        a = b = 0
        while True:
            n = a + b
            a_serving = _tb_server_is_a(n) == a_first
            if point(a_serving):
                a += 1
            else:
                b += 1
            if a >= target and a - b >= 2:
                return True, (a, b)
            if b >= target and b - a >= 2:
                return False, (a, b)

    sets_to_win = best_of // 2 + 1
    sa = sb = 0
    a_first = bool(rng.random() < 0.5)
    sets, tbs = [], []
    while sa < sets_to_win and sb < sets_to_win:
        is_final = sa == sb == sets_to_win - 1
        target = final_set_tb_target if is_final else 7
        ga = gb = 0
        tb = None
        while True:
            n = ga + gb
            a_serving = (n % 2 == 0) == a_first
            if ga == gb == 6:
                won, tb = tiebreak(a_serving, target)
                ga, gb = (7, 6) if won else (6, 7)
                break
            if game(a_serving):
                ga += 1
            else:
                gb += 1
            if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
                break
        if ga > gb:
            sa += 1
        else:
            sb += 1
        sets.append((ga, gb))
        tbs.append(tb)
        if (ga + gb) % 2 == 1:
            a_first = not a_first

    return SimulatedMatch(
        a_won=sa > sb,
        sets=sets,
        tiebreaks=tbs,
        serve_points={"a_svpt": stats["a_svpt"], "a_svpt_won": stats["a_won"],
                      "b_svpt": stats["b_svpt"], "b_svpt_won": stats["b_won"]},
        longest_game=longest["seq"],
        longest_game_server_is_a=longest["a_serving"],
    )


def longest_game_in(p_server: float, n_games: int, rng: np.random.Generator | None = None) -> str:
    """Play n_games service games and return the longest one's point sequence (S/R).

    Vectorised: one row per game, columns are points; a game lasting k points
    has probability ~(2pq)^((k-6)/2), so 60 columns covers billions of games.
    """
    rng = rng or np.random.default_rng()
    width = 60
    best = ""
    for start in range(0, n_games, 200_000):
        n = min(200_000, n_games - start)
        pts = rng.random((n, width)) < p_server            # True = server wins point
        s = np.cumsum(pts, axis=1)
        r = np.cumsum(~pts, axis=1)
        done = ((s >= 4) | (r >= 4)) & (np.abs(s - r) >= 2)
        has_end = done.any(axis=1)
        length = np.where(has_end, done.argmax(axis=1) + 1, width + 1)
        i = int(np.argmax(length))
        if length[i] > len(best) and length[i] <= width:
            best = "".join("S" if x else "R" for x in pts[i, : length[i]])
    return best
