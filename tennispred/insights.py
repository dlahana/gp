"""Detailed breakdown of one matchup, for on-demand "match insights" posts.

Everything here is computed, not written by an LLM: the serve-point
probabilities and what moved them, hold rates, the exact random-walk answer,
N simulated matches, the longest simulated game, and anything odd worth a
post. ``corny.py`` only turns these facts into tweets.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .features import FEATURES, Context
from .fixtures import Fixture, infer_context
from .markov import longest_game_in, match_probabilities, simulate_match
from .model import ServeModel, final_set_tb_target
from .names import NameResolver
from .pipeline import Built, _feature_date

# Features grouped into plain-English factors. Each matchup is scored from both
# serving directions, so per-player features combine into one factor.
GROUPS = {
    "overall Elo": ["elo_diff"], "surface Elo": ["surf_elo_diff"],
    "serve quality": ["s_serve", "r_serve"], "return quality": ["r_return", "s_return"],
    "recent form": ["form_diff"], "ranking": ["log_rank_ratio"],
    "lefty/righty matchup": ["s_left", "r_left", "lefty_srv_vs_righty", "lefty_ret_vs_righty"],
    "height": ["s_height", "r_height", "s_height_missing", "r_height_missing", "s_height_grass", "s_height_clay"],
    "age": ["s_age", "r_age", "s_age2", "r_age2"], "recent workload": ["s_fatigue", "r_fatigue"],
    "rest / rust": ["s_rust", "r_rust"], "experience": ["s_exp", "r_exp"], "head-to-head": ["h2h"],
}


def _contributions(model: ServeModel, x: np.ndarray) -> np.ndarray:
    """Per-feature contribution to the serve-point log-odds, relative to an average matchup."""
    return model.weights[1:] * (x - model.mean) / model.scale


def matchup_facts(built: Built, model: ServeModel, fx: Fixture, tour: str = "atp", n_sims: int = 500,
                  n_marathon_games: int = 1_000_000, seed: int | None = None) -> dict:
    hist = built.history
    fx = infer_context(fx, tour, hist.tourney_surface, hist.tourney_best_of, hist.tourney_level)
    resolver = NameResolver(hist)
    a, b = resolver.resolve(fx.player1), resolver.resolve(fx.player2)
    if a is None or b is None:
        missing = [n for n, p in ((fx.player1, a), (fx.player2, b)) if p is None]
        raise ValueError(f"could not find {', '.join(missing)} in the match history")
    sa, sb = hist.get(a), hist.get(b)
    ctx = Context(_feature_date(hist, fx.date), fx.surface, fx.best_of, fx.level)
    x_a, x_b = hist.features(a, b, ctx), hist.features(b, a, ctx)
    tb = final_set_tb_target(fx.level, ctx.date)
    p_a, p_b = float(model.p_serve(x_a)[0]), float(model.p_serve(x_b)[0])
    exact = match_probabilities(p_a, p_b, fx.best_of, tb)
    p_final = float(model.match_prob(x_a[None], x_b[None], fx.best_of, tb)[0])

    # What moved A's edge: (helps A on A's serve) - (helps B on B's serve), in log-odds.
    net = _contributions(model, x_a) - _contributions(model, x_b)
    grouped = {g: float(sum(net[FEATURES.index(f)] for f in fs)) for g, fs in GROUPS.items()}
    drivers = [{"factor": g, "favours": sa.name if v > 0 else sb.name, "log_odds": round(v, 3)}
               for g, v in sorted(grouped.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 0.01][:6]

    rng = np.random.default_rng(seed)
    sims = [simulate_match(p_a, p_b, fx.best_of, rng, tb) for _ in range(n_sims)]
    wins = sum(s.a_won for s in sims)
    sets_to_win = fx.best_of // 2 + 1
    score_counts = Counter(f"{sum(g[0] > g[1] for g in s.sets)}-{sum(g[1] > g[0] for g in s.sets)}" for s in sims)
    deciders = sum(len(s.sets) == fx.best_of for s in sims)
    tiebreaks = sum(sum(t is not None for t in s.tiebreaks) for s in sims)
    points = np.array([s.total_points for s in sims])
    longest_match = sims[int(points.argmax())]
    longest_game_sim = max(sims, key=lambda s: len(s.longest_game))
    longest_tb = max(((t, s) for s in sims for t in s.tiebreaks if t is not None), key=lambda ts: sum(ts[0]),
                     default=(None, None))[0]

    # A marathon: the longest of a million service games by the weaker server.
    weak_name, weak_p = (sa.name, p_a) if p_a < p_b else (sb.name, p_b)
    marathon = longest_game_in(weak_p, n_marathon_games, rng)

    facts = {
        "player1": sa.name, "player2": sb.name, "tournament": fx.tournament, "surface": fx.surface,
        "best_of": fx.best_of, "date": str(fx.date.date()),
        "serve_point_win": {sa.name: round(p_a, 4), sb.name: round(p_b, 4)},
        "hold_rate": {sa.name: round(exact.p_hold_a, 4), sb.name: round(exact.p_hold_b, 4)},
        "p1_win_exact_markov": round(exact.p_a, 4),
        "p1_win_final": round(p_final, 4),
        "set_score_probs": {k: round(v, 4) for k, v in exact.set_scores.items()},
        "drivers": drivers,
        "simulations": {
            "n": n_sims, "p1_wins": wins, "p1_win_rate": round(wins / n_sims, 4),
            "set_scores": dict(score_counts.most_common()),
            "deciding_set_matches": deciders, "tiebreaks_played": tiebreaks,
            "median_points": int(np.median(points)), "max_points": int(points.max()),
            "longest_match_sets": [f"{x}-{y}" for x, y in longest_match.sets],
            "longest_game_points": len(longest_game_sim.longest_game),
            "longest_game_sequence": longest_game_sim.longest_game,
            "longest_game_server": sa.name if longest_game_sim.longest_game_server_is_a else sb.name,
            "longest_tiebreak": f"{max(longest_tb)}-{min(longest_tb)}" if longest_tb else None,
        },
        "marathon": {"server": weak_name, "games_simulated": n_marathon_games,
                     "points": len(marathon), "deuces": max(0, (len(marathon) - 6) // 2), "sequence": marathon},
        "elo_favourite": sa.name if sa.elo >= sb.elo else sb.name,
    }
    facts["oddities"] = oddities(facts)
    return facts


def oddities(f: dict) -> list[str]:
    """Things in the numbers that might be worth a post on their own."""
    out = []
    p1, p2 = f["player1"], f["player2"]
    sim, exact = f["simulations"], f["p1_win_exact_markov"]
    model_fav = p1 if f["p1_win_final"] >= 0.5 else p2
    if model_fav != f["elo_favourite"]:
        out.append(f"The model picks {model_fav} even though {f['elo_favourite']} has the higher Elo rating.")
    if abs(sim["p1_win_rate"] - exact) > 0.03:
        out.append(f"{sim['n']} simulations gave {p1} {sim['p1_win_rate']:.1%} vs the exact {exact:.1%}: "
                   "a nice example of simulation noise.")
    holds = f["hold_rate"]
    if max(holds.values()) < 0.7:
        out.append(f"Break-fest alert: neither player is projected to hold even 70% of service games.")
    if min(holds.values()) > 0.9:
        out.append("Serve-bot showdown: both players hold over 90% of the time. Expect tiebreaks.")
    if sim["longest_game_points"] >= 20:
        out.append(f"One simulated game lasted {sim['longest_game_points']} points.")
    if f["marathon"]["points"] >= 30:
        out.append(f"In {f['marathon']['games_simulated']:,} simulated {f['marathon']['server']} service games, "
                   f"the longest went {f['marathon']['points']} points ({f['marathon']['deuces']} deuces).")
    for d in f["drivers"][:3]:
        if d["factor"] in ("lefty/righty matchup", "height", "head-to-head", "rest / rust") and abs(d["log_odds"]) > 0.05:
            out.append(f"Unusual driver: {d['factor']} is a top-3 factor here, favouring {d['favours']}.")
    return out


def render_text(f: dict) -> str:
    """Plain-text report for the terminal or an email."""
    p1, p2, sim = f["player1"], f["player2"], f["simulations"]
    lines = [f"{p1} vs {p2} | {f['tournament'] or 'n/a'} | {f['surface']} | best of {f['best_of']}",
             f"Win probability for {p1}: {f['p1_win_final']:.1%} (raw random walk {f['p1_win_exact_markov']:.1%})",
             "Serve points won: " + ", ".join(f"{k} {v:.1%}" for k, v in f["serve_point_win"].items()),
             "Hold rate: " + ", ".join(f"{k} {v:.1%}" for k, v in f["hold_rate"].items()),
             "Biggest factors:"]
    lines += [f"  {d['factor']}: favours {d['favours']} ({abs(d['log_odds']):.3f} log-odds)" for d in f["drivers"]]
    lines += [f"{sim['n']} simulated matches: {p1} won {sim['p1_wins']} ({sim['p1_win_rate']:.1%})",
              "  set scores: " + ", ".join(f"{k} x{v}" for k, v in sim["set_scores"].items()),
              f"  deciding sets: {sim['deciding_set_matches']}, tiebreaks: {sim['tiebreaks_played']}, "
              f"longest tiebreak: {sim['longest_tiebreak']}",
              f"  points per match: median {sim['median_points']}, max {sim['max_points']} "
              f"({' '.join(sim['longest_match_sets'])})",
              f"  longest game: {sim['longest_game_points']} points on {sim['longest_game_server']}'s serve",
              f"Marathon: longest of {f['marathon']['games_simulated']:,} {f['marathon']['server']} service games = "
              f"{f['marathon']['points']} points ({f['marathon']['deuces']} deuces)"]
    if f["oddities"]:
        lines += ["Oddities:"] + [f"  - {o}" for o in f["oddities"]]
    return "\n".join(lines)
