"""Glue: data -> features -> model -> today's predictions -> tweets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features
from .features import Context, History
from .fixtures import Fixture, infer_context
from .markov import match_probabilities
from .model import ServeModel, evaluate, final_set_tb_target
from .names import NameResolver

log = logging.getLogger(__name__)


@dataclass
class Built:
    history: History
    table: features.FeatureTable


def build_history(data_dir: Path, tour: str, start_year: int = 1985) -> Built:
    matches = data.load_matches(data_dir, tour, start_year)
    players = data.load_players(data_dir, tour)
    log.info("replaying %d %s matches", len(matches), tour)
    hist, table = features.build(matches, players)
    return Built(hist, table)


def train(built: Built, train_from: str = "1991-01-01", l2: float = 1.0) -> ServeModel:
    dates = built.table.matches.tourney_date
    mask = (dates >= pd.Timestamp(train_from)).to_numpy()
    model = ServeModel(l2=l2).fit(built.table, mask)
    log.info("trained on %d serve observations through %s (temperature %.3f)",
             model.n_obs, model.trained_through, model.temperature)
    return model


def backtest(built: Built, test_years: list[int], train_from: str = "1991-01-01", l2: float = 1.0) -> pd.DataFrame:
    """Walk-forward: for each year, train on everything before it, test on it."""
    t = built.table
    dates = t.matches.tourney_date
    rows = []
    for year in test_years:
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
        train_mask = ((dates >= pd.Timestamp(train_from)) & (dates < start)).to_numpy()
        test_mask = ((dates >= start) & (dates < end)).to_numpy() & t.matches.completed.to_numpy()
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        m = ServeModel(l2=l2).fit(t, train_mask)
        p = m.table_probs(t, test_mask)
        p_raw = m.table_probs(t, test_mask, calibrated=False)
        res = evaluate(p, t.elo_p[test_mask])
        res["raw_markov_log_loss"] = evaluate(p_raw)["log_loss"]
        res["year"] = year
        res["temperature"] = m.temperature
        rows.append(res)
    return pd.DataFrame(rows).set_index("year")


def predict_fixtures(built: Built, model: ServeModel, fixtures: list[Fixture], tour: str) -> list[dict]:
    hist = built.history
    active_since = (hist.last_date or pd.Timestamp.today()) - pd.Timedelta(days=730)
    resolver = NameResolver(hist, active_since)
    out = []
    for fx in fixtures:
        fx = infer_context(fx, tour, hist.tourney_surface, hist.tourney_best_of, hist.tourney_level)
        a, b = resolver.resolve(fx.player1), resolver.resolve(fx.player2)
        if a is None or b is None:
            log.warning("skipping %s vs %s: could not match %s", fx.player1, fx.player2,
                        " and ".join(n for n, p in ((fx.player1, a), (fx.player2, b)) if p is None))
            continue
        if a == b:
            log.warning("skipping %s vs %s: both names matched the same player", fx.player1, fx.player2)
            continue
        sa, sb = hist.get(a), hist.get(b)
        if sa.n < 5 or sb.n < 5:
            log.info("skipping %s vs %s: too little history", sa.name, sb.name)
            continue
        ctx = Context(date=_feature_date(hist, fx.date), surface=fx.surface, best_of=fx.best_of, level=fx.level)
        x_a, x_b = hist.features(a, b, ctx), hist.features(b, a, ctx)
        tb = final_set_tb_target(fx.level, ctx.date)
        p_a_serve, p_b_serve = float(model.p_serve(x_a)[0]), float(model.p_serve(x_b)[0])
        p1 = float(model.match_prob(x_a[None], x_b[None], fx.best_of, tb)[0])
        mp = match_probabilities(p_a_serve, p_b_serve, fx.best_of, tb)
        fav_is_1 = p1 >= 0.5
        sets_to_win = fx.best_of // 2 + 1
        fav_scores = {k: v for k, v in mp.set_scores.items()
                      if int(k.split("-")[0 if fav_is_1 else 1]) == sets_to_win}
        best = max(fav_scores, key=fav_scores.get)
        if not fav_is_1:
            best = "-".join(reversed(best.split("-")))
        out.append({
            "date": str(ctx.date.date()), "tournament": fx.tournament, "round": fx.round,
            "surface": fx.surface, "best_of": fx.best_of,
            "player1": sa.name, "player2": sb.name, "player1_id": a, "player2_id": b,
            "p1": p1, "p1_markov_raw": mp.p_a,
            "p1_serve_point": p_a_serve, "p2_serve_point": p_b_serve,
            "p1_hold": mp.p_hold_a, "p2_hold": mp.p_hold_b,
            "set_scores": mp.set_scores, "fav_likely_score": best,
            "prominence": max(sa.elo, sb.elo) + 0.5 * min(sa.elo, sb.elo),
        })
    return out


STALE_DAYS = 14


def _feature_date(hist: History, day: pd.Timestamp) -> pd.Timestamp:
    """Date used for date-based features (age, rest, fatigue).

    Public match data is often weeks behind. Measuring "days since last match"
    from today against stale data would make every player look rusty, so when
    the data is stale features are computed as of shortly after its last match.
    """
    day = day.normalize()
    if hist.last_date is not None and (day - hist.last_date).days > STALE_DAYS:
        return hist.last_date + pd.Timedelta(days=7)
    return day


def data_staleness_days(hist: History, day: pd.Timestamp) -> int | None:
    return None if hist.last_date is None else int((day.normalize() - hist.last_date).days)


def save_predictions(preds: list[dict], out_dir: Path, day: pd.Timestamp, tour: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tour}_{day.date()}.json"

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(preds, indent=2, default=conv))
    return path
