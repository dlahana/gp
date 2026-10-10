"""Command line entry point: python -m tennispred <command> ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from . import data, insights, name_pool, pipeline, synthetic, twitter
from .fixtures import ApiTennisSource, CsvFixtureSource, Fixture
from .model import ServeModel

log = logging.getLogger("tennispred")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tour", default="atp", choices=["atp", "wta"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--start-year", type=int, default=1985, help="first season to replay for player state")
    p.add_argument("--train-from", default="1991-01-01", help="first date used to fit the model")


def _fixture_source(args):
    if args.fixtures == "api":
        return ApiTennisSource(tour=args.tour)
    return CsvFixtureSource(Path(args.fixtures))


def cmd_download(args) -> None:
    data.download(args.data_dir, args.tour, args.start_year, include_challengers=args.challengers)


def cmd_synth(args) -> None:
    synthetic.write(args.data_dir, args.tour, n_years=args.years, seed=args.seed)
    print(f"wrote synthetic {args.tour} data to {args.data_dir}")


def cmd_train(args) -> None:
    built = pipeline.build_history(args.data_dir, args.tour, args.start_year)
    model = pipeline.train(built, args.train_from, args.l2)
    model.save(args.model)
    print(f"saved {args.model} (trained through {model.trained_through}, temperature {model.temperature:.3f})")
    print("\nServe-point log-odds per 1 SD of each feature (largest first):")
    print(model.coefficients().round(4).to_string())


def cmd_backtest(args) -> None:
    built = pipeline.build_history(args.data_dir, args.tour, args.start_year)
    res = pipeline.backtest(built, args.years, args.train_from, args.l2)
    pd.set_option("display.width", 200)
    print(res.round(4).to_string())


def _load_or_train(args, built: pipeline.Built) -> ServeModel:
    if args.model and Path(args.model).exists() and not args.retrain:
        return ServeModel.load(Path(args.model))
    model = pipeline.train(built, args.train_from, args.l2)
    if args.model:
        model.save(Path(args.model))
    return model


def cmd_predict(args) -> None:
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    built = pipeline.build_history(args.data_dir, args.tour, args.start_year)
    model = _load_or_train(args, built)

    stale = pipeline.data_staleness_days(built.history, day)
    if stale is not None and stale > pipeline.STALE_DAYS:
        log.warning("match data ends %s (%d days before %s); recent form is missing",
                    built.history.last_date.date(), stale, day.date())

    fixtures = _fixture_source(args).fixtures(day)
    preds = pipeline.predict_fixtures(built, model, fixtures, args.tour)
    path = pipeline.save_predictions(preds, args.out_dir, day, args.tour)
    print(f"{len(preds)} predictions ({len(fixtures)} fixtures) -> {path}")
    for p in sorted(preds, key=lambda d: -d["prominence"]):
        print(f"  {p['player1']:>24} {p['p1']:6.1%}  vs  {p['player2']:<24} "
              f"[{p['tournament']}, {p['surface']}, Bo{p['best_of']}]  "
              f"serve pts {p['p1_serve_point']:.3f}/{p['p2_serve_point']:.3f}")

    thread = twitter.build_thread(preds, day, args.tour, args.max_matches)
    if not thread:
        print("nothing to post")
        return
    print("\n--- thread ---")
    for i, t in enumerate(thread, 1):
        print(f"[{i}/{len(thread)}] ({len(t)} chars)\n{t}\n")

    if args.submit:
        submit_draft(built, model, preds, day, args.tour, args.max_matches)
        return
    if not args.post:
        print("dry run: pass --submit to send for approval (or --post to publish directly)")
        return
    marker = args.out_dir / f".posted_{args.tour}_{day.date()}"
    if marker.exists() and not args.force:
        print(f"already posted for {day.date()} ({marker}); use --force to post again")
        return
    creds = twitter.credentials_from_env()
    if creds is None:
        sys.exit("cannot post: X credentials are not set")
    ids = twitter.post_thread(thread, creds)
    marker.write_text("\n".join(ids))
    print(f"posted {len(ids)} tweets: https://x.com/i/status/{ids[0]}")


def _server() -> tuple[str, dict]:
    import os

    url, token = os.environ.get("BOT_SERVER_URL"), os.environ.get("BOT_ADMIN_TOKEN")
    if not url or not token:
        sys.exit("set BOT_SERVER_URL and BOT_ADMIN_TOKEN")
    return url.rstrip("/"), {"Authorization": f"Bearer {token}"}


MATCH_KEYS = ("player1", "player2", "p1", "tournament", "surface", "best_of", "round", "fav_likely_score")


def _post_draft(payload: dict) -> None:
    url, headers = _server()
    resp = requests.post(f"{url}/api/drafts", json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    out = resp.json()
    print(f"draft {out['id']} submitted; owner notified via: {', '.join(out['notified']) or 'nothing (see server log)'}")


def submit_draft(built, model, preds: list[dict], day: pd.Timestamp, tour: str, max_matches: int) -> None:
    """Send the most prominent matches to the bot server, which emails the owner for approval.

    Each match is also simulated, and anything odd is included in the email (not the post).
    """
    chosen = sorted(preds, key=lambda d: -d["prominence"])[:max_matches]
    odd = []
    for p in chosen:
        fx = Fixture(day, p["player1"], p["player2"], p["tournament"], p["surface"], p["best_of"])
        try:
            f = insights.matchup_facts(built, model, fx, tour, n_sims=500, n_marathon_games=200_000)
            odd += [f"{p['player1']} vs {p['player2']}: {o}" for o in f["oddities"]]
        except ValueError as exc:
            log.warning("no oddity check for %s vs %s: %s", p["player1"], p["player2"], exc)
    _post_draft({"day": f"{day.strftime('%a %b')} {day.day}", "tour": tour, "kind": "daily",
                 "matches": [{k: p[k] for k in MATCH_KEYS} for p in chosen], "oddities": odd})


def cmd_insights(args) -> None:
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    built = pipeline.build_history(args.data_dir, args.tour, args.start_year)
    model = _load_or_train(args, built)
    fx = Fixture(day, args.player1, args.player2, args.tournament or "", args.surface, args.best_of)
    try:
        facts = insights.matchup_facts(built, model, fx, args.tour, n_sims=args.sims)
    except ValueError as exc:
        sys.exit(str(exc))
    print(insights.render_text(facts))
    if not args.submit:
        print("\ndry run: pass --submit to have the bot write it up and email you for approval")
        return
    match = {"player1": facts["player1"], "player2": facts["player2"], "p1": facts["p1_win_final"],
             "tournament": facts["tournament"], "surface": facts["surface"], "best_of": facts["best_of"]}
    _post_draft({"day": f"{day.strftime('%a %b')} {day.day}", "tour": args.tour, "kind": "insights",
                 "matches": [match], "facts": facts, "oddities": facts["oddities"]})


def cmd_share_link(args) -> None:
    url, headers = _server()
    resp = requests.get(f"{url}/api/share-link", headers=headers, timeout=30)
    resp.raise_for_status()
    print(resp.json()["url"])


def cmd_name_pool(args) -> None:
    pool = []
    for tour in ("atp", "wta"):
        path = args.data_dir / f"{tour}_players.csv"
        if path.exists():
            pool += name_pool.from_players_file(path, tour, args.born_after)
    if not pool:
        sys.exit(f"no players files in {args.data_dir}; run download for atp and wta first")
    name_pool.save(pool, args.out)
    print(f"wrote {len(pool)} names to {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tennispred", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="fetch Sackmann match files")
    _common(p)
    p.add_argument("--challengers", action="store_true", help="also fetch ATP challenger/qualifying files")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("synth", help="write a synthetic dataset (offline demo / tests)")
    _common(p)
    p.add_argument("--years", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train", help="fit the serve-point model")
    _common(p)
    p.add_argument("--model", type=Path, default=Path("models/atp.json"))
    p.add_argument("--l2", type=float, default=1.0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="walk-forward evaluation by season")
    _common(p)
    p.add_argument("--years", type=int, nargs="+", required=True)
    p.add_argument("--l2", type=float, default=1.0)
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("predict", help="predict a day's matches and (optionally) tweet them")
    _common(p)
    p.add_argument("--date", help="YYYY-MM-DD, default today")
    p.add_argument("--fixtures", default="api", help="'api' (api-tennis.com) or a CSV/JSON path")
    p.add_argument("--model", type=Path, help="load this model (or save to it after training)")
    p.add_argument("--retrain", action="store_true", help="retrain even if --model exists")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("predictions"))
    p.add_argument("--max-matches", type=int, default=5, help="matches to include in the thread")
    p.add_argument("--submit", action="store_true",
                   help="send to the bot server for nicknames, a corny rewrite and SMS approval")
    p.add_argument("--post", action="store_true", help="post the plain thread to X directly (no approval)")
    p.add_argument("--force", action="store_true", help="post even if today's thread already went out")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("insights", help="full breakdown + simulations for one match (optionally submit as a post)")
    _common(p)
    p.add_argument("player1")
    p.add_argument("player2")
    p.add_argument("--tournament", help="tournament name (used to infer surface / best-of)")
    p.add_argument("--surface", choices=["Hard", "Clay", "Grass", "Carpet"])
    p.add_argument("--best-of", type=int, choices=[3, 5])
    p.add_argument("--date", help="YYYY-MM-DD, default today")
    p.add_argument("--sims", type=int, default=500, help="simulated matches")
    p.add_argument("--model", type=Path)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--submit", action="store_true", help="send to the bot server for write-up and approval")
    p.set_defaults(func=cmd_insights)

    p = sub.add_parser("share-link", help="print the one rating link to send friends")
    p.set_defaults(func=cmd_share_link)

    p = sub.add_parser("name-pool", help="build the rating page's surname pool from players files")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--born-after", type=int, default=1975)
    p.add_argument("--out", type=Path, default=Path("name_pool.json"))
    p.set_defaults(func=cmd_name_pool)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
