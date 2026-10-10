"""Command line entry point: python -m tennispred <command> ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import data, pipeline, synthetic, twitter
from .fixtures import ApiTennisSource, CsvFixtureSource
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


def cmd_predict(args) -> None:
    day = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    built = pipeline.build_history(args.data_dir, args.tour, args.start_year)
    if args.model and Path(args.model).exists() and not args.retrain:
        model = ServeModel.load(Path(args.model))
    else:
        model = pipeline.train(built, args.train_from, args.l2)
        if args.model:
            model.save(Path(args.model))

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

    if not args.post:
        print("dry run: pass --post to publish")
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
    p.add_argument("--post", action="store_true", help="actually post to X")
    p.add_argument("--force", action="store_true", help="post even if today's thread already went out")
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
