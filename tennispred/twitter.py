"""Formatting predictions as a tweet thread and posting it to X/Twitter.

Posting uses the X API v2 ``POST /2/tweets`` endpoint via tweepy, with OAuth
1.0a user-context credentials from the environment:

    X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET

The app needs "Read and write" permission, and the access token must be
generated *after* that permission is set.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

TWEET_LIMIT = 280
# X counts emoji as two characters; keep a margin so we never get rejected.
SAFE_LIMIT = 270


def _short(name: str) -> str:
    """Display surname: 'Carlos Alcaraz' -> 'Alcaraz', 'Alex de Minaur' -> 'de Minaur'."""
    toks = name.split()
    if len(toks) <= 1:
        return name
    particles = {"de", "del", "van", "von", "da", "di", "le", "la", "mc"}
    i = len(toks) - 1
    while i > 1 and toks[i - 1].lower() in particles:
        i -= 1
    return " ".join(toks[i:])


def format_line(pred: dict) -> str:
    fav, dog = (pred["player1"], pred["player2"]) if pred["p1"] >= 0.5 else (pred["player2"], pred["player1"])
    p = min(max(pred["p1"], 1 - pred["p1"]), 0.99)
    score = pred["fav_likely_score"]
    return f"• {_short(fav)} over {_short(dog)}: {p:.0%} (likeliest {score})"


def build_thread(preds: list[dict], day: pd.Timestamp, tour: str, max_matches: int = 5) -> list[str]:
    """A short thread: a header with picks, overflow lines in replies."""
    chosen = sorted(preds, key=lambda d: -d["prominence"])[:max_matches]
    if not chosen:
        return []
    by_event: dict[str, list[dict]] = {}
    for p in chosen:
        by_event.setdefault(p["tournament"] or "Today", []).append(p)
    header = f"🎾 {tour.upper()} picks for {day.strftime('%a %b')} {day.day}, from a random-walk point model"
    lines = []
    for event, ps in by_event.items():
        lines.append(f"\n{event} ({ps[0]['surface']})")
        lines.extend(format_line(p) for p in sorted(ps, key=lambda d: -max(d["p1"], 1 - d["p1"])))
    tweets, cur = [], header
    for line in lines:
        piece = "\n" + line
        if len(cur) + len(piece) > SAFE_LIMIT:
            tweets.append(cur.strip())
            cur = line.lstrip("\n")
        else:
            cur += piece
    tweets.append(cur.strip())
    footer = "\n\n#tennis"
    if len(tweets[-1]) + len(footer) <= SAFE_LIMIT:
        tweets[-1] += footer
    return tweets


def credentials_from_env() -> dict | None:
    keys = {"consumer_key": "X_API_KEY", "consumer_secret": "X_API_SECRET",
            "access_token": "X_ACCESS_TOKEN", "access_token_secret": "X_ACCESS_TOKEN_SECRET"}
    creds = {k: os.environ.get(v) for k, v in keys.items()}
    missing = [keys[k] for k, v in creds.items() if not v]
    if missing:
        log.warning("missing X credentials: %s", ", ".join(missing))
        return None
    return creds


def post_thread(tweets: list[str], creds: dict) -> list[str]:
    """Post tweets as a reply chain. Returns the tweet ids."""
    import tweepy

    client = tweepy.Client(**creds)
    ids: list[str] = []
    reply_to = None
    for text in tweets:
        resp = client.create_tweet(text=text, in_reply_to_tweet_id=reply_to)
        reply_to = resp.data["id"]
        ids.append(reply_to)
        log.info("posted tweet %s", reply_to)
    return ids
