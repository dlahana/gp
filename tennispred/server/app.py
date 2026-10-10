"""FastAPI app.

Run with:  uvicorn tennispred.server.app:create_app --factory --host 0.0.0.0 --port 8000

Friends:  /rate/<shared code>              one link for everyone: pick the funniest nickname, repeat
You:      /drafts/<id>?t=<token>           edit / rewrite / reject / approve the day's post
Pipeline: POST /api/drafts  (Bearer admin)  submit the day's predictions or a match-insights report

Configuration (environment):
    BOT_DB_PATH            SQLite file (default bot.db)
    BOT_ADMIN_TOKEN        bearer token for /api/drafts, /api/share-link, /api/admin/*
    BOT_PUBLIC_URL         e.g. https://tennisbot.example.com (used in texts and invite links)
    BOT_SHARE_CODE         optional fixed code for the shared rating link (else generated once)
    BOT_NAME_POOL          optional JSON from `tennispred name-pool`
    BOT_REWARD_PATH        where the fitted reward model is cached (default reward.json)
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, NOTIFY_EMAIL   (approval emails)
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, NOTIFY_PHONE  (optional SMS)
    X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .. import corny, name_pool, twitter
from ..reward import RewardModel, Vote
from . import notify, pages
from .store import Store

log = logging.getLogger(__name__)

REFIT_EVERY = 20          # refit the reward model after this many new votes
MIN_NICKS, MAX_NICKS = 2, 10
POST_TAU = 1.0            # policy temperature for nicknames offered in posts


class VoteIn(BaseModel):
    round_id: int
    chosen: int | None = None
    rater: str = ""


class DraftIn(BaseModel):
    day: str
    tour: str = "atp"
    kind: str = "daily"                  # "daily" or "insights"
    matches: list[dict]
    facts: dict | None = None            # insights: the computed breakdown
    oddities: list[str] = []             # emailed to the owner, never posted


class TweetsIn(BaseModel):
    tweets: list[str]


class RegenIn(BaseModel):
    direction: str = ""


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="tennis nickname bot", docs_url=None, redoc_url=None)
    store = Store(db_path or os.environ.get("BOT_DB_PATH", "bot.db"))
    pool = name_pool.load(Path(p) if (p := os.environ.get("BOT_NAME_POOL")) else None)
    reward_path = Path(os.environ.get("BOT_REWARD_PATH", "reward.json"))
    state = {"model": RewardModel.load(reward_path) if reward_path.exists() else RewardModel(),
             "fit_at": None, "lock": threading.Lock()}
    rng = np.random.default_rng()
    public = os.environ.get("BOT_PUBLIC_URL", "").rstrip("/")
    share_code = os.environ.get("BOT_SHARE_CODE") or store.shared_code()
    if share_code is None:
        share_code = secrets.token_urlsafe(6)
    if store.invite(share_code) is None:
        store.add_invite(share_code, "shared")

    def admin(authorization: str = Header(default="")) -> None:
        expected = os.environ.get("BOT_ADMIN_TOKEN")
        if not expected or not secrets.compare_digest(authorization, f"Bearer {expected}"):
            raise HTTPException(401, "bad admin token")

    def model() -> RewardModel:
        n = store.vote_count()
        if state["fit_at"] is None or n - state["fit_at"] >= REFIT_EVERY:
            with state["lock"]:
                if state["fit_at"] is None or n - state["fit_at"] >= REFIT_EVERY:
                    votes = [Vote(r["name_a"], r["name_b"], json.loads(r["shown"]), r["chosen"])
                             for r in store.votes()]
                    if votes:
                        state["model"] = RewardModel().fit(votes)
                        state["model"].save(reward_path)
                    state["fit_at"] = n
        return state["model"]

    def check_draft(draft_id: str, t: str) -> dict:
        d = store.draft(draft_id)
        if d is None or not secrets.compare_digest(t, d["token"]):
            raise HTTPException(404, "no such draft")
        return d

    # ------------------------------------------------------------- rating

    def new_round(code: str, rater: str) -> dict:
        a, b = name_pool.sample_pair(pool, rng)
        shown = model().rating_round(a.name, b.name, k=4, rng=rng)
        rid = store.add_round(code, a.name, b.name, shown)
        return {"round_id": rid, "name_a": a.name, "name_b": b.name, "shown": shown,
                "my_votes": store.vote_count(rater) if rater else 0}

    @app.get("/", response_class=HTMLResponse)
    def home():
        return "<p>🎾</p>"

    @app.get("/rate/{code}", response_class=HTMLResponse)
    def rate(code: str):
        if store.invite(code) is None:
            raise HTTPException(404, "invalid link")
        return pages.rate_page(code)

    @app.get("/api/round/{code}")
    def get_round(code: str, rater: str = ""):
        if store.invite(code) is None:
            raise HTTPException(404, "invalid link")
        return new_round(code, rater[:64])

    @app.post("/api/vote/{code}")
    def vote(code: str, body: VoteIn):
        if store.invite(code) is None:
            raise HTTPException(404, "invalid link")
        r = store.round(body.round_id)
        if r is None or r["invite"] != code:
            raise HTTPException(400, "unknown round")
        n_shown = len(json.loads(r["shown"]))
        if body.chosen is not None and not 0 <= body.chosen < n_shown:
            raise HTTPException(400, "bad choice")
        rater = body.rater[:64]
        store.add_vote(body.round_id, code, rater, body.chosen)
        return new_round(code, rater)

    # ------------------------------------------------------------- drafts

    def nicknames_for(m: dict) -> list[str]:
        k = int(rng.integers(MIN_NICKS, MAX_NICKS + 1))
        return model().sample(m["player1"], m["player2"], k, tau=POST_TAU, rng=rng)

    @app.post("/api/drafts", dependencies=[Depends(admin)])
    def create_draft(body: DraftIn):
        if not body.matches:
            raise HTTPException(400, "no matches")
        matches = []
        for m in body.matches:
            m = dict(m)
            m["nicknames"] = nicknames_for(m)
            matches.append(m)
        if body.kind not in ("daily", "insights"):
            raise HTTPException(400, "kind must be daily or insights")
        if body.kind == "insights" and not body.facts:
            raise HTTPException(400, "insights drafts need facts")
        tweets, used, source = corny.write_thread(matches, body.day, facts=body.facts)
        d = {"id": secrets.token_hex(4), "token": secrets.token_urlsafe(16), "kind": body.kind, "day": body.day,
             "tour": body.tour, "matches": matches, "facts": body.facts, "tweets": tweets, "nicknames_used": used,
             "source": source}
        store.add_draft(d)
        url = f"{public}/drafts/{d['id']}?t={d['token']}"
        what = "match insights" if body.kind == "insights" else f"{body.tour.upper()} picks"
        details = "Draft:\n\n" + "\n\n".join(tweets)
        if body.oddities:
            details += "\n\nOdd things spotted (not posted, just for you):\n" + "\n".join(f"- {o}" for o in body.oddities)
        sent = notify.notify(f"🎾 {what} for {body.day} ready to approve", url, details)
        return {"id": d["id"], "url": url, "notified": sent, "tweets": tweets}

    @app.get("/drafts/{draft_id}", response_class=HTMLResponse)
    def draft_page(draft_id: str, t: str = ""):
        return pages.approve_page(check_draft(draft_id, t))

    @app.post("/api/drafts/{draft_id}/regenerate")
    def regenerate(draft_id: str, body: RegenIn, t: str = ""):
        d = check_draft(draft_id, t)
        if d["status"] != "pending":
            raise HTTPException(409, f"draft is {d['status']}")
        tweets, used, source = corny.write_thread(d["matches"], d["day"], body.direction[:500], facts=d["facts"])
        store.update_draft(draft_id, tweets=tweets, nicknames_used=used, source=source)
        return {"tweets": tweets, "message": f"rewritten ({source})", "status": "pending"}

    @app.post("/api/drafts/{draft_id}/reject")
    def reject(draft_id: str, t: str = ""):
        d = check_draft(draft_id, t)
        if d["status"] != "pending":
            raise HTTPException(409, f"draft is {d['status']}")
        store.update_draft(draft_id, status="rejected")
        return {"status": "rejected", "message": "rejected, nothing posted", "locked": True}

    @app.post("/api/drafts/{draft_id}/approve")
    def approve(draft_id: str, body: TweetsIn, t: str = ""):
        d = check_draft(draft_id, t)
        if d["status"] != "pending":
            raise HTTPException(409, f"draft is {d['status']}")
        tweets = [x.strip() for x in body.tweets if x.strip()]
        if not tweets or any(len(x) > twitter.TWEET_LIMIT for x in tweets):
            raise HTTPException(400, "every tweet must be 1-280 characters")
        creds = twitter.credentials_from_env()
        if creds is None:
            raise HTTPException(500, "X credentials are not configured on the server")
        store.update_draft(draft_id, status="posting", tweets=tweets)
        try:
            ids = twitter.post_thread(tweets, creds)
        except Exception as exc:
            store.update_draft(draft_id, status="pending")
            log.exception("posting failed")
            raise HTTPException(502, f"X rejected the post: {exc}")
        store.update_draft(draft_id, status="posted", tweet_ids=ids)
        return {"status": "posted", "message": f"posted: https://x.com/i/status/{ids[0]}", "locked": True}

    # -------------------------------------------------------------- admin

    @app.get("/api/share-link", dependencies=[Depends(admin)])
    def share_link():
        return {"url": f"{public}/rate/{share_code}"}

    @app.get("/api/admin/stats", dependencies=[Depends(admin)])
    def stats():
        m = model()
        return {"votes": store.vote_count(), "raters": store.rater_count(), "votes_in_model": m.n_votes, "top_features": m.top_features(),
                "sample": {"Nadal x Federer": m.sample("Nadal", "Federer", 8, tau=0.7, rng=rng),
                           "Muchova x Alcaraz": m.sample("Muchova", "Alcaraz", 8, tau=0.7, rng=rng)}}

    return app

