"""Claude writes the corny post around the predictions and silly nicknames.

Needs ANTHROPIC_API_KEY. Without it (or if the call fails) a template
fallback is used, so a draft always reaches the approval page.
"""

from __future__ import annotations

import json
import logging
import random

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
TWEET_MAX = 270   # X counts emoji as 2; keep a margin under 280

SYSTEM = (
    "You write the daily post for a goofy tennis prediction account. A serious random-walk "
    "model produces the win probabilities; the account's personality is corny dad-joke energy. "
    "Every matchup gets a deliberately stupid portmanteau nickname from the list provided. Use "
    "one nickname per match, exactly as spelled. Puns, groan-worthy wordplay and the odd emoji "
    "are welcome. Never change the probabilities or who is favoured. Each tweet must be at most "
    f"{TWEET_MAX} characters. Mention every match given."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "tweets": {"type": "array", "items": {"type": "string"}},
        "nicknames_used": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"match": {"type": "integer"}, "nickname": {"type": "string"}},
                "required": ["match", "nickname"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["tweets", "nicknames_used"],
    "additionalProperties": False,
}


def _match_brief(i: int, m: dict) -> str:
    fav, dog = (m["player1"], m["player2"]) if m["p1"] >= 0.5 else (m["player2"], m["player1"])
    p = max(m["p1"], 1 - m["p1"])
    return (f"Match {i}: {fav} vs {dog} at {m.get('tournament') or 'today'} ({m.get('surface', '')}). "
            f"Model favours {fav} at {p:.0%}. Nickname options: {', '.join(m['nicknames'])}")


def fallback_thread(matches: list[dict], header: str) -> tuple[list[str], list[str]]:
    lines, used = [], []
    for m in matches:
        nick = random.choice(m["nicknames"]) if m["nicknames"] else ""
        used.append(nick)
        fav, dog = (m["player1"], m["player2"]) if m["p1"] >= 0.5 else (m["player2"], m["player1"])
        p = max(m["p1"], 1 - m["p1"])
        lines.append(f"• {nick.capitalize()}: {fav.split()[-1]} over {dog.split()[-1]}, {p:.0%}")
    tweets, cur = [], header
    for line in lines:
        if len(cur) + 1 + len(line) > TWEET_MAX:
            tweets.append(cur)
            cur = line
        else:
            cur += "\n" + line
    tweets.append(cur)
    return tweets, used


def write_thread(matches: list[dict], day_label: str, extra_instructions: str = "") -> tuple[list[str], list[str], str]:
    """Returns (tweets, nickname used per match, source) where source is 'claude' or 'template'."""
    header = f"🎾 Picks for {day_label}"
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (f"Write today's thread ({day_label}). Keep it to {max(1, len(matches) // 3 + 1)} tweet(s) "
                  "if you can.\n\n" + "\n".join(_match_brief(i, m) for i, m in enumerate(matches)))
        if extra_instructions:
            prompt += f"\n\nExtra direction from the account owner: {extra_instructions}"
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"Claude stopped with {response.stop_reason}")
        data = json.loads(next(b.text for b in response.content if b.type == "text"))
        tweets = [t.strip() for t in data["tweets"] if t.strip()]
        used = [""] * len(matches)
        for u in data["nicknames_used"]:
            if 0 <= u["match"] < len(matches):
                used[u["match"]] = u["nickname"]
        if not tweets or any(len(t) > 280 for t in tweets):
            raise ValueError("Claude's thread was empty or had an over-long tweet")
        return tweets, used, "claude"
    except Exception as exc:  # any failure falls back to the template; the human still reviews it
        log.warning("corny writer fell back to template: %s", exc)
        tweets, used = fallback_thread(matches, header)
        return tweets, used, "template"
