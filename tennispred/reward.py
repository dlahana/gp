"""Learning which nicknames are funny from people's picks (RLHF, small version).

1. Reward model. Each vote is "out of these k nicknames, I liked this one
   best" (or "none of them"). That is a multinomial-logit choice:

       P(pick i | shown set S) = exp(r_i) / (exp(r_null) + sum_{j in S} exp(r_j))

   with r(c) = w . phi(c). phi has hand-made features (length, how much of
   each name survives, where the cut is, silly letters, ...) plus hashed
   letter trigrams, so it can learn that some sounds are funnier than others.
   It is fit by L2-regularised maximum likelihood. With r_null fixed at 0,
   "none of these" votes teach it what bad looks like.

2. Policy. Every candidate can be listed and scored, so no PPO is needed. The
   policy that maximises expected reward minus a KL penalty to a uniform
   prior has a closed form, pi(c) ~ exp(r(c) / tau). Posting samples from
   it: a low tau gives the reliably funny names, a higher tau gives weirder
   ones.

3. Exploration. Rating rounds mix policy samples with uniform picks, so the
   model keeps seeing (and learning about) names it currently rates low.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .portmanteau import VOWELS, Candidate, enumerate_candidates

N_HASH = 256
SILLY = set("kzbpwvjx")
ENDINGS = ("er", "ez", "ic", "ov", "ova", "ka", "a", "o", "i", "y", "us", "in", "an", "on", "el", "al")

HAND_FEATURES = [
    "len", "len2", "n_vowel_groups", "frac_first", "frac_second", "kind_syllable", "kind_letter",
    "kind_whole", "double_at_join", "vv_join", "cc_join", "ends_vowel", "silly_letters",
    "max_cons_run", "repeat_bigram", "has_oo", "head_is_whole", "tail_is_whole",
] + [f"end_{e}" for e in ENDINGS]


def _hash(s: str) -> int:
    return zlib.crc32(s.encode()) % N_HASH


def features(c: Candidate) -> np.ndarray:
    t = c.text
    h = np.zeros(len(HAND_FEATURES))
    f = dict.fromkeys(HAND_FEATURES, 0.0)
    f["len"] = len(t) / 8
    f["len2"] = (len(t) / 8) ** 2
    groups, run, max_run, prev_v = 0, 0, 0, False
    for ch in t:
        v = ch in VOWELS
        if v and not prev_v:
            groups += 1
        run = 0 if v else run + 1
        max_run = max(max_run, run)
        prev_v = v
    f["n_vowel_groups"] = groups / 3
    f["frac_first"] = len(c.head) / max(len(c.first), 1)
    f["frac_second"] = len(c.tail) / max(len(c.second), 1)
    f[f"kind_{c.kind}"] = 1.0
    if c.head and c.tail:
        a, b = c.head[-1], c.tail[0]
        f["double_at_join"] = float(a == b)
        f["vv_join"] = float(a in VOWELS and b in VOWELS)
        f["cc_join"] = float(a not in VOWELS and b not in VOWELS)
    f["ends_vowel"] = float(t[-1] in VOWELS)
    f["silly_letters"] = sum(ch in SILLY for ch in t) / 3
    f["max_cons_run"] = max_run / 3
    bigrams = [t[i:i + 2] for i in range(len(t) - 1)]
    f["repeat_bigram"] = float(len(bigrams) != len(set(bigrams)))
    f["has_oo"] = float("oo" in t or "uu" in t)
    f["head_is_whole"] = float(c.head == c.first)
    f["tail_is_whole"] = float(c.tail == c.second)
    for e in ENDINGS:
        if t.endswith(e):
            f[f"end_{e}"] = 1.0
    h[:] = [f[k] for k in HAND_FEATURES]
    hashed = np.zeros(N_HASH)
    padded = f"^{t}$"
    for i in range(len(padded) - 2):
        hashed[_hash(padded[i:i + 3])] += 1.0
    return np.concatenate([h, hashed / max(len(t), 1) * 3])


@dataclass
class Vote:
    """One rating screen: the shown nicknames (with their source names) and the pick."""
    name_a: str
    name_b: str
    shown: list[str]
    chosen: int | None   # index into shown; None = "none of these are funny"


@dataclass
class RewardModel:
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(HAND_FEATURES) + N_HASH))
    n_votes: int = 0
    l2_hand: float = 0.5
    l2_hash: float = 3.0

    def score(self, cands: list[Candidate]) -> np.ndarray:
        if not cands:
            return np.zeros(0)
        return np.array([features(c) for c in cands]) @ self.w

    # ------------------------------------------------------------------ fit

    def fit(self, votes: list[Vote]) -> "RewardModel":
        sets = []
        for v in votes:
            lookup = {c.text: c for c in enumerate_candidates(v.name_a, v.name_b)}
            cands = [lookup[t] for t in v.shown if t in lookup]
            if len(cands) < 2 or (v.chosen is not None and v.shown[v.chosen] not in lookup):
                continue
            chosen = None if v.chosen is None else [c.text for c in cands].index(v.shown[v.chosen])
            sets.append((np.array([features(c) for c in cands]), chosen))
        self.n_votes = len(sets)
        if not sets:
            return self
        d = len(self.w)
        pen = np.concatenate([np.full(len(HAND_FEATURES), self.l2_hand), np.full(N_HASH, self.l2_hash)])

        def loss_grad(w):
            loss = 0.5 * np.sum(pen * w * w)
            grad = pen * w
            for X, chosen in sets:
                r = X @ w
                m = max(r.max(), 0.0)
                e = np.exp(r - m)
                z = e.sum() + math.exp(-m)            # + outside option with r_null = 0
                p = e / z
                loss += m + math.log(z) - (r[chosen] if chosen is not None else 0.0)
                grad += X.T @ p - (X[chosen] if chosen is not None else 0.0)
            return loss, grad

        res = minimize(loss_grad, self.w if len(self.w) == d else np.zeros(d), jac=True, method="L-BFGS-B")
        self.w = res.x
        return self

    # --------------------------------------------------------------- policy

    def sample(self, name_a: str, name_b: str, k: int, tau: float = 1.0,
               rng: np.random.Generator | None = None) -> list[str]:
        """k distinct nicknames from pi(c) ~ exp(r(c) / tau) (Gumbel top-k)."""
        rng = rng or np.random.default_rng()
        cands = enumerate_candidates(name_a, name_b)
        if not cands:
            return []
        r = self.score(cands) / max(tau, 1e-6)
        keys = r + rng.gumbel(size=len(r))
        order = np.argsort(-keys)[: min(k, len(cands))]
        return [cands[i].text for i in order]

    def rating_round(self, name_a: str, name_b: str, k: int = 4, explore: float = 0.5,
                     rng: np.random.Generator | None = None) -> list[str]:
        """Nicknames to show raters: some from the policy, some uniformly random."""
        rng = rng or np.random.default_rng()
        cands = [c.text for c in enumerate_candidates(name_a, name_b)]
        if len(cands) <= k:
            return list(rng.permutation(cands))
        n_explore = int(rng.binomial(k, explore))
        picked = self.sample(name_a, name_b, k - n_explore, tau=1.5, rng=rng)
        rest = [c for c in cands if c not in picked]
        picked += list(rng.choice(rest, size=n_explore, replace=False))
        return list(rng.permutation(picked))

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"w": self.w.tolist(), "n_votes": self.n_votes, "n_hash": N_HASH,
                                    "hand_features": HAND_FEATURES}))

    @classmethod
    def load(cls, path: Path) -> "RewardModel":
        d = json.loads(path.read_text())
        if d.get("hand_features") != HAND_FEATURES or d.get("n_hash") != N_HASH:
            return cls()   # feature set changed: start fresh (votes are kept, just refit)
        return cls(w=np.array(d["w"]), n_votes=d["n_votes"])

    def top_features(self, n: int = 12) -> list[tuple[str, float]]:
        names = HAND_FEATURES + [f"trigram#{i}" for i in range(N_HASH)]
        idx = np.argsort(-np.abs(self.w))[:n]
        return [(names[i], float(self.w[i])) for i in idx]
