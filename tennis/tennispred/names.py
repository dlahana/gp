"""Match fixture names ("C. Alcaraz", "Alcaraz C.", "Carlos Alcaraz") to player ids."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import pandas as pd

from .features import History


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace("-", " "))
    return re.sub(r"\s+", " ", s).strip()


class NameResolver:
    def __init__(self, history: History, active_since: pd.Timestamp | None = None):
        self.hist = history
        self.full: dict[str, list[int]] = defaultdict(list)
        self.by_last: dict[str, list[int]] = defaultdict(list)
        for pid, st in history.players.items():
            if not st.name:
                continue
            if active_since is not None and (not st.dates or st.dates[-1] < active_since):
                continue
            norm = normalize(st.name)
            self.full[norm].append(pid)
            toks = norm.split()
            # Index every suffix as a possible surname ("del potro", "potro").
            for i in range(1, len(toks)):
                self.by_last[" ".join(toks[i:])].append(pid)

    def _first_tokens(self, pid: int) -> list[str]:
        return normalize(self.hist.players[pid].name).split()

    def _most_active(self, pids: list[int]) -> int | None:
        if not pids:
            return None
        return max(set(pids), key=lambda p: (self.hist.players[p].dates[-1] if self.hist.players[p].dates
                                             else pd.Timestamp.min, self.hist.players[p].n))

    def resolve(self, name: str) -> int | None:
        norm = normalize(name)
        if not norm:
            return None
        if norm in self.full:
            return self._most_active(self.full[norm])
        toks = norm.split()
        # Try every split into (initials/first names, surname), in both orders.
        candidates: list[int] = []
        for i in range(1, len(toks)):
            for first, last in ((toks[:i], toks[i:]), (toks[len(toks) - i:], toks[:len(toks) - i])):
                surname = " ".join(last)
                for pid in self.by_last.get(surname, []):
                    given = self._first_tokens(pid)
                    if given and all(any(g.startswith(f) for g in given) for f in first):
                        candidates.append(pid)
            if candidates:
                break
        if not candidates and len(toks) == 1:
            candidates = self.by_last.get(toks[0], [])
        return self._most_active(candidates)
