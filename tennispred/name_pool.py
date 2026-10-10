"""Surnames used for rating rounds.

A built-in list of well-known players gets the rating page working on day
one. ``from_players_file`` builds a much bigger pool from Sackmann's players
file, weighted towards the big tennis countries so the reward model learns
on the kinds of names it will actually see (Spanish, Czech, Italian, ...).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Relative weight when sampling names for rating, by IOC code.
COUNTRY_WEIGHT = {"ESP": 3.0, "CZE": 3.0, "ITA": 2.5, "FRA": 2.0, "SRB": 2.0, "RUS": 2.0, "ARG": 2.0,
                  "USA": 1.5, "GER": 1.5, "AUS": 1.5, "CRO": 1.5, "POL": 1.5, "SUI": 1.5, "SVK": 1.5}


@dataclass
class PoolName:
    name: str
    country: str
    tour: str


SEED = [(n, c, "atp") for n, c in [
    ("Alcaraz", "ESP"), ("Sinner", "ITA"), ("Djokovic", "SRB"), ("Zverev", "GER"), ("Fritz", "USA"),
    ("Draper", "GBR"), ("de Minaur", "AUS"), ("Rune", "DEN"), ("Ruud", "NOR"), ("Tsitsipas", "GRE"),
    ("Musetti", "ITA"), ("Paul", "USA"), ("Shelton", "USA"), ("Rublev", "RUS"), ("Medvedev", "RUS"),
    ("Hurkacz", "POL"), ("Mensik", "CZE"), ("Lehecka", "CZE"), ("Machac", "CZE"), ("Fokina", "ESP"),
    ("Bautista Agut", "ESP"), ("Munar", "ESP"), ("Carreno Busta", "ESP"), ("Nadal", "ESP"),
    ("Federer", "SUI"), ("Murray", "GBR"), ("Wawrinka", "SUI"), ("Berdych", "CZE"), ("Stepanek", "CZE"),
    ("Ferrer", "ESP"), ("Verdasco", "ESP"), ("Kyrgios", "AUS"), ("Monfils", "FRA"), ("Tsonga", "FRA"),
    ("Gasquet", "FRA"), ("Fils", "FRA"), ("Humbert", "FRA"), ("Cerundolo", "ARG"), ("Etcheverry", "ARG"),
    ("Baez", "ARG"), ("Navone", "ARG"), ("Tabilo", "CHI"), ("Jarry", "CHI"), ("Fonseca", "BRA"),
    ("Auger-Aliassime", "CAN"), ("Shapovalov", "CAN"), ("Tiafoe", "USA"), ("Korda", "USA"),
    ("Nakashima", "USA"), ("Khachanov", "RUS"), ("Popyrin", "AUS"), ("Bublik", "KAZ"),
    ("Griekspoor", "NED"), ("Dimitrov", "BUL"), ("Cilic", "CRO"), ("Berrettini", "ITA"),
    ("Cobolli", "ITA"), ("Arnaldi", "ITA"), ("Darderi", "ITA"), ("Sonego", "ITA"), ("Struff", "GER"),
    ("Altmaier", "GER"), ("Kecmanovic", "SRB"), ("Medjedovic", "SRB"), ("Landaluce", "ESP"),
    ("Thiem", "AUT"), ("Nishikori", "JPN"), ("del Potro", "ARG"), ("Isner", "USA"), ("Raonic", "CAN"),
]] + [(n, c, "wta") for n, c in [
    ("Sabalenka", "BLR"), ("Swiatek", "POL"), ("Gauff", "USA"), ("Rybakina", "KAZ"), ("Pegula", "USA"),
    ("Paolini", "ITA"), ("Zheng", "CHN"), ("Navarro", "USA"), ("Keys", "USA"), ("Andreeva", "RUS"),
    ("Muchova", "CZE"), ("Krejcikova", "CZE"), ("Vondrousova", "CZE"), ("Kvitova", "CZE"),
    ("Siniakova", "CZE"), ("Noskova", "CZE"), ("Bouzkova", "CZE"), ("Pliskova", "CZE"), ("Badosa", "ESP"),
    ("Bouzas Maneiro", "ESP"), ("Muguruza", "ESP"), ("Sorribes Tormo", "ESP"), ("Ostapenko", "LAT"),
    ("Svitolina", "UKR"), ("Kostyuk", "UKR"), ("Jabeur", "TUN"), ("Osaka", "JPN"), ("Raducanu", "GBR"),
    ("Collins", "USA"), ("Kasatkina", "RUS"), ("Samsonova", "RUS"), ("Alexandrova", "RUS"),
    ("Haddad Maia", "BRA"), ("Fernandez", "CAN"), ("Mertens", "BEL"), ("Azarenka", "BLR"),
    ("Vekic", "CRO"), ("Williams", "USA"), ("Halep", "ROU"), ("Kerber", "GER"), ("Errani", "ITA"),
]]


def seed_pool() -> list[PoolName]:
    return [PoolName(n, c, t) for n, c, t in SEED]


def from_players_file(path: Path, tour: str, born_after: int = 1975, limit: int = 3000) -> list[PoolName]:
    """Players from Sackmann's {tour}_players.csv, recent generations only."""
    df = pd.read_csv(path, low_memory=False)
    df = df[pd.to_numeric(df["dob"], errors="coerce") >= born_after * 10000]
    df = df.dropna(subset=["name_last"])
    df = df[df["name_last"].str.len() >= 3]
    out = [PoolName(str(r.name_last), str(r.ioc), tour) for r in df.head(limit).itertuples()]
    return out


def save(pool: list[PoolName], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in pool], indent=0))


def load(path: Path | None) -> list[PoolName]:
    if path is None or not path.exists():
        return seed_pool()
    return [PoolName(**d) for d in json.loads(path.read_text())]


def sample_pair(pool: list[PoolName], rng: np.random.Generator) -> tuple[PoolName, PoolName]:
    """Two names from the same tour, weighted towards the big tennis countries."""
    w = np.array([COUNTRY_WEIGHT.get(p.country, 1.0) for p in pool])
    i = rng.choice(len(pool), p=w / w.sum())
    same = [j for j, p in enumerate(pool) if p.tour == pool[i].tour and j != i]
    ws = w[same]
    j = same[rng.choice(len(same), p=ws / ws.sum())]
    return pool[i], pool[j]
