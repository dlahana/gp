"""Candidate selection for explore and exploit modes."""
import warnings

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from . import model

MODES = ("explore", "exploit")
# Exclude sensitive compounds? Off for explore, on for exploit unless overridden.
DEFAULT_EXCLUDE_SENSITIVE = {"explore": False, "exploit": True}

_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(smiles):
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return _fpgen.GetFingerprint(mol) if mol is not None else None


def eligible(chemicals, tests, exclude_sensitive):
    """Candidates: in stock, never tested, and (optionally) not sensitive."""
    mask = chemicals["available"] & ~chemicals["chem_id"].isin(tests["chem_id"])
    if exclude_sensitive:
        mask &= ~chemicals["sensitive"]
    return chemicals[mask]


def pick_diverse(candidate_smiles, tested_smiles, n, seed=0):
    """Greedy MaxMin: repeatedly take the candidate farthest (Tanimoto distance)
    from everything already tested or picked. Returns positions into candidate_smiles.
    Candidates with unparseable SMILES are skipped.
    """
    cand_fps = [fingerprint(s) for s in candidate_smiles]
    valid = [i for i, fp in enumerate(cand_fps) if fp is not None]
    if len(valid) < len(cand_fps):
        warnings.warn(f"{len(cand_fps) - len(valid)} candidate(s) have unparseable SMILES; skipped")
    tested_fps = [fp for fp in (fingerprint(s) for s in tested_smiles) if fp is not None]

    fps = [cand_fps[i] for i in valid]
    min_dist = np.full(len(fps), np.inf)
    for t in tested_fps:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(t, fps))
        min_dist = np.minimum(min_dist, 1.0 - sims)

    chosen = []
    taken = np.zeros(len(fps), dtype=bool)
    for _ in range(min(n, len(fps))):
        if np.isinf(min_dist).all():  # nothing tested yet: seed with a random pick
            k = int(np.random.default_rng(seed).choice(np.flatnonzero(~taken)))
        else:
            k = int(np.argmax(np.where(taken, -1.0, min_dist)))
        chosen.append(valid[k])
        taken[k] = True
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[k], fps))
        min_dist = np.minimum(min_dist, 1.0 - sims)
    return chosen


def suggest(chemicals, tests, mode, n, exclude_sensitive=None, seed=0):
    """Return up to n rows of `chemicals` to test next, in suggested order."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if exclude_sensitive is None:
        exclude_sensitive = DEFAULT_EXCLUDE_SENSITIVE[mode]
    cands = eligible(chemicals, tests, exclude_sensitive)
    if cands.empty:
        return cands

    if mode == "explore":
        tested = chemicals[chemicals["chem_id"].isin(tests["chem_id"])]
        idx = pick_diverse(cands["smiles"].tolist(), tested["smiles"].tolist(), n, seed)
        return cands.iloc[idx]

    scores = np.asarray(model.score_candidates(cands, chemicals, tests), dtype=float)
    if scores.shape != (len(cands),):
        raise ValueError("model.score_candidates must return one score per candidate")
    order = np.argsort(-scores, kind="stable")[:n]
    out = cands.iloc[order].copy()
    out["score"] = scores[order]
    return out
