"""Learning the random walk's step probabilities.

The model is a binomial logistic regression for the probability that the
server wins a point:

    logit P(server wins point) = w . x(server, returner, context)

It is fit on serve points won / played per player per match (two observations
per match). The fitted serve probabilities for both players feed the exact
Markov chain in ``markov.py`` to give P(match).

The independent-points assumption makes the chain a little over-confident
(real players' point probabilities vary from match to match), so a single
temperature on the match log-odds is fit on held-out matches:

    P_final = sigmoid(t * logit(P_markov))
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, FeatureTable
from .markov import p_match


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def fit_binomial_logit(X: np.ndarray, k: np.ndarray, n: np.ndarray, l2: float = 1.0,
                       max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Newton-Raphson for a binomial GLM with logit link and L2 penalty.

    X must include an intercept column (index 0, unpenalised).
    """
    d = X.shape[1]
    w = np.zeros(d)
    p0 = k.sum() / n.sum()
    w[0] = np.log(p0 / (1 - p0))
    pen = np.full(d, l2)
    pen[0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(X @ w)
        grad = X.T @ (k - n * p) - pen * w
        W = n * p * (1 - p)
        H = (X * W[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, grad)
        w += step
        if np.max(np.abs(step)) < tol:
            break
    return w


def final_set_tb_target(level: str, date: pd.Timestamp) -> int:
    """Grand Slams have used a 10-point final-set tiebreak since 2022."""
    return 10 if level == "G" and date >= pd.Timestamp("2022-01-01") else 7


def markov_match_prob(p_a: np.ndarray, p_b: np.ndarray, best_of: np.ndarray,
                      tb_target: np.ndarray) -> np.ndarray:
    """Vectorised P(A wins match), grouping by match format."""
    out = np.empty(len(p_a))
    for bo in np.unique(best_of):
        for tb in np.unique(tb_target):
            m = (best_of == bo) & (tb_target == tb)
            if m.any():
                out[m] = p_match(p_a[m], p_b[m], int(bo), int(tb))
    return out


def fit_temperature(p_raw: np.ndarray, y: np.ndarray) -> float:
    """1-D Newton on log loss of sigmoid(t * logit(p_raw)) against y."""
    z = logit(p_raw)
    t = 1.0
    for _ in range(50):
        p = sigmoid(t * z)
        g = np.sum((p - y) * z)
        h = np.sum(p * (1 - p) * z * z) + 1e-9
        step = g / h
        t -= step
        if abs(step) < 1e-10:
            break
    return float(np.clip(t, 0.2, 3.0))


@dataclass
class ServeModel:
    weights: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    temperature: float = 1.0
    l2: float = 1.0
    feature_names: tuple = tuple(FEATURES)
    trained_through: str | None = None
    n_obs: int = 0

    # --------------------------------------------------------------- serve p

    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mean) / self.scale
        return np.hstack([np.ones((len(Z), 1)), Z])

    def p_serve(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return sigmoid(self._design(X) @ self.weights)

    def _fit_glm(self, X, k, n) -> None:
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        self.weights = fit_binomial_logit(self._design(X), k, n, self.l2)
        self.n_obs = len(X)

    # ----------------------------------------------------------------- match

    def match_prob(self, X_a: np.ndarray, X_b: np.ndarray, best_of, tb_target,
                   calibrated: bool = True) -> np.ndarray:
        """P(A wins). X_a: A serving to B; X_b: B serving to A."""
        p_a, p_b = self.p_serve(X_a), self.p_serve(X_b)
        n = len(p_a)
        raw = markov_match_prob(p_a, p_b, np.broadcast_to(best_of, n), np.broadcast_to(tb_target, n))
        return sigmoid(self.temperature * logit(raw)) if calibrated else raw

    def table_probs(self, table: FeatureTable, mask: np.ndarray, calibrated: bool = True) -> np.ndarray:
        """P(winner wins) for the masked matches in a feature table."""
        m = table.matches[mask]
        tb = np.array([final_set_tb_target(lv, d) for lv, d in zip(m.tourney_level, m.tourney_date)])
        return self.match_prob(table.X_w[mask], table.X_l[mask], m.best_of.to_numpy(), tb, calibrated)

    # ------------------------------------------------------------------- fit

    def fit(self, table: FeatureTable, mask: np.ndarray, calib_frac: float = 0.15) -> "ServeModel":
        """Fit the serve model and the match-level temperature.

        The temperature is estimated on the chronologically last `calib_frac`
        of the training window using a GLM fit on the rest, then the GLM is
        refit on the whole window.
        """
        dates = table.matches.tourney_date
        train_dates = dates[mask]
        cut = train_dates.quantile(1 - calib_frac)
        early = mask & (dates < cut).to_numpy()
        late = mask & (dates >= cut).to_numpy() & table.matches.completed.to_numpy()

        if calib_frac > 0 and early.sum() > 50 and late.sum() > 50:
            self._fit_glm(*table.serve_observations(early))
            self.temperature = 1.0
            raw = self.table_probs(table, late, calibrated=False)
            # Every row is from the winner's side, so y = 1. Mirror the
            # rows (y = 0, p -> 1 - p) so the fit sees both labels.
            self.temperature = fit_temperature(np.concatenate([raw, 1 - raw]),
                                               np.concatenate([np.ones(len(raw)), np.zeros(len(raw))]))

        self._fit_glm(*table.serve_observations(mask))
        self.trained_through = str(train_dates.max().date())
        return self

    # ------------------------------------------------------------ inspection

    def coefficients(self) -> pd.Series:
        """Effect of a one-standard-deviation change, in serve-point log-odds."""
        return pd.Series(self.weights[1:], index=self.feature_names).sort_values(key=np.abs, ascending=False)

    # ------------------------------------------------------------------ I/O

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "weights": self.weights.tolist(), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "temperature": self.temperature, "l2": self.l2, "feature_names": list(self.feature_names),
            "trained_through": self.trained_through, "n_obs": self.n_obs,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ServeModel":
        d = json.loads(path.read_text())
        if d["feature_names"] != list(FEATURES):
            raise ValueError(f"{path} was trained with a different feature set; retrain it")
        return cls(weights=np.array(d["weights"]), mean=np.array(d["mean"]), scale=np.array(d["scale"]),
                   temperature=d["temperature"], l2=d["l2"], feature_names=tuple(d["feature_names"]),
                   trained_through=d["trained_through"], n_obs=d["n_obs"])


def evaluate(p_winner: np.ndarray, elo_p: np.ndarray | None = None) -> dict:
    """Log loss, Brier score and accuracy for P(actual winner wins)."""
    p = np.clip(p_winner, 1e-9, 1 - 1e-9)
    out = {"n": int(len(p)), "log_loss": float(-np.mean(np.log(p))),
           "brier": float(np.mean((1 - p) ** 2)), "accuracy": float(np.mean(p > 0.5))}
    if elo_p is not None:
        e = np.clip(elo_p, 1e-9, 1 - 1e-9)
        out.update({"elo_log_loss": float(-np.mean(np.log(e))), "elo_brier": float(np.mean((1 - e) ** 2)),
                    "elo_accuracy": float(np.mean(e > 0.5))})
    return out
