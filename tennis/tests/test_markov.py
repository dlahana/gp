import numpy as np
import pytest

from tennispred.markov import (match_probabilities, p_game, p_match, p_tiebreak, set_outcome,
                               simulate_match)


def brute_game(p: float) -> float:
    """Direct DP over point scores, with deuce capped far out."""
    from functools import lru_cache

    @lru_cache(None)
    def win(a, b):
        if a >= 4 and a - b >= 2:
            return 1.0
        if b >= 4 and b - a >= 2:
            return 0.0
        if a + b > 60:
            return 0.5
        return p * win(a + 1, b) + (1 - p) * win(a, b + 1)

    return win(0, 0)


@pytest.mark.parametrize("p", [0.3, 0.5, 0.62, 0.7, 0.9])
def test_game_closed_form_matches_dp(p):
    assert p_game(p) == pytest.approx(brute_game(p), abs=1e-8)


def test_known_values():
    assert p_game(0.5) == pytest.approx(0.5)
    assert p_game(0.6) == pytest.approx(0.7357, abs=1e-4)
    assert p_tiebreak(0.6, 0.6) == pytest.approx(0.5)
    assert p_match(0.64, 0.64, 3) == pytest.approx(0.5)
    assert p_match(0.64, 0.64, 5, 10) == pytest.approx(0.5)


def test_symmetry_and_monotonicity():
    a = p_match(0.66, 0.62, 3)
    assert a + p_match(0.62, 0.66, 3) == pytest.approx(1.0)
    assert p_match(0.67, 0.62, 3) > a
    # A longer match favours the stronger player more.
    assert p_match(0.66, 0.62, 5) > a > 0.5


def test_tiebreak_first_server_nearly_irrelevant():
    # Who serves first in a tiebreak has (almost) no effect with alternating serve.
    assert p_tiebreak(0.7, 0.6) == pytest.approx(1 - p_tiebreak(0.6, 0.7), abs=0.01)


def test_set_distribution_sums_to_one():
    for tb in (6, None):
        o = set_outcome(0.65, 0.6, True, tb)
        assert o.a_even + o.a_odd + o.b_even + o.b_odd == pytest.approx(1.0)
    m = match_probabilities(0.65, 0.6, 5, 10)
    assert sum(m.set_scores.values()) == pytest.approx(1.0)
    assert set(m.set_scores) == {"3-0", "3-1", "3-2", "0-3", "1-3", "2-3"}


def test_vectorised_matches_scalar():
    pa = np.array([0.55, 0.62, 0.7, 0.66])
    pb = np.array([0.6, 0.62, 0.58, 0.69])
    vec = p_match(pa, pb, 5, 10)
    for i in range(len(pa)):
        assert vec[i] == pytest.approx(p_match(float(pa[i]), float(pb[i]), 5, 10))


@pytest.mark.parametrize("best_of", [3, 5])
def test_monte_carlo_agrees(best_of):
    rng = np.random.default_rng(1)
    pa, pb, n = 0.66, 0.61, 6000
    wins = sum(simulate_match(pa, pb, best_of, rng).a_won for _ in range(n))
    exact = p_match(pa, pb, best_of)
    assert wins / n == pytest.approx(exact, abs=4 * np.sqrt(exact * (1 - exact) / n))
