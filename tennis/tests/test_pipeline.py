import json

import numpy as np
import pandas as pd
import pytest

from tennispred import data, features, pipeline, synthetic, twitter
from tennispred.cli import main
from tennispred.fixtures import CsvFixtureSource, Fixture
from tennispred.model import ServeModel, evaluate, fit_binomial_logit, fit_temperature, sigmoid
from tennispred.names import NameResolver


def test_glm_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = np.hstack([np.ones((5000, 1)), rng.normal(size=(5000, 3))])
    w_true = np.array([0.5, 0.3, -0.2, 0.0])
    n = rng.integers(40, 120, 5000).astype(float)
    k = rng.binomial(n.astype(int), sigmoid(X @ w_true)).astype(float)
    w = fit_binomial_logit(X, k, n, l2=0.0)
    assert w == pytest.approx(w_true, abs=0.02)


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1.5, 20000)
    y = rng.random(20000) < sigmoid(0.7 * z)
    assert fit_temperature(sigmoid(z), y.astype(float)) == pytest.approx(0.7, abs=0.05)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    synthetic.write(d, n_players=120, n_years=5, seed=3)
    return d


@pytest.fixture(scope="module")
def built(synth_dir):
    return pipeline.build_history(synth_dir, "atp", start_year=2000)


def test_no_leakage_first_match_has_neutral_elo(built):
    # Before anyone has played, Elo features must be zero.
    i = features.FEATURES.index("elo_diff")
    assert built.table.X_w[0, i] == 0.0 and built.table.X_l[0, i] == 0.0


def test_model_learns_planted_effects_and_beats_elo(built):
    t = built.table
    dates = t.matches.tourney_date
    train = (dates < "2019-07-01").to_numpy()
    test = ~train & t.matches.completed.to_numpy()
    m = ServeModel().fit(t, train)
    coef = m.coefficients()
    assert coef["lefty_srv_vs_righty"] > 0
    assert coef["surf_grass"] > 0 and coef["surf_clay"] < 0
    res = evaluate(m.table_probs(t, test), t.elo_p[test])
    assert res["log_loss"] < res["elo_log_loss"]
    assert res["accuracy"] > 0.7


def test_backtest_runs(built):
    res = pipeline.backtest(built, [2019], train_from="2016-01-01")
    assert list(res.index) == [2019] and res.loc[2019, "n"] > 100


def test_name_resolution(built):
    r = NameResolver(built.history)
    pid = 100000 + 7   # "Holger Synthh"
    assert built.history.players[pid].name == "Holger Synthh"
    for variant in ("Holger Synthh", "H. Synthh", "Synthh H.", "holger synthh"):
        assert r.resolve(variant) == pid
    assert r.resolve("Roger Nobody") is None


def test_predict_and_thread(built, synth_dir, tmp_path):
    model = pipeline.train(built, "2016-01-01")
    day = built.history.last_date + pd.Timedelta(days=2)
    fx = [Fixture(day, "Holger Synthh", "Carlos Synthc", "Synth Open 26"),
          Fixture(day, "A. Syntha", "Synthl B.", "Wimbledon")]
    preds = pipeline.predict_fixtures(built, model, fx, "atp")
    assert len(preds) == 2
    for p in preds:
        assert 0 < p["p1"] < 1
        assert sum(p["set_scores"].values()) == pytest.approx(1.0)
    wimbledon = next(p for p in preds if p["tournament"] == "Wimbledon")
    assert wimbledon["surface"] == "Grass" and wimbledon["best_of"] == 5
    thread = twitter.build_thread(preds * 10, day, "atp", max_matches=12)
    assert thread and all(len(t) <= twitter.TWEET_LIMIT for t in thread)
    path = pipeline.save_predictions(preds, tmp_path, day, "atp")
    assert len(json.loads(path.read_text())) == 2


def test_cli_dry_run(synth_dir, tmp_path, capsys):
    matches = data.load_matches(synth_dir)
    day = matches.tourney_date.max() + pd.Timedelta(days=1)
    fx = tmp_path / "fx.csv"
    pd.DataFrame([{"date": day.date(), "player1": "Alex Syntha", "player2": "Ben Synthb",
                   "tournament": "Synth Open 30"}]).to_csv(fx, index=False)
    main(["predict", "--data-dir", str(synth_dir), "--start-year", "2000", "--train-from", "2016-01-01",
          "--fixtures", str(fx), "--date", str(day.date()), "--out-dir", str(tmp_path / "out"),
          "--model", str(tmp_path / "m.json")])
    out = capsys.readouterr().out
    assert "1 predictions" in out and "dry run" in out
    assert ServeModel.load(tmp_path / "m.json").weights is not None
    assert CsvFixtureSource(fx).fixtures(day)[0].player1 == "Alex Syntha"


def test_post_thread_chains_replies(monkeypatch):
    import tweepy

    calls = []

    class FakeClient:
        def __init__(self, **creds):
            assert set(creds) == {"consumer_key", "consumer_secret", "access_token", "access_token_secret"}

        def create_tweet(self, text, in_reply_to_tweet_id=None):
            calls.append((text, in_reply_to_tweet_id))
            return type("R", (), {"data": {"id": str(len(calls))}})()

    monkeypatch.setattr(tweepy, "Client", FakeClient)
    for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET"):
        monkeypatch.setenv(k, "x")
    ids = twitter.post_thread(["one", "two", "three"], twitter.credentials_from_env())
    assert ids == ["1", "2", "3"]
    assert calls == [("one", None), ("two", "1"), ("three", "2")]


def test_insights_report(built):
    from tennispred import insights

    model = pipeline.train(built, "2016-01-01")
    day = built.history.last_date + pd.Timedelta(days=2)
    f = insights.matchup_facts(built, model, Fixture(day, "Holger Synthh", "Carlos Synthc", "Synth Open 26"),
                               n_sims=200, n_marathon_games=20_000, seed=0)
    sim = f["simulations"]
    assert sim["n"] == 200 and sum(sim["set_scores"].values()) == 200
    assert 0 < f["p1_win_final"] < 1 and f["drivers"]
    assert len(f["marathon"]["sequence"]) == f["marathon"]["points"] >= 4
    assert "simulated matches" in insights.render_text(f)
    json.dumps(f)  # must be serialisable for the server
    with pytest.raises(ValueError):
        insights.matchup_facts(built, model, Fixture(day, "Nobody Atall", "Carlos Synthc"))
