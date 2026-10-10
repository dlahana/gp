import numpy as np
import pytest
from fastapi.testclient import TestClient

from tennispred import corny, name_pool, twitter
from tennispred.portmanteau import clean, enumerate_candidates, syllables
from tennispred.reward import RewardModel, Vote


def texts(a, b):
    return {c.text for c in enumerate_candidates(a, b)}


def test_syllables():
    assert syllables("alcaraz") == ["al", "ca", "raz"]
    assert syllables("federer") == ["fe", "de", "rer"]
    assert syllables("muchova") == ["mu", "cho", "va"]


def test_clean():
    assert clean("Alex de Minaur") == "deminaur"
    assert clean("Alejandro Davidovich Fokina") == "fokina"
    assert clean("Félix Auger-Aliassime") == "augeraliassime"


def test_known_blends_are_enumerated():
    assert {"alcarer", "fedal"} <= texts("Alcaraz", "Federer") | texts("Nadal", "Federer")
    assert "naderer" in texts("Nadal", "Federer")
    assert "sincaraz" in texts("Sinner", "Alcaraz")
    for t in texts("Sinner", "Alcaraz"):
        assert t not in ("sinner", "alcaraz") and 4 <= len(t) <= 14


def test_reward_learns_simulated_taste():
    rng = np.random.default_rng(0)
    names = [p.name for p in name_pool.seed_pool()][:40]

    def taste(t):   # this "friend" loves names ending in -er
        return 3.0 * t.endswith("er")

    m, votes = RewardModel(), []
    for _ in range(300):
        a, b = rng.choice(names, 2, replace=False)
        shown = m.rating_round(a, b, 4, rng=rng)
        u = np.array([taste(s) for s in shown] + [-2.0])
        i = rng.choice(len(u), p=np.exp(u) / np.exp(u).sum())
        votes.append(Vote(a, b, shown, None if i == len(shown) else int(i)))
    m.fit(votes)
    picks = m.sample("Nadal", "Federer", 10, tau=0.3, rng=rng)
    assert sum(p.endswith("er") for p in picks) >= 7


def test_sample_counts_and_distinct():
    picks = RewardModel().sample("Sinner", "Alcaraz", 7, rng=np.random.default_rng(1))
    assert len(picks) == len(set(picks)) == 7


def test_fallback_thread_fits():
    m = [{"player1": "Jannik Sinner", "player2": "Carlos Alcaraz", "p1": 0.6, "nicknames": ["sincaraz"]}] * 12
    tweets, used = corny.fallback_thread(m, "🎾 Picks")
    assert all(len(t) <= 280 for t in tweets) and used[0] == "sincaraz"


# ------------------------------------------------------------------ server

@pytest.fixture
def client(tmp_path, monkeypatch):
    from tennispred.server import app as app_mod

    monkeypatch.setenv("BOT_ADMIN_TOKEN", "secret")
    monkeypatch.setenv("BOT_PUBLIC_URL", "https://bot.test")
    monkeypatch.setenv("BOT_REWARD_PATH", str(tmp_path / "reward.json"))
    texts_sent, posted = [], []
    monkeypatch.setattr(app_mod.notify, "send_sms", lambda body: texts_sent.append(body) or True)
    monkeypatch.setattr(app_mod.corny, "write_thread",
                        lambda matches, day, extra="": (["corny " + day + (" " + extra if extra else "")],
                                                        [m["nicknames"][0] for m in matches], "claude"))
    monkeypatch.setattr(app_mod.twitter, "credentials_from_env", lambda: {"k": "v"})
    monkeypatch.setattr(app_mod.twitter, "post_thread", lambda tweets, creds: posted.append(tweets) or ["111"])
    c = TestClient(app_mod.create_app(str(tmp_path / "bot.db")))
    c.texts, c.posted = texts_sent, posted
    return c


AUTH = {"Authorization": "Bearer secret"}


def test_rating_flow(client):
    assert client.post("/api/invites", json={"name": "sam"}).status_code == 401
    url = client.post("/api/invites", json={"name": "sam"}, headers=AUTH).json()["url"]
    code = url.rsplit("/", 1)[1]
    assert client.get(f"/rate/{code}").status_code == 200
    assert client.get("/rate/nope").status_code == 404
    r = client.get(f"/api/round/{code}").json()
    assert len(r["shown"]) == 4
    nxt = client.post(f"/api/vote/{code}", json={"round_id": r["round_id"], "chosen": 2}).json()
    assert nxt["my_votes"] == 1 and nxt["round_id"] != r["round_id"]
    client.post(f"/api/vote/{code}", json={"round_id": nxt["round_id"], "chosen": None})
    assert client.post(f"/api/vote/{code}", json={"round_id": nxt["round_id"], "chosen": 9}).status_code == 400
    assert client.get("/api/admin/stats", headers=AUTH).json()["votes"] == 2


def test_draft_approval_flow(client):
    match = {"player1": "Jannik Sinner", "player2": "Carlos Alcaraz", "p1": 0.58, "tournament": "Shanghai",
             "surface": "Hard"}
    assert client.post("/api/drafts", json={"day": "Sat Oct 10", "matches": [match]}).status_code == 401
    out = client.post("/api/drafts", json={"day": "Sat Oct 10", "matches": [match]}, headers=AUTH).json()
    assert out["texted"] and out["url"] in client.texts[0]
    did, tok = out["id"], out["url"].split("t=")[1]
    page = client.get(f"/drafts/{did}?t={tok}")
    assert page.status_code == 200 and "corny Sat Oct 10" in page.text
    assert client.get(f"/drafts/{did}?t=wrong").status_code == 404
    regen = client.post(f"/api/drafts/{did}/regenerate?t={tok}", json={"direction": "more puns"}).json()
    assert regen["tweets"] == ["corny Sat Oct 10 more puns"]
    assert not client.posted
    assert client.post(f"/api/drafts/{did}/approve?t={tok}", json={"tweets": ["x" * 281]}).status_code == 400
    ok = client.post(f"/api/drafts/{did}/approve?t={tok}", json={"tweets": ["final edit"]}).json()
    assert ok["status"] == "posted" and client.posted == [["final edit"]]
    assert client.post(f"/api/drafts/{did}/approve?t={tok}", json={"tweets": ["again"]}).status_code == 409


def test_draft_has_2_to_10_nicknames(client, tmp_path):
    from tennispred.server.store import Store

    match = {"player1": "Rafael Nadal", "player2": "Roger Federer", "p1": 0.5}
    for _ in range(5):
        did = client.post("/api/drafts", json={"day": "d", "matches": [match]}, headers=AUTH).json()["id"]
        nicks = Store(tmp_path / "bot.db").draft(did)["matches"][0]["nicknames"]
        assert 2 <= len(nicks) <= 10
