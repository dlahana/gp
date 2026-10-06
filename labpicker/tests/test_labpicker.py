import pytest

from labpicker import cli, picker, store


def run(tmp_path, *argv):
    cli.main(["--data-dir", str(tmp_path), "--no-sync", "--tester", "Ann", "--lab-id", "L1", *argv])


@pytest.fixture
def data(tmp_path):
    (tmp_path / "chemicals.csv").write_text(
        (store.DEFAULT_DATA_DIR / "chemicals.csv").read_text())
    return tmp_path


def test_explore_picks_diverse_and_skips_tested(data):
    run(data, "log", "C00003")  # benzene tested
    chems, tests = store.load_chemicals(data), store.load_tests(data)
    s = picker.suggest(chems, tests, "explore", 3)
    assert "C00003" not in set(s["chem_id"])
    # toluene/phenol/aniline are near benzene; should not lead the picks
    assert s.iloc[0]["chem_id"] not in {"C00004", "C00005", "C00006"}


def test_sensitivity_defaults(data):
    chems, tests = store.load_chemicals(data), store.load_tests(data)
    assert "C00011" in set(picker.suggest(chems, tests, "explore", 12)["chem_id"])
    assert "C00011" not in set(picker.suggest(chems, tests, "explore", 12, exclude_sensitive=True)["chem_id"])
    assert "C00011" in set(picker.eligible(chems, tests, False)["chem_id"])
    assert "C00011" not in set(picker.eligible(chems, tests, picker.DEFAULT_EXCLUDE_SENSITIVE["exploit"])["chem_id"])


def test_exploit_requires_model(data):
    chems, tests = store.load_chemicals(data), store.load_tests(data)
    with pytest.raises(NotImplementedError):
        picker.suggest(chems, tests, "exploit", 3)


def test_exploit_uses_model(data, monkeypatch):
    monkeypatch.setattr(picker.model, "score_candidates",
                        lambda c, a, t: c["name"].str.len().to_numpy())
    chems, tests = store.load_chemicals(data), store.load_tests(data)
    s = picker.suggest(chems, tests, "exploit", 2)
    assert list(s["name"]) == ["acetic acid", "pyridine"]
    assert "C00011" not in set(s["chem_id"])


def test_log_records_who_and_blocks_duplicates(data):
    run(data, "log", "C00001", "C00002")
    t = store.load_tests(data)
    assert set(t["tested_by"]) == {"Ann"} and set(t["tester_lab_id"]) == {"L1"}
    with pytest.raises(SystemExit):
        run(data, "log", "C00001")
    with pytest.raises(SystemExit):
        run(data, "log", "NOPE")


def test_add_assigns_id_validates_and_dedups(data):
    run(data, "add", "--name", "methanol", "--smiles", "CO")
    assert store.load_chemicals(data).iloc[-1]["chem_id"] == "C00013"
    run(data, "add", "--name", "methanol again", "--smiles", "CO")
    assert len(store.load_chemicals(data)) == 13
    with pytest.raises(SystemExit):
        run(data, "add", "--name", "bad", "--smiles", "not a smiles(((")


def test_suggest_logs_only_confirmed(data, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "1")
    run(data, "suggest", "--mode", "explore", "-n", "3")
    assert len(store.load_tests(data)) == 1
    monkeypatch.setattr("builtins.input", lambda *_: "")
    run(data, "suggest", "--mode", "explore", "-n", "3")
    assert len(store.load_tests(data)) == 1


def test_git_sync_roundtrip(tmp_path):
    import subprocess
    sh = lambda *a, cwd: subprocess.run(a, cwd=cwd, check=True, capture_output=True)
    bare, a, b = tmp_path / "r.git", tmp_path / "a", tmp_path / "b"
    sh("git", "init", "--bare", "-b", "main", str(bare), cwd=tmp_path)
    sh("git", "clone", str(bare), str(a), cwd=tmp_path)
    (a / "data").mkdir()
    (a / "data" / "chemicals.csv").write_text((store.DEFAULT_DATA_DIR / "chemicals.csv").read_text())
    for c in (("git", "add", "."), ("git", "commit", "-m", "init"), ("git", "push", "-u", "origin", "HEAD")):
        sh(*c, cwd=a)
    sh("git", "clone", str(bare), str(b), cwd=tmp_path)
    for r in (a, b):
        sh("git", "config", "user.name", "x", cwd=r); sh("git", "config", "user.email", "x@x", cwd=r)
    # both log at once -> separate files, no conflict
    cli.main(["--data-dir", str(a / "data"), "--tester", "Ann", "--lab-id", "L1", "log", "C00001"])
    cli.main(["--data-dir", str(b / "data"), "--tester", "Bob", "--lab-id", "L2", "log", "C00002"])
    cli.main(["--data-dir", str(a / "data"), "--tester", "Ann", "--lab-id", "L1", "status"])
    assert set(store.load_tests(a / "data")["tested_by"]) == {"Ann", "Bob"}
