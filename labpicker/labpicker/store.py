"""Data storage: CSV files in a git repo, which is the shared "server".

Layout (under the data dir):
  chemicals.csv          master list; edit via `labpicker add` or by hand
  tests/<stamp>_<who>.csv  one file per logging session, never edited afterwards

Test logs are one-file-per-session so two people logging at the same time can
never produce a git merge conflict.
"""
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CHEM_COLUMNS = ["chem_id", "name", "smiles", "sensitive", "sensitivity_note", "available"]
TEST_COLUMNS = ["chem_id", "tested_by", "tester_lab_id", "tested_at", "mode", "result", "notes"]


def _to_bool(series, default):
    s = series.astype("string").str.strip().str.lower()
    out = s.map({"true": True, "1": True, "yes": True, "y": True,
                 "false": False, "0": False, "no": False, "n": False})
    return out.fillna(default).astype(bool)


def load_chemicals(data_dir=DEFAULT_DATA_DIR):
    path = Path(data_dir) / "chemicals.csv"
    if not path.exists():
        return pd.DataFrame(columns=CHEM_COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in CHEM_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["sensitive"] = _to_bool(df["sensitive"], default=False)
    df["available"] = _to_bool(df["available"], default=True)
    return df[CHEM_COLUMNS]


def save_chemicals(df, data_dir=DEFAULT_DATA_DIR):
    path = Path(data_dir) / "chemicals.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[CHEM_COLUMNS].copy()
    out["sensitive"] = out["sensitive"].map({True: "true", False: "false"})
    out["available"] = out["available"].map({True: "true", False: "false"})
    out.to_csv(path, index=False)
    return path


def next_chem_id(chemicals):
    nums = chemicals["chem_id"].str.extract(r"^C(\d+)$")[0].dropna().astype(int)
    return f"C{(nums.max() + 1) if len(nums) else 1:05d}"


def load_tests(data_dir=DEFAULT_DATA_DIR):
    files = sorted((Path(data_dir) / "tests").glob("*.csv"))
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]
    if not frames:
        return pd.DataFrame(columns=TEST_COLUMNS)
    return pd.concat(frames, ignore_index=True)[TEST_COLUMNS]


def write_test_session(rows, tester, data_dir=DEFAULT_DATA_DIR):
    """Write one new file of test rows; returns its path."""
    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^a-z0-9]+", "-", tester.lower()).strip("-") or "unknown"
    path = Path(data_dir) / "tests" / f"{now:%Y%m%dT%H%M%SZ}_{slug}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TEST_COLUMNS).to_csv(path, index=False)
    return path


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- git sync -------------------------------------------------------------

def _git(repo, *args, check=True, identity=None):
    cmd = ["git", "-C", str(repo)]
    if identity:
        cmd += ["-c", f"user.name={identity[0]}", "-c", f"user.email={identity[1]}"]
    r = subprocess.run(cmd + list(args), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def _has_remote(repo):
    return bool(_git(repo, "remote", check=False).stdout.strip())


def sync_pull(data_dir=DEFAULT_DATA_DIR):
    if not _has_remote(data_dir):
        return
    _git(data_dir, "pull", "--rebase", "--autostash")


def commit_and_push(paths, message, tester, lab_id, data_dir=DEFAULT_DATA_DIR, retries=3):
    """Commit the given files and push, re-pulling if someone else pushed first."""
    identity = None
    if not _git(data_dir, "config", "user.name", check=False).stdout.strip():
        identity = (tester, f"{lab_id or 'unknown'}@labpicker.invalid")
    _git(data_dir, "add", "--", *[str(p) for p in paths])
    _git(data_dir, "commit", "-m", message, identity=identity)
    if not _has_remote(data_dir):
        return
    for attempt in range(retries):
        if _git(data_dir, "push", check=False).returncode == 0:
            return
        _git(data_dir, "pull", "--rebase", "--autostash")
    _git(data_dir, "push")  # last try; raises with git's message
