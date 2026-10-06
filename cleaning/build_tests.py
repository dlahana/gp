"""Build the tested-compound log from the DART contact assays and the assay-kit shipment sheet.

Usage: python -I cleaning/build_tests.py FLIES.xlsx KIT.csv COMPOUNDS_ALL.csv OUT_DIR
Writes OUT_DIR/tests_legacy.csv, OUT_DIR/tested_not_in_inventory.csv and prints a reconciliation report.
Only compounds that appear in the 'trials' sheet count as tested; abbreviations that are in the key
but never used in a trial do not.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from py2opsin import py2opsin
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
DOMAIN = "swd"  # both sources are SWD per the lab


def fuzzy(n):
    """Loose name for matching essential-oil style entries ('Basil Exotic (florihana)' ~ 'Basil Exotic')."""
    import re
    n = re.sub(r"\(.*?\)", " ", str(n).lower())
    n = re.sub(r"\b(essential oil|oil|extract.*|brand|hllozzi|steam distilled|commercially available)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def parse_date(v):
    s = str(int(v)).zfill(8)  # mmddyyyy stored as int, leading zero lost
    return f"{s[4:]}-{s[:2]}-{s[2:4]}"


def main(flies, kit, comp_csv, out_dir):
    out = Path(out_dir)
    comp = pd.read_csv(comp_csv, keep_default_na=False)
    by_key = {k: c for k, c in zip(comp["inchikey"], comp["chem_id"]) if k}
    by_block = {}
    for k, c in zip(comp["inchikey"], comp["chem_id"]):
        if k:
            by_block.setdefault(k[:14], []).append(c)
    by_name = {}
    for c, n, o in zip(comp["chem_id"], comp["name"], comp["other_names"]):
        for x in [n] + [y.strip() for y in o.split("|") if y.strip()]:
            by_name.setdefault(x.lower(), c)

    by_fuzzy = {}
    for c, n in zip(comp["chem_id"], comp["name"]):
        by_fuzzy.setdefault(fuzzy(n), c)

    def lookup(name, key):
        """Return (chem_id, how)."""
        if key and key in by_key:
            return by_key[key], "inchikey"
        if key and key[:14] in by_block and len(by_block[key[:14]]) == 1:
            return by_block[key[:14]][0], "inchikey (connectivity only)"
        if name.lower() in by_name:
            return by_name[name.lower()], "name"
        if not key and fuzzy(name) in by_fuzzy and fuzzy(name):
            return by_fuzzy[fuzzy(name)], "loose name (oil/mixture)"
        smi = py2opsin(name, output_format="SMILES")
        if smi:
            k = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
            if k in by_key:
                return by_key[k], "structure from name"
            if k[:14] in by_block and len(by_block[k[:14]]) == 1:
                return by_block[k[:14]][0], "structure from name (connectivity only)"
        return None, ""

    rows, unmatched, how_counts = [], [], {}

    # --- DART contact assays ---
    xl = pd.ExcelFile(flies)
    trials, key = xl.parse("trials"), xl.parse("key to chem names and abbrev")
    key["Inchikey"] = key["Inchikey"].fillna("").astype(str).str.strip()
    abbr = {r.Abbreviation: (r._2, r.Inchikey) for r in key.rename(columns={"Chemical name": "_2"}).itertuples()}
    trials["date_iso"] = trials["date"].map(parse_date)
    tested = trials[~trials["treatment"].isin(["control", "CO"])]
    for ab, df in tested.groupby("treatment"):
        name, ik = abbr[ab]
        cid, how = lookup(name, ik)
        how_counts[how] = how_counts.get(how, 0) + 1
        info = dict(abbrev=ab, chemical=name, inchikey=ik, n_trials=len(df), n_sessions=df["filename"].nunique())
        if cid is None:
            unmatched.append(info)
            continue
        rows.append(dict(
            chem_id=cid, domain=DOMAIN, tested_by="legacy import", tester_lab_id="",
            tested_at=df["date_iso"].min(), mode="legacy",
            result="", notes=(f"DART contact assay ({ab}); {len(df)} trials, dilutions "
                              f"{sorted(df['dilution'].dropna().unique().tolist())}% v/v; "
                              f"dates {df['date_iso'].min()}..{df['date_iso'].max()}; matched by {how}")))

    # --- assay kit shipment ---
    k = pd.read_csv(kit, encoding="latin-1").dropna(subset=["Chemical"])
    recipient = None
    for r in k.itertuples():
        if isinstance(r.Recipient, str) and r.Recipient.strip():
            recipient = r.Recipient.strip()
        if str(r.Chemical).strip().upper() == "DPG":
            continue  # solvent control
        cid, how = lookup(str(r.Chemical), str(r.Inchikey).strip())
        info = dict(abbrev="kit", chemical=r.Chemical, inchikey=r.Inchikey, n_trials=0, n_sessions=0)
        if cid is None:
            unmatched.append(info)
            continue
        how_counts[how] = how_counts.get(how, 0) + 1
        d = pd.to_datetime(r._1, format="%m/%d/%Y").strftime("%Y-%m-%d")
        rows.append(dict(chem_id=cid, domain=DOMAIN, tested_by="legacy import", tester_lab_id="",
                         tested_at=d, mode="legacy", result="",
                         notes=f"assay kit shipped {d} to {recipient}; {str(r.Quantity).replace(chr(65533), 'u')}; matched by {how}"))

    t = pd.DataFrame(rows)
    # same compound can appear in both sources: keep both rows (they're different events)
    t.to_csv(out / "tests_legacy.csv", index=False)
    pd.DataFrame(unmatched).to_csv(out / "tested_not_in_inventory.csv", index=False)
    print(f"tested rows: {len(t)}  distinct compounds: {t['chem_id'].nunique()}")
    print("match methods:", how_counts)
    print(f"treatments in trials: {tested['treatment'].nunique()}; not matched to inventory: {len(unmatched)}")
    for u in unmatched:
        print("  ", u["abbrev"], "|", u["chemical"], "|", u["inchikey"])
    dup = t[t.duplicated(['chem_id'], keep=False)]
    print("compounds tested in both sources:", dup["chem_id"].nunique())
    print("untested abbreviations in key:", sorted(set(abbr) - set(trials["treatment"])))
    print("CO trials:", int((trials['treatment'] == 'CO').sum()), " control trials:", int((trials['treatment'] == 'control').sum()))


if __name__ == "__main__":
    main(*sys.argv[1:5])
