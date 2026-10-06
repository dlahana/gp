"""Clean the lab's chemical inventory into a master compound list.

Usage: python -I cleaning/clean_inventory.py INVENTORY.xlsx OUT_DIR

Reads the 'Master' sheet (the per-location sheets are subsets of it, plus
location info we use to repair missing/garbled Location/Bin cells), and writes:
  OUT_DIR/compounds_all.csv     one row per distinct compound (vendors collapsed)
  OUT_DIR/triage_flags.csv      compounds that are probably NOT for testing, with reasons

Structures: the inventory has InChIKeys but no SMILES, so SMILES come from
OPSIN (name -> structure, runs offline). A structure is only trusted if its
InChIKey matches the recorded one; otherwise the status says so.
"""
import re
import sys
import unicodedata
import warnings
from pathlib import Path

import pandas as pd
from py2opsin import py2opsin
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

SUB_SHEETS = ["Room Temp", "Fridge", "-20C Freezer", "Flammables", "Corrosive Acids", "Corrosive Bases"]


# ---------- helpers ----------

def norm_name(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).replace("\t", " ").replace("_x000D_", " ")
    return re.sub(r"\s+", " ", s).strip()


GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ω": "omega", "−": "-", "′": "'"}


def opsin_variants(name):
    """Name -> cleaner strings OPSIN has a better chance with (original first)."""
    out = [name]
    n = name
    for k, v in GREEK.items():
        n = n.replace(k, v)
    n = re.sub(r"\s*\([^)]*(contains|isomer|mixture|standard|natural|%)[^)]*\)", "", n, flags=re.I)
    n = re.sub(r",?\s*(mixture of isomers|for synthesis|anhydrous).*$", "", n, flags=re.I)
    n = re.sub(r",?\s*[>≥]?=?\s*\d+(\.\d+)?\s*%.*$", "", n).strip(" ,")
    out.append(n)
    # drop optical-rotation / D-L prefixes (stereo is checked at connectivity level)
    n2 = re.sub(r"^(\((\+|-|±|\+/-|1[RS]|[RS])\)-?\s*)+", "", n)
    n2 = re.sub(r"^(dl|d|l)-(?=[a-z])", "", n2, flags=re.I)
    n2 = re.sub(r"\ba-(?=[a-z])", "alpha-", n2)
    n2 = re.sub(r"^n-\(n-(\w+)\)", r"N-\1", n2)
    n2 = re.sub(r"^n-(?=\w+amide)", "N-", n2)
    out.append(n2)
    return [x for x in dict.fromkeys(out) if x]


def storage_class(loc):
    """Map a Location/Bin string to a storage class."""
    if not isinstance(loc, str) or not loc.strip():
        return "unknown"
    l = loc.lower()
    if l.startswith("fridge"):
        return "fridge"
    if l.startswith("-20c") or "freezer" in l:
        return "freezer"
    if l.startswith("flammable"):
        return "flammable"
    if l.startswith("corrosive"):
        return "corrosive"
    if l.startswith("room"):
        return "room_temp"
    return "unknown"


SENSITIVE_CLASSES = {"fridge", "freezer", "flammable"}


def inchikey_of(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


# Trivial names OPSIN can't parse. Each SMILES is only accepted if its InChIKey matches the
# inventory's recorded key, so a wrong guess here is rejected rather than trusted.
MANUAL_SMILES = {
    "(+)-nootkatone": ["CC1CC(=O)C=C2CCC(C(C)=C)CC12C"],
    "(-)-fenchone": ["CC1(C)C2CCC(C2)(C)C1=O"],
    "(1r)-(+)-a-pinene": ["CC1=CCC2CC1C2(C)C"],
    "(±)-limonene": ["CC1=CCC(CC1)C(C)=C"],
    "dipentene": ["CC1=CCC(CC1)C(C)=C"],
    "acetic acid, glacial": ["CC(=O)O"],
    "beta-cyclocitral": ["CC1=C(C=O)C(C)(C)CCC1"],
    "d-(+)-glucose anhydrous": ["OCC1OC(O)C(O)C(O)C1O"],
    "dipropylene glycol": ["CC(CO)OCC(C)O", "CC(CO)OC(C)CO", "CC(O)COC(C)CO", "CC(O)COCC(C)O"],
    "dipropylene glycol monomethyl ether (mixture of isomeres) for synthesis":
        ["COC(C)COCC(C)O", "COCC(C)OCC(C)O", "COC(C)COC(C)CO"],
    "distilled water": ["O"],
    "dl-alpha-tocopherol": ["CC1=C(C)C2=C(CCC(C)(CCCC(C)CCCC(C)CCCC(C)C)O2)C(C)=C1O"],
    "ethyl l-(-)-lactate": ["CCOC(=O)C(C)O"],
    "transfluthrin": ["CC1(C)C(C=C(Cl)Cl)C1C(=O)OCc1c(F)c(F)cc(F)c1F"],
    "β-caryophyllene": ["CC1=CCCC(=C)C2CC(C2CC1)(C)C"],
    "γ-nonalactone": ["CCCCCC1CCC(=O)O1"],
    "γ-octalactone": ["CCCCC1CCC(=O)O1"],
    "γ-undecalactone": ["CCCCCCCC1CCC(=O)O1"],
    "δ-octanolactone": ["CCCC1CCCC(=O)O1"],
}


def resolve_structures(groups):
    """Fill smiles/structure_status for each group dict (in place).

    groups: list of dicts with keys: names (list[str]), synonyms (list[str]), inchikey (str|'')
    """
    # pass 1: names; pass 2: synonyms (only for groups with a recorded InChIKey to verify against)
    def run(cands):
        uniq = sorted({c for _, c in cands})
        smi = dict(zip(uniq, py2opsin(uniq, output_format="SMILES"))) if uniq else {}
        return smi

    for pass_no in (1, 2):
        cands = []
        for i, g in enumerate(groups):
            if g.get("smiles"):
                continue
            if pass_no == 1:
                strings = [v for n in g["names"] for v in opsin_variants(n)]
            else:
                if not g["inchikey"]:
                    continue  # can't verify synonym guesses without a key
                strings = g["synonyms"][:25]
            cands += [(i, s) for s in strings if s]
        smi = run(cands)
        for i, s in cands:
            g = groups[i]
            if g.get("smiles") or not smi.get(s):
                continue
            mol = Chem.MolFromSmiles(smi[s])
            if mol is None:
                continue
            key = inchikey_of(mol)
            if not key:
                continue
            if g["inchikey"]:
                if key == g["inchikey"]:
                    g.update(smiles=Chem.MolToSmiles(mol), structure_status="verified")
                elif key[:14] == g["inchikey"][:14]:
                    g.update(smiles=Chem.MolToSmiles(mol), structure_status="verified_no_stereo")
            elif pass_no == 1:
                g.update(smiles=Chem.MolToSmiles(mol), structure_status="opsin_unverified", opsin_key=key)
    for g in groups:  # pass 3: hand-written SMILES, verified against the recorded InChIKey
        if g.get("smiles") or not g["inchikey"]:
            continue
        for n in g["names"]:
            for cand in MANUAL_SMILES.get(n.lower(), []):
                mol = Chem.MolFromSmiles(cand)
                key = inchikey_of(mol) if mol else None
                if key and key[:14] == g["inchikey"][:14]:
                    status = "verified" if key == g["inchikey"] else "verified_no_stereo"
                    g.update(smiles=Chem.MolToSmiles(mol), structure_status=status)
                    break
            if g.get("smiles"):
                break
    for g in groups:
        g.setdefault("smiles", "")
        g.setdefault("structure_status", "unresolved")


# ---------- triage ----------

def _key(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m)[:14]


SOLVENTS = {  # connectivity-layer InChIKey prefix -> label
    _key(s): lbl for s, lbl in [
        ("CCO", "ethanol"), ("CO", "methanol"), ("CC(C)O", "isopropanol"), ("CC(C)=O", "acetone"),
        ("CS(C)=O", "DMSO"), ("CCCCCC", "hexane"), ("CCCCCCC", "heptane"), ("C1CCOC1", "THF"),
        ("CC#N", "acetonitrile"), ("CCOCC", "diethyl ether"), ("ClCCl", "dichloromethane"),
        ("ClC(Cl)Cl", "chloroform"), ("CC(O)COC(C)CO", "DPG"), ("OCC(O)CO", "glycerol"),
        ("CC(O)CO", "propylene glycol"), ("OCCO", "ethylene glycol"), ("O", "water"),
        ("CN(C)C=O", "DMF"), ("CCOC(C)=O", "ethyl acetate"), ("CCCCO", "butanol"),
    ]
}
SUGARS_ETC = {
    _key(s): lbl for s, lbl in [
        ("OCC1OC(O)C(O)C(O)C1O", "glucose/hexopyranose"), ("OCC(O)C(O)C(O)C(O)C=O", "glucose (open)"),
        ("OCC1OC(CO)(OC2OC(CO)C(O)C(O)C2O)C(O)C1O", "sucrose"),
    ]
}
PRODUCT_WORDS = re.compile(
    r"\b(oil|essential|extract|spray|soap|agarose|surfactant|detergent|resin|polymer|buffer|"
    r"ethoxylat\w*|denatured|proof|solution|powder|gel|wax|emulsifier|concentrate)\b|"
    r"[®™]|-LQ-|\bLQ\b|\bCAS\b",
    re.I)


def triage(row):
    reasons = []
    name = row["name"]
    status = row["structure_status"]
    mol = Chem.MolFromSmiles(row["smiles"]) if row["smiles"] else None
    if PRODUCT_WORDS.search(name):
        reasons.append("product/mixture-like name")
    if mol is None:
        reasons.append("no structure (mixture/product, or name not parseable)")
    else:
        k = Chem.MolToInchiKey(mol)[:14]
        if k in SOLVENTS:
            reasons.append(f"common solvent ({SOLVENTS[k]})")
        if k in SUGARS_ETC:
            reasons.append(f"sugar ({SUGARS_ETC[k]})")
        mw = Descriptors.MolWt(mol)
        if mw > 300:
            reasons.append(f"MW {mw:.0f} > 300 (likely not volatile)")
        if len(Chem.GetMolFrags(mol)) > 1 or any(a.GetFormalCharge() for a in mol.GetAtoms()):
            reasons.append("salt / charged / multi-component")
        if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
            reasons.append("inorganic (no carbon)")
        n_oh = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))
        if n_oh >= 4:
            reasons.append(f"{n_oh} hydroxyls (sugar/polyol-like)")
        if any(a.GetSymbol() not in "C H O N S F Cl Br I P".split() for a in mol.GetAtoms()):
            reasons.append("contains unusual element")
    if status == "opsin_unverified" and not reasons:
        pass
    return "; ".join(reasons)


# ---------- main ----------

def main(inv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(inv_path)
    m = xl.parse("Master")
    m["row"] = m.index + 2  # excel row number
    m["name"] = m["Name"].map(norm_name)

    # repair from per-location sheets: names, locations
    subs = pd.concat([xl.parse(s).assign(sheet=s) for s in SUB_SHEETS]).dropna(subset=["Name"])
    sub_loc = {(str(v), str(c)): l for v, c, l in zip(subs["Vendor"], subs["Catalog Number"], subs["Location/Bin"])}
    sub_name = {(str(v), str(c)): n for v, c, n in zip(subs["Vendor"], subs["Catalog Number"], subs["Name"])}
    sub_sheet = {(str(v), str(c)): s for v, c, s in zip(subs["Vendor"], subs["Catalog Number"], subs["sheet"])}
    fixes = []
    for i, r in m.iterrows():
        k = (str(r["Vendor"]), str(r["Catalog Number"]))
        if not r["name"] and k in sub_name:
            m.at[i, "name"] = norm_name(sub_name[k])
            fixes.append(f"row {r['row']}: name filled from {sub_sheet[k]} sheet -> {m.at[i, 'name']}")
        if k in sub_loc:
            loc = sub_loc[k]
            if storage_class(r["Location/Bin"]) == "unknown" and isinstance(loc, str):
                fixes.append(f"row {r['row']}: location '{r['Location/Bin']}' -> '{loc}' (from {sub_sheet[k]} sheet)")
                m.at[i, "Location/Bin"] = loc
    m["storage_class"] = m["Location/Bin"].map(storage_class)
    m["inchikey"] = m["inchikey"].astype("string").str.strip()
    bad = ~m["inchikey"].fillna("").str.match(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$") & m["inchikey"].notna()
    m.loc[bad, "inchikey"] = pd.NA
    m["inchikey"] = m["inchikey"].fillna("")

    # group key: InChIKey if known, else normalized name
    m["gkey"] = [ik if ik else "name:" + n.lower() for ik, n in zip(m["inchikey"], m["name"])]

    groups = []
    for gk, df in m.groupby("gkey", sort=False):
        names = df["name"].value_counts()
        syns = []
        for s in df["Synonyms"].dropna().astype(str):
            syns += [norm_name(x) for x in re.split(r"[;,](?=\s)|;", s)]
        syns = list(dict.fromkeys(x for x in syns if x))
        groups.append(dict(
            gkey=gk, inchikey=df["inchikey"].iloc[0], names=list(names.index), name_counts=names,
            synonyms=sorted(syns, key=len), df=df))
    # name-only entries whose name matches a keyed entry's name (case-insensitive) are the same compound
    keyed_by_name = {}
    for g in groups:
        if g["inchikey"]:
            for n in g["names"]:
                keyed_by_name.setdefault(n.lower(), g)
    kept = []
    for g in groups:
        tgt = None if g["inchikey"] else keyed_by_name.get(g["names"][0].lower())
        if tgt is not None:
            tgt["df"] = pd.concat([tgt["df"], g["df"]])
        else:
            kept.append(g)
    groups = kept
    resolve_structures(groups)

    # second merge: unverified-OPSIN groups may collapse onto an existing InChIKey
    by_key = {g["inchikey"]: g for g in groups if g["inchikey"]}
    merged = []
    for g in groups:
        k = g.get("opsin_key")
        if k and k in by_key:
            tgt = by_key[k]
            tgt["df"] = pd.concat([tgt["df"], g["df"]])
            tgt["names"] += g["names"]
            tgt["merged_note"] = f"merged by structure with name-only entry '{g['names'][0]}'"
        else:
            merged.append(g)
    groups = merged

    rows = []
    for g in groups:
        df = g["df"]
        classes = set(df["storage_class"]) - {"unknown"} or {"unknown"}
        # a compound is 'sensitive' only if EVERY copy is stored somewhere sensitive
        sensitive = all(c in SENSITIVE_CLASSES for c in classes) if classes else False
        pick = g["name_counts"].index[0] if len(g["name_counts"]) else g["names"][0]
        locs = sorted({str(x) for x in df["Location/Bin"].dropna()})
        rows.append(dict(
            name=pick,
            other_names=" | ".join(n for n in dict.fromkeys(g["names"]) if n != pick),
            inchikey=g["inchikey"] or g.get("opsin_key", ""),
            inchikey_source="inventory" if g["inchikey"] else ("opsin" if g.get("opsin_key") else ""),
            smiles=g["smiles"], structure_status=g["structure_status"],
            sensitive=sensitive,
            sensitivity_note="; ".join(sorted(c for c in classes if c in SENSITIVE_CLASSES)) if sensitive else "",
            storage_classes="; ".join(sorted(classes)),
            locations="; ".join(locs),
            n_vendor_entries=len(df),
            vendors="; ".join(sorted({str(v) for v in df["Vendor"].dropna()})),
            catalog_numbers="; ".join(f"{v}:{c}" for v, c in zip(df["Vendor"], df["Catalog Number"])),
            synonyms="; ".join(g["synonyms"][:15]),
            source_rows=",".join(str(r) for r in df["row"]),
            note=g.get("merged_note", ""),
        ))
    res = pd.DataFrame(rows).sort_values("name", key=lambda s: s.str.lower()).reset_index(drop=True)
    res.insert(0, "chem_id", [f"C{i:05d}" for i in range(1, len(res) + 1)])
    res["triage_reasons"] = res.apply(triage, axis=1)
    res.to_csv(out / "compounds_all.csv", index=False)
    res[res["triage_reasons"] != ""].to_csv(out / "triage_flags.csv", index=False)
    (out / "repair_log.txt").write_text("\n".join(fixes) + "\n")

    print(f"rows in Master: {len(m)}  -> distinct compounds: {len(res)}")
    print(res["structure_status"].value_counts().to_string())
    print(f"flagged for triage: {(res['triage_reasons'] != '').sum()}")
    print("repairs:", *fixes, sep="\n  ")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
