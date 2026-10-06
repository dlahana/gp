"""Command line interface: labpicker {suggest,log,add,status}."""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import picker, store


def _ask(prompt, env, given):
    if given:
        return given
    if os.environ.get(env):
        return os.environ[env]
    return input(f"{prompt}: ").strip()


def _who(args):
    name = _ask("Your name", "LABPICKER_NAME", args.tester)
    lab_id = _ask("Your lab ID", "LABPICKER_LAB_ID", args.lab_id)
    if not name or not lab_id:
        sys.exit("Tester name and lab ID are required (or set LABPICKER_NAME / LABPICKER_LAB_ID).")
    return name, lab_id


def _sync(args):
    if not args.no_sync:
        store.sync_pull(args.data_dir)


def _log_rows(chem_ids, name, lab_id, mode, result="", notes=""):
    now = store.now_iso()
    return [dict(chem_id=c, tested_by=name, tester_lab_id=lab_id, tested_at=now,
                 mode=mode, result=result, notes=notes) for c in chem_ids]


def _record(args, chem_ids, name, lab_id, mode, result="", notes=""):
    path = store.write_test_session(_log_rows(chem_ids, name, lab_id, mode, result, notes),
                                    name, args.data_dir)
    print(f"Logged {len(chem_ids)} test(s) -> {path.name}")
    if not args.no_sync:
        store.commit_and_push([path], f"Log {len(chem_ids)} test(s) by {name} ({lab_id})",
                              name, lab_id, args.data_dir)
        print("Pushed to shared repo.")


def _parse_picks(text, ids):
    text = text.strip().lower()
    if text in ("", "none"):
        return []
    if text == "all":
        return list(ids)
    picks = []
    for tok in text.replace(",", " ").split():
        if not (tok.isdigit() and 1 <= int(tok) <= len(ids)):
            raise ValueError(f"'{tok}' is not a number from 1 to {len(ids)}")
        picks.append(ids[int(tok) - 1])
    return picks


def cmd_suggest(args):
    _sync(args)
    chems, tests = store.load_chemicals(args.data_dir), store.load_tests(args.data_dir)
    excl = args.exclude_sensitive
    mode_default = picker.DEFAULT_EXCLUDE_SENSITIVE[args.mode]
    sugg = picker.suggest(chems, tests, args.mode, args.n, excl, args.seed)
    if sugg.empty:
        sys.exit("No eligible compounds to suggest.")
    shown_excl = mode_default if excl is None else excl
    print(f"\n{args.mode} mode, sensitive compounds "
          f"{'excluded' if shown_excl else 'allowed'}; {len(sugg)} suggestion(s):\n")
    cols = ["chem_id", "name", "sensitive"] + (["score"] if "score" in sugg else [])
    table = sugg[cols].reset_index(drop=True)
    table.index += 1
    print(table.to_string())
    if args.dry_run:
        return
    ids = sugg["chem_id"].tolist()
    print("\nWhich did you actually go ahead with?")
    print("  numbers like '1 3 4', 'all', or just press Enter to log nothing now")
    print("  (you can log later with: labpicker log C00001 C00002 ...)")
    while True:
        try:
            picks = _parse_picks(input("> "), ids)
            break
        except ValueError as e:
            print(e)
    if not picks:
        print("Nothing logged.")
        return
    name, lab_id = _who(args)
    _record(args, picks, name, lab_id, args.mode)


def cmd_log(args):
    _sync(args)
    chems = store.load_chemicals(args.data_dir)
    unknown = sorted(set(args.ids) - set(chems["chem_id"]))
    if unknown:
        sys.exit(f"Unknown chem_id(s): {', '.join(unknown)}")
    tests = store.load_tests(args.data_dir)
    dup = sorted(set(args.ids) & set(tests["chem_id"]))
    if dup and not args.force:
        sys.exit(f"Already logged as tested: {', '.join(dup)} (use --force to log again)")
    name, lab_id = _who(args)
    _record(args, args.ids, name, lab_id, args.mode, args.result, args.notes)


def cmd_add(args):
    from .picker import fingerprint
    _sync(args)
    chems = store.load_chemicals(args.data_dir)
    if args.csv:
        new = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
        new = new.reindex(columns=store.CHEM_COLUMNS, fill_value="")
        new["sensitive"] = store._to_bool(new["sensitive"], False)
        new["available"] = store._to_bool(new["available"], True)
    else:
        if not (args.name and args.smiles):
            sys.exit("Provide --name and --smiles, or --csv FILE.")
        new = pd.DataFrame([dict(chem_id="", name=args.name, smiles=args.smiles,
                                 sensitive=args.sensitive, sensitivity_note=args.note,
                                 available=True)])
    bad = [r["name"] or r["smiles"] for _, r in new.iterrows() if fingerprint(r["smiles"]) is None]
    if bad:
        sys.exit(f"Invalid SMILES for: {', '.join(bad)}")
    existing = set(chems["smiles"])
    dups = new[new["smiles"].isin(existing)]
    if len(dups):
        print(f"Skipping {len(dups)} already in master list: {', '.join(dups['name'])}")
    new = new[~new["smiles"].isin(existing)].drop_duplicates("smiles").copy()
    if new.empty:
        return
    for i in new.index:
        if not new.at[i, "chem_id"]:
            new.at[i, "chem_id"] = store.next_chem_id(chems)
        chems = pd.concat([chems, new.loc[[i]]], ignore_index=True)
    if chems["chem_id"].duplicated().any():
        sys.exit("Duplicate chem_id in input.")
    path = store.save_chemicals(chems, args.data_dir)
    print(f"Added {len(new)} chemical(s).")
    if not args.no_sync:
        name, lab_id = _who(args)
        store.commit_and_push([path], f"Add {len(new)} chemical(s) ({name})", name, lab_id, args.data_dir)
        print("Pushed to shared repo.")


def cmd_status(args):
    _sync(args)
    chems, tests = store.load_chemicals(args.data_dir), store.load_tests(args.data_dir)
    tested = set(tests["chem_id"])
    left = chems[~chems["chem_id"].isin(tested)]
    print(f"Master list: {len(chems)} chemicals "
          f"({int(chems['sensitive'].sum())} sensitive, {int((~chems['available']).sum())} unavailable)")
    print(f"Tested: {len(tested)}   Untested: {len(left)} "
          f"({int((~left['sensitive']).sum())} not sensitive)")
    if len(tests):
        print("\nTests by person:")
        print(tests.groupby(["tested_by", "tester_lab_id"]).size().to_string())


def build_parser():
    p = argparse.ArgumentParser(prog="labpicker", description=__doc__)
    p.add_argument("--data-dir", type=Path, default=store.DEFAULT_DATA_DIR)
    p.add_argument("--no-sync", action="store_true", help="skip git pull/push (offline use)")
    p.add_argument("--tester", help="your name (or env LABPICKER_NAME)")
    p.add_argument("--lab-id", help="your lab ID (or env LABPICKER_LAB_ID)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suggest", help="suggest compounds to test, then log what you did")
    s.add_argument("--mode", choices=picker.MODES, required=True)
    s.add_argument("-n", type=int, default=10, help="how many to suggest")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--exclude-sensitive", dest="exclude_sensitive", action="store_true", default=None,
                   help="skip refrigerated/sensitive compounds (default in exploit mode)")
    g.add_argument("--include-sensitive", dest="exclude_sensitive", action="store_false",
                   help="allow sensitive compounds (default in explore mode)")
    s.add_argument("--seed", type=int, default=0, help="only used to seed the first explore pick")
    s.add_argument("--dry-run", action="store_true", help="show suggestions without logging")
    s.set_defaults(func=cmd_suggest)

    l = sub.add_parser("log", help="log compounds you tested")
    l.add_argument("ids", nargs="+", metavar="CHEM_ID")
    l.add_argument("--mode", default="manual")
    l.add_argument("--result", default="")
    l.add_argument("--notes", default="")
    l.add_argument("--force", action="store_true", help="log even if already tested")
    l.set_defaults(func=cmd_log)

    a = sub.add_parser("add", help="add chemicals to the master list")
    a.add_argument("--name")
    a.add_argument("--smiles")
    a.add_argument("--sensitive", action="store_true", help="needs refrigeration / special handling")
    a.add_argument("--note", default="", help="why it's sensitive")
    a.add_argument("--csv", type=Path, help="bulk add from a CSV (columns: name, smiles, ...)")
    a.set_defaults(func=cmd_add)

    st = sub.add_parser("status", help="summary of master list and tests")
    st.set_defaults(func=cmd_status)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (NotImplementedError, RuntimeError) as e:
        sys.exit(str(e))
