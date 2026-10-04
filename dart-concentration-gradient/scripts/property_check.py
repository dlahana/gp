"""Sanity check of vapor-pressure, diffusivity and activity-coefficient estimates
against literature values for the default compounds.  Prints a table.

Flag rule: any estimate/literature ratio outside [1/FLAG, FLAG] is flagged LARGE.
"""
from __future__ import annotations
import math, warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from thermo import Chemical
from thermo.group_contribution.joback import Joback
from thermo.unifac import UNIFAC, UFSG
from chemicals.acentric import LK_omega
from chemicals.vapor_pressure import Lee_Kesler, Ambrose_Walton

FLAG = 3.0
T = 298.15
MMHG = 133.322

# literature vapor pressure at 25 C [Pa], with provenance.  See notes in docs.
COMPOUNDS = {
    "DEET": dict(cas="134-62-3", smiles="CCN(CC)C(=O)c1cccc(C)c1", rho=0.998,
                 lit=0.00167 * MMHG, lit_src="ATSDR TP-185 Ch.4 (0.00167 mmHg @25C; a second entry 0.00013 mmHg also listed)"),
    "linalool": dict(cas="78-70-6", smiles="CC(C)=CCCC(C)(O)C=C", rho=0.862,
                     lit=0.10 * MMHG, lit_src="web-compiled listing 0.10 mmHg @25C; primary source not verified"),
    "citronellal": dict(cas="106-23-0", smiles="CC(C)=CCCC(C)CC=O", rho=0.851,
                        lit=0.28 * MMHG, lit_src="web-compiled listing 0.28 mmHg @25C; primary source not verified"),
    "geraniol": dict(cas="106-24-1", smiles="CC(C)=CCC/C(C)=C/CO", rho=0.889,
                     lit=None, lit_src="Landolt-Bornstein fit via thermo (342-503 K), extrapolated to 298 K"),
    "DPG": dict(cas="110-98-5", smiles="CC(O)COCC(C)O", rho=1.023,
                lit=None, lit_src="NIST WebBook Antoine via thermo (347-505 K), extrapolated to 298 K"),
}
# FSG atomic diffusion volumes (Fuller, Schettler, Giddings 1966)
V_ATOM = {"C": 15.9, "H": 2.31, "O": 6.11, "N": 4.54}
V_AIR, M_AIR = 19.7, 28.97
# hand-assigned original-UNIFAC subgroup ids (thermo UFSG numbering)
UNIFAC_GROUPS = {
    "linalool": {1: 3, 2: 2, 8: 1, 5: 1, 4: 1, 14: 1},
    "citronellal": {1: 3, 2: 3, 3: 1, 8: 1, 20: 1},
    "geraniol": {1: 3, 2: 3, 8: 2, 14: 1},
    "DPG": {1: 2, 3: 2, 2: 1, 25: 1, 14: 2},
    # DEET: aryl-C(=O)N(Et)2 has no parameterised original-UNIFAC group -> expect fallback
}


def fsg(MW: float, formula: dict, T=T, P_atm=1.0) -> float:
    vsum = sum(V_ATOM[a] * n for a, n in formula.items())
    return 1e-3 * T**1.75 * math.sqrt(1 / MW + 1 / M_AIR) / (
        P_atm * (vsum ** (1 / 3) + V_AIR ** (1 / 3)) ** 2) * 1e-4


def flag(est, lit):
    r = est / lit
    return r, ("LARGE" if (r > FLAG or r < 1 / FLAG) else "ok")


def main():
    print(f"{'compound':12s} {'lit Pa':>9s} | {'DB-Tb LK':>9s} {'DB AW':>9s} {'Joback LK':>10s} | ratio(DB-AW/lit) ratio(Joback/lit)")
    rows = {}
    for name, c in COMPOUNDS.items():
        ch = Chemical(c["cas"])
        vp = ch.VaporPressure
        lit = c["lit"]
        if lit is None:
            m = "LANDOLT" if "LANDOLT" in vp.all_methods else "ANTOINE_WEBBOOK"
            lit = vp.calculate(T, m)
        # estimate 1: database Tb/Tc/Pc (exp. critical props where known) -> corresponding states
        om = LK_omega(ch.Tb, ch.Tc, ch.Pc)
        e_lk = Lee_Kesler(T, ch.Tc, ch.Pc, om)
        e_aw = Ambrose_Walton(T, ch.Tc, ch.Pc, om)
        # estimate 2: pure structure (SMILES -> Joback Tb,Tc,Pc -> Lee-Kesler)
        j = Joback(Chem.MolFromSmiles(c["smiles"])).estimate()
        om_j = LK_omega(j["Tb"], j["Tc"], j["Pc"])
        e_jb = Lee_Kesler(T, j["Tc"], j["Pc"], om_j)
        r_aw, f_aw = flag(e_aw, lit)
        r_jb, f_jb = flag(e_jb, lit)
        rows[name] = dict(lit=lit, e_lk=e_lk, e_aw=e_aw, e_jb=e_jb, Tb_exp=ch.Tb, Tb_jb=j["Tb"],
                          mw=ch.MW, formula=ch.atoms)
        print(f"{name:12s} {lit:9.3g} | {e_lk:9.3g} {e_aw:9.3g} {e_jb:10.3g} | {r_aw:8.2f} {f_aw:5s}  {r_jb:8.2f} {f_jb:5s}   Tb exp/Joback {ch.Tb:.0f}/{j['Tb']:.0f} K")
        print(f"{'':12s}   lit source: {c['lit_src']}")
    print("\nDiffusivity in air, 25 C, 1 atm (FSG), and tube time scales (L = 8 in)")
    L = 8 * 0.0254
    for name, r in rows.items():
        D = fsg(r["mw"], r["formula"])
        print(f"{name:12s} MW {r['mw']:7.2f}  D = {D:.3e} m2/s   L^2/D = {L*L/D/3600:.2f} h")
    print("\nUNIFAC gamma of compound in DPG (loading A: 10 % v/v DPG, balance compound)")
    rho_dpg, mw_dpg = COMPOUNDS["DPG"]["rho"], rows["DPG"]["mw"]
    for name, c in COMPOUNDS.items():
        if name == "DPG":
            continue
        n_dpg = 0.10 * rho_dpg / mw_dpg
        n_c = 0.90 * c["rho"] / rows[name]["mw"]
        x = [n_c / (n_c + n_dpg), n_dpg / (n_c + n_dpg)]
        if name not in UNIFAC_GROUPS:
            print(f"{name:12s} x = {x[0]:.3f}  gamma = n/a (no UNIFAC groups) -> fallback gamma = 1, WARN")
            continue
        U = UNIFAC.from_subgroups(T=T, xs=x, chemgroups=[UNIFAC_GROUPS[name], UNIFAC_GROUPS["DPG"]],
                                  subgroups=UFSG, interaction_data=None, version=0) if False else \
            UNIFAC.from_subgroups(T=T, xs=x, chemgroups=[UNIFAC_GROUPS[name], UNIFAC_GROUPS["DPG"]], version=0)
        g = U.gammas()
        print(f"{name:12s} x = {x[0]:.3f}  gamma_compound = {g[0]:.3f}   gamma_DPG = {g[1]:.3f}")


if __name__ == "__main__":
    main()
