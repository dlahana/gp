"""Stage-1 demo: analytical solutions for the default tube (concentration in units of c_s)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dart_cg import Tube
from dart_cg.dimensionless import slowest_mode_time
from dart_cg.solvers.analytical import AnalyticalSolution

out = Path(sys.argv[1] if len(sys.argv) > 1 else "out"); out.mkdir(parents=True, exist_ok=True)
tube, D = Tube.default(), 5.84e-6          # linalool-like D in air (FSG estimate)
L = tube.length
x_in = np.linspace(0, L, 400) / 0.0254
cases = {"source at x=0 only (far end closed)": AnalyticalSolution.for_tube(tube, D, c_left=1.0),
         "sources at both ends": AnalyticalSolution.for_tube(tube, D, c_left=1.0, c_right=1.0)}
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
times_min = [1, 5, 15, 30, 60, 120]
for j, (name, s) in enumerate(cases.items()):
    for tm in times_min:
        ax[0, j].plot(x_in, s.concentration(x_in * 0.0254, tm * 60.0), label=f"{tm} min")
    ax[0, j].set(title=name, xlabel="x [in]", ylabel="c / c_s"); ax[0, j].legend(fontsize=8)
    t = np.linspace(0, 180 * 60, 600)
    for xp in (2, 4, 6):
        ax[1, j].plot(t / 60, s.probe_series(xp * 0.0254, t), label=f"probe x={xp} in")
    ax[1, j].axvline(slowest_mode_time(L, D, both_ends=(j == 1)) / 60, color="gray", ls=":", label="slowest-mode tau")
    ax[1, j].set(xlabel="t [min]", ylabel="c / c_s"); ax[1, j].legend(fontsize=8)
fig.suptitle(f"Analytical solution, D = {D:.2e} m$^2$/s, tube {L/0.0254:.0f} in x {tube.diameter/0.0254:.2f} in")
fig.tight_layout(); fig.savefig(out / "stage1_profiles_probe.png", dpi=110)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
s = cases["source at x=0 only (far end closed)"]
t = np.linspace(1, 180 * 60, 200)
im = ax[0].imshow(s.field(x_in * 0.0254, t), aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, L / 0.0254, 0, 180])
ax[0].set(xlabel="x [in]", ylabel="t [min]", title="space-time heatmap, c / c_s"); fig.colorbar(im, ax=ax[0])
xs = np.linspace(0, L, 201)
for tau in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    tt = tau * L**2 / D
    ax[1].semilogy(xs / 0.0254, np.abs(s.concentration(xs, tt, "series") - s.concentration(xs, tt, "images")) + 1e-18,
                   label=f"tau={tau}")
ax[1].set(xlabel="x [in]", ylabel="|series - images|", title="two representations agree (to round-off)", ylim=(1e-18, 1e-8))
ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "stage1_heatmap_agreement.png", dpi=110)
print("wrote", list(out.glob("stage1*.png")))
