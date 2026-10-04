# DART concentration gradient

1D vapor-transport simulator for a diffusion tube (assay design aid + delivery features for ML).
Import name: `dart_cg`. SI internally; unit conversion at the edges (`dart_cg.units`).

## Status
- [x] Stage 1: units, geometry, dimensionless numbers, analytical solver + tests
- [ ] Stage 2: numerical solver (FV / method of lines) + validation against stage 1
- [ ] Stage 3: compound properties (literature first, estimates flagged), UNIFAC, editable table
- [ ] Stage 4: features, sweeps, CSV export
- [ ] Stage 5: Streamlit app, 2D/3D visualization, example notebook

## Quick start
```python
from dart_cg import Tube, AnalyticalSolution
tube = Tube.default()                                   # 0.75 in ID x 8 in
sol = AnalyticalSolution.for_tube(tube, 5.84e-6, c_left=1.0)   # source at x=0, far end closed
sol.at(0.1, 600.0)                                      # c at x=0.1 m, t=600 s (any point)
```
Run tests: `pip install -e .[dev] && pytest`.  Figures: `python scripts/stage1_demo.py out/`.
`scripts/property_check.py` compares estimated vs literature vapor pressures (needs `.[props]`).
