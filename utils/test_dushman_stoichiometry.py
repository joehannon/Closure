"""
Verify that the mole-balance (stoichiometric) coefficients for Dushman R2 are
parsed correctly and applied correctly in the ODE right-hand side.

Expected for R2: 5 Iodide + Iodate + 6 Hplus -> 3 Iodine + 3 Water
  nu_reactants  :  Iodide=5, Iodate=1, Hplus=6,  all others 0
  nu_products   :  Iodine=3, Water=3,             all others 0
  nu_net        :  Iodide=-5, Iodate=-1, Hplus=-6, Iodine=+3, Water=+3

ODE check: if rate(R2) = r, then dy/dt = nu_net @ [0, r], which must give
  d[Iodide]/dt = -5r,  d[Iodate]/dt = -1r,  d[Hplus]/dt = -6r
  d[Iodine]/dt = +3r,  d[Water]/dt  = +3r
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from species_limits import parse_reactions
import numpy as np

SPECIES = ["Borate", "Hplus", "Boric", "Iodide", "Iodate", "Iodine", "Water", "Tri_iodide"]
REACTIONS = [
    "R1: Borate + Hplus -> Boric",
    "R2: 5 Iodide + Iodate + 6 Hplus -> 3 Iodine + 3 Water",
]

nu_r, nu_p = parse_reactions(REACTIONS, SPECIES)
nu_net = nu_p - nu_r          # positive = produced, negative = consumed

# ── Print full matrices ───────────────────────────────────────────────────────
col_w = 10
print("=" * 70)
print("Dushman stoichiometry check")
print("=" * 70)
header = f"{'Species':<14}" + "".join(f"{'nu_r['+r+']':>{col_w}}" for r in ['R1','R2']) \
       + "".join(f"{'nu_p['+r+']':>{col_w}}" for r in ['R1','R2']) \
       + "".join(f"{'nu_net['+r+']':>{col_w+2}}" for r in ['R1','R2'])
print(header)
print("-" * len(header))
for i, sp in enumerate(SPECIES):
    row = f"{sp:<14}"
    for j in range(2): row += f"{nu_r[i,j]:>{col_w}.0f}"
    for j in range(2): row += f"{nu_p[i,j]:>{col_w}.0f}"
    for j in range(2): row += f"{nu_net[i,j]:>{col_w+2}.0f}"
    print(row)

# ── Expected values for R2 (column 1) ────────────────────────────────────────
expected_nu_net_R2 = {
    "Borate":     0,
    "Hplus":     -6,
    "Boric":      0,
    "Iodide":    -5,
    "Iodate":    -1,
    "Iodine":    +3,
    "Water":     +3,
    "Tri_iodide": 0,
}

print()
print("─" * 70)
print("Checking nu_net for R2 against expected values:")
print("─" * 70)
all_ok = True
for i, sp in enumerate(SPECIES):
    got  = int(nu_net[i, 1])
    want = expected_nu_net_R2[sp]
    ok   = got == want
    flag = "PASS ✓" if ok else f"FAIL  (got {got}, want {want})"
    print(f"  {sp:<14}  nu_net = {got:>3}   {flag}")
    if not ok:
        all_ok = False

# ── ODE step check ────────────────────────────────────────────────────────────
print()
print("─" * 70)
print("ODE step check:  dy/dt = nu_net @ [r_R1, r_R2]  with r_R2 = 1, r_R1 = 0")
print("─" * 70)
rates = np.array([0.0, 1.0])        # isolate R2 only
dydt  = nu_net @ rates

print(f"  {'Species':<14}  {'dy/dt':>8}  {'expected':>10}  {'':>8}")
for i, sp in enumerate(SPECIES):
    got  = dydt[i]
    want = float(expected_nu_net_R2[sp])
    ok   = abs(got - want) < 1e-12
    flag = "PASS ✓" if ok else f"FAIL  (got {got:.4g}, want {want:.4g})"
    print(f"  {sp:<14}  {got:>8.1f}  {want:>10.1f}  {flag}")
    if not ok:
        all_ok = False

# ── Ratio check: Hplus/Iodate and Iodide/Iodate ──────────────────────────────
print()
print("─" * 70)
print("Consumption-ratio check (per mole of Iodate consumed):")
print("─" * 70)
d_iodate  = dydt[SPECIES.index("Iodate")]
d_iodide  = dydt[SPECIES.index("Iodide")]
d_hplus   = dydt[SPECIES.index("Hplus")]
d_iodine  = dydt[SPECIES.index("Iodine")]
d_water   = dydt[SPECIES.index("Water")]

checks = [
    ("Iodide consumed / Iodate consumed", abs(d_iodide/d_iodate), 5.0),
    ("Hplus  consumed / Iodate consumed", abs(d_hplus /d_iodate), 6.0),
    ("Iodine produced / Iodate consumed", abs(d_iodine/d_iodate), 3.0),
    ("Water  produced / Iodate consumed", abs(d_water /d_iodate), 3.0),
]
for desc, got, want in checks:
    ok   = abs(got - want) < 1e-12
    flag = "PASS ✓" if ok else f"FAIL  (got {got:.6g}, want {want:.6g})"
    print(f"  {desc:<42}  {got:>6.1f}  (expected {want:.0f})  {flag}")
    if not ok:
        all_ok = False

print()
print("=" * 70)
print("Overall:", "ALL PASS ✓" if all_ok else "FAILURES DETECTED")
print("=" * 70)
sys.exit(0 if all_ok else 1)
