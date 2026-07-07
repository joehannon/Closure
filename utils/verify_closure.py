"""Check stoichiometric closure of the blended limit at fs.

At the final ODE snapshot we take the blended LIMIT B (its value at the kink fs is
vfs, with the no-reaction mixing line M(fs) as the unreacted reference) and ask:
is the reaction-induced change  Δ = B(fs) − M(fs)  reachable by non-negative
reaction extents, i.e. does there exist ξ ≥ 0 with N·ξ = Δ?  We report the
least-squares residual plus the concrete element checks for this network.
"""
import json
import pathlib
import tempfile
import numpy as np

import species_limits as sl

cfg = json.load(open('inputs/input_2025_paper_kinetics.json'))
species = cfg['species']
rxns = cfg['reactions']
Y1 = np.array(cfg['stream_feeds']['stream_1'], float)
Y2 = np.array(cfg['stream_feeds']['stream_2'], float)
rate_constants = cfg.get('rate_constants', {})
mean_f = cfg.get('mean_f', 0.2)

nu_r, nu_p = sl.parse_reactions(rxns, species)
N = (nu_p - nu_r).astype(float)                 # net stoichiometry (n_sp, n_rxn)

res = sl.analyze_stream_limit_system(species, rxns, nu_r, nu_p, Y1, Y2)
seg = sl.generate_line_segments(species, Y1, Y2, res)

with tempfile.TemporaryDirectory() as d:
    lines = sl.export_all_subsets_json(species, rxns, nu_r, nu_p, Y1, Y2,
                                       pathlib.Path(d) / 'lines.json')
    lines_keep = sl.filter_subsets_by_num(lines, keep=cfg.get('keep_subsets'),
                                          discard=cfg.get('discard_subsets'))
    usp = sl.export_unique_species_json(lines_keep, pathlib.Path(d) / 'sp.json')

idx = {s: i for i, s in enumerate(species)}


def check(method, key):
    out = sl.integrate_species_odes(usp, Y1, Y2, rate_constants=rate_constants,
                                    mean_f=mean_f, weight_method=method)
    bd = out.get(key)
    if bd is None:
        print(f"\n=== {method}: no {key} data ===")
        return
    t_last = -1
    fsb = float(bd['fsb'][t_last])
    # Beta-averaged blended limit E[B] vs the no-reaction beta-average E[M] = y0.
    bavg = out['blend_avgs']
    y0 = mean_f * Y1 + (1.0 - mean_f) * Y2              # E[M]
    B = y0.copy()
    for sp in bavg:
        B[idx[sp]] = float(bavg[sp][t_last])           # E[B]_sp
    delta = B - y0                                      # reaction-induced change in E

    # Closure: solve N ξ = delta in least squares; residual measures inconsistency.
    xi, *_ = np.linalg.lstsq(N, delta, rcond=None)
    resid = N @ xi - delta

    print(f"\n=== {method}  (fs={fsb:.4f}) ===")
    print(f"  implied extents ξ (R1,R2,R3) = {np.round(xi, 5)}   (want all ≥ 0)")
    print(f"  closure residual ‖Nξ − Δ‖    = {np.linalg.norm(resid):.3e}   (want ≈ 0)")
    # Per-species: how much each species' Δ is left unexplained by N ξ.
    bad = [(species[i], delta[i], (N @ xi)[i]) for i in range(len(species))
           if abs(resid[i]) > 1e-9]
    for nm, dl, fit in bad:
        print(f"    {nm}: Δ={dl:+.5f}  but Nξ={fit:+.5f}  (unexplained {dl-fit:+.5f})")

    # Concrete element balances for this A/B/C → P/S/Q network.
    def cons(s):  # amount consumed (reactant) = M - B
        return M[idx[s]] - B[idx[s]]

    def form(s):  # amount formed (product) = B - M
        return B[idx[s]] - M[idx[s]]
    print("  element checks (consumed vs formed):")
    print(f"    C consumed = {cons('C'):.5f}   Q formed       = {form('Q'):.5f}"
          f"   mismatch {cons('C')-form('Q'):+.5f}")
    print(f"    B consumed = {cons('B'):.5f}   (P+S) formed   = {form('P')+form('S'):.5f}"
          f"   mismatch {cons('B')-(form('P')+form('S')):+.5f}")
    print(f"    A consumed = {cons('A'):.5f}   (P+2S+Q) formed= "
          f"{form('P')+2*form('S')+form('Q'):.5f}"
          f"   mismatch {cons('A')-(form('P')+2*form('S')+form('Q')):+.5f}")


check('ray_limit', 'raylimit')
check('blend_fs', 'blendfs')
