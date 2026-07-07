"""Diagnostic: compare ray_limit vs blend_fs fs(t), reaction rates, trajectories."""
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

res = sl.analyze_stream_limit_system(species, rxns, nu_r, nu_p, Y1, Y2)
sl.generate_line_segments(species, Y1, Y2, res)
with tempfile.TemporaryDirectory() as d:
    lines = sl.export_all_subsets_json(species, rxns, nu_r, nu_p, Y1, Y2, pathlib.Path(d)/'l.json')
    lk = sl.filter_subsets_by_num(lines, keep=cfg.get('keep_subsets'), discard=cfg.get('discard_subsets'))
    usp = sl.export_unique_species_json(lk, pathlib.Path(d)/'s.json')

idx = {s: i for i, s in enumerate(species)}
show = ['A', 'B', 'C', 'P', 'S', 'Q']


def run(method, key):
    out = sl.integrate_species_odes(usp, Y1, Y2, rate_constants=rate_constants,
                                    mean_f=mean_f, weight_method=method)
    t = out['t']
    rates = out['rates']
    fsb = out.get(key, {}).get('fsb')
    print(f"\n===== {method} =====")
    hdr = f"{'t':>8} {'fs':>7} " + " ".join(f"r_{l.split(':')[0]:>4}" for l in rxns) \
        + "  " + " ".join(f"{s:>7}" for s in show)
    print(hdr)
    for frac in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        k = min(int(frac * (len(t) - 1)), len(t) - 1)
        fs_str = f"{fsb[k]:.4f}" if fsb is not None and np.isfinite(fsb[k]) else "  -  "
        rr = " ".join(f"{rates[k, j]:6.3f}" for j in range(len(rxns)))
        yy = " ".join(f"{out['y'][k, idx[s]]:7.4f}" for s in show)
        print(f"{t[k]:8.4f} {fs_str:>7} {rr}  {yy}")


run('ray_limit', 'raylimit')
run('blend_fs', 'blendfs')
