#!/usr/bin/env python3
"""
One-off plots: blend_fs control-point tent profiles for (R+T) and S versus f,
for input_fuller_BC_azo_coupling_kinetics at 5 values of w = S/(R+T+S).

Produces two PNG files:
  blendfs_RT_S_tent_vs_ratio.png       — standard feed (B=1.2, C=1.2)
  blendfs_RT_S_tent_vs_ratio_noC.png   — no C in stream 2 (B=1.2, C=0)

Subsets 20={R1,R3,R5} and 31={all}, per JSON blend_subsets.
"""

import json
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import combinations

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from species_limits import (
    parse_reactions, analyze_stream_limit_system,
    generate_line_segments, _eval_segments,
)

# ── Load system ───────────────────────────────────────────────────────────────
cfg_path = HERE / 'inputs' / 'input_fuller_BC_azo_coupling_kinetics.json'
with open(cfg_path) as fh:
    cfg = json.load(fh)

species    = cfg['species']
rxns       = cfg['reactions']
Y1         = np.array(cfg['stream_feeds']['stream_1'])
blend_subs = cfg['blend_subsets']   # [20, 31]

nu_r, nu_p = parse_reactions(rxns, species)
n_rxns = len(rxns)

_enum_bf = [()] + [c for size in range(1, n_rxns + 1)
                   for c in combinations(range(n_rxns), size)]

plots_dir = HERE / 'plots'
plots_dir.mkdir(exist_ok=True)

ratios  = [0.0, 0.3, 0.6, 0.9, 1.0]
f_vals  = np.linspace(0.0, 1.0, 400)
colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(ratios)))


# ── Core routine ─────────────────────────────────────────────────────────────
def run_scenario(Y2, stem, title_note):
    """Compute CP-blend profiles, save plot and two CSVs for a given Y2."""

    # Compute 3-point limit data for each blend subset.
    bf_fs, bf_c0, bf_cfs, bf_c1 = [], [], [], []
    for subnum in blend_subs:
        rxn_idx = list(_enum_bf[subnum])
        res_s = analyze_stream_limit_system(
            species, [rxns[i] for i in rxn_idx],
            nu_r[:, rxn_idx], nu_p[:, rxn_idx], Y1, Y2)
        segs_s = generate_line_segments(species, Y1, Y2, res_s)['all_reactions']
        fsk = res_s['fs']
        bf_fs.append(fsk)
        bf_c0.append( {sp: _eval_segments(segs_s[sp], 0.0) for sp in species})
        bf_cfs.append({sp: _eval_segments(segs_s[sp], fsk)  for sp in species})
        bf_c1.append( {sp: _eval_segments(segs_s[sp], 1.0) for sp in species})
        print(f"  Subset {subnum} (fs={fsk:.6f}): "
              + "  ".join(f"{sp}={bf_cfs[-1][sp]:.5f}" for sp in ('R', 'T', 'S')))

    def cp_tent(w20, sp, f_arr=None):
        if f_arr is None:
            f_arr = f_vals
        v0  = w20 * bf_c0[0][sp]  + (1 - w20) * bf_c0[1][sp]
        v1  = w20 * bf_c1[0][sp]  + (1 - w20) * bf_c1[1][sp]
        Y1_A    = float(Y1[species.index('A')])
        Y2_B_v  = float(Y2[species.index('B')])
        r_k0    = Y1_A * bf_fs[0] / (Y2_B_v * (1 - bf_fs[0]))
        r_k1    = Y1_A * bf_fs[1] / (Y2_B_v * (1 - bf_fs[1]))
        r_blend = w20 * r_k0 + (1 - w20) * r_k1
        fsb     = Y2_B_v * r_blend / (Y1_A + Y2_B_v * r_blend)
        # vfs from stoichiometric yield: cfs_k/(1-fs_k) = mol sp per mol B consumed in subset k.
        vfs = (1 - fsb) * (w20  * bf_cfs[0][sp] / (1 - bf_fs[0])
                         + (1 - w20) * bf_cfs[1][sp] / (1 - bf_fs[1]))
        prof = np.where(
            f_arr <= fsb,
            v0  + (vfs - v0)  / fsb       * f_arr,
            vfs + (v1  - vfs) / (1 - fsb) * (f_arr - fsb),
        )
        return prof, fsb, vfs

    # Build curves for each ratio.
    curves_RT, curves_S = {}, {}
    for r in ratios:
        w20 = 1.0 - r
        pR, fsb, vR = cp_tent(w20, 'R')
        pT, _,   vT = cp_tent(w20, 'T')
        curves_RT[r] = (pR + pT, fsb, vR + vT)
        pS, fsb_S, vS = cp_tent(w20, 'S')
        curves_S[r]  = (pS, fsb_S, vS)

    # Profile CSV — each ratio's f grid includes its exact fsb kink point.
    prof_csv = plots_dir / f'{stem}_profiles.csv'
    with open(prof_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ratio_S_over_RTS', 'f', 'B_RplusT_CP', 'B_S_CP', 'is_kink'])
        for r in ratios:
            _, fsb_r, _ = curves_RT[r]
            f_grid = np.unique(np.concatenate([f_vals, [fsb_r]]))
            w20 = 1.0 - r
            pR_g, _, vR_g = cp_tent(w20, 'R', f_grid)
            pT_g, _, vT_g = cp_tent(w20, 'T', f_grid)
            pS_g, _, _    = cp_tent(w20, 'S', f_grid)
            for fi, fv in enumerate(f_grid):
                is_kink = abs(fv - fsb_r) < 1e-12
                writer.writerow([f'{r:.2f}', f'{fv:.8f}',
                                 f'{(pR_g[fi]+pT_g[fi]):.8f}',
                                 f'{pS_g[fi]:.8f}',
                                 '1' if is_kink else '0'])
    print(f"  CSV saved: {prof_csv}")

    # Segment-equations CSV.
    seg_csv = plots_dir / f'{stem}_segments.csv'
    with open(seg_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'ratio_S_over_RTS', 'fsb',
            'RT_seg1_m', 'RT_seg1_c', 'RT_seg2_m', 'RT_seg2_c',
            'S_seg1_m',  'S_seg1_c',  'S_seg2_m',  'S_seg2_c',
        ])
        for r in ratios:
            _, fsb, vfs_RT = curves_RT[r]
            _,   _, vfs_S  = curves_S[r]
            m1_RT = vfs_RT / fsb         if fsb > 1e-12 else 0.0
            m2_RT = -vfs_RT / (1 - fsb)
            m1_S  = vfs_S  / fsb         if fsb > 1e-12 else 0.0
            m2_S  = -vfs_S  / (1 - fsb)
            writer.writerow([
                f'{r:.2f}', f'{fsb:.6f}',
                f'{m1_RT:.6f}', '0', f'{m2_RT:.6f}', f'{-m2_RT:.6f}',
                f'{m1_S:.6f}',  '0', f'{m2_S:.6f}',  f'{-m2_S:.6f}',
            ])
    print(f"  CSV saved: {seg_csv}")

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
    panel_data   = [curves_RT, curves_S]
    panel_titles = ['R + T', 'S']

    for col, (data, title) in enumerate(zip(panel_data, panel_titles)):
        ax = axes[col]
        ax.axvline(bf_fs[0], color='grey',    ls=':', lw=0.9, alpha=0.55,
                   label=f'$f_{{s,20}}$ = {bf_fs[0]:.4f}')
        ax.axvline(bf_fs[1], color='dimgray', ls=':', lw=0.9, alpha=0.55,
                   label=f'$f_{{s,31}}$ = {bf_fs[1]:.4f}')
        for ri, r in enumerate(ratios):
            prof, fsb, vfs = data[r]
            lbl = f'$S/(R+T+S)$={r:.1f}  $f_{{sb}}$={fsb:.4f}  $v_{{fs}}$={vfs:.4f}'
            ax.plot(f_vals, prof, color=colors[ri], lw=2.0, label=lbl)
            ax.plot(fsb, vfs, 'o', color=colors[ri], ms=7, zorder=5,
                    markeredgecolor='k', markeredgewidth=0.6)
        ax.set_xlabel('$f$', fontsize=11)
        ax.set_ylabel('concentration  (mol/L)', fontsize=10)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.85, handlelength=2)

    fig.suptitle(
        f'blend_fs control-point tent profiles  '
        f'[subsets {blend_subs[0]}={{R1,R3,R5}} and {blend_subs[1]}={{all}}]'
        f'  —  {title_note}\n'
        r'ratio $= S/(R+T+S)$;  dot = control-point kink $(f_{sb},\,v_{fs})$',
        fontsize=10)
    fig.tight_layout()

    plot_path = plots_dir / f'{stem}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {plot_path}")
    plt.close(fig)


# ── Scenario 1: standard feed ─────────────────────────────────────────────────
print("\n=== Standard feed (B=1.2, C=1.2) ===")
Y2_std = np.array(cfg['stream_feeds']['stream_2'])
run_scenario(Y2_std, 'blendfs_RT_S_tent_vs_ratio', 'stream 2: B=1.2, C=1.2')

# ── Scenario 2: no C in stream 2 ─────────────────────────────────────────────
print("\n=== No C in stream 2 (B=1.2, C=0) ===")
Y2_noC = Y2_std.copy()
Y2_noC[species.index('C')] = 0.0
run_scenario(Y2_noC, 'blendfs_RT_S_tent_vs_ratio_noC', 'stream 2: B=1.2, C=0')
