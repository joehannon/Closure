"""Standalone plot: measured X vs jet Reynolds number Re_j, against two model
curves (6 mm and 1 mm jet, the latter at 0.25x epsilon).  Styled to match the
other line plots in this project (species_limits.py): colour-blind-safe
palette, no gridlines, unboxed legend, fully-filled markers.

Run with: python3 plot_X_vs_Rej.py
"""
import pathlib

import species_limits as sl

plt = sl.plt
np = sl.np

# ── Data (transcribed from the spreadsheet) ──────────────────────────────
data_Rej = np.array([
    455,
    608,
    660,
    855,
    815,
    1000,
    1000,
    1100,
    1200,
    1250,
    1250,
    1400,
    1400,
    1600,
    2000,
    2200,
    2500,
    2600,
    2900,
    3150,
    3600,
    4150,
    4500,
])
data_X = np.array([
    0.03,
    0.024,
    0.0225,
    0.02,
    0.0175,
    0.016,
    0.009,
    0.011,
    0.013,
    0.007,
    0.006,
    0.011,
    0.0065,
    0.0047,
    0.004,
    0.0035,
    0.0035,
    0.0039,
    0.0029,
    0.0031,
    0.0028,
    0.0026,
    0.00235,
]
)
_keep = data_Rej >= 455
data_Rej, data_X = data_Rej[_keep], data_X[_keep]

model_Rej = np.array([5.00E+02,7.50E+02,1.00E+03,1.50E+03,2.00E+03,2.50E+03,3.00E+03,3.50E+03,4.00E+03])
model_X_6mm = np.array([
    0.021133622,
    0.013807033,
    0.010593176,
    0.007067195,
    0.005397532,
    0.004355897,
    0.003620695,
    0.003070229,
    0.002645973,
])
model_X_1mm_025eps = np.array([
    0.012565718,
    0.007604399,
    0.005632544,
    0.003621671,
    0.002716619,
    0.002172097,
    0.001795898,
    0.001514425,
    0.001300975,
])

# Curve colour: the same colour species_limits.py assigns to Acetone when
# plotting the acetal hydrolysis case (CB_PALETTE, species-index-based).
acetal_species = ['HCl', 'NaOH', 'NaCl', 'H2O', 'DMP', 'Acetone', 'MeOH', 'HCl_star']
acetone_color = sl._sp_color('Acetone', acetal_species)

fig, ax = plt.subplots(figsize=(8, 5.5))
_ms, _lw, _leg_fs = sl._scaled_marker_lw(8, 5.5)

ax.plot(data_Rej, data_X, ls='none', marker='o', ms=_ms, color='black',
        mfc=sl._face('black'), label='data')
ax.plot(model_Rej, model_X_6mm, ls='-', lw=_lw, marker='<', ms=_ms,
        color=acetone_color, mfc=sl._face(acetone_color), label='model, λ=6mm')
ax.plot(model_Rej, model_X_1mm_025eps, ls='--', lw=_lw, marker='<', ms=_ms,
        color=acetone_color, mfc=sl._face(acetone_color), label='model, λ=1mm, 0.25ε')

ax.set_xlabel('$Re_j$', fontsize=11)
ax.set_ylabel('$X_{Acetone}$', fontsize=11)
ax.set_title('$X_{Acetone}$ versus jet Reynolds number', fontsize=12)
ax.tick_params(labelsize=10)
ax.legend(loc='upper right', bbox_to_anchor=(0.96, 0.96), frameon=False, fontsize=_leg_fs + 2)
fig.tight_layout()

save_stem = pathlib.Path(__file__).parent / 'x_vs_rej'
sl._SAVE_PDF = True  # also save a vector .pdf alongside the .png
sl._save_fig(fig, str(save_stem), 'X_vs_Rej_comparison.png')
