# Closure — mixing-limited reaction closures for two-stream reactors

Closure models for fast reactions in turbulent liquids: given a reaction network and
two feed streams, `species_limits.py` computes the mixing-limited species
concentration profile `C(f)` across mixture fraction `f`, and integrates the resulting
reaction-rate ODEs under turbulent mixing.

> **Status:** research code, still evolving — not yet published on PyPI. Install
> directly from GitHub for now.

## Installation

```bash
pip install git+https://github.com/joehannon/Closure.git
```

Requires Python 3.10+, `numpy`, `scipy`, `matplotlib`, and `scikit-sundae` (imported
as `sksundae`, used for CVODE-based ODE integration).

## Quickstart

```bash
python3 species_limits.py examples/input_one_rxn.json
```

The JSON describes species, reactions, and the two feed streams; results (plots,
enumerated reaction-subset limits, and per-species profiles) are written to
`plots/`, `lines/`, and `species/` next to the config file. See `examples/` for
ready-to-run configs spanning single-reaction, multi-reaction, and catalytic systems.

Minimal config:

```json
{
  "species": ["A", "B", "P"],
  "reactions": ["R1: A + B -> P"],
  "rate_constants": {"R1": 10},
  "stream_feeds": {
    "stream_1": [15.0, 0.0, 0.0],
    "stream_2": [0.0, 10.0, 0.0]
  },
  "mean_f": 0.33,
  "epsilon": [1.0e-6, 1.0, 1.0e6]
}
```

## How it works

There are two parts which complement each other:

In the first part, under the mixing-limited assumption, all $2^{N_R}% reaction subsets run infinitely fast relative to
turbulent mixing and the achievable concentration at each mixture fraction is bounded
by an exact, LP(linear programming)-derived reaction-subset limit. Plots of the limits are generated and subsets that produce duplicate limits are removed.

In the second part, the reaction system is solved in a plug-flow reaction zone, integrating the related ordinary differential equations (ODE). Three alternative closures build the ODE's working
profile `C_w(f)` for each species as the reactions progress:

| Method | Approach |
|---|---|
| `ray_limit` | A single, continuously-rotating "selectivity ray" complete-reaction limit (the default) |
| `blend_fs` | A weighted blend of a handful of user-chosen enumerated subset limits |
| `linear_interp` | Bracketing interpolation between the two enumerated subset limits nearest the current state |

The most general and useful of the closures is ray_limit. 

More detail and worked derivations are available in the author's [blog](https://joehannon.github.io/blog/)
and the [ChemRxiv preprint](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006522/v1).

## License

See the repository for current licensing terms.
