# Problem-definition JSON options

Config file passed as `argv[1]` to `species_limits.py` (defaults to
`inputs/input_example.json`). Keys below are read in `if __name__ ==
'__main__'` (species_limits.py:7213-7429).

## Required

| Key | Type | Notes |
|---|---|---|
| `species` | `[str, ...]` | Species names, in the order used for all stream/vector indexing. |
| `reactions` | `[str, ...]` | One string per reaction, e.g. `"R1: A + B -> P"`. Optional `label:` prefix. Terms are `<coeff> <species>` or `<species>` (coeff=1), joined by `+`, split by `->`. Species names are case-insensitive. Append the keyword `elementary` to a reaction string to make its rate law order = stoichiometric coefficients (all other reactions default to order 1 per reactant unless overridden by `orders`). |
| `stream_feeds.stream_1` | `[float, ...]` | Feed concentrations for stream 1, same length/order as `species`. |
| `stream_feeds.stream_2` | `[float, ...]` | Feed concentrations for stream 2. |

## Kinetics

| Key | Type | Default | Notes |
|---|---|---|---|
| `rate_constants` | `{label: float}` | all `1000.0` (or errors out) | Rate constant per reaction label. If every reaction's constant is zero, the run aborts (nothing can react). |
| `orders` | `[null \| number \| {species: number}, ...]` | `null` per reaction | One entry per reaction, same order as `reactions`. `null` → default order (stoichiometry if `elementary`, else 1 per reactant). A number sets a single overall order; a `{species: exponent}` dict sets a per-species rate-law exponent. |

## Mixing model

| Key | Type | Default | Notes |
|---|---|---|---|
| `mean_f` | float | `0.2` | Mean mixture fraction. |
| `epsilon` | float or `[float, ...]` | `100.0` | Turbulent dissipation rate. A list runs/plots the full case at every ε value. |
| `m_lambda` | float | `0.006` | Mixing model constant λ. |
| `m_nu` | float | `1.0e-6` | Mixing model constant ν. |
| `m_Sc` | float | `4000` | Mixing model Schmidt number. |

## Weighting method (mixing-limited closure)

| Key | Type | Default | Notes |
|---|---|---|---|
| `weight_methods` | `[int, ...]` | `[2]` | Which method(s) to run: `1`=`blend_fs`, `2`=`ray_limit`, `3`=`linear_interp`. Runs each once, in the order given (dupes collapsed). |
| `initial_ray` | float | none (uses real kⱼ) | `ray_limit` only: uniform rate constant for the initial mass-action-rate fallback ray. |
| `blend_subsets` | `[int, ...]` | none (all eligible subsets) | `blend_fs` only: restrict which subset numbers are used for blending. Does **not** shrink the subset pool used by other methods (see `keep_subsets`/`remove_subsets` below). |

## Subset selection

| Key | Type | Default | Notes |
|---|---|---|---|
| `keep_subsets` | `[int, ...]` | none (all) | Whitelist of subset numbers to retain, applied before deduping into per-species profiles. Affects **all** methods. |
| `remove_subsets` (alias `discard_subsets`) | `[int, ...]` | none | Blacklist of subset numbers to drop. Affects **all** methods. |

## ODE solver

| Key | Type | Default | Notes |
|---|---|---|---|
| `rtol` | float | solver default | ODE relative tolerance. |
| `atol` | float | solver default | ODE absolute tolerance. |

## CLI flags (not JSON, passed on the command line)

`--sweep` (ε sweep of product fractions), `--movie` (ray_limit B(f) movies), `--pdf` (also save vector PDFs), `--heatmaps-only` (skip everything except Cf/Bf heatmaps).
