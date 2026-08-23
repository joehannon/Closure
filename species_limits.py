import re
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.special import betainc as _betainc_top
from scipy import stats as _stats

# Default turbulent dissipation rate (ε) used by the mixing model when no
# m_epsilon is supplied.  Set this in one place to change the default everywhere.
DEFAULT_M_EPSILON = 1.0E+2

# Default mixing-model constants, used when m_lambda / m_nu / m_Sc are not
# supplied.  Set these in one place to change the defaults everywhere.
DEFAULT_M_LAMBDA = 0.006
DEFAULT_M_NU = 1.0E-6
DEFAULT_M_SC = 4000

# Fixed per-method line/marker style, keyed by weight_method name (not by
# position in a list), so a given method always looks the same across every
# plot that overlays results from multiple methods.
_WM_STYLE = {
    'blend_fs':      dict(ls='-',  mk='o', mfc='none', mew=1.4),  # hollow circle,   solid
    'ray_limit':     dict(ls='-',  mk='s', mfc='none', mew=1.4),  # hollow square,   solid
    'linear_interp': dict(ls='--', mk='^', mfc='none', mew=1.4),  # hollow triangle, dashed
}
_WM_STYLE_FALLBACK = dict(ls=(0, (3, 1, 1, 1)), mk='v', mfc='none', mew=1.4)


def _wm_style(weight_method):
    """Fixed marker/linestyle for *weight_method*, same everywhere it's plotted."""
    return _WM_STYLE.get(weight_method, _WM_STYLE_FALLBACK)


# Colour-blind-safe categorical palette: 8 hues, fixed order, chosen so that
# adjacent series clear a colour-vision-deficiency separation check (not just
# ordinary vision) -- species/products/subsets are assigned a slot by a fixed
# index (never remapped by value), and CB_LINESTYLES cycles in once a series
# count exceeds 8 so identity never relies on a repeated colour alone.
CB_PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#4a3aa7', '#e34948']
CB_LINESTYLES = ['-', '--', '-.', ':']
# Marker cycled in on every line series (not just past 8) so identity never
# relies on colour/linestyle alone -- every chart of lines gets symbols too.
_SP_MARKERS = ['o', '^', 's', 'D', 'v', '<', '>', 'p', 'h', '*']

# Experiment: plot without colour at all.  When False, every series-identity
# colour (_cb_style/_sp_color/_sp_linestyle, and the small non-species colour
# cycles below) resolves to one of 8 fixed black/grey (colour, linestyle)
# pairs instead of CB_PALETTE -- revert by flipping this back to True.
_USE_COLOR = True
_MONO_STYLES = [
    ('black', '-'),
    ('grey',  '-'),
    ('black', '--'),
    ('grey',  '--'),
    ('black', ':'),
    ('grey',  ':'),
    ('black', (0, (6, 2, 2, 2))),
    ('grey',  (0, (6, 2, 2, 2))),
]


def _active_palette():
    """Flat colour cycle for non-species-indexed usages (e.g. one colour per
    weight method or per reaction index) -- CB_PALETTE when coloured, the
    black/grey cycle under the no-colour experiment."""
    return CB_PALETTE if _USE_COLOR else [c for c, _ in _MONO_STYLES]


def _cb_style(idx):
    """(colour, linestyle, marker) for series *idx*, cycling linestyle once
    colours repeat and marker on every index."""
    if _USE_COLOR:
        return (CB_PALETTE[idx % len(CB_PALETTE)],
                CB_LINESTYLES[(idx // len(CB_PALETTE)) % len(CB_LINESTYLES)],
                _SP_MARKERS[idx % len(_SP_MARKERS)])
    color, ls = _MONO_STYLES[idx % len(_MONO_STYLES)]
    return (color, ls, _SP_MARKERS[idx % len(_SP_MARKERS)])


def _scaled_marker_lw(cell_w, cell_h):
    """(marker size, line width, legend fontsize) scaled to one subplot's
    physical size in inches (*cell_w* x *cell_h*, i.e. figsize / (ncols,
    nrows)), so symbols, lines, and legend text look proportionate whether a
    figure has one large panel or a dense grid of small ones -- rather than
    a single fixed size everywhere."""
    size = min(cell_w, cell_h)
    ms = max(3.0, min(8.0, size * 1.15))
    lw = max(0.8, min(2.0, size * 0.3))
    fs = max(7.0, min(9.0, size * 1.8))
    return ms, lw, fs


_MARKER_FACE_ALPHA = 1.0


def _face(color, alpha=_MARKER_FACE_ALPHA):
    """RGBA for a semi-transparent marker face matching *color* -- the filled,
    half-alpha marker experiment (in place of the previous hollow mfc='none')."""
    import matplotlib.colors as mcolors
    return mcolors.to_rgba(color, alpha)

# CVODE (BDF) solve tolerances for the species ODEs.  Tighter than a casual
# default because ray_limit's complete-reaction limit is rebuilt from the live
# state, giving a non-smooth RHS: at loose tolerances the adaptive stepper places
# sparse points and thrashes near the limit's kinks, which shows up as a stepped
# trajectory and a noisy (slow-channel) rate — and is often *slower*.  1e-8/1e-10
# removes that while keeping the smooth methods cheap.
_ODE_RTOL = 1.0e-8
_ODE_ATOL = 1.0e-10

# A reaction line may end with the keyword `elementary` to request rate-law
# orders equal to the stoichiometric coefficients (mass-action) rather than the
# default order-1-per-reactant.  The keyword is metadata, not stoichiometry, so
# it is stripped before a line is parsed.
_ELEMENTARY_RE = re.compile(r'\s+elementary\s*$', re.IGNORECASE)


def parse_reactions(reaction_labels, species_labels):
    """Parse human-readable reaction strings into reactant/product matrices.

    The label may carry an optional `prefix:` (e.g. `R6: B + Cat -> X + Cat`),
    which is stripped. Each side of `->` is split on `+`; each tosken may be
    `<coeff> <species>` or just `<species>` (coefficient 1). Species matching
    is case-insensitive.  A trailing `elementary` keyword (a rate-law hint
    consumed by :func:`integrate_species_odes`) is ignored here.

    Returns
    -------
    nu_reactants, nu_products : ndarray of shape (n_species, n_reactions)
        Non-negative left- and right-hand-side stoichiometric coefficients.
        The net stoichiometric matrix N is then `nu_products - nu_reactants`.
    """
    n_sp = len(species_labels)
    n_rxn = len(reaction_labels)
    nu_r = np.zeros((n_sp, n_rxn))
    nu_p = np.zeros((n_sp, n_rxn))
    name_to_idx = {name.lower(): i for i, name in enumerate(species_labels)}

    token_re = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*$")

    for r, label in enumerate(reaction_labels):
        text = label.split(':', 1)[1] if ':' in label else label
        text, _ = _strip_elementary(text)
        if '->' not in text:
            raise ValueError(f"Reaction {r} '{label}' missing '->' separator")
        left, right = text.split('->', 1)
        for side_text, target in ((left, nu_r), (right, nu_p)):
            for token in side_text.split('+'):
                token = token.strip()
                if not token:
                    continue
                m = token_re.match(token)
                if not m:
                    raise ValueError(f"Could not parse term '{token}' in reaction '{label}'")
                coeff = float(m.group(1)) if m.group(1) else 1.0
                key = m.group(2).lower()
                if key not in name_to_idx:
                    raise ValueError(f"Unknown species '{m.group(2)}' in reaction '{label}'")
                target[name_to_idx[key], r] += coeff

    return nu_r, nu_p


def _strip_elementary(text):
    """Return (text without a trailing `elementary` keyword, is_elementary)."""
    m = _ELEMENTARY_RE.search(text)
    if m:
        return text[:m.start()], True
    return text, False


def analyze_stream_limit_system(species_labels, reaction_labels, nu_reactants, nu_products, Y1, Y2):
    N = nu_products - nu_reactants
    stream_labels, reaction_info = classify_reactions(N, nu_reactants, nu_products, Y1, Y2,
                                                     species_labels, reaction_labels)
    print_reaction_classification(reaction_info, stream_labels, species_labels)

    N_reduced, nu_r_reduced, nu_p_reduced, reduced_labels = eliminate_product_intermediates(
        N, nu_reactants, nu_products, reaction_labels, Y1, Y2, species_labels)
    print("\nIntermediate elimination produced reactions:")
    for idx, label in enumerate(reduced_labels):
        print(f"  [{idx}] {label}")

    # Reclassify after eliminating intermediates
    _, reduced_info = classify_reactions(N_reduced, nu_r_reduced, nu_p_reduced, Y1, Y2,
                                         species_labels, reduced_labels)
    catalytic = [info['index'] for info in reduced_info if info['is_catalytic']]
    single_stream = [info['index'] for info in reduced_info if info['kind'] in ('stream1-only', 'stream2-only')]
    cross_stream = [info['index'] for info in reduced_info if info['kind'] == 'cross-stream' and not info['is_catalytic']]
    joint_candidates = [info['index'] for info in reduced_info]

    print("\nPost-elimination reaction roles:")
    print(f"  catalytic reactions: {[reduced_labels[i] for i in catalytic]}")
    print(f"  single-stream reactions: {[reduced_labels[i] for i in single_stream]}")
    print(f"  cross-stream reactions: {[reduced_labels[i] for i in cross_stream]}")

    if len(cross_stream) == 0:
        print("\nNo cross-stream coupling; mixture-fraction fs is not limiting this system.")
        fs = None
        fs_extents = {}
    else:
        # Joint LP includes every active reaction (cross-stream, single-stream that
        # shares a reactant, and catalytic). Pure catalysts contribute an upper-bound
        # inequality instead of a stoichiometric equation so they cannot drive the
        # system below zero.
        fs, fs_extents = derive_fs_from_cross_stream_reactions(
            N_reduced, nu_r_reduced, nu_p_reduced, joint_candidates, Y1, Y2, stream_labels)
        if fs is None:
            print("\nLP produced no valid fs; falling back to per-reaction limits.")
        else:
            print(f"\nDerived mixture fraction limit fs = {fs:.6f}")
            # Recompute extents at fs using the phased model (single-stream reactions
            # resolved first, then all cross-stream reactions jointly).
            all_active_at_fs = _cascade_active_reactions(
                nu_r_reduced, nu_p_reduced, joint_candidates,
                np.asarray(Y1, dtype=float), np.asarray(Y2, dtype=float))
            fs_extents = solve_max_extents_at_f(
                N_reduced, nu_r_reduced, nu_p_reduced, all_active_at_fs, fs,
                np.asarray(Y1, dtype=float), np.asarray(Y2, dtype=float),
                stream_labels)
            if fs_extents:
                print("Joint extents at fs:")
                for r, xi in fs_extents.items():
                    print(f"  - {reduced_labels[r]}: xi = {xi:.6f}")

            residuals = _fs_residuals(
                {'fs': fs, 'fs_extents': fs_extents, 'reduced_matrix': N_reduced},
                species_labels, Y1, Y2)
            feed_scale = float(max(np.asarray(Y1, dtype=float).max(),
                                   np.asarray(Y2, dtype=float).max()))
            if residuals:
                print("\n*** WARNING: fed species consumed but not fully depleted at fs: ***")
                for name, val in residuals:
                    print(f"  {name}: {val:.6g}  (feed scale = {feed_scale:.6g})")
            else:
                print("\nDiagnostic: all consumed fed species depleted at fs (OK).")

    results = {
        'stream_labels': stream_labels,
        'reaction_info': reduced_info,
        'reduced_matrix': N_reduced,
        'reduced_nu_reactants': nu_r_reduced,
        'reduced_nu_products': nu_p_reduced,
        'reduced_labels': reduced_labels,
        'fs': fs,
        'fs_extents': fs_extents,
        'catalytic_indices': catalytic,
        'single_stream_indices': single_stream,
        'cross_stream_indices': cross_stream,
        'regime_breakpoints': None,  # populated by generate_line_segments
    }
    return results


def classify_reactions(N: np.ndarray, nu_reactants: np.ndarray, nu_products: np.ndarray,
                       Y1: np.ndarray, Y2: np.ndarray, species_labels, reaction_labels):
    stream_labels = identify_stream_feeds(Y1, Y2)
    reaction_info = []
    for r, label in enumerate(reaction_labels):
        reactants = np.where(N[:, r] < 0)[0]
        products = np.where(N[:, r] > 0)[0]
        # Use nu_reactants (gross) so catalysts count toward stream membership
        # even though their net N entry is zero.
        required = np.where(nu_reactants[:, r] > 0)[0]
        reactant_streams = set()
        for s in required:
            if stream_labels[s] == 1:
                reactant_streams.add(1)
            elif stream_labels[s] == 2:
                reactant_streams.add(2)
            elif stream_labels[s] == 12:
                reactant_streams.update([1, 2])

        if reactant_streams == {1}:
            kind = 'stream1-only'
        elif reactant_streams == {2}:
            kind = 'stream2-only'
        elif reactant_streams == {1, 2}:
            kind = 'cross-stream'
        elif len(required) == 0:
            kind = 'no-reactants'
        else:
            kind = 'unfed-intermediate'

        catalytic = catalyst_species(nu_reactants, nu_products, r).size > 0
        reaction_info.append({
            'index': r,
            'label': label,
            'reactants': reactants,
            'products': products,
            'kind': kind,
            'is_catalytic': catalytic,
            'stream_membership': sorted(list(reactant_streams)),
        })

    return stream_labels, reaction_info


def identify_stream_feeds(Y1: np.ndarray, Y2: np.ndarray) -> np.ndarray:
    stream_labels = np.zeros(Y1.shape, dtype=int)
    stream_labels[(Y1 > 0) & (Y2 == 0)] = 1
    stream_labels[(Y2 > 0) & (Y1 == 0)] = 2
    stream_labels[(Y1 > 0) & (Y2 > 0)] = 12
    return stream_labels


def catalyst_species(nu_reactants: np.ndarray, nu_products: np.ndarray, r: int):
    return np.where((nu_reactants[:, r] > 0) & (nu_products[:, r] > 0))[0]


def print_reaction_classification(reaction_info, stream_labels, species_labels):
    print("\nReaction classification summary:")
    for info in reaction_info:
        reactant_names = [species_labels[s] for s in info['reactants']]
        product_names = [species_labels[s] for s in info['products']]
        print(f" - {info['label']}: kind={info['kind']}, catalytic={info['is_catalytic']}, "
              f"reactants={reactant_names}, products={product_names}")
    print("\nSpecies stream labels:")
    _stream_text = {1: 'stream1', 2: 'stream2', 12: 'stream1+2'}
    for s, label in enumerate(stream_labels):
        print(f"  - {species_labels[s]}: {_stream_text.get(label, 'unfed')}")


def eliminate_product_intermediates(N: np.ndarray, nu_reactants: np.ndarray, nu_products: np.ndarray,
                                    reaction_labels, Y1: np.ndarray, Y2: np.ndarray,
                                    species_labels=None):
    """Combine direct product-reactant pairs when an intermediate is produced and consumed.

    A species is only treated as an intermediate when it is absent from both
    feeds. The reactant/product matrices are kept in sync: when reactions
    `r_prod` and `r_cons` are fused, both columns are summed, then the
    cancellation of the intermediate species is applied to nu_r and nu_p so
    catalyst status of the merged reaction is preserved.
    """
    N_mod = N.copy().astype(float)
    nu_r = nu_reactants.copy().astype(float)
    nu_p = nu_products.copy().astype(float)
    labels = list(reaction_labels)
    S = N_mod.shape[0]
    fed_species = (np.asarray(Y1) > 0) | (np.asarray(Y2) > 0)
    done = False

    while not done:
        done = True
        for s in range(S):
            if fed_species[s]:
                continue
            producers = np.where(N_mod[s, :] > 0)[0]
            consumers = np.where(N_mod[s, :] < 0)[0]
            if producers.size == 1 and consumers.size == 1:
                r_prod = producers[0]
                r_cons = consumers[0]
                if r_prod == r_cons:
                    continue

                # Skip elimination if r_prod is a primary reaction (all reactants fed)
                # that shares its fed-reactant stoichiometry with another primary reaction.
                # Combining r_prod+r_cons would inflate the chain's effective fed-consumption
                # weight, causing the LP to incorrectly route all flux through the chain
                # instead of splitting equally with the competitor.
                r_prod_req = np.where(nu_r[:, r_prod] > 0)[0]
                if r_prod_req.size > 0 and all(fed_species[int(s)] for s in r_prod_req):
                    fp_prod = frozenset((int(s), float(nu_r[s, r_prod])) for s in r_prod_req)
                    has_competitor = False
                    for r2 in range(N_mod.shape[1]):
                        if r2 == int(r_prod) or r2 == int(r_cons):
                            continue
                        req2 = np.where(nu_r[:, r2] > 0)[0]
                        if (req2.size > 0
                                and all(fed_species[int(s)] for s in req2)
                                and frozenset((int(s), float(nu_r[s, r2])) for s in req2) == fp_prod):
                            has_competitor = True
                            break
                    if has_competitor:
                        continue

                # Scale so the intermediate cancels exactly:
                # r_prod runs at scale_p extents, r_cons at scale_c extents.
                scale_p = float(nu_r[s, r_cons])   # units P consumed per r_cons extent
                scale_c = float(nu_p[s, r_prod])   # units P produced per r_prod extent
                combined_N = N_mod[:, r_prod] * scale_p + N_mod[:, r_cons] * scale_c
                combined_r = nu_r[:, r_prod] * scale_p + nu_r[:, r_cons] * scale_c
                combined_p = nu_p[:, r_prod] * scale_p + nu_p[:, r_cons] * scale_c
                combined_r[s] = 0.0
                combined_p[s] = 0.0

                def _fmt(c, lbl):
                    if c == 1.0:
                        return lbl
                    n = int(c) if c == int(c) else c
                    return f'{n}·{lbl}'
                combined_label = f"({_fmt(scale_p, labels[r_prod])} + {_fmt(scale_c, labels[r_cons])})"

                keep = [j for j in range(N_mod.shape[1]) if j not in {r_prod, r_cons}]
                N_mod = np.column_stack([N_mod[:, keep], combined_N])
                nu_r = np.column_stack([nu_r[:, keep], combined_r])
                nu_p = np.column_stack([nu_p[:, keep], combined_p])
                labels = [labels[j] for j in keep] + [combined_label]
                done = False
                break

    remaining = [s for s in range(S)
                 if not fed_species[s]
                 and np.any(N_mod[s, :] > 0)
                 and np.any(N_mod[s, :] < 0)]
    if remaining:
        names = ([species_labels[s] for s in remaining] if species_labels is not None
                 else [str(s) for s in remaining])
        print(f"Warning: unfed intermediates not eliminated (multiple producers or consumers): {names}")

    return N_mod, nu_r, nu_p, labels


def _fed_derived_mask(N, nu_reactants, active_indices, Y1, Y2):
    """Species that are fed directly, or produced entirely by a reaction (among
    `active_indices`) whose every reactant is itself fed(-derived) -- the
    transitive closure of "fed".

    Used to weight LP objectives/tie-breaks consistently regardless of how
    many reaction steps a species is removed from the original feeds: a
    species produced by combining two directly-fed reactants (e.g. R in
    A+B->R) represents a full unit of fed material one step removed, and a
    reaction consuming it (e.g. A+R->S) should be weighted the same as one
    consuming two directly-fed species -- not penalised for reaching its
    reactant through an intermediate.  Restricted to `active_indices` so a
    species reachable only via a reaction that's absent from the current
    subset correctly stays unmarked (it doesn't get fed within this subset)."""
    mask = (np.asarray(Y1) > 0) | (np.asarray(Y2) > 0)
    changed = True
    while changed:
        changed = False
        for r in active_indices:
            reactant_idx = np.where(nu_reactants[:, r] > 0)[0]
            if reactant_idx.size == 0 or not all(mask[s] for s in reactant_idx):
                continue
            for s in np.where(N[:, r] > 0)[0]:
                if not mask[s]:
                    mask[s] = True
                    changed = True
    return mask


def derive_fs_from_cross_stream_reactions(N, nu_reactants, nu_products, candidate_indices,
                                          Y1, Y2, stream_labels):
    """Two-step LP for fs: single-stream reactions run to completion first.

    Step 1: solve single-stream reaction extents at their respective stream
    boundaries (f=1 for stream-1-only, f=0 for stream-2-only) to get effective
    feeds Y1_eff and Y2_eff for the cross-stream balance.

    Step 2: LP over [f, xi_coupling...] maximising total extent subject to
    species non-negativity (as inequalities).  Using inequalities rather than
    equalities means reactions with multiple co-reactants from the same
    stream are handled correctly: only the most limiting co-reactant is
    binding at fs; the rest remain as slack.

    A candidate reaction is only a fs-DETERMINING "coupling" reaction if its
    NET-consumed fed reactants span both streams (real stoichiometric
    coupling).  A reaction that reaches stream-1 only through a catalyst
    (net-zero, regenerated -- e.g. DMP + H2O + HCl -> Acetone + 2MeOH + HCl)
    has no genuine mass-balance tie to the other stream and must not be
    allowed to pull fs towards wherever ITS OWN (single-stream) reactants are
    most abundant; it doesn't set fs, it just needs its catalyst present at
    whatever fs the coupling reactions determine.  Such reactions are solved
    afterwards, at the resulting mixed feed, via `_solve_extent_lp` -- which
    already blocks a reaction outright if a catalyst is entirely absent, but
    otherwise leaves its extent bounded only by its real (net-consumed)
    reactants, not by catalyst quantity.  Returned extents include both
    phases scaled to fs.
    """
    if not candidate_indices:
        return None, {}

    Y1f = np.asarray(Y1, dtype=float)
    Y2f = np.asarray(Y2, dtype=float)
    fed_mask = _fed_derived_mask(N, nu_reactants, candidate_indices, Y1f, Y2f)

    ss1, ss2, cs_cands = _split_by_stream(nu_reactants, stream_labels, candidate_indices)

    # --- Step 1: single-stream pre-solve ---
    Y1_eff, Y2_eff, xi1_max, xi2_max = _presolve_single_stream(
        N, nu_reactants, nu_products, ss1, ss2, Y1f, Y2f, fed_mask)

    if not cs_cands:
        return None, {}

    # --- Step 2: cross-stream LP with effective feeds ---
    active = _cascade_active_reactions(nu_reactants, nu_products, cs_cands, Y1_eff, Y2_eff)
    if not active:
        return None, {}

    # Split into fs-determining "coupling" reactions vs. reactions only
    # linked to the other stream via a catalyst -- see docstring.
    coupling, catalyzed_only = _split_coupling_vs_catalyzed(N, stream_labels, active, fed_mask,
                                                            nu_reactants, nu_products)

    if not coupling:
        return None, {}

    n_rxn = len(coupling)
    n_var = 1 + n_rxn

    # Build inequality constraints: Y_mix(f)[s] + N[s,r]*xi >= 0 for each
    # consumed fed species s.  Written as a <= row: -(Y1[s]-Y2[s])*f - N[s,r]*xi <= Y2[s].
    # Using inequalities (not equalities) allows excess co-reactants from the
    # same stream to remain as slack — only the limiting one binds at fs.
    # Include all species that can be driven negative — fed species define the
    # stoichiometric mixing point; unfed intermediates (e.g. R produced by R1,
    # consumed by R2) need their own rows to enforce xi_R2 <= xi_R1 when R1+R2
    # are no longer pre-combined by eliminate_product_intermediates.
    consumed_any = sorted({
        int(s) for r in coupling
        for s in np.where(N[:, r] < 0)[0]
    })
    if consumed_any:
        ca = np.array(consumed_any)
        A_ub_lp = np.zeros((len(ca), n_var))
        A_ub_lp[:, 0] = -(Y1_eff[ca] - Y2_eff[ca])
        A_ub_lp[:, 1:] = -N[np.ix_(ca, coupling)]
        b_ub_lp = Y2_eff[ca]
    else:
        A_ub_lp = b_ub_lp = None

    # Equality constraints for competing primaries in the fs LP.
    # Variables: [f, xi_coupling[0], xi_coupling[1], ...]; competing primaries forced equal.
    idx_of_fs = {r: j for j, r in enumerate(coupling)}
    A_eq_fs, b_eq_fs = [], []
    for group in _competing_primary_groups(nu_reactants, coupling, fed_mask):
        j0 = idx_of_fs[group[0]]
        for r in group[1:]:
            jk = idx_of_fs[r]
            row = np.zeros(n_var)
            row[1 + j0] = 1.0
            row[1 + jk] = -1.0
            A_eq_fs.append(row)
            b_eq_fs.append(0.0)
    A_eq_fs = np.array(A_eq_fs) if A_eq_fs else None
    b_eq_fs = np.array(b_eq_fs) if b_eq_fs else None

    # Weight each reaction by its total fed-reactant consumption per extent
    # (summed over all fed species from either stream).  This makes the objective
    # "maximise total cross-stream reactant consumed" so the LP finds the joint fs
    # where all cross-stream reactants are simultaneously exhausted, regardless of
    # which stream they come from.  With a flat -1 objective the LP is biased toward
    # reactions with low per-extent consumption of the shared stream-1 reactant.
    c = np.zeros(n_var)
    for j, r in enumerate(coupling):
        w = sum(float(nu_reactants[s, r]) for s in range(len(Y1f)) if fed_mask[s])
        c[1 + j] = -max(w, 1e-9)
    res = linprog(c,
                  A_ub=A_ub_lp, b_ub=b_ub_lp,
                  A_eq=A_eq_fs, b_eq=b_eq_fs,
                  bounds=[(0.0, 1.0)] + [(0.0, None)] * n_rxn,
                  method='highs')
    if not res.success:
        return None, {}

    fs = float(res.x[0])
    if not (0.0 <= fs <= 1.0):
        return None, {}

    extents = {int(coupling[j]): float(res.x[1 + j]) for j in range(n_rxn)}
    # Scale single-stream extents to fs
    for r, xi in xi1_max.items():
        extents[int(r)] = xi * fs
    for r, xi in xi2_max.items():
        extents[int(r)] = xi * (1.0 - fs)

    # Reactions only linked to the other stream via a catalyst: their real
    # (net-consumed) reactants are still bounded by the feed REMAINING after
    # the coupling reaction(s) have taken their share (e.g. H2O, which R1
    # produces and R2 consumes -- a genuine shared resource).  But the
    # catalyst's mere PRESENCE is checked against the raw mixing line, not
    # that post-coupling pool: a catalyst has net stoichiometry zero, so a
    # coupling reaction's real consumption of it (e.g. R1 driving leftover
    # HCl to zero for f<fs) doesn't mean it was ever truly unavailable --
    # only that its FINAL net balance ends up spent.  Since it's regenerated,
    # not consumed, by the catalyzed reaction, presence is a question of
    # whether it exists in the feed at all.
    if catalyzed_only:
        Y_mix_fs = Y1_eff * fs + Y2_eff * (1.0 - fs)
        Y_after_coupling = _apply_extents(Y_mix_fs, {r: extents[r] for r in coupling}, N)
        Y_raw_fs = Y1f * fs + Y2f * (1.0 - fs)
        xi_cat = _solve_extent_lp(N, nu_reactants, nu_products, catalyzed_only, Y_after_coupling, fed_mask,
                                  Y_presence=Y_raw_fs)
        extents.update(xi_cat)

    return fs, extents


def _split_by_stream(nu_reactants, stream_labels, candidate_indices):
    """Classify each candidate into ss1 (stream-1-only), ss2, or other.

    Uses gross reactant matrix so catalysts (nu_r>0, N=0) count as required reactants.
    Any reaction with an unfed reactant (stream_label==0, i.e. an intermediate) is
    placed in 'other' regardless of which streams its fed reactants come from, because
    it depends on an intermediate that can only be supplied by other reactions.
    """
    ss1, ss2, other = [], [], []
    for r in candidate_indices:
        required = np.where(nu_reactants[:, r] > 0)[0]
        streams = set()
        has_unfed_reactant = False
        for s in required:
            sl = int(stream_labels[s])
            if sl == 1:
                streams.add(1)
            elif sl == 2:
                streams.add(2)
            elif sl == 12:
                streams.update([1, 2])
            else:
                has_unfed_reactant = True
        if has_unfed_reactant:
            other.append(r)
        elif streams == {1}:
            ss1.append(r)
        elif streams == {2}:
            ss2.append(r)
        else:
            other.append(r)
    return ss1, ss2, other


def _split_coupling_vs_catalyzed(N, stream_labels, candidate_indices, fed_mask,
                                 nu_reactants=None, nu_products=None):
    """Split cross-stream candidates into "coupling" (net-consumed fed
    reactants span both streams -- genuine mass-balance coupling) vs.
    "catalyzed" (net-consumed fed reactants confined to ONE stream, reached
    from the other stream only via a regenerated catalyst -- net coefficient
    zero, e.g. DMP + H2O + HCl -> Acetone + 2MeOH + HCl).

    A catalyzed reaction has no real stoichiometric claim on the other
    stream: it doesn't determine fs, and its catalyst can be fully consumed
    by a coupling reaction that DOES really consume it (e.g. R1 uses up all
    available HCl for every f up to fs, since HCl is the tighter constraint
    there) -- so it must be solved AFTER the coupling reactions, against
    whatever they leave behind, not against the raw mixing line.

    Being confined to one FED stream is only "catalyzed" if the reaction
    actually HAS a genuine (net-zero) catalyst species -- `nu_reactants`/
    `nu_products`, if given, are used to check this via `catalyst_species`.
    Without them, a reaction whose only OTHER net-consumed reactant is an
    unfed intermediate (e.g. R2: A + R -> S, R produced by another reaction)
    would be wrongly excluded from the fs-determining LP: R is a real,
    irreversibly-consumed reactant, not a regenerated catalyst, so R2 still
    has a genuine (indirect) claim on the other stream and must count as
    "coupling", not "catalyzed"."""
    coupling, catalyzed = [], []
    for r in candidate_indices:
        net_streams = set()
        for s in np.where(N[:, r] < 0)[0]:
            if not fed_mask[s]:
                continue
            sl = int(stream_labels[s])
            if sl == 1:
                net_streams.add(1)
            elif sl == 2:
                net_streams.add(2)
            elif sl == 12:
                net_streams.update((1, 2))
        is_catalyzed = len(net_streams) < 2
        if is_catalyzed and nu_reactants is not None and nu_products is not None:
            is_catalyzed = catalyst_species(nu_reactants, nu_products, r).size > 0
        (catalyzed if is_catalyzed else coupling).append(r)
    return coupling, catalyzed


def _presolve_single_stream(N, nu_r, nu_p, ss1, ss2, Y1f, Y2f, fed_mask=None):
    """Run single-stream reactions to completion on their own feeds.

    Returns (Y1_eff, Y2_eff, xi1_max, xi2_max) — effective feed vectors and
    extent dicts used for scaling single-stream extents to fs later.
    """
    Y1_eff, Y2_eff, xi1, xi2 = Y1f.copy(), Y2f.copy(), {}, {}
    if ss1:
        active = _cascade_active_reactions(nu_r, nu_p, ss1, Y1f, np.zeros_like(Y2f))
        xi1 = _solve_extent_lp(N, nu_r, nu_p, active, Y1f, fed_mask)
        Y1_eff = _apply_extents(Y1_eff, xi1, N)
    if ss2:
        active = _cascade_active_reactions(nu_r, nu_p, ss2, np.zeros_like(Y1f), Y2f)
        xi2 = _solve_extent_lp(N, nu_r, nu_p, active, Y2f, fed_mask)
        Y2_eff = _apply_extents(Y2_eff, xi2, N)
    return Y1_eff, Y2_eff, xi1, xi2


def _cascade_active_reactions(nu_reactants, nu_products, candidates, Y1, Y2):
    """Return the candidate reactions whose reactants are supplied — either
    directly fed or produced by another active reaction. Uses `nu_reactants`
    so catalysts are recognised as required reactants even though their net
    column entry in N is zero."""
    available = (np.asarray(Y1) > 0) | (np.asarray(Y2) > 0)
    active = []
    active_set = set()
    changed = True
    while changed:
        changed = False
        for r in candidates:
            if r in active_set:
                continue
            reactant_idx = np.where(nu_reactants[:, r] > 0)[0]
            if reactant_idx.size == 0:
                continue
            if all(available[int(s)] for s in reactant_idx):
                active.append(r)
                active_set.add(r)
                for s in np.where(nu_products[:, r] > 0)[0]:
                    if not available[int(s)]:
                        available[int(s)] = True
                changed = True
    return active


def _solve_extent_lp(N, nu_reactants, nu_products, active_indices, Y_avail, fed_mask=None,
                     Y_presence=None):
    """LP: maximise total reaction extent, subject to species non-negativity.

    Reflects the mixing-limited assumption: all kinetics are infinitely fast so
    reactions proceed as far as stoichiometry allows.  Uses HiGHS via linprog
    so constraint boundaries are hit exactly.

    *Y_presence*, if given, is used instead of *Y_avail* for the catalyst
    presence-only gate below.  A catalyst has net stoichiometry zero -- it is
    regenerated, not consumed -- so its "availability" is a question of
    whether it exists in the feed AT ALL, not how much of it some other
    (real, net-consuming) reaction leaves behind.  A reaction's own real
    extent is still correctly bounded by *Y_avail* (e.g. a shared reactant
    genuinely produced/consumed by another reaction); only the binary
    presence check for its catalyst(s) uses *Y_presence*.  Defaults to
    *Y_avail* when not given (unchanged behaviour).
    """
    if not active_indices:
        return {}
    n_rxn = len(active_indices)
    Y = np.asarray(Y_avail, dtype=float)
    weights = np.empty(n_rxn)
    for j, r in enumerate(active_indices):
        reactant_idx = np.where(nu_reactants[:, r] > 0)[0]
        # Weight by limiting reactant availability; fall back to 1 if no reactants.
        weights[j] = float(np.min(Y[reactant_idx])) if reactant_idx.size > 0 else 1.0
    # Tiebreaker: score each reaction by min_availability × total_fed_consumption.
    # min_availability rewards reactions whose limiting reactant is most plentiful;
    # total_fed_consumption (summed over all fed species from either stream) rewards
    # reactions that use more of the available feeds per extent, keeping routing
    # consistent with the fs LP so cross-stream reactants are exhausted at fs.
    # The product handles both criteria in a single value above solver tolerance.
    mask = fed_mask if fed_mask is not None else (Y > 1e-9)
    total_consumption = (nu_reactants[:, active_indices] * mask[:, None]).sum(axis=0)
    scores = weights * total_consumption
    s_max = float(np.max(scores)) if np.any(scores > 0) else 1.0
    c = -(1.0 + 1e-6 * scores / (s_max + 1e-300))
    A_ub = (-N[:, active_indices]).astype(float)
    b_ub = np.asarray(Y_avail, dtype=float)
    # Catalyst presence-only: if absent the reaction is blocked outright.
    Y_pres = np.asarray(Y_presence, dtype=float) if Y_presence is not None else Y
    bounds = [(0.0, None)] * n_rxn
    _PRESENCE_TOL = 1e-9
    for j, r in enumerate(active_indices):
        for k in catalyst_species(nu_reactants, nu_products, r):
            if float(Y_pres[k]) < _PRESENCE_TOL:
                # pyrefly: ignore [unsupported-operation]
                bounds[j] = (0.0, 0.0)
                break
    # Equality constraints: competing primary reactions share identical fed-reactant
    # stoichiometry and must run at equal extents (equal selectivity assumption).
    idx_of = {r: j for j, r in enumerate(active_indices)}
    A_eq_rows, b_eq_vals = [], []
    if fed_mask is not None:
        for group in _competing_primary_groups(nu_reactants, active_indices, fed_mask):
            unblocked = [r for r in group if bounds[idx_of[r]] != (0.0, 0.0)]
            if len(unblocked) < 2:
                continue
            j0 = idx_of[unblocked[0]]
            for r in unblocked[1:]:
                jk = idx_of[r]
                row = np.zeros(n_rxn)
                row[j0] = 1.0
                row[jk] = -1.0
                A_eq_rows.append(row)
                b_eq_vals.append(0.0)
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        return {int(r): 0.0 for r in active_indices}
    xi = np.maximum(res.x, 0.0)
    return {int(active_indices[j]): float(xi[j]) for j in range(n_rxn)}


def _competing_primary_groups(nu_reactants, active_indices, fed_mask):
    """Return groups of reactions that are all-primary (every reactant is a fed species)
    and share identical fed-reactant stoichiometry.  Only groups with >=2 members returned.
    Competing primaries must run at equal extents (equal selectivity assumption)."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in active_indices:
        reactant_idx = np.where(nu_reactants[:, r] > 0)[0]
        if reactant_idx.size == 0:
            continue
        if not all(bool(fed_mask[int(s)]) for s in reactant_idx):
            continue
        fp = frozenset((int(s), float(nu_reactants[s, r])) for s in reactant_idx)
        groups[fp].append(int(r))
    return [sorted(g) for g in groups.values() if len(g) > 1]


def _apply_extents(Y, xi_map, N):
    if xi_map:
        cols = list(xi_map.keys())
        Y = Y + N[:, cols] @ np.array([xi_map[r] for r in cols])
    return np.maximum(Y, 0.0)


def solve_max_extents_at_f(N, nu_reactants, nu_products, active_indices, f, Y1, Y2,
                           stream_labels=None, fs=None):
    """Three-phase extent calculation at a given mixture fraction f.

    Phase 1: single-stream reactions consume their stream's feed to completion.
    Phase 2: cross-stream "coupling" reactions (genuine mass-balance ties to
    both streams) run on the remaining species.
    Phase 3: "catalyzed" cross-stream reactions (reach the other stream only
    via a regenerated catalyst, not real net consumption) have their real
    (net-consumed) reactants bounded by whatever phase 2 leaves behind (a
    genuinely shared resource, e.g. H2O produced by phase 2 and consumed
    here).

    A catalyzed reaction's catalyst presence is handled in one of two ways,
    depending on whether *fs* is given and a genuine coupling reaction (phase
    2) is ALSO active and net-consumes that same catalyst species:
      - No such competitor (or fs unknown): presence is checked against the
        raw mixing line at this f -- a catalyst's net stoichiometry is zero,
        so it's a question of whether it exists in the feed at all, not how
        much some other reaction leaves behind.  Reacts fully for any f>0
        (up to its own real reactants' limits).
      - A genuine competitor IS active (e.g. R1 also consuming HCl): the
        catalyst is merely LIMITED below fs, not absent -- present in the
        raw feed, just scarcer than the coupling reaction's demand.  The
        catalyzed reaction's extent then ramps linearly from 0 at f=0 to its
        complete-reaction value at f=fs (where the competing demand is
        exactly met), rather than being gated on/off.
    """
    if not active_indices:
        return {}
    Y1f = np.asarray(Y1, dtype=float)
    Y2f = np.asarray(Y2, dtype=float)
    fed_mask = _fed_derived_mask(N, nu_reactants, active_indices, Y1f, Y2f)
    Y_avail = Y1f * f + Y2f * (1.0 - f)
    if stream_labels is None:
        stream_labels = identify_stream_feeds(Y1f, Y2f)
    ss1, ss2, other = _split_by_stream(nu_reactants, stream_labels, active_indices)

    all_extents = {}

    if ss1 or ss2:
        xi_ss = _solve_extent_lp(N, nu_reactants, nu_products, ss1 + ss2, Y_avail, fed_mask)
        all_extents.update(xi_ss)
        Y_avail = _apply_extents(Y_avail, xi_ss, N)

    if other:
        coupling, catalyzed = _split_coupling_vs_catalyzed(N, stream_labels, other, fed_mask,
                                                           nu_reactants, nu_products)
        if coupling:
            xi_coupling = _solve_extent_lp(N, nu_reactants, nu_products, coupling, Y_avail, fed_mask)
            all_extents.update(xi_coupling)
            Y_avail = _apply_extents(Y_avail, xi_coupling, N)
        if catalyzed:
            contested, uncontested = [], []
            for r in catalyzed:
                cats = catalyst_species(nu_reactants, nu_products, r)
                has_competitor = fs is not None and coupling and any(
                    float(N[k, r2]) < -1e-9 for k in cats for r2 in coupling)
                (contested if has_competitor else uncontested).append(r)

            if uncontested:
                Y_raw = Y1f * f + Y2f * (1.0 - f)
                xi_cat = _solve_extent_lp(N, nu_reactants, nu_products, uncontested, Y_avail, fed_mask,
                                          Y_presence=Y_raw)
                all_extents.update(xi_cat)

            if contested:
                if f >= fs:
                    Y_raw = Y1f * f + Y2f * (1.0 - f)
                    xi_cat = _solve_extent_lp(N, nu_reactants, nu_products, contested, Y_avail, fed_mask,
                                              Y_presence=Y_raw)
                    all_extents.update(xi_cat)
                else:
                    # Catalyst is limited, not absent, below fs: ramp this
                    # reaction's extent linearly from 0 at f=0 to its
                    # complete-reaction value at f=fs (found by recursing to
                    # f=fs, where the competing demand is exactly met and
                    # the catalyst is no longer contested).
                    xi_at_fs = solve_max_extents_at_f(N, nu_reactants, nu_products, active_indices,
                                                      fs, Y1, Y2, stream_labels, fs=fs)
                    for r in contested:
                        all_extents[int(r)] = float(xi_at_fs.get(r, 0.0)) * (f / fs)

    return all_extents


def _fs_residuals(results, species_labels, Y1, Y2):
    """Return list of (species_name, residual_value) for fed species that are
    net-consumed by the reactions at fs but still have significant concentration.
    Returns [] when fs is None or no reactions run."""
    fs = results.get('fs')
    fs_extents = results.get('fs_extents', {})
    N_red = results.get('reduced_matrix')
    if fs is None or N_red is None or not fs_extents:
        return []
    Y1a = np.asarray(Y1, dtype=float)
    Y2a = np.asarray(Y2, dtype=float)
    fed_mask = (Y1a > 0) | (Y2a > 0)
    Y_reacted = Y1a * fs + Y2a * (1.0 - fs)
    cols = list(fs_extents.keys())
    delta = N_red[:, cols] @ np.array([fs_extents[r] for r in cols])
    Y_reacted = np.maximum(Y_reacted + delta, 0.0)
    feed_scale = float(max(Y1a.max(), Y2a.max()))
    _TOL = 1e-4
    return [
        (species_labels[s], float(Y_reacted[s]))
        for s in range(len(species_labels))
        if fed_mask[s] and delta[s] < 0 and Y_reacted[s] > _TOL * feed_scale
    ]


def generate_line_segments(species_labels, Y1, Y2, results):
    """Return piecewise-linear segment equations for each species.

    For both the no-reaction (pure mixing) and all-reactions (maximum extent)
    cases, each species concentration profile is piecewise affine in f with
    breakpoints at 0, fs (if it exists), and 1 (plus any intermediate basis-
    change points).  Each segment is described by:

        y(f) = slope·f + intercept    for f in [f_start, f_end]

    Returns
    -------
    dict with keys:
        'breakpoints'   : sorted list of f values
        'fs'            : stoichiometric mixture fraction (float or None)
        'no_reaction'   : dict mapping species name -> list of segment dicts
        'all_reactions' : dict mapping species name -> list of segment dicts

    Each segment dict contains:
        f_start, f_end          : float interval boundaries
        f_start_label, f_end_label : symbolic labels ('0', 'fs', '1', or numeric)
        slope, intercept        : float coefficients of the line y = slope·f + intercept
        equation                : formatted string
    """
    fs = results.get('fs')
    N_reduced = results['reduced_matrix']
    nu_r = results['reduced_nu_reactants']
    nu_p = results['reduced_nu_products']
    reaction_info = results['reaction_info']

    Y1a = np.asarray(Y1, dtype=float)
    Y2a = np.asarray(Y2, dtype=float)
    fed_mask = (Y1a > 0) | (Y2a > 0)
    stream_labels = results['stream_labels']

    active_indices = _cascade_active_reactions(
        nu_r, nu_p, [info['index'] for info in reaction_info], Y1a, Y2a)

    breakpoints = _parametric_breakpoints(N_reduced, nu_r, nu_p, active_indices, Y1a, Y2a, fs)
    results['regime_breakpoints'] = breakpoints
    n_pts = len(breakpoints)
    n_species = len(species_labels)
    n_rxn = N_reduced.shape[1]

    Y_no_rxn = np.zeros((n_species, n_pts))
    Y_with_rxn = np.zeros((n_species, n_pts))
    xi_at_breakpoints = np.zeros((n_rxn, n_pts))

    for i, f in enumerate(breakpoints):
        y_mix = Y1a * f + Y2a * (1.0 - f)
        Y_no_rxn[:, i] = y_mix

        xi_map = solve_max_extents_at_f(N_reduced, nu_r, nu_p, active_indices, f, Y1a, Y2a,
                                        stream_labels, fs=fs)
        xi_vec = np.zeros(n_rxn)
        for r, xi in xi_map.items():
            xi_vec[r] = xi
        xi_at_breakpoints[:, i] = xi_vec
        Y_with_rxn[:, i] = y_mix + N_reduced @ xi_vec

    label_map = {0.0: '0', 1.0: '1'}
    if fs is not None and 0.0 < fs < 1.0:
        label_map[float(fs)] = 'fs'

    def _make_segments(Y_matrix):
        segments = {}
        for s, name in enumerate(species_labels):
            segs = []
            for i in range(n_pts - 1):
                f0, f1 = breakpoints[i], breakpoints[i + 1]
                y0, y1 = float(Y_matrix[s, i]), float(Y_matrix[s, i + 1])
                df = f1 - f0
                slope = (y1 - y0) / df if df > 1e-15 else 0.0
                intercept = y0 - slope * f0
                segs.append({
                    'f_start': f0,
                    'f_end': f1,
                    'f_start_label': label_map.get(f0, f'{f0:.6f}'),
                    'f_end_label': label_map.get(f1, f'{f1:.6f}'),
                    'slope': slope,
                    'intercept': intercept,
                    'equation': _fmt_equation(slope, intercept),
                })
            segments[name] = segs
        return segments

    return {
        'breakpoints': breakpoints,
        'fs': fs,
        'no_reaction': _make_segments(Y_no_rxn),
        'all_reactions': _make_segments(Y_with_rxn),
        # Raw arrays for plotting — prefixed to distinguish from public segment data.
        '_f_grid': np.array(breakpoints),
        '_Y_no_rxn': Y_no_rxn,
        '_Y_with_rxn': Y_with_rxn,
        '_xi_at_breakpoints': xi_at_breakpoints,
        '_fed_mask': fed_mask,
    }


def _parametric_breakpoints(N, nu_r, nu_p, all_active, Y1, Y2, fs=None):
    """Return the exact f-values where the two-phase LP solution changes slope.

    The LP solution is piecewise affine in f; breakpoints occur at:
      - the endpoints 0 and 1,
      - fs (the stoichiometric mixture fraction, if known),
      - any f in (0, 1) where the effective cross-stream available feed
        b(f) = Y1_eff·f + Y2_eff·(1−f) crosses zero for a fed species.

    Solving the LP only at these points and connecting with straight lines gives
    exact — not merely approximate — concentration profiles.
    """
    Y1f = np.asarray(Y1, dtype=float)
    Y2f = np.asarray(Y2, dtype=float)
    fed_mask = (Y1f > 0) | (Y2f > 0)
    stream_labels = identify_stream_feeds(Y1f, Y2f)
    ss1, ss2, _ = _split_by_stream(nu_r, stream_labels, all_active)

    Y1_eff, Y2_eff, _, _ = _presolve_single_stream(N, nu_r, nu_p, ss1, ss2, Y1f, Y2f)

    # b(f) = Y2_eff + f*(Y1_eff - Y2_eff); zeros in (0,1) are basis-change candidates.
    breakpoints = {0.0, 1.0}
    if fs is not None and 0.0 < fs < 1.0:
        breakpoints.add(float(fs))
    b0 = Y2_eff
    db = Y1_eff - Y2_eff
    for s in np.where(fed_mask)[0]:
        if abs(db[s]) > 1e-12:
            f_zero = float(-b0[s] / db[s])
            if 1e-9 < f_zero < 1.0 - 1e-9:
                breakpoints.add(f_zero)

    # For a catalytic reaction with NO genuine coupling competitor for its
    # catalyst active in this subset, the catalyst's raw feed is untouched by
    # any other reaction, so it goes from truly absent (f=0, stream-1-only
    # catalyst; or f=1, stream-2-only) to fully present at any f just past
    # that boundary -- a real near-discontinuity, so add a small interior
    # breakpoint to capture it (e.g. subset "R2 alone": DMP reacts for any
    # f>0).  But when a genuine coupling reaction is ALSO active and net-
    # consumes that same catalyst species (e.g. R1+R2 together, HCl shared),
    # the catalyst is merely LIMITED (not absent) below fs -- present in the
    # raw feed, just scarcer than the coupling reaction's demand -- and the
    # catalyzed reaction's extent ramps linearly from the mixing line at f=0
    # to fully reacted at fs (see `_solve_extent_lp`'s Y_presence handling).
    # That's already exactly linear between the plain endpoints 0 and fs, so
    # no extra interior breakpoint should be added for it.
    _EPS = 1e-3
    cat_map = {r: catalyst_species(nu_r, nu_p, r) for r in all_active}
    has_uncontested_catalytic = False
    for r in all_active:
        cats = cat_map[r]
        if cats.size == 0:
            continue
        _CAT_TOL = 1e-9
        for k in cats:
            has_coupling_competitor = any(
                float(N[k, r2]) < -_CAT_TOL for r2 in all_active if r2 != r)
            if has_coupling_competitor:
                continue
            has_uncontested_catalytic = True
            if float(Y1_eff[k]) > _CAT_TOL and float(Y2_eff[k]) < _CAT_TOL:
                breakpoints.add(_EPS)        # step at f=0 → f=ε
            elif float(Y2_eff[k]) > _CAT_TOL and float(Y1_eff[k]) < _CAT_TOL:
                breakpoints.add(1.0 - _EPS)  # step at f=1 → f=1-ε
    if fs is not None and has_uncontested_catalytic:
        f_plus = float(fs) + _EPS
        if f_plus < 1.0 - _EPS:
            breakpoints.add(f_plus)
        f_minus = float(fs) - _EPS
        if f_minus > _EPS:
            breakpoints.add(f_minus)

    return sorted(breakpoints)


def _fmt_equation(slope, intercept):
    """Format y = slope·f + intercept with clean sign handling."""
    if abs(slope) < 1e-10 and abs(intercept) < 1e-10:
        return 'y = 0'
    parts = []
    if abs(slope) > 1e-10:
        parts.append(f'{slope:.4f}·f')
    if abs(intercept) > 1e-10:
        if parts:
            parts.append(f'{"+" if intercept >= 0 else "-"} {abs(intercept):.4f}')
        else:
            parts.append(f'{intercept:.4f}')
    return 'y = ' + ' '.join(parts)


def print_line_segments(segments, species_labels):
    """Print the piecewise-linear segment equations produced by generate_line_segments."""
    fs = segments['fs']
    breakpoints = segments['breakpoints']

    print(f"\n{'=' * 70}")
    print('Piecewise-linear segment equations')
    if fs is not None:
        print(f'  fs = {fs:.6f}')
    bp_strs = []
    for b in breakpoints:
        lbl = {0.0: '0', 1.0: '1'}.get(b)
        if fs is not None and abs(b - fs) < 1e-12:
            lbl = 'fs'
        bp_strs.append(lbl if lbl else f'{b:.6f}')
    print(f'  Breakpoints: [{", ".join(bp_strs)}]')
    print(f"{'=' * 70}")

    for case_label, case_data in [
        ('No reaction (pure mixing)', segments['no_reaction']),
        ('All reactions (maximum extent)', segments['all_reactions']),
    ]:
        print(f'\n--- {case_label} ---')
        for name in species_labels:
            segs = case_data[name]
            if all(abs(seg['slope']) < 1e-10 and abs(seg['intercept']) < 1e-10 for seg in segs):
                continue
            print(f'\n  {name}:')
            for seg in segs:
                print(f"    f ∈ [{seg['f_start_label']}, {seg['f_end_label']}]"
                      f"  ({seg['f_start']:.4f} – {seg['f_end']:.4f}):  {seg['equation']}")


def report_subset_fs_residuals(species_labels, reaction_labels, nu_reactants, nu_products, Y1, Y2):
    """Check every non-empty reaction subset for fed species that are consumed
    but not fully depleted at the computed fs, and print a summary report."""
    import contextlib, io as _io
    from itertools import combinations

    n_rxn = len(reaction_labels)
    Y1a = np.asarray(Y1, dtype=float)
    Y2a = np.asarray(Y2, dtype=float)
    feed_scale = float(max(Y1a.max(), Y2a.max()))
    hits = []

    total = 2 ** n_rxn - 1
    print(f"\nScanning {total} reaction subsets for fs residuals...", flush=True)
    for size in range(1, n_rxn + 1):
        for subset in combinations(range(n_rxn), size):
            sub_labels = [reaction_labels[i] for i in subset]
            sub_nu_r = nu_reactants[:, list(subset)]
            sub_nu_p = nu_products[:, list(subset)]
            with contextlib.redirect_stdout(_io.StringIO()):
                res = analyze_stream_limit_system(
                    species_labels, sub_labels, sub_nu_r, sub_nu_p, Y1a, Y2a)
            residuals = _fs_residuals(res, species_labels, Y1a, Y2a)
            if residuals:
                hits.append((sub_labels, res['fs'], residuals))

    print(f"\n{'=' * 70}")
    if not hits:
        print("fs residual report: no issues found across all reaction subsets.")
    else:
        print(f"fs residual report: {len(hits)} subset(s) with unconsumed fed species at fs:\n")
        for sub_labels, fs, residuals in hits:
            print(f"  Reactions: {sub_labels}")
            print(f"  fs = {fs:.6f}")
            for name, val in residuals:
                print(f"    {name}: {val:.6g}  (feed scale = {feed_scale:.6g})")
            print()
    print('=' * 70)


def export_all_subsets_json(species_labels, reaction_labels, nu_reactants, nu_products,
                             Y1, Y2, output_path):
    """Write a compact JSON of line-segment profiles for every reaction subset.

    Schema
    ------
    {
      "meta": { "species", "reactions", "stream_1_feed", "stream_2_feed" },
      "subsets": [
        {
          "reactions":     [...],   // labels of active reactions ([] = no-reaction)
          "fs":            float | null,
          "breakpoints":   [f0, f1, ...],
          "no_reaction":   { species: [[slope, intercept], ...], ... },
          "all_reactions": { species: [[slope, intercept], ...], ... }
        },
        ...
      ]
    }

    Segments are indexed 0..len(breakpoints)-2; segment i covers
    f in [breakpoints[i], breakpoints[i+1]].  All-zero profiles are omitted.
    """
    import contextlib, io, json, pathlib
    from itertools import combinations

    n_rxn = len(reaction_labels)
    Y1a = np.asarray(Y1, dtype=float)
    Y2a = np.asarray(Y2, dtype=float)

    all_subsets = [()] + [c for size in range(1, n_rxn + 1)
                          for c in combinations(range(n_rxn), size)]
    total = len(all_subsets)
    progress_step = max(1, total // 10)
    print(f"Computing line segments for {total} reaction subsets...", flush=True)

    subsets_out = []
    for done, subset in enumerate(all_subsets, 1):
        sub_labels = [reaction_labels[i] for i in subset]
        if subset:
            sub_nu_r = nu_reactants[:, list(subset)]
            sub_nu_p = nu_products[:,  list(subset)]
        else:
            sub_nu_r = np.zeros((len(species_labels), 0))
            sub_nu_p = np.zeros((len(species_labels), 0))

        with contextlib.redirect_stdout(io.StringIO()):
            res = analyze_stream_limit_system(species_labels, sub_labels,
                                              sub_nu_r, sub_nu_p, Y1a, Y2a)
            seg = generate_line_segments(species_labels, Y1a, Y2a, res)

        subsets_out.append({
            'subset_num':    done - 1,
            'reactions':     sub_labels,
            'fs':            seg['fs'],
            'breakpoints':   seg['breakpoints'],
            'no_reaction':   _nonzero_profiles(species_labels, seg['no_reaction']),
            'all_reactions': _nonzero_profiles(species_labels, seg['all_reactions']),
        })

        if done % progress_step == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    out = {
        'meta': {
            'species':        list(species_labels),
            'reactions':      list(reaction_labels),
            'stream_1_feed':  Y1a.tolist(),
            'stream_2_feed':  Y2a.tolist(),
        },
        'subsets': subsets_out,
    }
    pathlib.Path(output_path).write_text(json.dumps(out, indent=2))
    return out


def _nonzero_profiles(species_labels, case_data):
    """Return {species: [[slope, intercept], ...]} omitting all-zero profiles."""
    out = {}
    for sp in species_labels:
        pairs = _seg_to_pairs(case_data[sp])
        if any(abs(s) > 1e-10 or abs(i) > 1e-10 for s, i in pairs):
            out[sp] = pairs
    return out


def _seg_to_pairs(segs):
    return [[seg['slope'], seg['intercept']] for seg in segs]


def count_unique_profiles(subsets, species_labels):
    """Group subsets by identical all_reactions profiles and report uniqueness.

    Two subsets are equivalent when every species has the same piecewise-linear
    slope/intercept pairs (rounded to 9 decimal places to ignore solver noise).
    Works directly on the 'subsets' list from the lines_*.json schema.
    """
    from collections import defaultdict

    def _key(all_reactions):
        return tuple(
            tuple(round(v, 9) for pair in all_reactions.get(sp, []) for v in pair)
            for sp in species_labels
        )

    groups = defaultdict(list)
    for i, subset in enumerate(subsets):
        groups[_key(subset.get('all_reactions', {}))].append(i)

    n_total = len(subsets)
    n_unique = len(groups)

    def _label(idx):
        rxns = subsets[idx].get('reactions', [])
        return ', '.join(rxns) if rxns else '∅ (no reaction)'

    print(f"\n{'=' * 60}")
    print(f"Unique profile analysis  ({n_total} subsets total)")
    print(f"  Unique profiles: {n_unique}")
    print(f"  Duplicates:      {n_total - n_unique}")
    print('=' * 60)

    duplicate_groups = sorted(
        [g for g in groups.values() if len(g) > 1], key=lambda g: g[0])
    if not duplicate_groups:
        print("  All subsets produce distinct profiles.")
    else:
        print(f"\n  {len(duplicate_groups)} group(s) sharing the same profile:")
        for members in duplicate_groups:
            print(f"\n    {len(members)} subsets with identical profile:")
            for idx in members:
                print(f"      [{_label(idx)}]")

    return {'n_total': n_total, 'n_unique': n_unique, 'groups': list(groups.values())}


def filter_subsets_by_num(lines_data, keep=None, discard=None):
    """Retain only selected reaction subsets before per-species deduplication.

    Subsets are selected by their integer ``subset_num`` (as printed by
    :func:`count_unique_profiles` and stored in the ``lines_*.json`` schema).
    ``keep`` is an optional whitelist of subset_nums; ``discard`` an optional
    blacklist applied afterwards.  With both ``None`` every subset is retained
    (the default), so omitting the corresponding config fields leaves the ODE
    integration unchanged.

    Returns a new lines_data dict that shares the original 'meta' block; the
    input is not mutated.
    """
    subsets = lines_data['subsets']
    if keep is None and discard is None:
        return lines_data

    valid_nums = {s['subset_num'] for s in subsets}
    selected = subsets
    if keep is not None:
        keep_set = set(keep)
        unknown = keep_set - valid_nums
        if unknown:
            print(f"[filter] warning: keep_subsets has unknown subset_num(s): {sorted(unknown)}")
        selected = [s for s in selected if s['subset_num'] in keep_set]
    if discard is not None:
        discard_set = set(discard)
        unknown = discard_set - valid_nums
        if unknown:
            print(f"[filter] warning: discard_subsets has unknown subset_num(s): {sorted(unknown)}")
        selected = [s for s in selected if s['subset_num'] not in discard_set]

    kept = [s['subset_num'] for s in selected]
    dropped = sorted(valid_nums - set(kept))
    print(f"[filter] retaining {len(kept)}/{len(subsets)} subset(s) for ODE "
          f"integration; kept subset_num={kept}, discarded={dropped}")
    if not selected:
        print("[filter] warning: no subsets retained — unique species data will be empty.")
    return {**lines_data, 'subsets': selected}


def export_unique_species_json(lines_data, output_path):
    """Write a JSON organised by species, containing one entry per distinct profile.

    For each species, all subsets that produce identical all_reactions profiles
    (same breakpoints and slope/intercept pairs, rounded to 9 d.p.) are grouped
    together.  Each group is written once under a key formed by joining the
    subset_nums of every member with underscores.  All-zero / missing profiles
    are omitted.

    Schema
    ------
    {
      "meta": { "species", "reactions", "stream_1_feed", "stream_2_feed" },
      "species": {
        "SpeciesName": {
          "0_2_5": { "breakpoints": [...], "segments": [[slope, intercept], ...] },
          "1_3":   { "breakpoints": [...], "segments": [...] },
          ...
        },
        ...
      }
    }
    """
    import json, pathlib
    from collections import defaultdict

    meta = lines_data['meta']
    subsets = lines_data['subsets']
    species_list = meta['species']

    def _normalize(bps, raw_segs, tol=9):
        """Merge adjacent segments with identical rounded slope/intercept.

        Returns (norm_bps, norm_segs) where redundant intermediate breakpoints
        have been removed, so two profiles that are the same function but were
        evaluated at different intermediate points compare as equal.
        """
        if not raw_segs:
            return [round(bps[0], tol), round(bps[-1], tol)], []
        rounded = [(round(s[0], tol), round(s[1], tol)) for s in raw_segs]
        out_bps = [bps[0]]
        out_segs = []
        j = 0
        while j < len(rounded):
            k = j + 1
            while k < len(rounded) and rounded[k] == rounded[j]:
                k += 1
            out_segs.append(list(rounded[j]))
            out_bps.append(bps[k])
            j = k
        return [round(b, tol) for b in out_bps], out_segs

    def _profile_key(subset, sp):
        bps = subset.get('breakpoints', [0.0, 1.0])
        raw_segs = subset.get('all_reactions', {}).get(sp, [])
        norm_bps, norm_segs = _normalize(bps, raw_segs)
        return (tuple(norm_bps),
                tuple(v for pair in norm_segs for v in pair))

    species_out = {}
    for sp in species_list:
        groups = defaultdict(list)
        zero_group = []
        for subset in subsets:
            k = _profile_key(subset, sp)
            if not k[1] or all(abs(v) < 1e-10 for v in k[1]):
                zero_group.append(subset)
            else:
                groups[k].append(subset)

        if not groups:
            continue

        sp_out = {}
        for k, members in sorted(groups.items(), key=lambda kv: kv[1][0]['subset_num']):
            label = '_'.join(str(s['subset_num']) for s in members)
            n_segs = len(k[0]) - 1
            sp_out[label] = {
                'breakpoints': list(k[0]),
                'segments': [[k[1][2 * j], k[1][2 * j + 1]] for j in range(n_segs)],
            }
        if zero_group:
            label = '_'.join(str(s['subset_num']) for s in zero_group)
            sp_out[label] = {
                'breakpoints': [0.0, 1.0],
                'segments': [[0.0, 0.0]],
            }
        species_out[sp] = sp_out

    out = {'meta': meta, 'species': species_out}
    pathlib.Path(output_path).write_text(json.dumps(out, indent=2))
    return out


def plot_concentration_profiles(species_labels, results, segments, save_stem=None):
    """Plot concentration profiles using pre-computed segment data.

    Accepts the dict returned by generate_line_segments so that the solver
    is not called a second time.
    """
    fs = results.get('fs')
    n_species = len(species_labels)
    reduced_labels = results['reduced_labels']

    f_grid = segments['_f_grid']
    Y_no_rxn = segments['_Y_no_rxn']
    Y_with_rxn = segments['_Y_with_rxn']
    xi_at_bp = segments['_xi_at_breakpoints']
    fed_mask = segments['_fed_mask']

    nonnegativity_violation = np.sum(
        np.maximum(0.0, -Y_with_rxn[fed_mask, :]), axis=0)

    active = [s for s in range(n_species)
              if np.max(np.abs(Y_no_rxn[s])) > 1e-9 or np.max(np.abs(Y_with_rxn[s])) > 1e-9]

    stream_labels_arr = results['stream_labels']
    s1_active = [s for s in active if stream_labels_arr[s] in (1, 12)]
    s2_active = [s for s in active if stream_labels_arr[s] not in (1, 12)]
    color_map = {s: _sp_color(s, species_labels) for s in active}
    ls_map = {s: _sp_linestyle(s, species_labels) for s in active}

    # Detect any species whose global max dwarfs every other species sharing
    # the same axis (left: stream-2 species, right: stream-1 species on the
    # twin axis) -- e.g. a solvent fed in vast excess (water in the acetal
    # case) would otherwise crush the others -- and scale it down to ~1000.
    def _outlier_scale(group):
        gmax = {s: max(np.max(np.abs(Y_no_rxn[s])), np.max(np.abs(Y_with_rxn[s]))) for s in group}
        return _outlier_scale_divisors(gmax)

    div_of = {**_outlier_scale(s1_active), **_outlier_scale(s2_active)}
    _lbl = lambda s: f'{species_labels[s]}/{div_of[s]:g}' if s in div_of else species_labels[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].sharey(axes[1])
    _ms, _lw, _leg_fs = _scaled_marker_lw(17.0 / 3.0, 5.0)

    n_s2 = len(s2_active)
    n_s1 = len(s1_active)
    fracs_s2 = np.linspace(0.15, 0.85, max(n_s2, 1))
    fracs_s1 = np.linspace(0.15, 0.85, max(n_s1, 1))

    _mk_every = max(1, len(f_grid) // 15)
    twin0 = twin1 = None
    for ax_idx, Y_matrix in [(0, Y_no_rxn), (1, Y_with_rxn)]:
        ax = axes[ax_idx]
        for s in s2_active:
            ax.plot(f_grid, _yv(s, Y_matrix[s]), color=color_map[s], linestyle=ls_map[s],
                    marker=_sp_marker(s, species_labels), markevery=_mk_every,
                    mfc=_face(color_map[s]), ms=_ms, lw=_lw, label=_lbl(s))
        if s1_active:
            ax_r = ax.twinx()
            for s in s1_active:
                ax_r.plot(f_grid, _yv(s, Y_matrix[s]), color=color_map[s], linestyle=ls_map[s],
                          marker=_sp_marker(s, species_labels), markevery=_mk_every,
                          mfc=_face(color_map[s]), ms=_ms, lw=_lw, label=_lbl(s))
            if ax_idx == 0:
                twin0 = ax_r
            else:
                twin1 = ax_r
                ax_r.set_ylabel('Stream 1 species (right axis)', fontsize=9)
        _h, _l = ax.get_legend_handles_labels()
        if ax_idx == 1 and twin1 is not None:
            _h_r, _l_r = twin1.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        elif ax_idx == 0 and twin0 is not None:
            _h_r, _l_r = twin0.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs)
    if twin0 is not None and twin1 is not None:
        twin1.sharey(twin0)
        twin0.tick_params(axis='y', labelright=False)

    # Third panel: reaction extents ξ(f) and total non-negativity violation.
    rxn_plotted = []
    _rxn_i = 0
    for r, label in enumerate(reduced_labels):
        if np.max(np.abs(xi_at_bp[r])) > 1e-9:
            _rc, _rls, _rmk = _cb_style(_rxn_i)
            _rxn_i += 1
            line, = axes[2].plot(f_grid, xi_at_bp[r], color=_rc, linestyle=_rls,
                                  marker=_rmk, markevery=max(1, len(f_grid) // 15),
                                  mfc=_face(_rc), ms=_ms, lw=_lw, label=label)
            rxn_plotted.append((r, label, line.get_color()))
    if len(rxn_plotted) > 1:
        axes[2].legend(loc='best', frameon=False, fontsize=_leg_fs)
    ax2_right = axes[2].twinx()
    ax2_right.fill_between(f_grid, nonnegativity_violation, alpha=0.15, color='red')
    ax2_right.set_ylabel('Non-negativity violation (shaded)', color='red')
    ax2_right.tick_params(axis='y', labelcolor='red')

    axes[0].set_title('(i) No reaction (pure mixing)')
    axes[1].set_title('(ii) All reactions at maximum extent')
    axes[2].set_title('(iii) Reaction extents ξ(f)')
    for ax in axes:
        ax.set_xlabel('Mixture fraction f')
        if fs is not None and 0.0 < fs < 1.0:
            ax.axvline(fs, color='k', linestyle=':', alpha=0.5)
    axes[0].set_ylabel('Species amount')
    axes[2].set_ylabel('Extent ξ')
    fig.tight_layout()

    # Second figure: one small subplot per active species.
    n_active = len(active)
    ncols = min(4, n_active)
    nrows = (n_active + ncols - 1) // ncols
    fig2, axes2 = plt.subplots(nrows, ncols,
                               figsize=(4 * ncols, 3 * nrows),
                               squeeze=False)
    _ms2, _lw2, _leg_fs2 = _scaled_marker_lw(4.0, 3.0)
    _role0_color, _role1_color = _active_palette()[0], _active_palette()[1]
    for idx, s in enumerate(active):
        ax = axes2[idx // ncols][idx % ncols]
        _mk2 = max(1, len(f_grid) // 15)
        ax.plot(f_grid, Y_no_rxn[s], color=_role0_color, linestyle='-',
                marker=_SP_MARKERS[0], markevery=_mk2, mfc=_face(_role0_color), ms=_ms2, lw=_lw2, label='no rxn')
        ax.plot(f_grid, Y_with_rxn[s], color=_role1_color, linestyle='--',
                marker=_SP_MARKERS[1], markevery=_mk2, mfc=_face(_role1_color), ms=_ms2, lw=_lw2, label='reacted')
        ax.set_title(species_labels[s], fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        if fs is not None and 0.0 < fs < 1.0:
            ax.axvline(fs, color='k', linestyle=':', alpha=0.5)
        ax.legend(loc='best', frameon=False, fontsize=_leg_fs2)

    # Hide unused subplot cells.
    for idx in range(n_active, nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)

    fig2.suptitle('Species profiles: no reaction (blue) vs reacted (orange)', fontsize=11)
    fig2.tight_layout()

    if save_stem is not None:
        import pathlib as _pathlib
        _stem_name = _pathlib.Path(save_stem).name
        _save_fig(fig, save_stem, f'{_stem_name}_1.png')
        _save_fig(fig2, save_stem, f'{_stem_name}_2.png')
    else:
        plt.show()


def _sp_index(sp, all_species):
    """Global index of *sp* (name or int) within *all_species*, for consistent
    colour/linestyle assignment regardless of which subset is being displayed."""
    if isinstance(sp, (int, np.integer)):
        return int(sp)
    try:
        return list(all_species).index(sp)
    except ValueError:
        return abs(hash(sp))


def _sp_color(sp, all_species, palette=None):
    """Return a consistent colour for *sp* based on its position in *all_species*.

    *sp* may be a species name (str) or a global species index (int).
    *all_species* is the full ordered species list for the system.
    Using the global index ensures the same species always gets the same colour
    across every plot, regardless of which subset of species is being displayed.
    """
    idx = _sp_index(sp, all_species)
    if not _USE_COLOR:
        return _MONO_STYLES[idx % len(_MONO_STYLES)][0]
    if palette is None:
        palette = CB_PALETTE
    return palette[idx % len(palette)]


def _sp_linestyle(sp, all_species):
    """Linestyle to pair with _sp_color(sp, all_species): cycles in once the
    colour palette wraps, so identity past 8 species never relies on colour alone."""
    idx = _sp_index(sp, all_species)
    if not _USE_COLOR:
        return _MONO_STYLES[idx % len(_MONO_STYLES)][1]
    return CB_LINESTYLES[(idx // len(CB_PALETTE)) % len(CB_LINESTYLES)]


def _sp_marker(sp, all_species):
    """Marker shape to pair with _sp_color(sp, all_species): the same species
    gets the same marker in every plot, mirroring _sp_color's global index."""
    idx = _sp_index(sp, all_species)
    return _SP_MARKERS[idx % len(_SP_MARKERS)]


def _outlier_scale_divisors(peak_of, target=1000.0, ratio=10.0):
    """Species whose peak value dwarfs every other species sharing the same
    plot axis (e.g. a solvent fed in vast excess, like water in the acetal
    case) get a divisor that brings THEIR OWN peak down to about *target* --
    so e.g. a peak of 55000 gets divided by round(55000/1000) = 55, landing
    it near 1000 rather than always dividing by a flat, one-size-fits-all
    factor.  *peak_of* maps species -> its peak value on the shared axis.
    Only species whose peak exceeds *ratio* times every other species'
    trigger scaling at all -- ordinary spread between species is left alone.
    Returns {species: divisor} for the species that should be scaled."""
    divisor_of = {}
    for s, peak in peak_of.items():
        others_max = max((v for o, v in peak_of.items() if o != s), default=0.0)
        if others_max > 1e-12 and peak > ratio * others_max:
            divisor_of[s] = max(1.0, round(peak / target))
    return divisor_of


def _outlier_scale_divisors_by_stream(peak_of, is_stream1, target=1000.0, ratio=10.0):
    """Like _outlier_scale_divisors, but only compares a species against
    OTHER species fed by the same stream -- a stream-1 concentrate (e.g. a
    concentrated acid) is only an outlier relative to other stream-1 species,
    never relative to a much larger stream-2 species (e.g. a solvent) that
    may share a plot but not a comparable feed scale (and, on plots where the
    two streams are drawn on separate/twin axes, don't even share a visual
    scale at all).  *is_stream1* maps species -> True if fed by stream 1,
    False otherwise (stream-2-fed or unfed, which conventionally shares
    stream 2's scale)."""
    s1 = {s: v for s, v in peak_of.items() if is_stream1[s]}
    s2 = {s: v for s, v in peak_of.items() if not is_stream1[s]}
    return {**_outlier_scale_divisors(s1, target, ratio), **_outlier_scale_divisors(s2, target, ratio)}


def _heatmap_vmax_divisors(species_list, active, Y1, Y2):
    """Per-species colour-scale vmax for the C(f)/B(f) heatmaps, plus a /N
    divisor+label for whichever species' own feed concentration dwarfs its
    stream co-feeds (e.g. water fed in vast excess).

    Stream-1 and stream-2 species are treated as separate groups (each
    stream can have its own natural scale) and outlier-detected within their
    own group using each species' own feed concentration.  A group's
    outlier(s) get scaled down to ~1000; every other stream-1 species shares
    that stream's max feed concentration EXCLUDING any outlier, and every
    other stream-2 species (plus unfed reaction products, which have no feed
    of their own) shares stream 2's max feed concentration EXCLUDING any
    outlier -- an outlier left in that max would otherwise still dominate it
    and crush everything else back to near-floor colour."""
    stream1 = [sp for sp in active if Y1[species_list.index(sp)] > 0]
    own1 = {sp: float(Y1[species_list.index(sp)]) for sp in stream1}
    div1 = _outlier_scale_divisors(own1)

    stream2 = [sp for sp in active if sp not in stream1 and Y2[species_list.index(sp)] > 0]
    own2 = {sp: float(Y2[species_list.index(sp)]) for sp in stream2}
    div2 = _outlier_scale_divisors(own2)

    div_of = {**div1, **div2}
    outlier1_sgs = {species_list.index(sp) for sp in div1}
    outlier2_sgs = {species_list.index(sp) for sp in div2}
    non_outlier_Y1 = [float(Y1[sg]) for sg in range(len(Y1)) if Y1[sg] > 0 and sg not in outlier1_sgs]
    non_outlier_Y2 = [float(Y2[sg]) for sg in range(len(Y2)) if Y2[sg] > 0 and sg not in outlier2_sgs]
    max_Y1_excl = max(non_outlier_Y1) if non_outlier_Y1 else float(np.max(Y1))
    max_Y2_excl = max(non_outlier_Y2) if non_outlier_Y2 else float(np.max(Y2))

    vmax_of = {}
    for sp in active:
        if sp in div_of:
            own = own1[sp] if sp in own1 else own2[sp]
            vmax_of[sp] = own / div_of[sp]
        elif sp in stream1:
            vmax_of[sp] = max_Y1_excl
        else:
            vmax_of[sp] = max_Y2_excl
    return vmax_of, div_of


def _ray_equation_str(d, species_list, active_indices, tol=1e-9, rel_tol=0.0):
    """Compact 'reactants → products' string for a selectivity ray *d*
    (indexed like *active_indices*), e.g. '0.66A+0.33B → 0.28R+0.29Q'.

    *rel_tol*, if >0, additionally drops terms smaller than that fraction of
    the largest |d_i| (for space-constrained callers, e.g. small subplot titles).

    Returns None if *d* has no species above tolerance on either side (e.g.
    the ray is ~0 right at t=0)."""
    if rel_tol > 0.0 and len(d):
        tol = max(tol, rel_tol * float(np.max(np.abs(d))))
    reac = [(species_list[i], -d[k]) for k, i in enumerate(active_indices) if d[k] < -tol]
    prod = [(species_list[i], d[k]) for k, i in enumerate(active_indices) if d[k] > tol]
    if not reac and not prod:
        return None
    lhs = '+'.join(f'{v:.2g}{name}' for name, v in reac)
    rhs = '+'.join(f'{v:.2g}{name}' for name, v in prod)
    return f'{lhs} → {rhs}'


_SAVE_PDF = False  # --pdf CLI flag: also save each figure as a vector PDF alongside the PNG


def _save_fig(fig, save_stem, filename):
    """Save fig to <save_stem>.parent/plots/<filename>, close it, and print path.

    When _SAVE_PDF is set (the --pdf CLI flag), also saves a vector .pdf next
    to the .png -- for figures submitted to journals, which typically require
    vector graphics rather than a rasterized image.  The PDF is saved with a
    blank figure title (journals normally caption figures in the manuscript
    text, not on the figure itself), but keeps each subplot's own title; the
    .png keeps its figure title too.
    """
    import pathlib as _pathlib
    plots_dir = _pathlib.Path(save_stem).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    path = plots_dir / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved to {path}")
    if _SAVE_PDF:
        pdf_path = path.with_suffix('.pdf')
        fig.suptitle('')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved to {pdf_path}")
    plt.close(fig)


def _fit_legend_avoiding_curves(ax, handles, labels, fontsize, other_axes=(),
                                loc='upper left', frameon=False):
    """Place a legend inside *ax*, widening it from thin-and-tall (ncol=1) to
    wide-and-short (more columns) if the default shape would otherwise cover
    a plotted curve -- checked against *ax* and any twin axes in
    *other_axes*, whose data can share the same physical box on a different
    scale.  As a last resort (still overlapping at the widest layout), adds
    headroom to each axis' y-limits so the legend has genuinely empty space
    to sit in, per the fallback the plots should degrade to."""
    fig = ax.figure
    axes_to_check = (ax,) + tuple(other_axes)

    def _overlaps(bbox):
        for a in axes_to_check:
            inv = a.transData.inverted()
            x0, y0 = inv.transform((bbox.x0, bbox.y0))
            x1, y1 = inv.transform((bbox.x1, bbox.y1))
            lo_x, hi_x = sorted((x0, x1))
            lo_y, hi_y = sorted((y0, y1))
            for line in a.get_lines():
                xd = np.asarray(line.get_xdata(), dtype=float)
                yd = np.asarray(line.get_ydata(), dtype=float)
                # Skip reference/marker artifacts (axhline's 2-point span,
                # single-point endpoint dots) -- only real plotted curves count.
                if xd.size <= 2:
                    continue
                mask = (xd >= lo_x) & (xd <= hi_x)
                if mask.any() and np.any((yd[mask] >= lo_y) & (yd[mask] <= hi_y)):
                    return True
        return False

    max_ncol = min(4, len(labels))
    leg = None
    for ncol in range(1, max_ncol + 1):
        leg = ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=frameon, ncol=ncol)
        fig.canvas.draw()
        if not _overlaps(leg.get_window_extent(fig.canvas.get_renderer())):
            return leg
    for _ in range(4):
        for a in axes_to_check:
            ylo, yhi = a.get_ylim()
            a.set_ylim(ylo, yhi + 0.15 * (yhi - ylo))
        leg = ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=frameon, ncol=max_ncol)
        fig.canvas.draw()
        if not _overlaps(leg.get_window_extent(fig.canvas.get_renderer())):
            return leg
    return leg


def plot_unique_species_profiles(unique_sp_data, save_stem=None):
    """One subplot per species; one line per unique all_reactions profile.

    Each line is labelled with its concatenated subset key (e.g. '0_2_5'),
    using the same tab palette and inline annotation style as the other plots.
    """
    import pathlib as _pathlib

    meta = unique_sp_data['meta']
    species_data = unique_sp_data['species']
    active = [sp for sp in meta['species'] if sp in species_data]
    if not active:
        return

    n_active = len(active)
    ncols = min(3, max(1, int(np.ceil(np.sqrt(n_active)))))
    nrows = (n_active + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.5, 3.5)

    _tab_palette = _build_tab_palette()

    # A label (e.g. '0_2_5') names the same subset/group of subsets in every
    # species it appears in, so assign it one style globally -- giving that
    # group the same colour and the same left-to-right legend position in
    # every subplot it shows up in, instead of a per-subplot index.
    _all_labels = []
    _seen_labels = set()
    for sp in active:
        for label in species_data[sp]:
            if label not in _seen_labels:
                _seen_labels.add(label)
                _all_labels.append(label)
    _label_style = {label: _tab_palette[i % len(_tab_palette)] for i, label in enumerate(_all_labels)}

    for idx, sp in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        profiles = species_data[sp]

        _MAX_LABEL = 20
        curves = []
        for label in _all_labels:
            if label not in profiles:
                continue
            profile = profiles[label]
            bps = profile['breakpoints']
            segs = profile['segments']
            if not bps or not segs:
                continue
            f_grid = np.array(bps)
            y_grid = np.array(
                [segs[0][0] * bps[0] + segs[0][1]] +
                [segs[j][0] * bps[j + 1] + segs[j][1] for j in range(len(segs))]
            )
            color, ls, mk = _label_style[label]
            display = label if len(label) <= _MAX_LABEL else label[:_MAX_LABEL - 3] + '...'
            ax.plot(f_grid, y_grid, color=color, linestyle=ls, marker=mk,
                    markevery=max(1, len(f_grid) // 15), mfc=_face(color), ms=_ms, lw=_lw, label=display)
            curves.append((f_grid, y_grid, label, color))

        if len(curves) > 1:
            ax.legend(loc='best', frameon=False, fontsize=_leg_fs)

        ax.set_title(sp, fontsize=10)
        ax.set_xlabel('f', fontsize=8)

    for idx in range(n_active, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Unique species profiles by subset combination', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_species_limits.png')
    else:
        plt.show()


def _build_tab_palette():
    """(colour, linestyle) pairs for higher-cardinality series (subsets, profiles).

    Cycles the 8-colour CB_PALETTE against CB_LINESTYLES (32 combinations)
    instead of a large bank of similar hues, so adjacent series stay
    distinguishable by colour+linestyle rather than by hue alone once the
    count exceeds 8."""
    return [_cb_style(i) for i in range(len(CB_PALETTE) * len(CB_LINESTYLES))]


def plot_all_subset_limits(species_labels, reaction_labels, nu_reactants, nu_products,
                           Y1, Y2, save_stem=None, keep_subsets=None, discard_subsets=None):
    """Compute and plot species profiles for every reaction subset (2^n_rxn total).

    Iterates all combinations of 1..n_rxn reactions plus the no-reaction baseline,
    runs the full mixing-limit analysis for each, and overlays every profile on a
    per-species subplot grid.  Curves are coloured by subset size; the no-reaction
    line is dashed grey and the full scheme is bold black.

    ``keep_subsets`` / ``discard_subsets`` select which subsets to plot by their
    integer ``subset_num`` (same numbering as :func:`export_all_subsets_json` and
    :func:`filter_subsets_by_num`: 0 is the no-reaction baseline).  With both
    ``None`` every subset is plotted (the default); when set they mirror the
    filter applied before ODE integration so the sweep shows only the retained
    limits.  Skipped subsets are not computed, so the filter also saves LP solves.

    Note: runtime is O(2^n_rxn) LP solves — practical up to ~12 reactions (~4 k solves).
    """
    import io
    import pathlib
    import contextlib
    from itertools import combinations

    n_rxn = len(reaction_labels)
    if n_rxn > 12:
        print(f"Warning: {2**n_rxn} subsets — this may be slow for n_rxn > 12.")

    Y1a = np.asarray(Y1, dtype=float)
    Y2a = np.asarray(Y2, dtype=float)
    n_species = len(species_labels)

    # Retain-by-subset_num predicate (keep applied first, then discard) — mirrors
    # filter_subsets_by_num so the sweep plots the same subsets fed to the ODE.
    _keep = set(keep_subsets) if keep_subsets is not None else None
    _discard = set(discard_subsets) if discard_subsets is not None else None

    def _retained(num):
        if _keep is not None and num not in _keep:
            return False
        if _discard is not None and num in _discard:
            return False
        return True

    profiles = []

    if _retained(0):
        profiles.append({
            'subset_num': 0, 'size': 0, 'fs': None,
            'reactions': [],
            'f_grid': np.array([0.0, 1.0]),
            'Y': np.column_stack([Y2a, Y1a]),
        })

    total = 2 ** n_rxn - 1
    done = 0
    progress_step = max(1, total // 10)
    print(f"Computing {total} reaction subsets...", flush=True)
    for size in range(1, n_rxn + 1):
        for subset in combinations(range(n_rxn), size):
            done += 1
            if not _retained(done):
                continue
            sub_labels = [reaction_labels[i] for i in subset]
            sub_nu_r = nu_reactants[:, list(subset)]
            sub_nu_p = nu_products[:, list(subset)]
            with contextlib.redirect_stdout(io.StringIO()):
                res = analyze_stream_limit_system(
                    species_labels, sub_labels, sub_nu_r, sub_nu_p, Y1a, Y2a)
                seg = generate_line_segments(species_labels, Y1a, Y2a, res)
            # pyrefly: ignore [bad-argument-type]
            profiles.append({
                'subset_num': done, 'size': size, 'fs': res['fs'],
                'reactions': sub_labels,
                'f_grid': seg['_f_grid'],
                'Y': seg['_Y_with_rxn'],
            })
            if done % progress_step == 0 or done == total:
                print(f"  {done}/{total}", flush=True)

    if not profiles:
        print("[filter] no subsets retained for plot_all_subset_limits; skipping plots.")
        return

    active = [s for s in range(n_species)
              # pyrefly: ignore [bad-index, unsupported-operation]
              if any(np.max(np.abs(p['Y'][s])) > 1e-9 for p in profiles)]

    n_active = len(active)
    ncols = min(4, n_active)
    nrows = (n_active + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.5, 3.5)

    n_profiles = len(profiles)
    _tab_palette = _build_tab_palette()
    subset_styles = {p['subset_num']: _tab_palette[i % len(_tab_palette)]
                     for i, p in enumerate(profiles)}

    # Baseline (no-reaction) and full-scheme profiles, located by size rather
    # than list position since the filter may drop either one.
    baseline = next((p for p in profiles if p['size'] == 0), None)
    full = next((p for p in profiles if p['size'] == n_rxn), None)

    for idx, s in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        # Draw intermediate subsets first (background), then highlights on top.
        curves = []
        for p in profiles:
            if p['size'] == 0:
                continue
            if p['size'] == n_rxn:
                continue
            # pyrefly: ignore [bad-index, unsupported-operation]
            y_arr = p['Y'][s]
            color, ls, mk = subset_styles[p['subset_num']]
            ax.plot(p['f_grid'], y_arr, color=color, linestyle=ls, marker=mk, mfc=_face(color),
                    markevery=max(1, len(p['f_grid']) // 15), ms=_ms, lw=_lw * 0.6, alpha=0.7,
                    zorder=1, label=str(p['subset_num']))
            curves.append((p['f_grid'], y_arr, p['subset_num'], color))
        # No-reaction baseline (dashed) — only when retained.
        if baseline is not None:
            # pyrefly: ignore [bad-index, unsupported-operation]
            y0 = baseline['Y'][s]
            color0, _, mk0 = subset_styles[baseline['subset_num']]
            ax.plot(baseline['f_grid'], y0, color=color0, lw=_lw, ls='--', zorder=4, alpha=0.9,
                    marker=mk0, mfc=_face(color0), markevery=max(1, len(baseline['f_grid']) // 15), ms=_ms,
                    label=str(baseline['subset_num']))
            curves.append((baseline['f_grid'], y0, baseline['subset_num'], color0))
        # Full scheme on top (bold) — only when retained.
        if full is not None and full is not baseline:
            # pyrefly: ignore [bad-index, unsupported-operation]
            yN = full['Y'][s]
            colorN, _, mkN = subset_styles[full['subset_num']]
            ax.plot(full['f_grid'], yN, color=colorN, lw=_lw * 1.33, zorder=5,
                    marker=mkN, mfc=_face(colorN), markevery=max(1, len(full['f_grid']) // 15), ms=_ms + 1,
                    label=str(full['subset_num']))
            curves.append((full['f_grid'], yN, full['subset_num'], colorN))
        if len(curves) > 1:
            ax.legend(loc='best', frameon=False, fontsize=_leg_fs,
                      ncol=max(1, (len(curves) + 5) // 6))
        ax.set_title(species_labels[s], fontsize=10)
        ax.set_xlabel('f', fontsize=8)

    for idx in range(n_active, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _n_shown = len(profiles)
    _total_all = 2 ** n_rxn
    _shown_str = (f'all {_total_all}' if _n_shown == _total_all
                  else f'{_n_shown} of {_total_all}')
    fig.suptitle(f'Species mixing limits — {_shown_str} reaction subsets', fontsize=12)
    # pyrefly: ignore [bad-argument-type]
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{pathlib.Path(save_stem).name}_species_by_subset.png')
    else:
        plt.show()

    # One subplot per reaction subset: all species vs f for that combination.
    stream_labels_sub = identify_stream_feeds(Y1a, Y2a)
    s1_sub = [s for s in active if stream_labels_sub[s] in (1, 12)]
    s2_sub = [s for s in active if stream_labels_sub[s] not in (1, 12)]
    color_map3 = {s: _sp_color(s, species_labels) for s in active}
    ls_map3 = {s: _sp_linestyle(s, species_labels) for s in active}
    mk_map3 = {s: _sp_marker(s, species_labels) for s in active}

    # Detect any species whose global max dwarfs every OTHER SPECIES FED BY
    # THE SAME STREAM and scale it down to ~1000, for a readable legend/axis
    # -- stream-1 species (on the twin axis) are never compared against
    # stream-2 species (on the main axis), since they don't share a scale.
    # pyrefly: ignore [bad-index, unsupported-operation]
    global_max3 = {s: max(np.max(np.abs(p['Y'][s])) for p in profiles) for s in active}
    div_of3 = _outlier_scale_divisors_by_stream(global_max3, {s: s in s1_sub for s in active})

    def _sub_label(s):
        return f'{species_labels[s]}/{div_of3[s]:g}' if s in div_of3 else species_labels[s]

    def _sub_y(s, y_arr):
        return y_arr / div_of3[s] if s in div_of3 else y_arr

    n_subsets = len(profiles)
    ncols_sub = min(2 if n_subsets <= 8 else 3, n_subsets)
    nrows_sub = (n_subsets + ncols_sub - 1) // ncols_sub
    fig3, axes3 = plt.subplots(nrows_sub, ncols_sub,
                               figsize=(4 * ncols_sub, 3.2 * nrows_sub),
                               squeeze=False)
    _ms3, _lw3, _leg_fs3 = _scaled_marker_lw(4.0, 3.2)

    for idx, p in enumerate(profiles):
        ax = axes3[idx // ncols_sub][idx % ncols_sub]
        _mk_every3 = max(1, len(p['f_grid']) // 15)
        for s in s2_sub:
            # pyrefly: ignore [bad-index, unsupported-operation]
            ax.plot(p['f_grid'], _sub_y(s, p['Y'][s]), color=color_map3[s], linestyle=ls_map3[s],
                    marker=mk_map3[s], mfc=_face(color_map3[s]), markevery=_mk_every3, ms=_ms3, lw=_lw3, label=_sub_label(s))
        ax_r3 = None
        if s1_sub:
            ax_r3 = ax.twinx()
            for s in s1_sub:
                # pyrefly: ignore [bad-index, unsupported-operation]
                ax_r3.plot(p['f_grid'], _sub_y(s, p['Y'][s]), color=color_map3[s], linestyle=ls_map3[s],
                           marker=mk_map3[s], mfc=_face(color_map3[s]), markevery=_mk_every3, ms=_ms3, lw=_lw3, label=_sub_label(s))
            ax_r3.tick_params(axis='y', labelsize=6)
        _h, _l = ax.get_legend_handles_labels()
        if ax_r3 is not None:
            _h_r, _l_r = ax_r3.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs3)
        rxn_list = p['reactions']
        # pyrefly: ignore [no-matching-overload]
        rxn_str = '∅ — no reaction' if not rxn_list else '\n'.join(rxn_list)
        title = f"Subset {p['subset_num']}\n{rxn_str}"
        fs_p = p.get('fs')
        # pyrefly: ignore [unsupported-operation]
        if fs_p is not None and 0.0 < fs_p < 1.0:
            title += f'\nfs = {fs_p:.4f}'
        ax.set_title(title, fontsize=7)
        ax.set_xlabel('f', fontsize=7)
        # pyrefly: ignore [unsupported-operation]
        if fs_p is not None and 0.0 < fs_p < 1.0:
            ax.axvline(fs_p, color='k', linestyle=':', alpha=0.5)

    for idx in range(n_subsets, nrows_sub * ncols_sub):
        axes3[idx // ncols_sub][idx % ncols_sub].set_visible(False)

    fig3.suptitle(f'Species profiles for {n_subsets} reaction subset(s)', fontsize=12)
    # pyrefly: ignore [bad-argument-type]
    fig3.tight_layout(rect=[0, 0, 1, 0.97])

    if save_stem is not None:
        _save_fig(fig3, save_stem, f'{pathlib.Path(save_stem).name}_species_by_subset_grid.png')
    else:
        plt.show()


def plot_ode_trajectories(ode_results, save_stem=None):
    """Plot species concentration trajectories and reaction rates vs time.

    ode_results may be a single result dict or a list of result dicts.
    When multiple results are passed their trajectories are overlaid on the
    same axes, distinguished by weight_method label and linestyle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as _gs
    import pathlib as _pathlib

    if isinstance(ode_results, dict):
        ode_results = [ode_results]

    ref = ode_results[0]
    species_list = ref['species']
    active_indices = ref.get('active_indices', np.arange(len(species_list)))
    active_species = [species_list[i] for i in active_indices]
    n_plot = len(active_species)
    m_epsilon = ref.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ref.get('m_lambda', DEFAULT_M_LAMBDA)
    m_nu = ref.get('m_nu', DEFAULT_M_NU)
    m_Sc = ref.get('m_Sc', DEFAULT_M_SC)
    mean_f = ref['mean_f']
    var_end = ref.get('var_end', 0.0)

    has_rates = any(
        r.get('rates') is not None and r['rates'].shape[1] > 0
        for r in ode_results
    )
    rxn_labels = ref.get('rxn_labels', [])

    single_method = (len(ode_results) == 1)
    n_cols = min(3, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols

    if single_method:
        res0 = ode_results[0]
        sp_peaks  = {sp: float(np.max(res0['y'][:, i]))
                     for i, sp in zip(active_indices, active_species)}
        sp_max    = max(sp_peaks.values()) if sp_peaks else 1.0
        sp_thresh = sp_max * 0.1
        has_minor = any(sp_peaks[sp] < sp_thresh for sp in active_species)
        primary_sp   = [(i, sp) for i, sp in zip(active_indices, active_species)
                        if sp_peaks[sp] >= sp_thresh]
        secondary_sp = [(i, sp) for i, sp in zip(active_indices, active_species)
                        if sp_peaks[sp] < sp_thresh]
    else:
        sp_peaks = sp_thresh = None
        has_minor = False

    n_rxns = len(rxn_labels) if has_rates else 0

    def _lambda_source_of(res):
        """blend_fs/ray_limit only: the 'blendfs'/'raylimit' sub-dict, whose
        'prof' gives {species: array[:, 3]} -- that species' blend weight
        λ(t) (0 = at the complete-reaction limit B(f), 1 = at the
        no-reaction limit M(f)) -- and whose 'fsb' gives the run's fs(t)."""
        bf = res.get('blendfs')
        if bf is not None and bf.get('prof'):
            return bf
        rl = res.get('raylimit')
        if rl is not None and rl.get('prof'):
            return rl
        return None

    has_lambda = any(_lambda_source_of(r) is not None for r in ode_results)

    # ── Unified cell-based panel grid ─────────────────────────────────────────
    n_sp_panels    = (2 if has_minor else 1) if single_method else n_plot
    ios_panel      = n_sp_panels
    lambda_panel   = ios_panel + 1
    rate_panel0    = lambda_panel + (1 if has_lambda else 0)
    n_total_panels = n_sp_panels + 1 + (1 if has_lambda else 0) + n_rxns
    n_rows         = (n_total_panels + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    gs  = _gs.GridSpec(n_rows, n_cols, figure=fig)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4, 3)

    def _cell(panel_idx):
        r, c = divmod(panel_idx, n_cols)
        return fig.add_subplot(gs[r, c])

    method_colors = _active_palette()

    # ── Species panels ────────────────────────────────────────────────────────
    if single_method:
        t_arr = res0['t']
        every = max(1, len(t_arr) // 20)

        ax_maj = _cell(0)
        primary_curves = []
        for sp_idx, (i, sp) in enumerate(primary_sp):
            color = _sp_color(i, species_list, method_colors)
            mkr = _sp_marker(i, species_list)
            y_arr = res0['y'][:, i]
            ax_maj.plot(t_arr, y_arr, lw=_lw, color=color, label=sp,
                        marker=mkr, ms=_ms, markevery=every, mfc=_face(color), mew=1.2)
            primary_curves.append((sp, color, y_arr))
        ax_maj.set_xlabel('t (s)', fontsize=8)
        ax_maj.set_ylabel('concentration (mol/m³)', fontsize=8)
        ax_maj.tick_params(labelsize=7)
        ax_maj.set_title('Major species', fontsize=10)
        if primary_curves:
            ax_maj.legend(loc='best', frameon=False, fontsize=_leg_fs)

        if has_minor:
            ax_min = _cell(1)
            secondary_curves = []
            for sp_idx, (i, sp) in enumerate(secondary_sp):
                color = _sp_color(i, species_list, method_colors)
                mkr = _sp_marker(i, species_list)
                y_arr = res0['y'][:, i]
                ax_min.plot(t_arr, y_arr, lw=_lw, color=color, label=sp,
                            marker=mkr, ms=_ms, markevery=every, mfc=_face(color), mew=1.2)
                secondary_curves.append((sp, color, y_arr))
            ax_min.set_xlabel('t (s)', fontsize=8)
            ax_min.set_ylabel('concentration (mol/m³)', fontsize=8)
            ax_min.tick_params(labelsize=7)
            ax_min.set_title('Minor species', fontsize=10)
            if secondary_curves:
                ax_min.legend(loc='best', frameon=False, fontsize=_leg_fs)
    else:
        for plot_idx, (i, sp) in enumerate(zip(active_indices, active_species)):
            ax = _cell(plot_idx)
            for res_idx, res in enumerate(ode_results):
                method = res.get('weight_method', f'run{res_idx}')
                color = method_colors[res_idx % len(method_colors)]
                st = _wm_style(method)
                every = max(1, len(res['t']) // 15)
                mfc = color if st['mfc'] is None else _face(color)
                ax.plot(res['t'], res['y'][:, i], lw=_lw, color=color, ls=st['ls'],
                        marker=st['mk'], ms=_ms, markevery=every, mfc=mfc, mew=st['mew'],
                        label=method)
            ax.axhline(ref['y'][0, i], color='gray', lw=0.7, ls='--', alpha=0.5)
            ax.set_title(sp, fontsize=10)
            ax.set_xlabel('t (s)', fontsize=8)
            ax.set_ylabel('C', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=_leg_fs, loc='best', frameon=False)

    # ── Intensity of segregation ──────────────────────────────────────────────
    ax_var = _cell(ios_panel)
    t_dense = np.linspace(0.0, float(ref['t'][-1]), 500)
    max_var = mean_f * (1.0 - mean_f)
    ios_curve = np.array([mixing_variance(t, mean_f, m_epsilon, m_lambda, m_nu, m_Sc) / max_var for t in t_dense])
    ax_var.plot(t_dense, ios_curve, color='steelblue', lw=_lw)
    ax_var.set_xlabel('t (s)', fontsize=8)
    ax_var.set_ylabel('$I_s$', fontsize=8)
    ax_var.set_title(f'Intensity of segregation  (ε={m_epsilon:.4g})', fontsize=10)
    ax_var.tick_params(labelsize=7)

    # ── Blend weight λ(t) (blend_fs / ray_limit only), with fs(t) on a
    # ── linear right axis (minimum span 0.2, so a near-constant fs doesn't
    # ── get exaggerated by autoscaling to a razor-thin range) ────────────────
    if has_lambda:
        _lam_floor = 1e-8   # log-scale display floor; λ is often clamped to exactly 0
        ax_lam = _cell(lambda_panel)
        ax_fs = ax_lam.twinx()
        fs_vals = []
        for res_idx, res in enumerate(ode_results):
            src = _lambda_source_of(res)
            if src is None:
                continue
            method = res.get('weight_method', f'run{res_idx}')
            st = _wm_style(method)
            every = max(1, len(res['t']) // 15)
            # All species share the same λ almost always -- draw one
            # representative curve per method (not one per species) so the
            # single 'λ' legend entry's colour actually matches what's drawn,
            # rather than labelling just the first of several overlapping,
            # differently-coloured per-species lines.
            _, arr0 = next(iter(src['prof'].items()))
            lam = np.maximum(arr0[:, 3], _lam_floor)
            color = CB_PALETTE[0]
            mfc = color if st['mfc'] is None else _face(color)
            ax_lam.plot(res['t'], lam, color=color, ls=st['ls'], lw=_lw,
                       marker=st['mk'], ms=_ms, markevery=every, mfc=mfc, mew=st['mew'],
                       label=f'λ ({method})' if len(ode_results) > 1 else 'λ')
            fsb = np.asarray(src.get('fsb'), dtype=float)
            fs_mask = np.isfinite(fsb) & (fsb > 0.0) & (fsb < 1.0)
            if fs_mask.any():
                ax_fs.plot(res['t'][fs_mask], fsb[fs_mask], color='black', ls=st['ls'], lw=1.5,
                          marker='o', ms=_ms, markevery=every, mfc=_face('black'), mew=st['mew'],
                          label=f'fs ({method})' if len(ode_results) > 1 else 'fs')
                fs_vals.append(fsb[fs_mask])
        ax_lam.set_xlabel('t (s)', fontsize=8)
        ax_lam.set_ylabel('λ', fontsize=8)
        ax_lam.set_yscale('log')
        ax_lam.set_ylim(_lam_floor, 1.5)
        ax_lam.set_title('Blend weight λ  (0 = B(f), 1 = M(f))', fontsize=10)
        ax_lam.tick_params(labelsize=7)
        if fs_vals:
            fs_lo, fs_hi = float(np.min(np.concatenate(fs_vals))), float(np.max(np.concatenate(fs_vals)))
            if fs_hi - fs_lo < 0.2:
                mid = 0.5 * (fs_lo + fs_hi)
                fs_lo, fs_hi = mid - 0.1, mid + 0.1
            ax_fs.set_ylim(fs_lo, fs_hi)
        ax_fs.set_ylabel('fs', fontsize=8)
        ax_fs.tick_params(labelsize=7)
        _h, _l = ax_lam.get_legend_handles_labels()
        _h2, _l2 = ax_fs.get_legend_handles_labels()
        if len(_h) + len(_h2) > 1:
            ax_lam.legend(_h + _h2, _l + _l2, fontsize=_leg_fs, loc='best', frameon=False)

    # ── One subplot per reaction (colour = reaction; linestyle+marker = method) ──
    if has_rates:
        rxn_colors = _active_palette()
        cont_times = sorted({t for r in ode_results
                             for t, _ in r.get('continuous_events', [])})
        jump_times = sorted({t for r in ode_results
                             for t, _ in r.get('jump_events', [])})
        for j, lbl in enumerate(rxn_labels):
            ax_r = _cell(rate_panel0 + j)
            color = rxn_colors[j % len(rxn_colors)]
            for res_idx, res in enumerate(ode_results):
                rates = res.get('rates')
                if rates is None or j >= rates.shape[1]:
                    continue
                method = res.get('weight_method', f'run{res_idx}')
                st = _wm_style(method)
                every = max(1, len(res['t']) // 15)
                mfc = color if st['mfc'] is None else _face(color)
                ax_r.plot(res['t'], rates[:, j], color=color, ls=st['ls'], lw=_lw,
                          marker=st['mk'], ms=_ms, markevery=every, mfc=mfc, mew=st['mew'],
                          label=method if len(ode_results) > 1 else None)
            for t_c in cont_times:
                ax_r.axvline(t_c, color='green', lw=0.8, ls='--', alpha=0.4)
            for t_j in jump_times:
                ax_r.axvline(t_j, color='purple', lw=0.8, ls='--', alpha=0.4)
            ax_r.set_xlabel('t (s)', fontsize=8)
            ax_r.set_ylabel('rate  (mol/(m³·s))', fontsize=8)
            ax_r.set_title(lbl, fontsize=10)
            ax_r.tick_params(labelsize=7)
            if len(ode_results) > 1:
                ax_r.legend(fontsize=_leg_fs, loc='best', frameon=False)

    # hide unused cells in last row
    for p in range(n_total_panels, n_rows * n_cols):
        r, c = divmod(p, n_cols)
        fig.add_subplot(gs[r, c]).set_visible(False)

    methods_str = ', '.join(r.get('weight_method', '?') for r in ode_results)
    fig.suptitle(
        f'ODE trajectories  (mean_f={mean_f:.4f},  '
        f'I_s: 1 → 0  (ε={m_epsilon:.4g}, τ_s={mixing_timescale(m_epsilon, m_lambda):.4g} s),  '
        f'weights=[{methods_str}])',
        fontsize=9)
    plt.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_ode.png')
    else:
        plt.show()


def mixing_variance(t, mean_f, m_epsilon=DEFAULT_M_EPSILON,
                    m_lambda=DEFAULT_M_LAMBDA, m_nu=DEFAULT_M_NU, m_Sc=DEFAULT_M_SC):
    """Return Beta-distribution variance at time *t* for the given mixing parameters.

    Starts near the theoretical maximum max_var = mean_f*(1-mean_f) and decays
    with the intensity of segregation Ios(t), set by the turbulent dissipation
    rate ``m_epsilon`` (via the mixing timescale :func:`mixing_timescale`),
    floored at max_var*1e-6.
    """
    max_var   = mean_f * (1.0 - mean_f)
    var_start = max_var * (1.0 - 1e-6)
    var_end   = max_var * 1e-6
    tau_s=mixing_timescale(m_epsilon, m_lambda)
    tau_E=1/(0.05776*(m_epsilon/m_nu)**0.5)
    tau_D=1/(1/tau_E*(0.303+17051/m_Sc))
    m_M=tau_s/tau_E
    m_N=tau_s/tau_D
    Ios=np.exp(-t/tau_s)+1/(m_M-1)*(np.exp(-t/tau_s)-np.exp(-m_M*t/tau_s))+m_M/(m_M-1)*(1/(m_N-1)*(np.exp(-t/tau_s)-np.exp(-m_N*t/tau_s))-1/(m_N-m_M)*(np.exp(-m_M*t/tau_s)-np.exp(-m_N*t/tau_s)))
    return max(var_start * Ios, var_end)


def mixing_timescale(m_epsilon, m_lambda=DEFAULT_M_LAMBDA):
    """Scalar-dissipation (mechanical mixing) timescale tau_s in seconds.

    Sets the rate of the intensity-of-segregation decay; smaller for higher
    turbulent dissipation rate ``m_epsilon``.

    """
    return 0.75 * (m_lambda ** 0.666666) / (m_epsilon ** 0.333333)


def plot_product_fractions_vs_time(ode_results, Y1, nu_reactants, nu_products,
                                   save_stem=None, Y2=None):
    """Plot the product-fraction split versus time, one subplot per weight method.

    For each weighting method (one ``ode_result`` per method), the closure
    quantity ``X_p`` — the fraction of the consumed stream-1 limiting reactant
    that has ended up as each product p — is recomputed at every output time
    using :func:`stream1_reactant_to_product_fraction` (with ``y_final`` set to
    the running state ``y(t)``).  Each subplot overlays one line per product,
    plus a thin grey TOTAL line (their sum, ≈1 under perfect mass closure).

    Product colours are shared across subplots so methods are directly
    comparable.  Times before the limiting reactant has been measurably
    consumed are left blank (the fraction is 0/0 there).
    """
    import matplotlib.pyplot as plt
    import pathlib as _pathlib

    if isinstance(ode_results, dict):
        ode_results = [ode_results]

    Y1 = np.asarray(Y1, dtype=float)
    Y2 = np.asarray(Y2, dtype=float) if Y2 is not None else None
    nu_reactants = np.asarray(nu_reactants, dtype=float)
    nu_products = np.asarray(nu_products, dtype=float)

    # Consistent product ordering/colours across all methods: union of products
    # reported by each run (the limiting reactant is identical across methods).
    all_products = []
    for res in ode_results:
        for p in res.get('stream1_closure', {}).get('products', []):
            if p not in all_products:
                all_products.append(p)
    if not all_products:
        print("[plot] no products to plot product fractions for; skipping.")
        return
    _sp_list_ref = ode_results[0]['species']
    color_of = {p: _sp_color(p, _sp_list_ref) for p in all_products}
    ls_of = {p: _sp_linestyle(p, _sp_list_ref) for p in all_products}
    mk_of = {p: _sp_marker(p, _sp_list_ref) for p in all_products}

    n = len(ode_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.0), squeeze=False, sharey=True)
    _ms, _lw, _leg_fs = _scaled_marker_lw(5.0, 4.0)

    for res_idx, res in enumerate(ode_results):
        ax = axes[0][res_idx]
        method = res.get('weight_method', f'run{res_idx}')
        closure = res.get('stream1_closure', {})
        L_name = closure.get('limiting_reactant')
        t = np.asarray(res['t'], dtype=float)
        y = np.asarray(res['y'], dtype=float)
        species_list = res['species']

        if L_name is None:
            ax.set_title(f'{method}\n(no stream-1 limiting reactant)', fontsize=9)
            ax.set_xlabel('t (s)', fontsize=8)
            continue

        products = closure.get('products', [])
        fracs = {p: np.full(len(t), np.nan) for p in products}
        total = np.full(len(t), np.nan)
        # Recompute the L→product split at each output time (y_final = y(t)).
        for k in range(len(t)):
            c = stream1_reactant_to_product_fraction(
                species_list, y[0], y[k], Y1, nu_reactants, nu_products,
                limiting_species=L_name, silent=True, Y2=Y2)
            # 0/0 before anything is consumed -> leave as NaN (blank).
            if not (c['consumed'] > 1e-12):
                continue
            for p in products:
                fracs[p][k] = c['per_product_fraction'].get(p, np.nan)
            total[k] = c['fraction']

        for p in products:
            ax.plot(t, fracs[p], lw=_lw, color=color_of[p], linestyle=ls_of[p],
                    marker=mk_of[p], mfc=_face(color_of[p]), markevery=max(1, len(t) // 15), ms=_ms,
                    label=f'$X_{{{p}}}$')
        ax.plot(t, total, lw=1.0, color='grey', ls='--', label='total')
        ax.axhline(1.0, color='grey', lw=0.6, ls=':', alpha=0.6)
        ax.set_title(f'{method}  ({L_name} → products)', fontsize=10)
        ax.set_xlabel('t (s)', fontsize=8)
        ax.tick_params(labelsize=7)
        _h, _l = ax.get_legend_handles_labels()
        _fit_legend_avoiding_curves(ax, _h, _l, _leg_fs, loc='upper right')

    axes[0][0].set_ylabel('$X_{\\mathrm{species}}$', fontsize=8)
    fig.tight_layout()
    fig.suptitle('Product fractions $X_p$ vs time (by weighting method)', fontsize=11, y=1.0)

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_product_fractions_vs_time.png')
    else:
        plt.show()


def stream1_reactant_to_product_fraction(species_list, y_initial, y_final, Y1,
                                         nu_reactants, nu_products,
                                         limiting_species=None, silent=False, Y2=None):
    """Fraction of the consumed stream-1 limiting reactant that ended up as product.

    Intended to be called at the end of an ODE simulation as a closure check.
    The limiting reactant is the species fed in stream 1 (Y1 > 0, the f=1 feed)
    that is in shortest supply relative to its stoichiometric demand.  Its
    consumed amount over the simulation is ``y_initial - y_final``.

    Each terminal product (a species produced by some reaction but never itself a
    reactant, e.g. P, R, T, S, Q) is weighted by its *stoichiometric content* of
    the limiting reactant: the moles of L embodied per mole of product, derived
    from the reaction network (see :func:`_limiting_reactant_content`).  So if one
    mole of S requires two moles of the limiting reactant A, S contributes
    ``2 * formed_S`` to the numerator, and::

        fraction = sum_p content_p * formed_p / consumed_L

    With exact mass closure this is 1; deviations measure how much consumed
    reactant is unaccounted for by the product pool (or closure error).

    Parameters
    ----------
    species_list : list of str
        Species names, in the same order as the rows of the arrays below.
    y_initial, y_final : ndarray, shape (n_species,)
        Concentrations at the start and end of the simulation
        (e.g. ``y_full[0]`` and ``y_full[-1]``).
    Y1 : ndarray, shape (n_species,)
        Stream-1 feed composition (the f=1 feed); used to identify which
        reactants are fed in stream 1.
    nu_reactants, nu_products : ndarray, shape (n_species, n_reactions)
        Gross reactant and product stoichiometric matrices from
        ``parse_reactions``.
    limiting_species : str, optional
        Name of the limiting reactant.  If given it overrides auto-detection.

    Returns
    -------
    dict
        Keys: ``limiting_reactant``, ``fed`` (moles of L fed), ``consumed``
        (moles of L), ``conversion`` (fractional conversion of L, ``consumed/fed``),
        ``products`` (list of names), ``per_product`` (name -> net moles formed),
        ``stoich_factors`` (name -> moles of L per mole of product),
        ``per_product_equiv`` (name -> moles of L embodied), ``per_product_fraction``
        (name -> share of consumed L; these sum to ``fraction`` and equal unity
        under perfect mass closure), ``product_in_reactant_equiv`` (numerator, in
        moles of L), ``fraction``.  ``fraction`` is NaN when no stream-1 reactant
        is found or nothing was consumed.
    """
    y_initial = np.asarray(y_initial, dtype=float)
    y_final = np.asarray(y_final, dtype=float)
    Y1 = np.asarray(Y1, dtype=float)
    nu_reactants = np.asarray(nu_reactants, dtype=float)
    nu_products = np.asarray(nu_products, dtype=float)
    N = nu_products - nu_reactants

    is_product = nu_products.sum(axis=1) > 0

    # Candidate limiting reactants: reactants that are fed in stream 1.
    sp_to_idx = {sp: i for i, sp in enumerate(species_list)}
    if limiting_species is not None:
        if limiting_species not in sp_to_idx:
            raise ValueError(f"limiting_species {limiting_species!r} not in species_list")
        L = sp_to_idx[limiting_species]
    else:
        L = _stream1_limiting_index(Y1, nu_reactants, Y2, nu_products)
        if L is None:
            return {'limiting_reactant': None, 'fed': float('nan'),
                    'consumed': float('nan'), 'conversion': float('nan'),
                    'products': [], 'per_product': {}, 'stoich_factors': {},
                    'per_product_equiv': {}, 'per_product_fraction': {},
                    'product_in_reactant_equiv': float('nan'),
                    'fraction': float('nan')}

    fed = float(y_initial[L])
    consumed = float(y_initial[L] - y_final[L])
    # Fractional conversion of the limiting reactant (close to 1 when it is
    # almost fully consumed).
    conversion = consumed / fed if fed > 0 else float('nan')

    # Products = every species produced by some reaction (P, R, T, S, Q ...),
    # including intermediates that are later consumed again, e.g. P in
    # A+B->P then A+P->S.  Excluded: the limiting reactant, and any *fed reactant*
    # — a species present in the feed (y_initial>0) that is also consumed as a
    # reactant.  Without that exclusion a reverse reaction (e.g. P -> A + B) would
    # mark the fed reactant B as "produced" and report it as a spurious product.
    arr = np.arange(len(species_list))
    is_fed = y_initial > 1e-12
    is_reactant = nu_reactants.sum(axis=1) > 0
    fed_reactant = is_fed & is_reactant
    product_idx = np.where(is_product & ~fed_reactant & (arr != L))[0]
    product_names = [species_list[i] for i in product_idx]

    # Moles of L embodied per mole of each species, from network stoichiometry.
    content = _limiting_reactant_content(N, L, Y1=Y1, Y2=Y2)

    per_product = {}
    stoich_factors = {}
    per_product_equiv = {}  # name -> moles of L embodied in that product
    product_in_reactant_equiv = 0.0
    unresolved = []
    for i in product_idx:
        formed = float(y_final[i] - y_initial[i])
        factor = content[i]
        per_product[species_list[i]] = formed
        stoich_factors[species_list[i]] = float(factor) if not np.isnan(factor) else float('nan')
        equiv = factor * formed if not np.isnan(factor) else float('nan')
        per_product_equiv[species_list[i]] = equiv
        if not np.isnan(equiv):
            product_in_reactant_equiv += equiv
        else:
            unresolved.append(species_list[i])

    if unresolved:
        # Unresolvable L-content means the production network has a cycle — the
        # signature of a reversible reaction (forward + reverse).  The L balance
        # cannot close, so the product-fraction closure is not applicable here.
        if not silent:
            print(f"[closure] warning: L-content unresolvable for {unresolved} "
                  f"(cyclic/reversible network); product-fraction closure not applicable.")
        fraction = float('nan')
    elif len(product_idx) == 0:
        # No terminal products (every produced species is itself a fed reactant,
        # e.g. a reversible reaction with the product also fed).  Nothing to close.
        if not silent:
            print("[closure] no terminal products (all produced species are fed "
                  "reactants); product-fraction closure not applicable.")
        fraction = float('nan')
    else:
        fraction = product_in_reactant_equiv / consumed if consumed > 0 else float('nan')
    # Per-product share of the consumed limiting reactant.  These sum to
    # ``fraction``, which equals unity under perfect mass closure.
    per_product_fraction = {
        name: (equiv / consumed if consumed > 0 else float('nan'))
        for name, equiv in per_product_equiv.items()
    }
    # Fallback for a zero stoichiometric factor: L's content in this product's
    # own reaction network traces to nothing but L's own catalytic role (or
    # nothing at all), so its true share of *consumed* L is correctly zero.
    # Report it relative to how much L was fed instead, so a substantial
    # product doesn't display as a flat 0 -- this value is on a different
    # basis than the other products' and is not included in ``fraction``.
    for name in per_product_fraction:
        if stoich_factors.get(name) == 0.0:
            per_product_fraction[name] = per_product[name] / fed if fed > 0 else float('nan')
            if not silent:
                print(f"[closure]   note: {name}'s stoichiometric factor relative to "
                      f"{species_list[L]} is 0; reporting its fraction relative to "
                      f"{species_list[L]} fed (not consumed, and not part of the total).")

    return {
        'limiting_reactant': species_list[L],
        'fed': fed,
        'consumed': consumed,
        'conversion': conversion,
        'products': product_names,
        'per_product': per_product,
        'stoich_factors': stoich_factors,
        'per_product_equiv': per_product_equiv,
        'per_product_fraction': per_product_fraction,
        'product_in_reactant_equiv': product_in_reactant_equiv,
        'fraction': fraction,
    }


def _limiting_reactant_content(N, L, Y1=None, Y2=None):
    """Moles of limiting reactant L embodied in one mole of each species.

    Propagates the limiting-reactant content through the reaction network using
    the net stoichiometric matrix ``N = nu_products - nu_reactants`` (shape
    (n_species, n_reactions)).  Species that are never net-produced are sources:
    they contain 1 mole of L per mole (if they *are* L) or 0 otherwise.  For a
    species produced by reaction j the content is the L entering as reactants,
    divided by the moles of that species produced::

        content[i] = sum_{s consumed in j} (-N[s,j]) * content[s] / N[i,j]

    Net stoichiometry means catalysts (consumed and re-formed) contribute zero.
    Multiple producing reactions are averaged.  Unresolvable species (e.g. in a
    cycle) are left as NaN.  Examples: ``2A -> S`` gives S a content of 2; the
    chain ``A -> R`` then ``2R -> S`` also gives S a content of 2.

    Feed vectors Y1, Y2: fed species are treated as sources with content 0
    regardless of whether some reaction also produces them.  This prevents
    over-counting when a species is both a feed component (e.g. a solvent fed in
    large excess) and incidentally produced by a side reaction.
    """
    n_sp = N.shape[0]
    produced = (N > 0).any(axis=1)
    content = np.full(n_sp, np.nan)
    # Source species (never net-produced): contain L only if they are L.
    content[~produced] = 0.0
    # Fed species are external inputs — their L-content is 0 (they come from the
    # feed, not from L).  Override even if some reaction also produces them.
    if Y1 is not None:
        content[np.asarray(Y1) > 0] = 0.0
    if Y2 is not None:
        content[np.asarray(Y2) > 0] = 0.0
    content[L] = 1.0  # by definition, regardless of how L is fed

    for _ in range(n_sp):
        progressed = False
        for i in range(n_sp):
            if not np.isnan(content[i]):
                continue
            factors = []
            for j in np.where(N[i, :] > 0)[0]:
                consumed_idx = np.where(N[:, j] < 0)[0]
                if any(np.isnan(content[s]) for s in consumed_idx):
                    continue  # a precursor is not resolved yet
                l_in = sum((-N[s, j]) * content[s] for s in consumed_idx)
                factors.append(l_in / N[i, j])
            if factors:
                content[i] = float(np.mean(factors))
                progressed = True
        if not progressed:
            break
    return content


def _stream1_limiting_index(Y1, nu_reactants, Y2=None, nu_products=None):
    """Global index of the limiting reactant that termination/closure should track.

    Normally the stream-1 (f=1 feed) reactant least available relative to its
    stoichiometric demand (its largest coefficient as a reactant), restricted (when
    ``Y2`` is given) to reactants taking part in a CROSS-STREAM reaction (one with
    reactants from both feeds).  This is what termination should track: a premixed
    single-stream reactant is depleted almost instantly and would otherwise stop the
    run before the mixing-limited cross-stream chemistry develops.

    Two refinements that need the net stoichiometry (``nu_products``):
      * Catalysts and other never-net-consumed species are excluded.  A catalyst is
        regenerated, so its conversion stays 0 and the conversion event would never
        fire (the run would only stop at the steady-state / time-cap backstop).
      * If stream 1 feeds no consumable cross-stream reactant (e.g. it feeds only a
        catalyst), fall back to the consumable stream-2 reactant of the cross-stream
        reaction, so termination still keys on a species that is actually depleted.
    """
    Y1 = np.asarray(Y1, dtype=float)
    nu_reactants = np.asarray(nu_reactants, dtype=float)
    is_reactant = nu_reactants.sum(axis=1) > 0
    if nu_products is not None:
        N = np.asarray(nu_products, dtype=float) - nu_reactants
        consumable = is_reactant & (N < -1e-12).any(axis=1)   # net-consumed somewhere
    else:
        consumable = is_reactant
    s1 = (Y1 > 0) & consumable
    candidates = np.where(s1)[0]
    Yfeed = Y1
    if Y2 is not None:
        Y2 = np.asarray(Y2, dtype=float)
        Yfeed = Y1 + Y2
        s2 = (Y2 > 0) & consumable
        cross_s1, cross_s2 = set(), set()
        for j in range(nu_reactants.shape[1]):
            reac = nu_reactants[:, j] > 0
            if (reac & (Y1 > 0)).any() and (reac & (Y2 > 0)).any():   # cross-stream rxn
                cross_s1.update(int(i) for i in np.where(reac & s1)[0])
                cross_s2.update(int(i) for i in np.where(reac & s2)[0])
        if cross_s1:
            candidates = np.array(sorted(cross_s1), dtype=int)
        elif candidates.size == 0 and cross_s2:   # stream 1 has only a catalyst etc.
            candidates = np.array(sorted(cross_s2), dtype=int)
    if candidates.size == 0:
        return None
    # Least-available relative to stoichiometric demand (max coeff as reactant).
    demand = np.array([max(nu_reactants[i, :].max(), 1.0) for i in candidates])
    return int(candidates[int(np.argmin(Yfeed[candidates] / demand))])


def _blend_fs_check_runnable(unique_sp_data, stream_1_feed, stream_2_feed, blend_subsets=None):
    """Return True if blend_fs has >=2 blendable subsets (fast pre-check, no integration)."""
    if blend_subsets is None:
        return False
    from itertools import combinations as _comb
    meta = unique_sp_data['meta']
    species_list = meta['species']
    rxns = meta['reactions']
    species_data = unique_sp_data.get('species', {})
    n_rxns = len(rxns)
    nu_reactants, nu_products = parse_reactions(rxns, species_list)
    active_mask = (nu_reactants.sum(axis=1) + nu_products.sum(axis=1)) > 0
    active_indices = np.where(active_mask)[0]
    Y1 = np.asarray(stream_1_feed, dtype=float)
    Y2 = np.asarray(stream_2_feed, dtype=float)
    sp_profiles, sp_profile_arrays = {}, {}
    for sp in species_list:
        if sp not in species_data or not species_data[sp]:
            continue
        plist = list(species_data[sp].items())
        sp_profiles[sp] = plist
        sp_profile_arrays[sp] = [(np.array(prof['breakpoints'], dtype=float),
                      np.array(prof['segments'], dtype=float))
                     for _, prof in plist]
    def _psub(label):
        return [int(x) for x in str(label).split('_')]
    _avail = sorted({s for sp in sp_profiles for lbl, _ in sp_profiles[sp] for s in _psub(lbl)})
    _sub2prof = {s: {} for s in _avail}
    for sp in sp_profiles:
        for pidx, (lbl, _) in enumerate(sp_profiles[sp]):
            for s in _psub(lbl):
                if s in _sub2prof:
                    _sub2prof[s][sp] = pidx
    _is_prod = np.asarray(nu_products).sum(axis=1) > 0
    _fed = (Y1 > 0) | (Y2 > 0)
    full_products = [int(i) for i in active_indices
                     if _is_prod[i] and not _fed[i] and species_list[i] in sp_profiles]
    _fullset = set(full_products)
    def _prods_of(subnum):
        out = set()
        for i in full_products:
            p = _sub2prof[subnum].get(species_list[i])
            if p is None:
                continue
            segs = sp_profile_arrays[species_list[i]][p][1]
            if segs.size and float(np.abs(segs).max()) > 1e-12:
                out.add(i)
        return out
    _prods_by_subset = {s: _prods_of(s) for s in _avail}
    _req = [int(s) for s in blend_subsets]
    bf_subs = [s for s in _req if s in _avail]
    return len(bf_subs) >= 2


def sweep_epsilon_product_fractions(unique_sp_data, stream_1_feed, stream_2_feed,
                                    epsilons=None, rate_constants=None,
                                    mean_f=0.2, weight_methods=('linear_interp',),
                                    conversion_target=0.999, n_out=200, save_stem=None,
                                    blend_subsets=None,
                                    ode_rtol=None, ode_atol=None,
                                    reaction_orders=None, t_cap_mult=1.0e4,
                                    m_lambda=DEFAULT_M_LAMBDA, m_nu=DEFAULT_M_NU, m_Sc=DEFAULT_M_SC,
                                    raylimit_initial_ray_k=None):
    """Sweep the turbulent dissipation rate ε and plot the product fractions X_species.

    For each dissipation rate in ``epsilons`` (default 51 log-spaced points from
    1e-6 to 1e6) the species ODEs are integrated with :func:`integrate_species_odes`
    using its event-based termination: each run stops when the stream-1 limiting
    reactant reaches ``conversion_target`` fractional conversion, so CVODE — not a
    prescribed window — sets the integration time.  The stream-1 limiting-reactant
    closure is then evaluated.  The resulting per-product fractions ``X_species``
    (and the total and the conversion) are plotted against ε (log x; higher
    ε = faster mixing); multiple ``weight_methods`` are overlaid with distinct
    linestyles.

    Parameters
    ----------
    unique_sp_data, stream_1_feed, stream_2_feed, rate_constants, mean_f, n_out
        Passed straight through to :func:`integrate_species_odes`.
    epsilons : array-like, optional
        Turbulent dissipation rates to sweep.  Defaults to
        ``np.geomspace(1.0e-6, 1.0e6, 51)``.
    conversion_target : float
        Fractional conversion of the stream-1 limiting reactant at which the
        terminal event stops each integration.  Passed through to
        :func:`integrate_species_odes`.
    weight_methods : str or sequence of str
        Interpolation method(s) to evaluate.
    save_stem : path-like, optional
        If given, the figure is saved under ``<save_stem>.parent/plots/``.

    Returns
    -------
    dict
        ``{weight_method: {'epsilons', 'tau_s', 'integration_times', 'fractions'
        (species -> array), 'conversion', 'total', 'limiting_reactant'}}`` —
        where ``integration_times`` are the event-determined end times.  Ready
        for a later CSV export.  For ``weight_method == 'ray_limit'`` only,
        also includes ``'ode_results'`` — the full per-epsilon result dicts,
        for :func:`plot_raylimit_epsilon_sweep_movie`.
    """
    import pathlib as _pathlib

    if epsilons is None:
        epsilons = np.geomspace(1.0e-6, 1.0e6, 51)
    epsilons = np.asarray(epsilons, dtype=float)
    tau_s_vals = mixing_timescale(epsilons, m_lambda)
    if isinstance(weight_methods, str):
        weight_methods = (weight_methods,)

    print(f"[sweep] ε {epsilons.min():g}..{epsilons.max():g}, "
          f"integration time set by conversion event (target={100.0 * conversion_target:g}%):")
    for eps, ts in zip(epsilons, tau_s_vals):
        print(f"[sweep]   ε={eps:.4g}  ->  τ_s={ts:.4g} s")

    _all_species = None   # captured from first ODE result for consistent colouring
    sweep = {}
    for wm in weight_methods:
        if wm == 'blend_fs' and not _blend_fs_check_runnable(
                unique_sp_data, stream_1_feed, stream_2_feed,
                blend_subsets if wm == 'blend_fs' else None):
            print(f"[sweep]   [{wm}] not runnable (fewer than 2 products available "
                  f"for weighting); skipping method.")
            continue
        per_species = None
        products = None
        total = []
        conversion = []
        end_times = []
        limiting = None
        _skipped = False
        total_cpu_s = 0.0
        total_clamps = 0
        _track_fsb = wm in ('blend_fs', 'ray_limit')
        fsb_finals = [] if _track_fsb else None
        sp_clamp_pcts = {}  # {sp: [pct_per_epsilon]}
        # Keep the full per-epsilon ray_limit results (only this method needs
        # them) so the epsilon-sweep movie can show every one of the 51 runs,
        # not just the small base-case epsilon list.
        _rl_results = [] if wm == 'ray_limit' else None
        for eps in epsilons:
            res = integrate_species_odes(
                unique_sp_data, stream_1_feed, stream_2_feed,
                rate_constants=rate_constants, t_end=None, n_out=n_out,
                mean_f=mean_f, m_epsilon=float(eps), weight_method=wm,
                conversion_target=conversion_target,
                blend_subsets=blend_subsets if wm == 'blend_fs' else None,
                ode_rtol=ode_rtol, ode_atol=ode_atol,
                reaction_orders=reaction_orders, t_cap_mult=t_cap_mult,
                m_lambda=m_lambda, m_nu=m_nu, m_Sc=m_Sc,
                raylimit_initial_ray_k=raylimit_initial_ray_k)
            if not res.get('method_ran', True):
                print(f"[sweep]   [{wm}] method did not run at ε={eps:.4g}; skipping method.")
                _skipped = True
                break
            if _all_species is None:
                _all_species = res.get('species', [])
            if _rl_results is not None:
                _rl_results.append(res)
            closure = res['stream1_closure']
            limiting = closure['limiting_reactant']
            if products is None:
                products = list(closure['products'])
                per_species = {p: [] for p in products}
            for p in products:
                per_species[p].append(closure['per_product_fraction'].get(p, float('nan')))
            total.append(closure['fraction'])
            conversion.append(closure['conversion'])
            end_times.append(float(res['t'][-1]))
            total_cpu_s += float(res.get('solve_cpu_s', 0.0))
            total_clamps += _clamp_total(res)
            _n_t = len(res['t'])
            if wm == 'blend_fs':
                _lo_d = res.get('blendfs', {}).get('clamp_below_M', {})
                _hi_d = res.get('blendfs', {}).get('clamp_above_B', {})
            elif wm == 'ray_limit':
                _lo_d = res.get('raylimit', {}).get('clamp_below_M', {})
                _hi_d = res.get('raylimit', {}).get('clamp_above_B', {})
            elif wm == 'linear_interp':
                _lo_d = res.get('li_clamps', {}).get('clamp_below_min', {})
                _hi_d = res.get('li_clamps', {}).get('clamp_above_max', {})
            else:
                _lo_d, _hi_d = {}, {}
            for _sp in set(_lo_d) | set(_hi_d):
                _pct = 100.0 * (_lo_d.get(_sp, 0) + _hi_d.get(_sp, 0)) / _n_t
                sp_clamp_pcts.setdefault(_sp, []).append(_pct)
            if _track_fsb:
                if wm == 'blend_fs':
                    _arr = res.get('blendfs', {}).get('fsb')
                else:
                    _arr = res.get('raylimit', {}).get('fsb')
                if _arr is not None and len(_arr) > 0:
                    _fin = float(_arr[-1])
                else:
                    _fin = float('nan')
                fsb_finals.append(_fin)
        if _skipped:
            continue
        sweep[wm] = {
            'epsilons': epsilons,
            'tau_s': tau_s_vals,
            'integration_times': np.array(end_times),
            'fractions': {p: np.array(v) for p, v in (per_species or {}).items()},
            'conversion': np.array(conversion),
            'total': np.array(total),
            'limiting_reactant': limiting,
            'total_cpu_s': total_cpu_s,
            'total_clamps': total_clamps,
        }
        if _track_fsb:
            sweep[wm]['fsb_finals'] = np.array(fsb_finals)
        sweep[wm]['sp_clamp_pcts'] = {sp: np.array(v) for sp, v in sp_clamp_pcts.items()}
        if _rl_results is not None:
            sweep[wm]['ode_results'] = _rl_results

    if not sweep:
        print("[sweep] no method produced results; skipping plot.")
        return sweep

    # ── Plot X_species vs dissipation rate ε ──────────────────────────────────
    all_products = sorted({p for s in sweep.values() for p in s['fractions']})
    _present_wms_early = [wm for wm in weight_methods if wm in sweep]
    _has_fsb = any('fsb_finals' in sweep.get(wm, {}) for wm in _present_wms_early)
    if _has_fsb:
        fig, (ax, ax_chk, ax_fs) = plt.subplots(3, 1, figsize=(8, 12),
                                                gridspec_kw={'height_ratios': [1.5, 1, 1]})
    else:
        fig, (ax, ax_chk) = plt.subplots(2, 1, figsize=(8, 9),
                                         gridspec_kw={'height_ratios': [1.5, 1]})
        ax_fs = None
    _sweep_palette = _active_palette()
    color = {p: _sp_color(p, _all_species or [], _sweep_palette) for p in all_products}
    # Per-method visual encoding: linestyle, marker shape, fill (None=filled / 'none'=hollow),
    # and marker edge width, from the shared _WM_STYLE mapping (same symbol for
    # a given method in every plot).  Filled vs. hollow also gets a slightly
    # heavier/lighter line weight so methods stay distinguishable even when two
    # curves share the same product colour.
    _ms = 6   # marker size (uniform)
    # Conversion and TOTAL (closure checks) live on their own subplot below.
    _present_wms = _present_wms_early
    eps_x = sweep[_present_wms[0]]['epsilons']
    for m_idx, wm in enumerate(weight_methods):
        if wm not in sweep:
            continue
        st = _wm_style(wm)
        ls, mk, mew = st['ls'], st['mk'], st['mew']
        lw = 2.0 if st['mfc'] is None else 1.8
        s = sweep[wm]
        for p in all_products:
            if p not in s['fractions']:
                continue
            _mfc = color[p] if st['mfc'] is None else _face(color[p])
            ax.plot(s['epsilons'], s['fractions'][p], ls,
                    marker=mk, color=color[p], mfc=_mfc, mew=mew, ms=_ms, lw=lw)
        # Fractional conversion of the limiting reactant (close to 1 in many cases).
        _mfc_gray = 'tab:gray' if st['mfc'] is None else _face('tab:gray')
        ax_chk.plot(s['epsilons'], s['conversion'], ls,
                    marker=mk, color='tab:gray', mfc=_mfc_gray, mew=mew, ms=_ms, lw=1.5)
        _mfc_blk = 'black' if st['mfc'] is None else _face('black')
        ax_chk.plot(s['epsilons'], s['total'], ls,
                    marker=mk, color='black', mfc=_mfc_blk, mew=mew, ms=_ms, lw=2.0)

    lim_names = ', '.join(sorted({str(s['limiting_reactant']) for s in sweep.values()}))
    # Label every decade across the swept range.
    import matplotlib.ticker as _mticker
    _lo = int(np.floor(np.log10(epsilons.min())))
    _hi = int(np.ceil(np.log10(epsilons.max())))
    _decades = np.arange(_lo, _hi + 1)
    for _a in (ax, ax_chk):
        _a.set_xscale('log')
        _a.set_xticks(10.0 ** _decades.astype(float))
        _a.set_xticklabels([f'$10^{{{d}}}$' for d in _decades])
        _a.xaxis.set_minor_locator(_mticker.NullLocator())
        _a.set_xlabel('turbulent dissipation rate  ε  (W/kg)')
    # Tick marks every 0.1 on both y-axes.
    ax.yaxis.set_major_locator(_mticker.MultipleLocator(0.1))
    ax_chk.yaxis.set_major_locator(_mticker.MultipleLocator(0.1))
    ax_chk.set_ylim(0.0, 1.1)
    ax.set_ylabel('$X_{\\mathrm{species}}$')
    ax_chk.set_ylabel('conversion / closure total  (X_conv, TOTAL)')
    ax.set_title(f'Product fractions vs dissipation rate ε  '
                 f'(mean_f={mean_f:.4f},  limiting reactant {lim_names},  '
                 f'integration to {conversion_target:g} conversion)')
    # Legend: colours → products, markers/linestyles → methods.
    from matplotlib.lines import Line2D
    leg_h, leg_l = [], []
    for p in all_products:
        leg_h.append(Line2D([0], [0], color=color[p], lw=2.0))
        leg_l.append(f'$X_{{{p}}}$')
    if len(_present_wms_early) > 1:
        for m_idx, wm in enumerate(weight_methods):
            if wm not in sweep:
                continue
            st = _wm_style(wm)
            _mfc_leg = 'k' if st['mfc'] is None else _face('k')
            leg_h.append(Line2D([0], [0], color='k', lw=2.0 if st['mfc'] is None else 1.8,
                                 ls=st['ls'], marker=st['mk'],
                                 ms=_ms, mfc=_mfc_leg, mew=st['mew']))
            leg_l.append(wm)
    ax.legend(leg_h, leg_l, loc='best', frameon=False, ncol=2, fontsize=11, handlelength=2.5)
    # Conversion/TOTAL subplot gets its own small legend (method encoding already
    # explained in the plot above).
    chk_h = [Line2D([0], [0], color='black', lw=2.0),
             Line2D([0], [0], color='tab:gray', lw=1.5)]
    chk_l = ['TOTAL', 'X_conv']
    ax_chk.legend(chk_h, chk_l, loc='best', frameon=False, fontsize=11)

    # ── fs_final subplot ──────────────────────────────────────────────────────
    if ax_fs is not None:
        for m_idx, wm in enumerate(weight_methods):
            if wm not in sweep or 'fsb_finals' not in sweep[wm]:
                continue
            st = _wm_style(wm)
            s = sweep[wm]
            _mfc_fs = 'black' if st['mfc'] is None else _face('black')
            ax_fs.plot(s['epsilons'], s['fsb_finals'], st['ls'],
                       marker=st['mk'], color='black',
                       mfc=_mfc_fs, mew=st['mew'], ms=_ms,
                       lw=2.0 if st['mfc'] is None else 1.8,
                       label=wm)
        ax_fs.set_xscale('log')
        ax_fs.set_xticks(10.0 ** _decades.astype(float))
        ax_fs.set_xticklabels([f'$10^{{{d}}}$' for d in _decades])
        ax_fs.xaxis.set_minor_locator(_mticker.NullLocator())
        ax_fs.set_xlabel('turbulent dissipation rate  ε  (W/kg)')
        ax_fs.set_ylabel('$f_s^*$  (final)')
        ax_fs.legend(loc='best', frameon=False, fontsize=11)

    fig.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_Xspecies_vs_epsilon.png')
        _save_sweep_csv(sweep, weight_methods, all_products, save_stem)
        _plot_and_save_clamp_pcts(sweep, weight_methods, save_stem, _all_species or [])
    else:
        plt.show()

    _present_wms = [wm for wm in weight_methods if wm in sweep]
    _col = 50
    _bar = '=' * _col
    print(f"\n{_bar}")
    print(f"[sweep summary] ε sweep — {len(epsilons)} points, all methods combined")
    print(f"[sweep summary] {'method':<20} {'CPU (s)':>10} {'clamp steps':>12}")
    print(f"[sweep summary] {'-' * (_col - 16)}")
    for wm in _present_wms:
        s = sweep[wm]
        print(f"[sweep summary] {wm:<20} {s['total_cpu_s']:>10.4g} {s['total_clamps']:>12d}")
    print(_bar)

    return sweep


def _save_sweep_csv(sweep, weight_methods, all_products, save_stem):
    """Write the plotted ε-sweep points (X_species, X_conv, TOTAL) to a CSV.

    One row per ε; columns are epsilon, tau_s, integration_time and, for each
    weight method, X_<product> for every product plus X_conv and TOTAL.  Column
    names are suffixed with the method when more than one is swept.
    """
    import csv
    import pathlib as _pathlib

    present = [wm for wm in weight_methods if wm in sweep]
    multi = len(present) > 1
    epsilons = sweep[present[0]]['epsilons']
    tau_s = sweep[present[0]]['tau_s']
    integration_times = sweep[present[0]]['integration_times']

    header = ['epsilon', 'tau_s', 'integration_time']
    for wm in present:
        suffix = f' ({wm})' if multi else ''
        header += [f'X_{p}{suffix}' for p in all_products]
        header += [f'X_conv{suffix}', f'TOTAL{suffix}']
        if 'fsb_finals' in sweep[wm]:
            header += [f'fs_final{suffix}']

    rows = []
    for i, eps in enumerate(epsilons):
        row = [eps, tau_s[i], integration_times[i]]
        for wm in present:
            s = sweep[wm]
            for p in all_products:
                arr = s['fractions'].get(p)
                row.append(arr[i] if arr is not None else float('nan'))
            row.append(s['conversion'][i])
            row.append(s['total'][i])
            if 'fsb_finals' in s:
                row.append(s['fsb_finals'][i])
        rows.append(row)

    plots_dir = _pathlib.Path(save_stem).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    path = plots_dir / f'{_pathlib.Path(save_stem).name}_Xspecies_vs_epsilon.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved to {path}")


def _save_base_case_epsilon_csv(all_base_results, weight_methods, save_stem):
    """Write the final per-epsilon product fractions (X_<product>, X_conv,
    TOTAL) from the base-case runs (the config's 'epsilon' list, not the
    --sweep geomspace) to a CSV -- one row per epsilon, columns suffixed by
    weight method when more than one ran."""
    import csv
    import pathlib as _pathlib

    by_wm = {}
    for r in all_base_results:
        closure = r.get('stream1_closure', {})
        if closure.get('limiting_reactant') is None:
            continue
        by_wm.setdefault(r.get('weight_method'), {})[r['m_epsilon']] = closure

    present = [wm for wm in weight_methods if wm in by_wm]
    if not present:
        return
    epsilons = sorted({eps for wm in present for eps in by_wm[wm]})

    all_products = []
    for wm in present:
        for closure in by_wm[wm].values():
            for p in closure['products']:
                if p not in all_products:
                    all_products.append(p)

    multi = len(present) > 1
    header = ['epsilon']
    for wm in present:
        suffix = f' ({wm})' if multi else ''
        header += [f'X_{p}{suffix}' for p in all_products]
        header += [f'X_conv{suffix}', f'TOTAL{suffix}']

    rows = []
    for eps in epsilons:
        row = [eps]
        for wm in present:
            closure = by_wm[wm].get(eps)
            for p in all_products:
                row.append(closure['per_product_fraction'].get(p, float('nan')) if closure else float('nan'))
            row.append(closure['conversion'] if closure else float('nan'))
            row.append(closure['fraction'] if closure else float('nan'))
        rows.append(row)

    plots_dir = _pathlib.Path(save_stem).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    path = plots_dir / f'{_pathlib.Path(save_stem).name}_product_fractions_by_epsilon.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved to {path}")


def _plot_and_save_clamp_pcts(sweep, weight_methods, save_stem, all_species=None):
    """Separate figure + CSV: per-species clamp % vs ε for each weight method."""
    import csv as _csv
    import pathlib as _pathlib

    present = [wm for wm in weight_methods if wm in sweep and sweep[wm].get('sp_clamp_pcts')]
    if not present:
        return

    epsilons = sweep[present[0]]['epsilons']
    _lo = int(np.floor(np.log10(epsilons.min())))
    _hi = int(np.ceil(np.log10(epsilons.max())))
    _decades = np.arange(_lo, _hi + 1)
    import matplotlib.ticker as _mticker

    # ── figure: one subplot per method ────────────────────────────────────────
    n_m = len(present)
    fig, axes = plt.subplots(n_m, 1, figsize=(8, 3.5 * n_m), squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(8.0, 3.5)

    for row, wm in enumerate(present):
        ax = axes[row][0]
        scp = sweep[wm]['sp_clamp_pcts']
        all_sp = sorted(scp)
        for sp in all_sp:
            ax.plot(epsilons, scp[sp], marker=_sp_marker(sp, all_species or []),
                    color=_sp_color(sp, all_species or []),
                    linestyle=_sp_linestyle(sp, all_species or []),
                    markevery=max(1, len(epsilons) // 15), mfc=_face(_sp_color(sp, all_species or [])),
                    ms=_ms, lw=_lw, label=sp)
        ax.set_xscale('log')
        ax.set_xticks(10.0 ** _decades.astype(float))
        ax.set_xticklabels([f'$10^{{{d}}}$' for d in _decades])
        ax.xaxis.set_minor_locator(_mticker.NullLocator())
        ax.set_xlabel('turbulent dissipation rate  ε')
        ax.set_ylabel('% steps clamped')
        ax.set_ylim(bottom=0)
        ax.set_title(wm, fontsize=10)
        ax.legend(fontsize=_leg_fs, loc='best', frameon=False)

    fig.tight_layout()
    _save_fig(fig, save_stem,
              f'{_pathlib.Path(save_stem).name}_clamp_pct_vs_epsilon.png')

    # ── CSV ───────────────────────────────────────────────────────────────────
    all_sp_all = sorted({sp for wm in present for sp in sweep[wm]['sp_clamp_pcts']})
    multi = len(present) > 1
    header = ['epsilon']
    for wm in present:
        suffix = f' ({wm})' if multi else ''
        header += [f'{sp}_clamp_pct{suffix}' for sp in all_sp_all]

    rows = []
    for i, eps in enumerate(epsilons):
        row = [eps]
        for wm in present:
            scp = sweep[wm]['sp_clamp_pcts']
            for sp in all_sp_all:
                arr = scp.get(sp)
                row.append(float(arr[i]) if arr is not None else float('nan'))
        rows.append(row)

    plots_dir = _pathlib.Path(save_stem).parent / 'plots'
    plots_dir.mkdir(exist_ok=True)
    path = plots_dir / f'{_pathlib.Path(save_stem).name}_clamp_pct_vs_epsilon.csv'
    with open(path, 'w', newline='') as f:
        writer = _csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved to {path}")


def integrate_species_odes(unique_sp_data, stream_1_feed, stream_2_feed,
                            rate_constants=None, t_end=None, n_out=200,
                            mean_f=0.5, m_epsilon=DEFAULT_M_EPSILON,
                            m_lambda=DEFAULT_M_LAMBDA, m_nu=DEFAULT_M_NU, m_Sc=DEFAULT_M_SC,
                            weight_method='linear_interp',
                            conversion_target=0.999, blend_subsets=None,
                            ode_rtol=None, ode_atol=None,
                            reaction_orders=None,
                            raylimit_fallback_tol=1e-6,
                            raylimit_fallback_conv_ceiling=0.01,
                            raylimit_initial_ray_k=None,
                            rate_ramp=None, t_cap_mult=1.0e4):
    """Integrate species concentration ODEs with mixing-dependent reaction rates.

    State y_i(t) is the concentration of species i.  At each step the
    mixing state is characterised by a Beta(alpha(t), beta(t)) distribution
    whose variance decays as var_start·Ios(t), set by the turbulent dissipation
    rate ``m_epsilon`` (see :func:`mixing_variance`), floored at max_var * 1e-6.

    Weights for the blended C_w_i(f) profile are recomputed every step so
    that the beta-weighted average of C_w_i equals y_i(t).
    The rate of reaction j is then the exact beta-weighted integral of
    k_j * prod_i C_w_i(f)^nu_ij over [0, 1].

    Initial conditions: y_i(0) = (1 - mean_f)*stream_1_feed[i] + mean_f*stream_2_feed[i]
    — the unreacted linear mix at the mean mixture fraction.

    Integration end time
    --------------------
    When ``t_end is None`` (the default) the integration time is not prescribed:
    a CVODE terminal *event* stops the solve when the stream-1 limiting reactant
    reaches ``conversion_target`` fractional conversion (default 0.999), so the
    solver decides the end time from the solution itself.  A generous safety cap
    (1e4·tau_s) bounds the run if the target is never reached.  Passing an
    explicit ``t_end`` restores fixed-time integration with no event (used by the
    ε sweep, which gives every run the same mixing-time budget).

    ``raylimit_fallback_tol`` / ``raylimit_fallback_conv_ceiling`` (ray_limit
    only): ``_selectivity_ray`` uses the mass-action-rate fallback
    direction instead of the accumulated extent Δ=y-y0 only while BOTH (a) the
    absolute |Δ| is below ``raylimit_fallback_tol``, AND (b) the fractional
    conversion of whichever fed species has changed most (relative to its own
    feed magnitude) is below ``raylimit_fallback_conv_ceiling``.  (b) is the
    scale-invariant safety net: (a) alone is an absolute threshold and so isn't
    reliably small relative to a given system's concentration scale.

    ``raylimit_initial_ray_k`` (ray_limit only, default None): while the
    mass-action-rate fallback direction above is in effect, use this value as
    every reaction's rate constant instead of its own k_j.  ``None`` uses the
    real k_j.  Set from the JSON config's ``initial_ray`` key.

    ``rate_ramp``: optional callable ``t -> multiplier``, applied uniformly to
    every reaction's rate constant each step (``k_vals = k_vals_base *
    rate_ramp(t)``) — e.g. for stress-testing how a closure responds to
    reactions accelerating over the course of a single run.  A UNIFORM
    multiplier leaves each reaction's rate *relative* to the others unchanged
    (it's equivalent to a local rescaling of time), but it does change the
    reaction rate relative to the fixed mixing timescale as the run
    proceeds.  ``None`` (default) leaves rate constants fixed at their
    original values throughout.
    """
    from sksundae.cvode import CVODE
    betainc = _betainc_top

    meta = unique_sp_data['meta']
    species_list = meta['species']
    rxns = meta['reactions']
    species_data = unique_sp_data.get('species', {})
    n_sp = len(species_list)

    nu_reactants, nu_products = parse_reactions(rxns, species_list)
    nu_net = (nu_products - nu_reactants).astype(float)
    rxn_labels = [r.split(':')[0].strip() for r in rxns]
    n_rxns = len(rxn_labels)
    k_vals_base = np.array([(rate_constants or {}).get(lbl, 1000.0) for lbl in rxn_labels])
    k_vals = k_vals_base.copy()  # rescaled per-step by `rate_ramp(t)` in _rates_at, if given

    # Only integrate species that appear in at least one reaction
    active_mask = (nu_reactants.sum(axis=1) + nu_products.sum(axis=1)) > 0
    active_indices = np.where(active_mask)[0]  # global indices into species_list
    n_active = len(active_indices)
    nu_net_active = nu_net[active_indices, :]   # shape (n_active, n_rxns)

    max_var   = mean_f * (1.0 - mean_f)
    var_start = max_var * (1.0 - 1e-6)
    var_end   = max_var * 1e-6

    # ── Precompute per-species profile data ───────────────────────────────
    sp_profile_arrays = {}   # sp -> [(bps_arr, segs_arr), ...]  one entry per unique profile
    sp_profiles = {}   # sp -> [(label, profile_dict), ...]
    for sp in species_list:
        if sp not in species_data or not species_data[sp]:
            continue
        plist = list(species_data[sp].items())
        sp_profiles[sp] = plist
        sp_profile_arrays[sp] = [(np.array(prof['breakpoints'], dtype=float),
                      np.array(prof['segments'], dtype=float))
                     for _, prof in plist]

    # Union of all species breakpoints → integration sub-intervals
    all_bps_set = {0.0, 1.0}
    for pd_list in sp_profile_arrays.values():
        for bps, _ in pd_list:
            all_bps_set.update(bps.tolist())
    all_bps = np.array(sorted(all_bps_set))
    n_segs = len(all_bps) - 1

    # ── Initial conditions: linear mix of stream feeds at mean_f ─────────
    Y1 = np.asarray(stream_1_feed, dtype=float)
    Y2 = np.asarray(stream_2_feed, dtype=float)
    # Convention: f=1 is stream 1, f=0 is stream 2
    y0_full = mean_f * Y1 + (1.0 - mean_f) * Y2
    y0 = y0_full[active_indices]

    # ── Reaction order (rate-law exponents) ───────────────────────────────
    # Priority: JSON "orders" list > "elementary" keyword on reaction label > default 1.
    # "orders" is a list aligned with reactions; each entry is either:
    #   • a dict  {species_name: exponent} — per-reactant orders; missing reactants → 1
    #   • a number — all reactants in that reaction get that exponent
    #   • null/None — fall through to elementary/default for that reaction
    elementary = [_strip_elementary(lbl)[1] for lbl in rxns]
    if any(elementary):
        _el = [rxn_labels[j] for j in range(n_rxns) if elementary[j]]
        print(f"[kinetics] elementary (order = stoichiometry) reactions: {_el}; "
              f"all other reactants enter the rate law at order 1.")

    def _get_order(j, i):
        """Rate-law exponent for species i in reaction j.  Always returns int."""
        def _int_or_float(v):
            v = float(v)
            return int(v) if v == int(v) else v

        if reaction_orders is not None and j < len(reaction_orders):
            entry = reaction_orders[j]
            if entry is not None:
                if isinstance(entry, dict):
                    return _int_or_float(entry.get(species_list[i], 1))
                return _int_or_float(entry)
        return int(nu_reactants[i, j]) if elementary[j] else 1

    if reaction_orders is not None:
        _ord_log = []
        for j in range(n_rxns):
            entry = reaction_orders[j] if j < len(reaction_orders) else None
            if entry is not None:
                _ord_log.append(f"{rxn_labels[j]}: {entry}")
        if _ord_log:
            print(f"[kinetics] custom reaction orders from JSON: " + "; ".join(_ord_log))

    # ── Precompute which species (by global index) are reactants per rxn, paired
    #    with the rate-law order to which each reactant is raised ──
    rxn_reactants = [
        [(i, _get_order(j, i))
         for i in range(n_sp) if nu_reactants[i, j] > 0]
        for j in range(n_rxns)
    ]
    # Map global species index → position in active y vector (or None)
    global_to_active = {int(i): ai for ai, i in enumerate(active_indices)}
    # Map species name → global index (avoids O(n) list.index in the recording loop)
    sp_to_global = {sp: i for i, sp in enumerate(species_list)}

    # ── Precompute hot-path geometry (independent of alpha/beta/weights) ────
    # sp_bp_positions[sp][n] : indices into all_bps for profile n's breakpoints.
    # sp_seg_MB[sp]    : array [n_profiles, n_segs, 2] of (slope, intercept) of
    #                    each profile on each shared all_bps sub-interval.
    sp_bp_positions = {}
    sp_seg_MB = {}
    for sp, pd_list in sp_profile_arrays.items():
        sp_bp_positions[sp] = [np.searchsorted(all_bps, bps) for (bps, _segs) in pd_list]
        mb = np.zeros((len(pd_list), n_segs, 2))
        for n, (bps, segs) in enumerate(pd_list):
            for seg_k in range(n_segs):
                f_mid = 0.5 * (all_bps[seg_k] + all_bps[seg_k + 1])
                idx = min(int(np.searchsorted(bps, f_mid, side='right')) - 1, len(segs) - 1)
                idx = max(idx, 0)
                mb[n, seg_k] = segs[idx]
        sp_seg_MB[sp] = mb

    # Highest betainc order needed: 1 for E[C_n]; the rate-poly degree is the sum
    # of the reactant orders (each order-k factor adds k to the polynomial degree).
    _max_deg = int(max(max((sum(o for _, o in r) for r in rxn_reactants), default=1), 1))


    # ── blend_fs-method precompute ─────────────────────────────────────────
    # blend_fs is the product-weighted blend of the subsets listed in the JSON
    # config's ``blend_subsets`` key, built with the strict fs rule (single
    # kink at fs_blend = w_a*fs_a + w_b*fs_b + ...); the rate then interpolates
    # that blend against the no-reaction limit by the mean.  Weights come from
    # each subset's distinguishing products -- the species it produces that
    # the other blended subsets do not.  Everything that is constant in time
    # (the subsets, their fs, and each species' values at f=0, f=1 and at
    # each subset's own fs) is precomputed here.  blend_fs only runs when
    # >=2 valid subsets are given in blend_subsets.
    is_blendfs = weight_method == 'blend_fs'
    blendfs_ok = False
    if is_blendfs:
        import contextlib as _ctx
        import io as _io_bf
        from itertools import combinations as _comb_bf
        _enum_bf = [()] + [c for size in range(1, n_rxns + 1) for c in _comb_bf(range(n_rxns), size)]

        def _psub(label):
            return [int(x) for x in str(label).split('_')]

        _avail = sorted({s for sp in sp_profiles for lbl, _ in sp_profiles[sp] for s in _psub(lbl)})
        _sub2prof = {s: {} for s in _avail}   # subnum -> {species_name: profile idx}
        for sp in sp_profiles:
            for pidx, (lbl, _) in enumerate(sp_profiles[sp]):
                for s in _psub(lbl):
                    if s in _sub2prof:
                        _sub2prof[s][sp] = pidx
        _is_prod = np.asarray(nu_products).sum(axis=1) > 0
        _fed = (np.asarray(Y1) > 0) | (np.asarray(Y2) > 0)
        full_products = [int(i) for i in active_indices
                         if _is_prod[i] and not _fed[i] and species_list[i] in sp_profiles]

        def _prods_of(subnum):
            out = set()
            for i in full_products:
                p = _sub2prof[subnum].get(species_list[i])
                if p is None:
                    continue
                segs = sp_profile_arrays[species_list[i]][p][1]
                if segs.size and float(np.abs(segs).max()) > 1e-12:
                    out.add(i)
            return out

        _fullset = set(full_products)
        _prods_by_subset = {s: _prods_of(s) for s in _avail}
        # Which subsets to blend: exactly those listed in blend_subsets.
        if blend_subsets is None:
            bf_subs = []
        else:
            _req = [int(s) for s in blend_subsets]
            bf_subs = [s for s in _req if s in _avail]
            _bad = [s for s in _req if s not in _avail]
            if _bad:
                print(f"[blend_fs]   [blend] blend_subsets {_bad} not in the available pool "
                      f"{sorted(_avail)}; ignoring them.")
        if len(bf_subs) >= 2:
            def _ana(subnum):
                idx = list(_enum_bf[subnum])
                with _ctx.redirect_stdout(_io_bf.StringIO()):
                    res = analyze_stream_limit_system(
                        species_list, [rxns[i] for i in idx],
                        nu_reactants[:, idx], nu_products[:, idx], Y1, Y2)
                    seg = generate_line_segments(species_list, Y1, Y2, res)
                return res['fs'], seg['breakpoints'], seg['all_reactions']

            bf_fs, bf_distinguishing, bf_sizes, bf_misses = [], [], [], []
            bf_bp_list, bf_segs = [], []   # per subset: own breakpoints, {sp: segment list}
            _ok = True
            for k, s in enumerate(bf_subs):
                fs_s, bps_s, segs_s = _ana(s)
                if fs_s is None:
                    _ok = False
                    break
                bf_fs.append(fs_s)
                bf_bp_list.append(bps_s)
                bf_segs.append(segs_s)
                # Distinguishing products that weight this subset: products it
                # makes that the OTHER blended subsets do not.
                _others = set().union(*[_prods_by_subset[o] for o in bf_subs if o != s]) \
                    if len(bf_subs) > 1 else set()
                bf_distinguishing.append(sorted(_prods_by_subset[s] - _others))
                bf_sizes.append(len(_enum_bf[s]))
                bf_misses.append([species_list[i] for i in (_fullset - _prods_by_subset[s])])
            blendfs_ok = _ok
            if blendfs_ok:
                bf_n = len(bf_subs)
                print(f"[{weight_method}]   [blend] blend_subsets blend of subsets {bf_subs}: " + "; ".join(
                    f"subset {bf_subs[k]} (fs={bf_fs[k]:.4f}, distinguishing "
                    f"{[species_list[i] for i in bf_distinguishing[k]]})" for k in range(bf_n)))
                # Each blended subset's own profile (breakpoints, segment values)
                # is fixed -- computed once here from stoichiometry alone, not
                # per time-step.  Only the blend WEIGHTS w[k](t) vary with time
                # (see _blendfs_weights), so rather than collapsing each subset
                # to 3 control points (f=0, its own fs, f=1) and losing any
                # interior kink a catalytic reaction needs (e.g. the near-f=0
                # step where a catalyst confined to one stream first becomes
                # present), take the union of every blended subset's FULL
                # breakpoint set once, and precompute each subset's value at
                # every point in that union once -- then each step's blended
                # profile is just a weighted sum (bf_vals[sp] @ w) over this
                # fixed grid, no re-evaluation needed.
                bf_bp = np.array(sorted(set().union(*[set(np.round(bp, 12)) for bp in bf_bp_list])))
                bf_n_bp = len(bf_bp)
                # bf_vals[sp] has shape (bf_n_bp, bf_n): value of subset k's own
                # profile at each unioned breakpoint.
                bf_vals = {sp: np.array([[_eval_segments(bf_segs[k][sp], f) for k in range(bf_n)]
                                         for f in bf_bp])
                          for sp in species_list}
        if not blendfs_ok:
            if blend_subsets is None:
                print(f"[{weight_method}]   [blend] blend_fs requires 'blend_subsets' "
                      f"in the config; none given, method will not run.")
            else:
                print(f"[{weight_method}]   [blend] need >=2 blendable subsets "
                      f"(found {len(bf_subs)}); method will not run.")

    # ── ray_limit-method precompute ───────────────────────────────────────
    # ray_limit builds ONE complete-reaction limit each step whose reaction
    # selectivity is taken from the CURRENT reaction rates (mass action at the
    # current mean composition).  The mixing line M(f) is reacted to completion
    # along that fixed selectivity direction d: the limiting reactants reach 0 at a
    # single fs, products peak there, and because C − M = N·(c·d) with c·d ≥ 0 the
    # limit is stoichiometrically exact by construction.  No subset enumeration is
    # needed — the "blended subset" is whatever the live solution implies.
    is_raylimit = (weight_method == 'ray_limit')
    raylimit_ok = is_raylimit and (n_rxns > 0) and (n_active > 0)
    if raylimit_ok:
        print(f"[ray_limit]   [selectivity] from current reaction rates; "
              f"{n_rxns} reactions over {n_active} active species.")
        _reactant_rows = np.where(nu_reactants.sum(axis=1) > 0)[0]
        _impure_feeds = [species_list[i] for i in _reactant_rows
                         if Y1[i] > 0 and Y2[i] > 0]
        if _impure_feeds:
            raise ValueError(
                f"ray_limit requires every reactant to be fed by exactly one "
                f"stream; {', '.join(_impure_feeds)} appear(s) in both streams "
                f"(stream_1 and stream_2 both > 0). Reformulate the feeds so "
                f"each reactant is pure, or use a different weight_method.")
        for _cat_r in range(n_rxns):
            if catalyst_species(nu_reactants, nu_products, _cat_r).size == 0:
                continue
            for _i in np.where(nu_net[:, _cat_r] > 0)[0]:
                _downstream = [rxn_labels[_r2] for _r2 in range(n_rxns)
                               if _r2 != _cat_r and nu_reactants[_i, _r2] > 0]
                if _downstream:
                    print(f"[ray_limit]   note: {species_list[_i]}, a product of catalytic "
                          f"reaction {rxn_labels[_cat_r]}, is also a reactant in "
                          f"{', '.join(_downstream)}; its C(f) shape near f=0/1 feeds into "
                          f"that reaction's rate here, so compare the LP species-limit grid "
                          f"against the ray_limit limit grid for {species_list[_i]}.")
    elif is_raylimit:
        print("[ray_limit]   [selectivity] no active reactions/species; "
              "falling back to linear_interp behaviour.")

    # Feed vectors restricted to active species (for the mixing line M(f)),
    # needed by ray_limit's _raylimit_limit call.
    if raylimit_ok:
        rl_Y1a = np.array([float(Y1[i]) for i in active_indices])
        rl_Y2a = np.array([float(Y2[i]) for i in active_indices])
        # Per-species feed scale (larger of its two stream feeds) and which
        # species are fed at all — used by _selectivity_ray's conversion-based
        # fallback ceiling below, so the switch-over is scale-invariant rather
        # than relying solely on the absolute |Δ| threshold.
        _rl_feed_scale = np.maximum(rl_Y1a, rl_Y2a)
        _rl_fed_mask = _rl_feed_scale > 1e-300

    # ── Local helpers ─────────────────────────────────────────────────────
    def _e_vals(sp, Bt, a_over_ab):
        """Beta-weighted average of each unique profile for *sp* (via Bt)."""
        pd_list = sp_profile_arrays.get(sp)
        if pd_list is None:
            return np.array([])
        idx_list = sp_bp_positions[sp]
        ev = np.zeros(len(pd_list))
        for n, (bps, segs) in enumerate(pd_list):
            bidx = idx_list[n]
            for j in range(len(segs)):
                lo_i, hi_i = bidx[j], bidx[j + 1]
                dI0 = Bt[0, hi_i] - Bt[0, lo_i]
                dI1 = Bt[1, hi_i] - Bt[1, lo_i]
                ev[n] += segs[j, 0] * a_over_ab * dI1 + segs[j, 1] * dI0
        return ev

    def _poly_beta_int(poly, Bt, ratio, lo_i, hi_i):
        """Integrate polynomial (highest-degree first) * Beta on [all_bps[lo_i], all_bps[hi_i]]."""
        degree = len(poly) - 1
        total = 0.0
        for k_from_top, c in enumerate(poly):
            if abs(c) < 1e-300:
                continue
            d = degree - k_from_top
            dI = Bt[d, hi_i] - Bt[d, lo_i]
            total += c * ratio[d] * dI
        return total

    def _seg_linear(sp, weights, seg_k):
        """Slope and intercept of C_w_i(f) on all_bps sub-interval seg_k."""
        mb = sp_seg_MB.get(sp)
        if mb is None or len(weights) == 0:
            return 0.0, 0.0
        M, B = weights @ mb[:, seg_k, :]
        return M, B

    # ── blend_fs rate path: interpolate the product-weighted fs-rule blend of the
    #    blended subsets' limits against the no-reaction limit, by the mean ─────
    def _blendfs_weights(y_active):
        """(w[k] over the blended subsets, fs_blend) from each subset's
        distinguishing products' ODE amounts."""
        amts = np.array([
            sum(max(float(y_active[global_to_active[i]]), 0.0) for i in bf_distinguishing[k])
            for k in range(bf_n)])
        tot = float(amts.sum())
        if tot > 1e-300:
            w = amts / tot
        else:  # no distinguishing product yet -> all weight to the smallest subset
            w = np.zeros(bf_n)
            w[int(np.argmin(bf_sizes))] = 1.0
        bf_fs_arr = np.array(bf_fs)
        s_k       = bf_fs_arr / (1.0 - bf_fs_arr)   # odds ratio: fs/(1-fs) for each subset
        s_blend   = float(np.dot(w, s_k))            # blend A:B consumption ratios
        fsb = min(1.0 - 1e-9, max(1e-9, s_blend / (1.0 + s_blend)))
        return w, fsb

    _blendfs_lam_reported = [0]   # count out-of-range lam0 prints; cap at 10

    def _rates_blendfs(alpha_t, beta_t, ab, a_over_ab, y_active):
        w, fsb = _blendfs_weights(y_active)

        # Betainc table over the FULL union of every blended subset's own
        # breakpoints (bf_bp, fixed -- precomputed above), not just
        # [0, fsb, 1].  Collapsing to 3 points discards any interior kink a
        # catalytic reaction needs (e.g. the near-f=0 step where a catalyst
        # confined to one stream first becomes present), smearing it into a
        # long diagonal and biasing E[B]; using the full grid instead keeps
        # each blended subset's real shape intact.
        Bt = np.empty((_max_deg + 1, bf_n_bp))
        for k in range(_max_deg + 1):
            Bt[k] = betainc(alpha_t + k, beta_t, bf_bp)
        rat = np.empty(_max_deg + 1)
        rat[0] = 1.0
        for d in range(1, _max_deg + 1):
            rat[d] = rat[d - 1] * (alpha_t + (d - 1)) / (ab + (d - 1))

        n_bseg = bf_n_bp - 1
        # Per active species: C_i = B_i + (M_i - B_i)·λ, where M_i is the no-reaction
        # line and B_i the blended limit (piecewise-linear over bf_bp).
        # λ = (y_i - E[B_i])/(E[M_i] - E[B_i]) matches the mean so E[C_i] = y_i.
        Cseg = {}
        # Per-species info for recording/plotting:
        #   info[sp] = (E[B_i], raw λ, clamped λ, v0, vfs, v1, v_arr)
        # v0/v1/vfs (values at f=0, fs_blend, f=1) kept for existing
        # consumers; v_arr is the full blended value at every bf_bp point,
        # for the limit-grid plot's fuller-resolution curve.
        info = {}
        for ai, i in enumerate(active_indices):
            sp = species_list[i]
            v_arr = bf_vals[sp] @ w    # shape (bf_n_bp,) -- blend of fixed per-subset values
            mM = float(Y1[i] - Y2[i])     # no-reaction line slope
            mB = float(Y2[i])             # ... and intercept
            e_M = mM * a_over_ab + mB

            Bslopes = np.empty(n_bseg)
            Bints = np.empty(n_bseg)
            e_B = 0.0
            for s in range(n_bseg):
                fa, fb = bf_bp[s], bf_bp[s + 1]
                df = fb - fa
                sl = (v_arr[s + 1] - v_arr[s]) / df if df > 1e-15 else 0.0
                it = v_arr[s] - sl * fa
                Bslopes[s], Bints[s] = sl, it
                e_B += sl * a_over_ab * (Bt[1, s + 1] - Bt[1, s]) + it * (Bt[0, s + 1] - Bt[0, s])

            yi = max(float(y_active[ai]), 0.0)
            denom = e_M - e_B
            lam0 = (yi - e_B) / denom if abs(denom) > 1e-9 * (abs(e_M) + abs(e_B) + 1.0) else 1.0
            lam = min(1.0, max(0.0, lam0))
            if (lam0 < -1e-9 or lam0 > 1.0 + 1e-9) and _blendfs_lam_reported[0] < 10:
                print(f"[clamp]  {sp}: lam0={lam0:.4f}  lam={lam:.4f}")
                _blendfs_lam_reported[0] += 1
            v0, v1 = float(v_arr[0]), float(v_arr[-1])
            vfs = float(np.interp(fsb, bf_bp, v_arr))
            info[sp] = (e_B, lam0, lam, v0, vfs, v1, v_arr)
            Cseg[sp] = [(lam * mM + (1 - lam) * Bslopes[s], lam * mB + (1 - lam) * Bints[s])
                        for s in range(n_bseg)]
        rates = np.zeros(n_rxns)
        for j, reactants in enumerate(rxn_reactants):
            if not reactants:
                continue
            r_j = 0.0
            for segk in range(n_bseg):
                poly = np.array([k_vals[j]])
                for i, order in reactants:
                    s_i, b_i = Cseg[species_list[i]][segk]
                    for _ in range(order):
                        poly = np.polymul(poly, np.array([s_i, b_i], dtype=float))
                deg = len(poly) - 1
                for kf, c in enumerate(poly):
                    if abs(c) < 1e-300:
                        continue
                    d = deg - kf
                    r_j += c * rat[d] * (Bt[d, segk + 1] - Bt[d, segk])
            rates[j] = r_j
        return rates, info

    # ── ray_limit rate path ──────────────────────────────────────────────────
    # Build a complete-reaction limit B(f) by reacting the mixing line M(f) to
    # completion along the selectivity ray d=Δ=y−y₀, then mean-match it
    # against the no-reaction line.  Pure cross-stream only: one ray, one kink.

    def _massaction_rates(y_active, uniform_k=None):
        """Instantaneous mass-action reaction rates r_j = k_j·Π yᵢ^{νᵢⱼ}.

        If ``uniform_k`` is given, every reaction uses that value in place of
        its own k_j."""
        rj = np.empty(n_rxns)
        for j, reactants in enumerate(rxn_reactants):
            r = float(k_vals[j]) if uniform_k is None else float(uniform_k)
            for gi, order in reactants:
                r *= max(float(y_active[global_to_active[gi]]), 0.0) ** order
            rj[j] = r
        return rj

    _fallback_switch_reported = False

    def _selectivity_ray(y_active):
        """Selectivity ray d = Δ = y − y0 (accumulated extent direction).

        Fallback near y0 (Δ≈0, ill-conditioned): mass-action rates projected
        onto the stoichiometry matrix to get a direction in species space.
        The switch to Δ happens as soon as EITHER |Δ| exceeds the (absolute,
        not scale-invariant) `raylimit_fallback_tol`, OR the fractional
        conversion of whichever fed species has changed most exceeds
        `raylimit_fallback_conv_ceiling` — the second criterion bounds how
        much of the reaction the crude fallback direction can influence,
        regardless of the system's absolute concentration scale."""
        nonlocal _fallback_switch_reported
        delta = y_active - y0
        conv_frac = (float(np.max(np.abs(delta[_rl_fed_mask]) / _rl_feed_scale[_rl_fed_mask]))
                     if _rl_fed_mask.any() else 0.0)
        if (float(np.sum(np.abs(delta))) > raylimit_fallback_tol
                or conv_frac > raylimit_fallback_conv_ceiling):
            if not _fallback_switch_reported:
                _fallback_switch_reported = True
                if L_active is not None and y0_L:
                    L_conv = 1.0 - float(y_active[L_active]) / y0_L
                    print(f"[{weight_method}]   [selectivity] switching from mass-action "
                          f"fallback to the accumulated ray at conversion of "
                          f"{species_list[L_event]} (stream-1 limiting reactant) = "
                          f"{100.0 * L_conv:.4g}%")
            return delta
        rj = _massaction_rates(y_active, uniform_k=raylimit_initial_ray_k)
        if float(rj.sum()) > 1e-300:
            return nu_net_active @ rj
        return None

    def _cmax_along(Mvec, d):
        """Largest extent c≥0 with Mvec + c·d ≥ 0 (min over consumed species)."""
        cons = np.where(d < -1e-12)[0]
        if cons.size == 0:
            return 0.0
        return max(0.0, float(np.min(Mvec[cons] / (-d[cons]))))

    _mixed_feed_warned = [False]

    def _raylimit_limit(y_active):
        """Complete-reaction limit B(f) as a piecewise-linear profile.

        Returns (bps, Bvals) where bps is a 1-D array of breakpoints in [0,1]
        and Bvals has shape (len(bps), n_active), or (None, None) when nothing
        reacts.  Feeds must be pure (each reactant fed by exactly one stream):
        c_max(f) = min_i M_i(f)/(-d_i) then has at most one interior kink,
        because any two same-side candidates have M_i(f) profiles proportional
        to f (stream-1) or (1-f) (stream-2) and so can never cross each other
        — only the tightest stream-1 candidate and the tightest stream-2
        candidate are ever live, giving the closed-form crossing
        fs = kappa2/(kappa1+kappa2).  A reactant fed by both streams breaks
        this and is refused rather than handled by a general multi-kink
        search."""
        d = _selectivity_ray(y_active)
        if d is None:
            return None, None

        reac = [i for i in range(n_active) if d[i] < -1e-12]
        if not reac:
            return None, None

        kappa1_terms, kappa2_terms = [], []
        for i in reac:
            nu = -d[i]
            y1i, y2i = rl_Y1a[i], rl_Y2a[i]
            scale = 1e-9 * (abs(y1i) + abs(y2i) + 1.0)
            pure1, pure2 = abs(y2i) < scale, abs(y1i) < scale
            if pure1 and not pure2:
                kappa1_terms.append(y1i / nu)
            elif pure2 and not pure1:
                kappa2_terms.append(y2i / nu)
            else:
                # Fed by both streams (or by neither) -- the closed form
                # doesn't apply; refuse rather than fall back silently.
                if not _mixed_feed_warned[0]:
                    print(f"[ray_limit]   [closed-form fs] species "
                          f"{species_list[active_indices[i]]} is fed by both "
                          f"streams; ray_limit requires pure feeds -- "
                          f"refusing to compute a limit (rates set to 0).")
                    _mixed_feed_warned[0] = True
                return None, None

        kappa1 = min(kappa1_terms) if kappa1_terms else None
        kappa2 = min(kappa2_terms) if kappa2_terms else None
        if kappa1 is None and kappa2 is None:
            return None, None
        if kappa2 is None:
            fs = 0.0
        elif kappa1 is None:
            fs = 1.0
        else:
            fs = kappa2 / (kappa1 + kappa2)
        allbp = np.array([0.0, fs, 1.0] if 1e-9 < fs < 1.0 - 1e-9 else [0.0, 1.0])

        def _state(f):
            M = f * rl_Y1a + (1.0 - f) * rl_Y2a
            return M + _cmax_along(M, d) * d

        Bvals = np.array([_state(f) for f in allbp])

        Mbp = allbp[:, None] * rl_Y1a[None, :] + (1 - allbp)[:, None] * rl_Y2a[None, :]
        if float(np.max(np.abs(Bvals - Mbp))) < 1e-12:
            # Limit collapsed: try mass-action direction as fallback
            rj = _massaction_rates(y_active)
            if float(rj.sum()) > 1e-300:
                d_ma = nu_net_active @ rj
                Bvals_ma = np.array([_f * rl_Y1a + (1.0 - _f) * rl_Y2a + _cmax_along(
                    _f * rl_Y1a + (1.0 - _f) * rl_Y2a, d_ma) * d_ma for _f in allbp])
                if float(np.max(np.abs(Bvals_ma - Mbp))) >= 1e-12:
                    Bvals = Bvals_ma
                else:
                    return None, None
            else:
                return None, None
        return allbp, Bvals


    def _rates_raylimit(alpha_t, beta_t, ab, a_over_ab, y_active):
        bps, Bvals = _raylimit_limit(y_active)
        if bps is None:
            # No complete-reaction state available -> no reaction this step.
            info = {species_list[i]: (a_over_ab * (rl_Y1a[ai] - rl_Y2a[ai]) + rl_Y2a[ai],
                                      0.0, 0.0, float(rl_Y2a[ai]),
                                      float(0.5 * (rl_Y1a[ai] + rl_Y2a[ai])), float(rl_Y1a[ai]))
                    for ai, i in enumerate(active_indices)}
            return np.zeros(n_rxns), info

        n_seg = len(bps) - 1
        Bt = np.empty((_max_deg + 1, len(bps)))
        for kk in range(_max_deg + 1):
            Bt[kk] = betainc(alpha_t + kk, beta_t, bps)
        rat = np.empty(_max_deg + 1)
        rat[0] = 1.0
        for d in range(1, _max_deg + 1):
            rat[d] = rat[d - 1] * (alpha_t + (d - 1)) / (ab + (d - 1))

        # Peak-extent breakpoint (for the single-kink diagnostic v0/vfs/v1).
        Mbp = bps[:, None] * rl_Y1a[None, :] + (1 - bps)[:, None] * rl_Y2a[None, :]
        peak = int(np.argmax(np.abs(Bvals - Mbp).sum(axis=1)))

        # Per active species: C_i = B_i + (M_i - B_i)·λ; B_i piecewise-linear over
        # bps, M_i the no-reaction line; λ = (y_i - E[B_i])/(E[M_i] - E[B_i])
        # matches the mean so E[C_i]=y_i.
        Cseg = {}
        info = {}   # info[sp] = (E[B_i], raw λ, clamped λ, v0, vfs(peak), v1)
        for ai, i in enumerate(active_indices):
            sp = species_list[i]
            mM = float(rl_Y1a[ai] - rl_Y2a[ai])
            mB = float(rl_Y2a[ai])
            e_M = mM * a_over_ab + mB
            Bseg = []
            e_B = 0.0
            for s in range(n_seg):
                fa, fb = bps[s], bps[s + 1]
                if fb - fa < 1e-15:
                    Bseg.append((0.0, 0.0))
                    continue
                ya, yb = float(Bvals[s, ai]), float(Bvals[s + 1, ai])
                sl = (yb - ya) / (fb - fa)
                it = ya - sl * fa
                Bseg.append((sl, it))
                e_B += (sl * a_over_ab * (Bt[1, s + 1] - Bt[1, s])
                        + it * (Bt[0, s + 1] - Bt[0, s]))
            yi = max(float(y_active[ai]), 0.0)
            denom = e_M - e_B
            lam0 = (yi - e_B) / denom if abs(denom) > 1e-9 * (abs(e_M) + abs(e_B) + 1.0) else 1.0
            lam = min(1.0, max(0.0, lam0))
            info[sp] = (e_B, lam0, lam, float(Bvals[0, ai]),
                        float(Bvals[peak, ai]), float(Bvals[-1, ai]))
            Cseg[sp] = [(lam * mM + (1 - lam) * sl, lam * mB + (1 - lam) * it)
                        for (sl, it) in Bseg]

        rates = np.zeros(n_rxns)
        for j, reactants in enumerate(rxn_reactants):
            if not reactants:
                continue
            r_j = 0.0
            for s in range(n_seg):
                if bps[s + 1] - bps[s] < 1e-15:
                    continue
                poly = np.array([k_vals[j]])
                for i, order in reactants:
                    s_i, b_i = Cseg[species_list[i]][s]
                    for _ in range(order):
                        poly = np.polymul(poly, np.array([s_i, b_i], dtype=float))
                deg = len(poly) - 1
                for kf, c in enumerate(poly):
                    if abs(c) < 1e-300:
                        continue
                    d = deg - kf
                    r_j += c * rat[d] * (Bt[d, s + 1] - Bt[d, s])
            rates[j] = r_j
        return rates, info

    # ── Rate computation (shared by RHS and post-solve recording) ────────
    def _rates_at(t, y_active, silent=False, need_profiles=True):
        nonlocal k_vals
        if rate_ramp is not None:
            k_vals = k_vals_base * float(rate_ramp(t))
        var_t = mixing_variance(t, mean_f, m_epsilon, m_lambda, m_nu, m_Sc)
        s_t = max_var / var_t - 1.0
        alpha_t = mean_f * s_t
        beta_t = (1.0 - mean_f) * s_t
        ab = alpha_t + beta_t

        # Betainc table over all breakpoints: Bt[k, m] = I_{all_bps[m]}(alpha_t+k, beta_t).
        # All segment integrals reduce to differences of these, so each distinct
        # (order, breakpoint) value is evaluated exactly once per step.
        Bt = np.empty((_max_deg + 1, len(all_bps)))
        for k in range(_max_deg + 1):
            Bt[k] = betainc(alpha_t + k, beta_t, all_bps)
        # ratio[d] = prod_{k=0..d-1} (alpha_t+k)/(ab+k)
        ratio = np.empty(_max_deg + 1)
        ratio[0] = 1.0
        for d in range(1, _max_deg + 1):
            ratio[d] = ratio[d - 1] * (alpha_t + (d - 1)) / (ab + (d - 1))
        a_over_ab = alpha_t / ab

        # Per-profile beta-averages: computed when the selected method's own
        # rate calculation reads them directly, or when the caller explicitly
        # asks for them (need_profiles=True) to record for plotting.  Skipped
        # otherwise, since callers that don't set need_profiles discard this
        # return value.
        _uses_profiles_directly = not (is_blendfs or (is_raylimit and raylimit_ok))
        sp_ev = {}
        if need_profiles or _uses_profiles_directly:
            for ai, i in enumerate(active_indices):
                sp = species_list[i]
                if sp in sp_profile_arrays:
                    sp_ev[sp] = _e_vals(sp, Bt, a_over_ab)

        if is_blendfs:
            if not blendfs_ok:
                return np.zeros(n_rxns), {}, sp_ev
            # blend_fs has no per-profile weights; it returns a per-species info
            # dict in the sp_w slot for the recording loop (averages plot, clamp
            # tally, and the snapshot M/B/C curves).
            _rb, _info = _rates_blendfs(alpha_t, beta_t, ab, a_over_ab, y_active)
            return _rb, _info, sp_ev

        if is_raylimit and raylimit_ok:
            # ray_limit also returns a per-species info dict (E[B], λ, single-kink
            # v0/vfs/v1) in the sp_w slot for the recording/plotting loop.
            _rb, _info = _rates_raylimit(alpha_t, beta_t, ab, a_over_ab, y_active)
            return _rb, _info, sp_ev

        sp_w = {}
        for ai, i in enumerate(active_indices):
            sp = species_list[i]
            if sp not in sp_profile_arrays:
                continue
            sp_w[sp] = _compute_cw_weights(sp_profiles[sp], sp_ev[sp], max(float(y_active[ai]), 0.0),
                                           weight_method, species=sp, t=t, silent=silent)

        rates = np.zeros(n_rxns)
        for j, reactants in enumerate(rxn_reactants):
            if not reactants:
                continue
            r_j = 0.0
            for seg_k in range(n_segs):
                if all_bps[seg_k + 1] - all_bps[seg_k] < 1e-15:
                    continue
                prod_poly = np.array([k_vals[j]])
                for i, order in reactants:
                    sp = species_list[i]
                    Mi, Bi = _seg_linear(sp, sp_w.get(sp, np.array([])), seg_k)
                    for _ in range(order):
                        prod_poly = np.polymul(prod_poly, np.array([Mi, Bi], dtype=float))
                r_j += _poly_beta_int(prod_poly, Bt, ratio, seg_k, seg_k + 1)
            rates[j] = r_j
        return rates, sp_w, sp_ev

    # ── CVODE RHS ─────────────────────────────────────────────────────────
    # Active-vector reactant indices per reaction (for the depletion guard).
    rxn_reac_active = [[global_to_active[gi] for gi, _ in reactants]
                       for reactants in rxn_reactants]

    def rhsfn(t, y, yp):
        rates, _, _ = _rates_at(t, y, silent=True, need_profiles=False)
        # Depletion guard: a reaction cannot proceed if any of its reactants is
        # exhausted.  Zero the whole reaction's rate (not just one species' net
        # derivative) so stoichiometry — and mass — are preserved at depletion.
        # (Zeroing only the depleted species' yp would let the reaction's products
        #  keep growing, creating mass — which blows up catalytic/gated limits.)
        for j in range(n_rxns):
            if rates[j] != 0.0:
                for ai_r in rxn_reac_active[j]:
                    if y[ai_r] <= 0.0:
                        rates[j] = 0.0
                        break
        yp[:] = nu_net_active @ rates
        for ai in range(n_active):
            if y[ai] <= 0.0 and yp[ai] < 0.0:
                yp[ai] = 0.0

    # ── Integration end time ──────────────────────────────────────────────
    # With t_end given, integrate over a fixed window (the ε sweep relies on
    # this).  Otherwise let CVODE terminate on a conversion event: the solve
    # stops when the stream-1 limiting reactant hits `conversion_target`.
    L_event = _stream1_limiting_index(Y1, nu_reactants, Y2, nu_products)
    use_event = t_end is None and L_event is not None
    tau_s_run = mixing_timescale(m_epsilon, m_lambda)
    # Stream-1 limiting reactant, active-indexed and its fed (t=0) amount --
    # computed unconditionally (not just when use_event) so _selectivity_ray's
    # fallback-switch report below always has them available.
    L_active = global_to_active[int(L_event)] if L_event is not None else None
    y0_L = float(y0_full[L_event]) if L_event is not None else None

    if use_event:
        # Safety cap so the run is bounded if neither stop condition is met.
        t_cap = t_cap_mult * tau_s_run
        _thresh = (1.0 - conversion_target) * y0_L
        # Steady-state backstop: stop when the net rate ||dy/dt||_1 has decayed to
        # _SS_TOL of its t=0 value.  Scale-free (relative to the system's own
        # initial rate), so it fires at a reversible reaction's equilibrium plateau
        # — where conversion never reaches the target — without running to the cap.
        # At equilibrium ||dy/dt||->0; at conversion_target the rate is still well
        # above _SS_TOL, so for irreversible schemes the conversion event fires
        # first and the end time is unchanged.
        #
        # GATE on mixing progress: the backstop is only meaningful once mixing has
        # substantially completed (variance decayed below _SS_MIX_FRAC of its start).
        # While the variance is still decaying the system is mixing-limited, and a
        # collapsed rate just means the (fast) chemistry has reacted everything
        # currently co-located and is WAITING for mixing to supply fresh reactants —
        # not equilibrium.  Without this gate a fast reaction from a segregated start
        # (e.g. ester hydrolysis: a fast neutralisation competing with slow
        # hydrolysis) trips the backstop at t≈0 with ~0 conversion, stopping before
        # any mixing occurs.
        _SS_TOL = 1.0e-6
        _SS_MIX_FRAC = 1.0e-2
        _var0 = mixing_variance(0.0, mean_f, m_epsilon, m_lambda, m_nu, m_Sc)
        _yp0 = nu_net_active @ _rates_at(0.0, y0, silent=True, need_profiles=False)[0]
        yp0_norm = float(np.sum(np.abs(_yp0)))
        _use_ss = yp0_norm > 1e-300

        def eventsfn(t, y, events):
            # Crosses zero (downward) when conversion reaches conversion_target.
            events[0] = y[L_active] - _thresh
            if _use_ss:
                if mixing_variance(t, mean_f, m_epsilon, m_lambda, m_nu, m_Sc) > _SS_MIX_FRAC * _var0:
                    events[1] = 1.0       # mixing not yet complete: not steady state
                else:
                    yp = nu_net_active @ _rates_at(t, y, silent=True, need_profiles=False)[0]
                    events[1] = float(np.sum(np.abs(yp))) / yp0_norm - _SS_TOL
        _n_ev = 2 if _use_ss else 1
        eventsfn.terminal = [True] * _n_ev
        eventsfn.direction = [-1] * _n_ev

        # Pass 1: adaptive solve that stops at the first event, to discover t_final.
        _cpu0 = time.process_time()
        evt_solver = CVODE(rhsfn, method='BDF', rtol=ode_rtol or _ODE_RTOL, atol=ode_atol or _ODE_ATOL,
                           eventsfn=eventsfn, num_events=_n_ev)
        res_evt = evt_solver.solve([0.0, t_cap], y0)
        t_final = float(res_evt.t[-1])
        _conv_end = ((y0_L - float(res_evt.y[-1][L_active])) / y0_L) if y0_L > 0 else float('nan')
        if t_final >= t_cap * (1 - 1e-9):
            print(f"[{weight_method}]   [event] warning: {species_list[L_event]} reached "
                  f"neither conversion {100.0 * conversion_target:.4g}% nor steady state "
                  f"within cap t={t_cap:.4g} s; integrating to cap.")
        elif _conv_end >= conversion_target - 1e-6:
            print(f"[{weight_method}]   [event] {species_list[L_event]} reached conversion "
                  f"{100.0 * conversion_target:.4g}% at t={t_final:.4g} s "
                  f"({t_final / tau_s_run:.3g} τ_s); ending integration.")
        else:
            print(f"[{weight_method}]   [event] {species_list[L_event]} reached steady state "
                  f"(||dy/dt|| < {_SS_TOL:g} of initial, conversion {100.0 * _conv_end:.4g}%) "
                  f"at t={t_final:.4g} s ({t_final / tau_s_run:.3g} τ_s); ending integration.")
        # Output on a uniform grid over [0, t_final] for plotting/recording.
        # We RESAMPLE pass 1's adaptive trajectory (res_evt — saved at the solver's
        # own internal steps, since its tspan had length 2) rather than launching a
        # fresh fixed-grid solve.  ray_limit rebuilds its complete-reaction limit
        # from the live state, so its RHS is non-smooth and y0 is a near-unstable
        # fixed point: an independent re-solve can follow a different path and stall
        # at y0 while the event solve reacted, yielding spurious ~0 conversion and
        # NaN product fractions.  Reusing res_evt guarantees the recorded run is the
        # one the event actually detected.
        from types import SimpleNamespace as _NS
        tspan = np.linspace(0.0, t_final, n_out)
        _te = np.asarray(res_evt.t, dtype=float)
        _ye = np.asarray(res_evt.y, dtype=float)
        # Guard against non-increasing duplicate times before interpolating.
        _keep = np.concatenate(([True], np.diff(_te) > 0))
        _te, _ye = _te[_keep], _ye[_keep]
        _yg = np.column_stack([np.interp(tspan, _te, _ye[:, j])
                               for j in range(_ye.shape[1])])
        result = _NS(t=tspan, y=_yg)
        _native_t, _native_y = _te, _ye          # solver's own on-trajectory points
        solve_cpu_s = time.process_time() - _cpu0
    else:
        if t_end is None:
            # No stream-1 reactant to key an event on; fall back to a fixed span.
            t_end = 1.0e4 * tau_s_run
            print(f"[{weight_method}]   [event] no stream-1 reactant found; integrating to "
                  f"fixed t_end={t_end:.4g} s.")
        solver = CVODE(rhsfn, method='BDF', rtol=ode_rtol or _ODE_RTOL, atol=ode_atol or _ODE_ATOL)
        tspan = np.linspace(0.0, float(t_end), n_out)
        _cpu0 = time.process_time()
        result = solver.solve(tspan, y0)
        _native_t = np.asarray(result.t, dtype=float)   # grid solve is already
        _native_y = np.asarray(result.y, dtype=float)    # on-trajectory at the grid
        solve_cpu_s = time.process_time() - _cpu0

    print(f"[{weight_method}]   [timing] ODE solve CPU time: {solve_cpu_s:.4g} s")

    # Reconstruct full species array: inactive species stay at their initial value
    n_t = len(result.t)
    y_full = np.tile(y0_full, (n_t, 1))
    y_full[:, active_indices] = result.y

    # Solver-native reaction rates.  Evaluate the rate on the solver's OWN solution
    # points (the event solve's internal steps, or the grid solution) and resample
    # those onto the output grid — rather than recomputing at the linearly-resampled
    # output points.  ray_limit's complete-reaction limit is rebuilt from the live
    # state, so its RHS is non-smooth; recomputing it off-trajectory (between the
    # sparse steps the solver takes late in the reaction) produces spurious rate
    # noise even though the integrated curves are smooth.  On-trajectory evaluation
    # is the rate the solver actually used.  (For the fixed-grid solve the native
    # points ARE the output grid, so this is an identity there.)
    if n_rxns:
        _native_rates = np.array([_rates_at(float(_native_t[mm]), _native_y[mm],
                                            silent=True, need_profiles=False)[0]
                                  for mm in range(len(_native_t))])
        rates_native_grid = np.column_stack(
            [np.interp(result.t, _native_t, _native_rates[:, j]) for j in range(n_rxns)])
    else:
        rates_native_grid = np.zeros((n_t, 0))

    # Record rates at each output time point; detect linear_interp pair changes.
    # Each change is classified:
    #   CONTINUOUS (type 1) – the mean crossed the shared profile, so the
    #       departing profile's weight passed through zero: the rate is smooth.
    #       Signature: departing and arriving profiles lie on opposite sides of
    #       the mean (or a profile merely faded in/out with no partner swap).
    #   JUMP (type 2) – two profiles crossed in beta-average (rank swap), so a
    #       bracket member with non-zero weight was replaced abruptly: the rate
    #       steps.  Signature: departing and arriving profiles lie on the SAME
    #       side of the mean.
    rates_out = np.empty((n_t, n_rxns))
    # Per-limit beta-averages over time: sp -> array (n_t, n_profiles).
    limit_avgs = {sp: np.full((n_t, len(sp_profiles[sp])), np.nan) for sp in sp_profiles}
    # blend_fs only: the time-varying per-subset blend weights, blended fs, and
    # the beta-averaged blended limit E[B_i] per species (for the averages plot).
    _bf_diag = (is_blendfs and blendfs_ok)
    _rl_diag = (is_raylimit and raylimit_ok)
    bf_w_t = np.full((n_t, bf_n), np.nan) if _bf_diag else None
    bf_fsb_t = np.full(n_t, np.nan)
    # Beta-averaged blended limit E[B_i] per species, for the averages plot.
    blend_avgs = {sp: np.full(n_t, np.nan) for sp in sp_profiles} if (_bf_diag or _rl_diag) else {}
    # Absolute-magnitude floor for the clamp tallies.  A clamp is only counted when
    # the mean falls outside the [E[M], E[B]] band by more than this amount — not
    # merely when the raw λ leaves [0,1] by a hair.  This suppresses trivial
    # boundary-grazing: e.g. at the segregated (very low ε) limit the mean reacts
    # essentially to the complete-reaction average and rides along E[B], grazing it
    # by ~1e-4·(feed) — physically λ→1 (C→B) is the correct profile there, so those
    # steps are not representational failures.  Genuine clamps (mean off the band by
    # a sizeable fraction of the feed) are far above this floor and still reported.
    _clamp_atol = 1.0e-4 * float(max(np.asarray(Y1).max(), np.asarray(Y2).max(), 1.0))
    # blend_fs clamp tally: per species, # steps the mean fell below E[M] (pinned
    # to no-reaction) or above E[B] (pinned to the blended limit).
    bf_clamp_lo = {sp: 0 for sp in sp_profiles} if _bf_diag else {}
    bf_clamp_hi = {sp: 0 for sp in sp_profiles} if _bf_diag else {}
    # blend_fs: blend profile params per species over time [v0, vfs, v1, λ] (for
    # drawing the M / B / C curves on the snapshot plots).
    bf_prof = {sp: np.full((n_t, 4), np.nan) for sp in sp_profiles} if _bf_diag else {}
    # blend_fs: the FULL blended value at every point in bf_bp (the fixed
    # union of every blended subset's own breakpoints), per species over
    # time -- for the limit-grid plot, which needs the fuller resolution
    # v0/vfs/v1 alone can't represent (a catalytic reaction's interior kink).
    bf_prof_full = {sp: np.full((n_t, bf_n_bp), np.nan) for sp in sp_profiles} if _bf_diag else {}
    # ray_limit diagnostics: the time-varying complete-reaction fs, the
    # per-species single-kink limit params [v0, vfs, v1, λ] (for the snapshot
    # M/B/C curves), and the same clamp tally.
    rl_fsb_t = np.full(n_t, np.nan) if _rl_diag else None
    rl_prof = {sp: np.full((n_t, 4), np.nan) for sp in sp_profiles} if _rl_diag else {}
    rl_clamp_lo = {sp: 0 for sp in sp_profiles} if _rl_diag else {}
    rl_clamp_hi = {sp: 0 for sp in sp_profiles} if _rl_diag else {}
    # Full staged complete-reaction limit per output step: lists (length n_t) of
    # breakpoint arrays and B-value arrays (shape (len(bps), n_active)), or None.
    rl_limit_bps = [] if _rl_diag else None
    rl_limit_B = [] if _rl_diag else None
    # linear_interp clamp tally: per species, # steps the mean fell below the
    # lowest / above the highest subset-limit beta-average, so the bracket
    # interpolation pinned to an endpoint (E[C] != y there).
    _li_diag = (weight_method == 'linear_interp')
    li_clamp_lo = {sp: 0 for sp in sp_profiles} if _li_diag else {}
    li_clamp_hi = {sp: 0 for sp in sp_profiles} if _li_diag else {}
    # linear_interp: per-species bracket weights over its candidate profiles at
    # each output step (for reconstructing the blended C(f) = Σ w_n·profile_n(f)
    # on the snapshot plots).
    li_weights = {sp: np.full((n_t, len(sp_profiles[sp])), np.nan) for sp in sp_profiles} if _li_diag else {}
    jump_events = []  # (time, species) for each JUMP-type pair change
    continuous_events = []  # (time, species) for each CONTINUOUS pair change
    _prev_pairs = {}  # sp -> frozenset of active profile labels (previous step)
    for k in range(n_t):
        _rates_k, sp_w_k, sp_ev_k = _rates_at(result.t[k], y_full[k, active_indices])
        rates_out[k] = rates_native_grid[k]   # solver-native (see above), not _rates_k
        for sp, ev in sp_ev_k.items():
            limit_avgs[sp][k, :len(ev)] = ev
        if _bf_diag:
            bf_w_t[k], bf_fsb_t[k] = _blendfs_weights(y_full[k, active_indices])
            for sp, (e_B, lr, lam, v0, vfs, v1, v_arr) in sp_w_k.items():   # blend_fs info
                blend_avgs[sp][k] = e_B
                bf_prof[sp][k] = (v0, vfs, v1, lam)
                bf_prof_full[sp][k] = v_arr
                _band = abs(e_B - y0_full[sp_to_global[sp]])   # |E[B]-E[M]|, E[M]=y0
                if lr < -1e-9 and (-lr) * _band > _clamp_atol:
                    bf_clamp_hi[sp] += 1
                elif lr > 1.0 + 1e-9 and (lr - 1.0) * _band > _clamp_atol:
                    bf_clamp_lo[sp] += 1
        if _rl_diag:
            _bps_k, _Bv_k = _raylimit_limit(y_full[k, active_indices])
            rl_limit_bps.append(_bps_k)
            rl_limit_B.append(_Bv_k)
            if _bps_k is not None:                      # peak-extent f and selectivity
                _Mbp = (_bps_k[:, None] * rl_Y1a[None, :]
                        + (1 - _bps_k)[:, None] * rl_Y2a[None, :])
                rl_fsb_t[k] = float(_bps_k[int(np.argmax(np.abs(_Bv_k - _Mbp).sum(axis=1)))])

            for sp, (e_B, lr, lam, v0, vfs, v1) in sp_w_k.items():   # ray_limit info
                blend_avgs[sp][k] = e_B
                rl_prof[sp][k] = (v0, vfs, v1, lam)
                _band = abs(e_B - y0_full[sp_to_global[sp]])   # |E[B]-E[M]|, E[M]=y0
                if lr < -1e-9 and (-lr) * _band > _clamp_atol:
                    rl_clamp_hi[sp] += 1
                elif lr > 1.0 + 1e-9 and (lr - 1.0) * _band > _clamp_atol:
                    rl_clamp_lo[sp] += 1
        if weight_method == 'linear_interp':
            for sp, w in sp_w_k.items():
                li_weights[sp][k, :len(w)] = w
                # Tally clamps: mean outside the [lowest, highest] profile-average
                # range (mirrors the clamp in _compute_cw_weights; trivial near-zero
                # means are skipped, as there the per-step warning is also suppressed).
                _ev = sp_ev_k.get(sp)
                _avg_sp = float(y_full[k, sp_to_global[sp]])
                if _ev is not None and len(_ev) >= 2 and _avg_sp >= 1e-15:
                    _emin, _emax = float(_ev.min()), float(_ev.max())
                    if _avg_sp < _emin - 1e-10 and (_emin - _avg_sp) > _clamp_atol:
                        li_clamp_lo[sp] += 1
                    elif _avg_sp > _emax + 1e-10 and (_avg_sp - _emax) > _clamp_atol:
                        li_clamp_hi[sp] += 1
                active_idx = frozenset(
                    sp_profiles[sp][idx][0]
                    for idx in np.where(w > 1e-10)[0]
                )
                prev = _prev_pairs.get(sp)
                if prev is not None and active_idx != prev:
                    _avg = y_full[k, sp_to_global[sp]]
                    if _avg >= 0.0:
                        outgoing = prev - active_idx
                        incoming = active_idx - prev
                        _lbl_e = {sp_profiles[sp][i][0]: sp_ev_k[sp][i]
                                  for i in range(len(sp_ev_k[sp]))}
                        _side = lambda lbls: {int(np.sign(_lbl_e[l] - _avg))
                                              for l in lbls if l in _lbl_e}
                        if not outgoing or not incoming:
                            # a profile only faded in/out at ~zero weight
                            tag = 'CONTINUOUS'
                            continuous_events.append((float(result.t[k]), sp))
                        elif _side(outgoing) & _side(incoming):
                            # departing and arriving on the same side of the mean
                            tag = 'JUMP'
                            jump_events.append((float(result.t[k]), sp))
                        else:
                            tag = 'CONTINUOUS'
                            continuous_events.append((float(result.t[k]), sp))
                        old_lbl = ', '.join(sorted(prev))
                        new_lbl = ', '.join(sorted(active_idx))
                        print(f"[{weight_method}]   [{tag}] {sp} ({_avg:.4g}): interpolation pair "
                              f"changed at t={result.t[k]:.4g} s  [{old_lbl}] → [{new_lbl}]")
                _prev_pairs[sp] = active_idx

    # Presentation-only repair of the RECORDED rates (touches ONLY `rates_out`, the
    # rate diagnostic — never the trajectory, the closure, or the solve).  Two
    # outliers can mar the rate plot that the smooth integrated curves don't show:
    #   (a) a t≈0 burst of a fast reaction (a one-point spike that dominates the
    #       plot scale) — common, so it is ALWAYS removed when present; and
    #   (b) short DROPOUTS to ~0 later in the run where ray_limit's state-rebuilt
    #       limit momentarily loses its reactant overlap — these are specific to
    #       problems that exercise the non-smooth limit, so they are bridged ONLY
    #       when a rate actually suffers them (≥2 dropout points).  Most problems
    #       have none and keep their native rates exactly.
    # Outliers are flagged against a robust rolling-median trend (its own max sets
    # the rate scale, so the startup spike can't inflate the floor) and replaced by
    # interpolation from the inlier neighbours — a smooth bridge that preserves the
    # genuine ramp/peak/decay (incl. the end-of-run roll-off, where rate and trend
    # fall together so nothing is flagged).
    if n_rxns and n_t >= 9:
        from scipy.ndimage import median_filter as _medfilt
        _w = max(9, (n_t // 12) | 1)       # odd trend window (~8% of the run)
        _idx = np.arange(n_t)
        _early = _idx < max(3, n_t // 20)  # the startup-burst region
        # Pass 1: flag each reaction's startup spike and its dropouts.
        _spk, _drp, _ndrop = [], [], []
        for j in range(n_rxns):
            col = rates_out[:, j]
            trend = _medfilt(col, size=_w, mode='mirror')
            scale = float(np.max(trend))               # robust peak (spike rejected)
            substantial = trend > 0.05 * scale          # skip genuine low-rate regions
            spike = (col > 4.0 * trend) & (col > 0.05 * scale) & _early if scale > 0 else \
                np.zeros(n_t, bool)
            drop = (col < 0.3 * trend) & substantial if scale > 0 else np.zeros(n_t, bool)
            _spk.append(spike); _drp.append(drop); _ndrop.append(int(drop.sum()))
        # The PLOT only "suffers" the dropout pathology when some rate has many of
        # them (≥ ~5% of points).  Then bridge the dropouts on all of this problem's
        # rates (for a consistent plot); otherwise leave the rates fully native and
        # only strip the universal t≈0 startup spike.  Most problems take this path.
        _drop_trigger = max(4, n_t // 20)
        _suffers = max(_ndrop) >= _drop_trigger
        _bridged = []
        for j in range(n_rxns):
            bad = (_spk[j] | _drp[j]) if _suffers else _spk[j]
            good = ~bad
            if bad.any() and good.sum() >= 2 and bad.sum() <= 0.3 * n_t:
                rates_out[:, j] = np.interp(_idx, _idx[good], rates_out[:, j][good])
                if _suffers and _ndrop[j]:
                    _bridged.append("%s×%d" % (rxn_labels[j] if j < len(rxn_labels)
                                               else "R%d" % (j + 1), _ndrop[j]))
        if _bridged:
            print(f"[{weight_method}]   [rates] presentation: rate plot is rough — bridged "
                  f"dropout(s) in {', '.join(_bridged)} (native rate has isolated zeros "
                  f"from the non-smooth limit; plot only, solution unchanged)")

    # blend_fs clamp summary: where the species mean left the [E[M], E[B]] band,
    # the interpolation pinned to a boundary (E[C] != y there).
    if _bf_diag:
        _clamped = [sp for sp in sp_profiles if bf_clamp_lo[sp] or bf_clamp_hi[sp]]
        if not _clamped:
            print(f"[{weight_method}]   [clamp] no species left the "
                  f"[no-reaction, blended-limit] band ({n_t} steps).")
        else:
            print(f"[{weight_method}]   [clamp] {n_t} output steps; mean outside "
                  f"[E[M], E[B]] band by >{_clamp_atol:.2g} ->")
            for sp in _clamped:
                lo, hi = bf_clamp_lo[sp], bf_clamp_hi[sp]
                print(f"[{weight_method}]     {sp}: {hi} step(s) above blended limit "
                      f"({100.0*hi/n_t:.0f}%), {lo} below no-reaction "
                      f"({100.0*lo/n_t:.0f}%)")

    # ray_limit clamp summary: the inverse-distance blend is meant to keep the
    # mean inside the [E[M], E[B]] band, so this should normally report no clamping.
    if _rl_diag:
        _clamped = [sp for sp in sp_profiles if rl_clamp_lo[sp] or rl_clamp_hi[sp]]
        if not _clamped:
            print(f"[{weight_method}]   [clamp] no species left the "
                  f"[no-reaction, blended-limit] band ({n_t} steps).")
        else:
            print(f"[{weight_method}]   [clamp] {n_t} output steps; mean outside "
                  f"[E[M], E[B]] band by >{_clamp_atol:.2g} ->")
            for sp in _clamped:
                lo, hi = rl_clamp_lo[sp], rl_clamp_hi[sp]
                print(f"[{weight_method}]     {sp}: {hi} step(s) above blended limit "
                      f"({100.0*hi/n_t:.0f}%), {lo} below no-reaction "
                      f"({100.0*lo/n_t:.0f}%)")

    # linear_interp clamp summary: where the species mean left the
    # [lowest-limit, highest-limit] profile-average range, the bracket
    # interpolation pinned to an endpoint (E[C] != y there).  Usually empty.
    if _li_diag:
        _clamped = [sp for sp in sp_profiles if li_clamp_lo[sp] or li_clamp_hi[sp]]
        if not _clamped:
            print(f"[{weight_method}]   [clamp] no species left the "
                  f"[lowest-limit, highest-limit] profile-average range ({n_t} steps).")
        else:
            print(f"[{weight_method}]   [clamp] {n_t} output steps; mean outside "
                  f"[lowest-limit, highest-limit] range by >1e-10 ->")
            for sp in _clamped:
                lo, hi = li_clamp_lo[sp], li_clamp_hi[sp]
                print(f"[{weight_method}]     {sp}: {hi} step(s) above highest limit "
                      f"({100.0*hi/n_t:.0f}%), {lo} below lowest limit "
                      f"({100.0*lo/n_t:.0f}%)")

    # Closure check: fraction of the consumed stream-1 limiting reactant that
    # ended up as product (P, R, T, S, Q ...).
    stream1_closure = stream1_reactant_to_product_fraction(
        species_list, y_full[0], y_full[-1], Y1, nu_reactants, nu_products, Y2=Y2)
    if stream1_closure['limiting_reactant'] is not None:
        L_name = stream1_closure['limiting_reactant']
        print(f"[{weight_method}]   [closure] ε={m_epsilon:.4g}, "
              f"τ_s={mixing_timescale(m_epsilon, m_lambda):.4g} s, "
              f"t_end={float(result.t[-1]):.4g} s; "
              f"limiting reactant {L_name}, "
              f"{stream1_closure['fed']:.6g} mol fed, "
              f"{stream1_closure['consumed']:.6g} mol consumed")
        print(f"[{weight_method}]     X_conv  {'':>12s} "
              f"{'(conversion of ' + L_name + ')':>20s} "
              f"= {100.0 * stream1_closure['conversion']:.4g}%")
        print(f"[{weight_method}]   [closure] fraction of consumed {L_name} ending up as each product:")
        for name in stream1_closure['products']:
            frac = stream1_closure['per_product_fraction'][name]
            factor = stream1_closure['stoich_factors'][name]
            formed = stream1_closure['per_product'][name]
            print(f"[{weight_method}]     X_{name:<5s} factor={factor:.4g}  "
                  f"formed={formed:.6g} mol  fraction={frac:.6f}")
        print(f"[{weight_method}]     {'TOTAL':<7s} {'':>12s} {'':>20s} "
              f"fraction={stream1_closure['fraction']:.6f}")

    # Mathematical check: the conserved scalar f_conserved(t), built from the
    # ODE-solved means of the stream-1 and stream-2 limiting reactants (L1,
    # L2), each scaled by its own selectivity-ray component
    # nu_i(t) = y_i(0) - y_i(t) (the amount of i consumed so far -- all ray
    # components sampled at the same t):
    #   f_conserved(t) = (y_L1(t)/nu_L1(t) - y_L2(t)/nu_L2(t) + kappa2(t))
    #                     / (kappa1(t) + kappa2(t))
    #   kappa1(t) = Y1[L1]/nu_L1(t),  kappa2(t) = Y2[L2]/nu_L2(t)
    # This is the same kappa1/kappa2 construction _raylimit_limit uses to get
    # its closed-form fs = kappa2/(kappa1+kappa2); here it's evaluated from
    # the actual solved trajectory at every t as a mass-balance sanity check
    # -- it should track mean_f throughout the run.  At t=0, nu1=nu2=0
    # exactly (0/0); resolved via the L'Hopital limit, substituting the
    # initial mass-action consumption RATE of L1/L2 (the same fallback ray
    # direction _selectivity_ray uses near y0) for the zero accumulated
    # extent -- kappa_i(0) = Y_i[Li] / (initial rate of Li).
    L1_fc = _stream1_limiting_index(Y1, nu_reactants, Y2, nu_products)
    L2_fc = _stream1_limiting_index(Y2, nu_reactants, Y1, nu_products)
    f_conserved = None
    if L1_fc is not None and L2_fc is not None and L1_fc != L2_fc:
        nu1_t = y_full[0, L1_fc] - y_full[:, L1_fc]
        nu2_t = y_full[0, L2_fc] - y_full[:, L2_fc]
        _fc_valid = (nu1_t > 0) & (nu2_t > 0)
        f_conserved = np.full(len(result.t), np.nan)
        if _fc_valid.any():
            kappa1_t = Y1[L1_fc] / nu1_t[_fc_valid]
            kappa2_t = Y2[L2_fc] / nu2_t[_fc_valid]
            f_conserved[_fc_valid] = (
                (y_full[_fc_valid, L1_fc] / nu1_t[_fc_valid]
                 - y_full[_fc_valid, L2_fc] / nu2_t[_fc_valid] + kappa2_t)
                / (kappa1_t + kappa2_t))

        if (np.isclose(result.t[0], 0.0) and int(L1_fc) in global_to_active
                and int(L2_fc) in global_to_active):
            rj0 = _massaction_rates(y_full[0, active_indices], uniform_k=raylimit_initial_ray_k)
            if float(rj0.sum()) > 1e-300:
                d_rate0 = nu_net_active @ rj0
                d1 = -float(d_rate0[global_to_active[int(L1_fc)]])
                d2 = -float(d_rate0[global_to_active[int(L2_fc)]])
                denom0 = Y1[L1_fc] * d2 + Y2[L2_fc] * d1
                if d1 > 0.0 and d2 > 0.0 and abs(denom0) > 1e-300:
                    f_conserved[0] = (
                        (y_full[0, L1_fc] * d2 - y_full[0, L2_fc] * d1 + Y2[L2_fc] * d1)
                        / denom0)
                    _fc_valid[0] = True

        if _fc_valid.any():
            _fc_err = np.abs(f_conserved[_fc_valid] - mean_f)
            print(f"[{weight_method}]   [check] f_conserved "
                  f"(L1={species_list[L1_fc]}, L2={species_list[L2_fc]}): "
                  f"mean_f={mean_f:.6f}, "
                  f"range=[{np.min(f_conserved[_fc_valid]):.6f}, "
                  f"{np.max(f_conserved[_fc_valid]):.6f}], "
                  f"max|error|={np.max(_fc_err):.3g}")

    out = {
        't': result.t,
        'y': y_full,
        'stream1_closure': stream1_closure,
        'f_conserved': f_conserved,
        'rates': rates_out,
        'rxn_labels': rxn_labels,
        'species': species_list,
        'mean_f': mean_f,
        'var_start': var_start,
        'var_end': var_end,
        'm_epsilon': m_epsilon,
        'm_lambda': m_lambda,
        'm_nu': m_nu,
        'm_Sc': m_Sc,
        'solve_cpu_s': solve_cpu_s,
        'active_indices': active_indices,
        'rxn_reactants': rxn_reactants,
        'Y1': Y1,
        'Y2': Y2,
        'weight_method': weight_method,
        'sp_profiles': sp_profiles,
        'jump_events': jump_events,
        'jump_times': sorted({t for t, _ in jump_events}),
        'continuous_events': continuous_events,
        'limit_avgs': limit_avgs,
        'blend_avgs': blend_avgs,
        'method_ran': not (is_blendfs and not blendfs_ok),
    }
    if _bf_diag:
        out['blendfs'] = {
            'subs': list(bf_subs), 'fs': list(bf_fs), 'misses': list(bf_misses),
            'w': bf_w_t, 'fsb': bf_fsb_t, 'prof': bf_prof,
            'bp': bf_bp, 'prof_full': bf_prof_full,
            'Y1': Y1, 'Y2': Y2,
            'clamp_below_M': dict(bf_clamp_lo), 'clamp_above_B': dict(bf_clamp_hi),
        }
    if _rl_diag:
        out['raylimit'] = {
            'rxn_labels': rxn_labels, 'fsb': rl_fsb_t, 'prof': rl_prof,
            'Y1': Y1, 'Y2': Y2, 'limit_bps': rl_limit_bps, 'limit_B': rl_limit_B,
            'clamp_below_M': dict(rl_clamp_lo), 'clamp_above_B': dict(rl_clamp_hi),
        }
    if _li_diag:
        out['li_clamps'] = {
            'clamp_below_min': dict(li_clamp_lo),
            'clamp_above_max': dict(li_clamp_hi),
        }
        out['li_weights'] = li_weights
    return out


def _compute_cw_weights(profile_list, e_vals, average_val, weight_method='linear_interp',
                        species=None, t=None, silent=False):
    """Non-negative weights for the C_w(f) blending.

    `linear_interp`: two-profile nearest-bracket interpolation —
        Σ w_n = 1 and Σ w_n·E[C_n] = average_val.

    weight_method: also the label used in diagnostic clamping messages.
    species / t:   optional labels included in those messages.

    Returns the weights array.
    """
    _t_str = f" t={t:.4g} s" if t is not None else ""
    _sp = f" {species} ({average_val:.4g}){_t_str}" if species is not None else f" ({average_val:.4g}){_t_str}"
    n_p = len(profile_list)

    if n_p == 0:
        return np.zeros(0)
    if n_p == 1:
        return np.ones(1)

    _e_min, _e_max = float(e_vals.min()), float(e_vals.max())
    _suppress = silent or average_val < 1e-15
    if average_val < _e_min - 1e-10:
        if not _suppress:
            print(f"[{weight_method}]   [clamp]{_sp}: below profile range "
                  f"[{_e_min:.4g}, {_e_max:.4g}]; clamping to {_e_min:.4g}")
        average_val = _e_min
    elif average_val > _e_max + 1e-10:
        if not _suppress:
            print(f"[{weight_method}]   [clamp]{_sp}: above profile range "
                  f"[{_e_min:.4g}, {_e_max:.4g}]; clamping to {_e_max:.4g}")
        average_val = _e_max

    return _linear_interp_weights(e_vals, average_val)


def _linear_interp_weights(e_vals, average_val):
    """Two-profile nearest-bracket weights (the kinky `linear_interp` scheme).

    Picks the adjacent pair of profiles whose beta-weighted averages bracket
    *average_val*.  The primary weight w_lo (on the lower profile, analogous to
    λ in the blend convention) is (e_hi - average_val) / (e_hi - e_lo); the upper
    profile gets 1 - w_lo.  Σ w = 1 and Σ w·E[C_n] = average_val are satisfied.
    Continuous in average_val but only C0 — the weight vector kinks each time the
    bracketing pair changes (i.e. when average_val crosses a profile's E[C_n]).
    """
    n_p = len(e_vals)
    _order = np.argsort(e_vals)
    _e_sorted = e_vals[_order]
    _idx_hi = int(np.searchsorted(_e_sorted, average_val, side='left'))
    _idx_hi = min(_idx_hi, n_p - 1)
    _idx_lo = max(_idx_hi - 1, 0)
    if _idx_hi == _idx_lo:
        _idx_hi = min(_idx_lo + 1, n_p - 1)
    e_lo, e_hi = _e_sorted[_idx_lo], _e_sorted[_idx_hi]
    span = e_hi - e_lo
    w_lo = (e_hi - average_val) / span if span > 1e-15 else 0.5
    weights = np.zeros(n_p)
    weights[_order[_idx_lo]] = w_lo
    weights[_order[_idx_hi]] = 1.0 - w_lo
    return weights



def _eval_segments(segs, f):
    """Evaluate a piecewise-linear profile (list of segment dicts with f_start,
    f_end, slope, intercept) at scalar f, clamping outside the covered range."""
    for s in segs:
        if s['f_start'] - 1e-12 <= f <= s['f_end'] + 1e-12:
            return s['slope'] * f + s['intercept']
    s = segs[0] if f <= segs[0]['f_start'] else segs[-1]
    return s['slope'] * f + s['intercept']


def plot_ode_limit_averages(ode_result, save_stem=None):
    """Plot the beta-average of each limit (profile) vs time, per active species.

    Mirrors ``plot_ode_trajectories`` but, instead of the single blended
    species average C_i(t), each subplot overlays one line per limit — its
    beta-weighted average over time.  This makes limit crossings (the instants
    where the interpolation bracket can swap) easy to spot.
    """
    import matplotlib.pyplot as plt
    import pathlib as _pathlib

    species_list = ode_result['species']
    active_indices = ode_result.get('active_indices', np.arange(len(species_list)))
    limit_avgs = ode_result.get('limit_avgs', {})
    blend_avgs = ode_result.get('blend_avgs', {})  # blend_fs: E[B_i] over time
    sp_profiles = ode_result['sp_profiles']
    t = ode_result['t']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ode_result.get('m_lambda', DEFAULT_M_LAMBDA)
    mean_f = ode_result['mean_f']
    weight_method = ode_result.get('weight_method', 'linear_interp')
    jump_events = ode_result.get('jump_events', [])
    continuous_events = ode_result.get('continuous_events', [])

    active_species = [species_list[i] for i in active_indices if species_list[i] in limit_avgs]
    n_plot = len(active_species)
    if n_plot == 0:
        return

    _tab_palette = _build_tab_palette()

    n_cols = min(3, n_plot)
    n_rows = (n_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows + 0.5),
                             squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.0, 3.0)

    for plot_idx, sp in enumerate(active_species):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row][col]
        avgs = limit_avgs[sp]                  # (n_t, n_profiles)
        labels = [lbl for lbl, _ in sp_profiles[sp]]
        n_profiles = avgs.shape[1]
        sp_idx = species_list.index(sp)
        y_sp = ode_result['y'][:, sp_idx]

        # Which profiles to highlight (bold) at each timestep.
        active_mask = np.zeros((len(t), n_profiles), dtype=bool)
        if 'blendfs' in ode_result or 'raylimit' in ode_result:
            # This closure interpolates only between the no-reaction limit and
            # the blend (the E[B] line), so bold only the no-reaction limit.
            noreac = next((j for j, l in enumerate(labels)
                           if '0' in str(l).split('_')), None)
            if noreac is not None:
                active_mask[:, noreac] = True
        else:
            for k in range(len(t)):
                ev = avgs[k]
                av = float(np.clip(max(y_sp[k], 0.0), ev.min(), ev.max()))
                w = _linear_interp_weights(ev, av)
                active_mask[k] = w > 1e-10

        _mk_every_la = max(1, len(t) // 15)
        _MAX_LABEL_LA = 15
        for n, lbl in enumerate(labels):
            color, ls, mk = _tab_palette[n % len(_tab_palette)]
            _disp = lbl if len(lbl) <= _MAX_LABEL_LA else lbl[:_MAX_LABEL_LA - 3] + '...'
            ax.plot(t, avgs[:, n], lw=_lw * (1.0 / 1.5), color=color, linestyle=ls, marker=mk, mfc=_face(color),
                    markevery=_mk_every_la, ms=_ms, label=_disp, alpha=0.4)
            # Bold overlay where this profile is in the active pair
            y_bold = np.where(active_mask[:, n], avgs[:, n], np.nan)
            ax.plot(t, y_bold, lw=_lw * (2.5 / 1.5), color=color, linestyle=ls, marker=mk, mfc=_face(color),
                    markevery=_mk_every_la, ms=_ms + 1)
        # beta-averaged complete-reaction limit E[B_i] — black solid (drawn first / underneath)
        if sp in blend_avgs:
            _elbl = {'ray_limit': 'ray limit E[B]'}.get(weight_method, 'blended limit E[B]')
            ax.plot(t, blend_avgs[sp], lw=2.4, color='black', ls='-',
                    label=_elbl, zorder=5)
        # blended species average C_i(t) — red, drawn on top with a wide dash so it
        # stays visible riding on the black E[B] line when the two nearly coincide.
        ax.plot(t, ode_result['y'][:, species_list.index(sp)],
                lw=1.6, color='red', ls=(0, (5, 4)), label='species avg', zorder=8)
        # mark instants where this species' limit pair changed
        for t_j, sp_j in continuous_events:
            if sp_j == sp:
                ax.axvline(t_j, color='green', lw=0.8, ls='--', alpha=0.6)
        for t_j, sp_j in jump_events:
            if sp_j == sp:
                ax.axvline(t_j, color='purple', lw=0.8, ls='--', alpha=0.6)
        ax.set_title(sp, fontsize=10)
        ax.set_xlabel('t (s)', fontsize=8)
        ax.set_ylabel('beta-avg of limit', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=_leg_fs, loc='best', frameon=False, ncol=max(1, len(labels) // 6))

    for plot_idx in range(n_plot, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f'Limit beta-averages vs time  (mean_f={mean_f:.4f},  '
        f'ε={m_epsilon:.4g}, τ_s={mixing_timescale(m_epsilon, m_lambda):.4g} s,  weights={weight_method})',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_limit_averages_{weight_method}.png')
    else:
        plt.show()


def plot_ode_beta_snapshots(unique_sp_data, ode_result, save_stem=None, f_xlim=None):
    """Beta-plot snapshots across an ODE integration result.

    Four columns: the first near 1/96 of the reaction time, the remaining three
    spread evenly to t_end.  Rows = active species.
    Each cell shows coloured horizontal lines at
    the per-profile beta-weighted average, black dashed blended C_w(f), Beta
    PDF on a secondary axis, and coloured weight values in the subplot title.
    """
    import pathlib as _pathlib

    t_arr = ode_result['t']
    y_arr = ode_result['y']
    active_indices = ode_result['active_indices']
    mean_f = ode_result['mean_f']
    var_start = ode_result['var_start']
    var_end = ode_result.get('var_end', 0.0)
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ode_result.get('m_lambda', DEFAULT_M_LAMBDA)
    m_nu = ode_result.get('m_nu', DEFAULT_M_NU)
    m_Sc = ode_result.get('m_Sc', DEFAULT_M_SC)
    weight_method = ode_result.get('weight_method', 'linear_interp')
    species_list = ode_result['species']

    sp_profiles = ode_result['sp_profiles']
    active_species = [species_list[i] for i in active_indices if species_list[i] in sp_profiles]
    n_sp = len(active_species)
    if n_sp == 0:
        return

    max_var = mean_f * (1.0 - mean_f)
    _tab_palette = _build_tab_palette()

    # Four time snapshots: the first near 1/96 of the reaction time (a very
    # segregated, early-mixing state rather than t=0), the rest spread evenly.
    n_t = len(t_arr)
    _T = float(t_arr[-1] - t_arr[0])
    target_times = np.linspace(t_arr[0], t_arr[-1], 4)
    target_times[0] = t_arr[0] + _T / 96.0
    snap_idxs = [int(np.argmin(np.abs(t_arr - tt))) for tt in target_times]
    # keep columns in chronological order
    snap_idxs = sorted(snap_idxs)
    snap_labels = [f't={t_arr[i]:.4g} s' for i in snap_idxs]

    def _var_at(t_snap):
        return mixing_variance(t_snap, mean_f, m_epsilon, m_lambda, m_nu, m_Sc)

    _bf = ode_result.get('blendfs')        # blend_fs: draw M/B/C curves per cell
    _rl = ode_result.get('raylimit')      # ray_limit: draw M/B/C curves per cell
    _bf_fgrid = np.linspace(0.0, 1.0, 200)

    # ray_limit only: the current selectivity ray d=y-y0, one compact
    # reactants→products string per snapshot column (species-independent).
    _ray_source = _rl
    _ray_eqs = None
    if _ray_source is not None:
        _Y1r, _Y2r = np.asarray(_ray_source['Y1']), np.asarray(_ray_source['Y2'])
        _y0_active = mean_f * _Y1r[active_indices] + (1.0 - mean_f) * _Y2r[active_indices]
        _ray_eqs = [_ray_equation_str(y_arr[ti, active_indices] - _y0_active,
                                       species_list, active_indices, rel_tol=0.02)
                    for ti in snap_idxs]

    ncols = 4
    nrows = n_sp
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for row, sp in enumerate(active_species):
        sp_global_idx = species_list.index(sp)
        profiles = dict(sp_profiles[sp])
        profile_list = list(profiles.items())
        styles = [_tab_palette[i % len(_tab_palette)] for i in range(len(profile_list))]

        for col, (t_idx, col_label) in enumerate(zip(snap_idxs, snap_labels)):
            ax = axes[row][col]
            t_snap = t_arr[t_idx]
            var_t = _var_at(t_snap)
            s_t = max_var / var_t - 1.0
            alpha_t = mean_f * s_t
            beta_t = (1.0 - mean_f) * s_t

            average_val = max(float(y_arr[t_idx, sp_global_idx]), 0.0)
            cell_title = f'{sp}  {col_label}'
            if _bf is not None and sp in _bf['prof']:
                # blend_fs: show M(f), the FULL blended B(f) (multi-segment
                # over bf['bp'], not just the single-kink v0/vfs/v1 -- a
                # catalyzed reaction's near-boundary kink needs the fuller
                # resolution) and the interpolated C(f)=B+(M-B)·λ.
                v0, vfs, v1, lam = _bf['prof'][sp][t_idx]
                fsb = float(_bf['fsb'][t_idx])
                Mi = float(_bf['Y1'][sp_global_idx]) * _bf_fgrid + \
                    float(_bf['Y2'][sp_global_idx]) * (1.0 - _bf_fgrid)
                _bprof = None
                if 'bp' in _bf and sp in _bf.get('prof_full', {}):
                    _v_arr = _bf['prof_full'][sp][t_idx]
                    if np.all(np.isfinite(_v_arr)):
                        _bprof = (_bf['bp'], _v_arr)
                _draw_blendfs_cell(ax, cell_title, _bf_fgrid, Mi, v0, vfs, v1, fsb, lam,
                                   alpha_t, beta_t, average_val, xlim=f_xlim, B_profile=_bprof)
            elif _rl is not None and sp in _rl['prof']:
                # ray_limit: draw the FULL staged complete-reaction limit B(f)
                # (multi-segment for catalytic systems) and C(f)=B+(M-B)·λ.
                v0, vfs, v1, lam = _rl['prof'][sp][t_idx]
                fsb = float(_rl['fsb'][t_idx])
                Mi = float(_rl['Y1'][sp_global_idx]) * _bf_fgrid + \
                    float(_rl['Y2'][sp_global_idx]) * (1.0 - _bf_fgrid)
                _bprof = None
                _bps_t = _rl.get('limit_bps', [None] * (t_idx + 1))[t_idx]
                if _bps_t is not None and sp_global_idx in list(active_indices):
                    _ai_pos = list(active_indices).index(sp_global_idx)
                    _bprof = (_bps_t, _rl['limit_B'][t_idx][:, _ai_pos])
                _draw_blendfs_cell(ax, cell_title, _bf_fgrid, Mi, v0, vfs, v1, fsb, lam,
                                   alpha_t, beta_t, average_val, xlim=f_xlim,
                                   fs_label='fs_ray', B_profile=_bprof)
            else:
                _draw_cw_cell(ax, cell_title, profiles, alpha_t, beta_t,
                              average_val, weight_method, styles, t=t_snap, xlim=f_xlim)

            if row == 0 and _ray_eqs is not None and _ray_eqs[col] is not None:
                _lhs, _rhs = _ray_eqs[col].split(' → ')
                ax.text(0.5, 0.99, f'd: {_lhs}\n→ {_rhs}', transform=ax.transAxes,
                        ha='center', va='top', fontsize=6, zorder=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.85, ec='none'))

        # Share left-axis y-range across all columns in this row.
        # For the zoomed plot, base the range only on data within the visible x-range.
        if f_xlim is not None:
            row_ymin, row_ymax = np.inf, -np.inf
            for col in range(ncols):
                for line in axes[row][col].get_lines():
                    xd = np.asarray(line.get_xdata(), dtype=float)
                    yd = np.asarray(line.get_ydata(), dtype=float)
                    if len(xd) == 0:
                        continue
                    mask = (xd >= f_xlim[0]) & (xd <= f_xlim[1])
                    if mask.any():
                        row_ymin = min(row_ymin, float(np.min(yd[mask])))
                        row_ymax = max(row_ymax, float(np.max(yd[mask])))
            if not np.isfinite(row_ymin) or row_ymin >= row_ymax:
                row_ymin, row_ymax = 0.0, 1.0
            _ym = 0.05 * (row_ymax - row_ymin)
            row_ymin -= _ym
            row_ymax += _ym
        else:
            ylims = [axes[row][col].get_ylim() for col in range(ncols)]
            row_ymin = min(lo for lo, _ in ylims)
            row_ymax = max(hi for _, hi in ylims)
        for col in range(ncols):
            axes[row][col].set_ylim(row_ymin, row_ymax)

    fig.suptitle(
        f'C_w(f) snapshots along ODE  (mean_f={mean_f:.4f},  '
        f'var {var_start:.4f}→{var_end:.2e}  (ε={m_epsilon:.4g}, τ_s={mixing_timescale(m_epsilon, m_lambda):.4g} s),  weights={weight_method})',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_stem is not None:
        _suffix = '_zoom' if f_xlim is not None else ''
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_snapshots_{weight_method}{_suffix}.png')
    else:
        plt.show()


def _draw_blendfs_cell(ax, title, f_grid, Mi, v0, vfs, v1, fsb, lam,
                       alpha, beta_p, average_val, xlim=None, fs_label='fs_b',
                       B_profile=None):
    """Draw one snapshot cell: the no-reaction limit M(f), the complete-reaction
    limit B(f), and the interpolated species profile C(f) = B + (M-B)·λ, with the
    Beta PDF and the species mean for reference.

    By default B(f) is the single-kink limit (v0, vfs, v1, kink at fsb).  Passing
    ``B_profile=(bps, Bvals_1d)`` instead draws the full piecewise-linear staged
    limit (used by ray_limit, whose catalytic limit is multi-segment)."""

    if B_profile is not None:
        _bps, _Bv = B_profile
        Bf = np.interp(f_grid, _bps, _Bv)
    else:
        # Single-kink limit B(f): two segments, kink at fsb.
        Bf = np.where(f_grid <= fsb,
                      v0 + (vfs - v0) * (f_grid / fsb),
                      vfs + (v1 - vfs) * (f_grid - fsb) / (1.0 - fsb))
    Cf = lam * Mi + (1.0 - lam) * Bf

    ax.plot(f_grid, Mi, color='grey', lw=1.4, ls='--', label='no reaction M')
    ax.plot(f_grid, Bf, color='black', lw=1.8, label='blend B')
    ax.plot(f_grid, Cf, color=CB_PALETTE[0], lw=2.0, label='species C')
    ax.axhline(average_val, color=CB_PALETTE[1], lw=1.5, ls=':', zorder=4)
    ax.annotate(f'average={average_val:.4g}', xy=(0.97, average_val),
                xycoords=('axes fraction', 'data'), fontsize=6, ha='right',
                va='bottom', color=CB_PALETTE[1], zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none'))

    # Beta-average of C(f): equals 'average' when the mean is matched, diverges
    # from it where λ has clamped.
    mM = Mi[-1] - Mi[0]                                  # M(f) slope (=Y1-Y2)
    mB = Mi[0]                                            # ... intercept (=Y2)
    e_M = _beta_segment_integral(mM, mB, 0.0, 1.0, alpha, beta_p)
    if B_profile is not None:
        _bps, _Bv = B_profile
        e_B = 0.0
        for _s in range(len(_bps) - 1):
            _fa, _fb = _bps[_s], _bps[_s + 1]
            if _fb - _fa < 1e-15:
                continue
            _sl = (_Bv[_s + 1] - _Bv[_s]) / (_fb - _fa)
            e_B += _beta_segment_integral(_sl, _Bv[_s] - _sl * _fa, _fa, _fb, alpha, beta_p)
    else:
        Bs0, Bi0 = (vfs - v0) / fsb, v0
        Bs1, Bi1 = (v1 - vfs) / (1.0 - fsb), vfs - (v1 - vfs) / (1.0 - fsb) * fsb
        e_B = (_beta_segment_integral(Bs0, Bi0, 0.0, fsb, alpha, beta_p)
               + _beta_segment_integral(Bs1, Bi1, fsb, 1.0, alpha, beta_p))
    e_C = lam * e_M + (1 - lam) * e_B
    ax.axhline(e_C, color=CB_PALETTE[0], lw=1.5, ls='-.', zorder=4)
    ax.annotate(f'E[C]={e_C:.4g}', xy=(0.03, e_C),
                xycoords=('axes fraction', 'data'), fontsize=6, ha='left',
                va='top', color=CB_PALETTE[0], zorder=7,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none'))

    f_dense = np.linspace(1e-4, 1.0 - 1e-4, 100)
    ax_pdf = ax.twinx()
    ax_pdf.plot(f_dense, _stats.beta.pdf(f_dense, alpha, beta_p), color='lightgrey', lw=1.0, zorder=0)
    ax_pdf.set_ylabel('β-PDF', fontsize=7, color='grey')
    ax_pdf.tick_params(axis='y', labelcolor='grey', labelsize=6)
    ax_pdf.set_ylim(bottom=0)
    ax.set_zorder(ax_pdf.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.set_title(f'{title}  (λ={lam:.2f}, {fs_label}={fsb:.3f})', fontsize=8)
    ax.set_xlabel('f', fontsize=8)
    ax.set_ylabel('C(f)', fontsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)


def _beta_segment_integral(slope, intercept, lo, hi, alpha, beta_p):
    """Exact integral of (slope*f + intercept) * Beta(f; alpha, beta_p) over [lo, hi]."""
    dI0 = _betainc_top(alpha,     beta_p, hi) - _betainc_top(alpha,     beta_p, lo)
    dI1 = _betainc_top(alpha + 1, beta_p, hi) - _betainc_top(alpha + 1, beta_p, lo)
    return slope * (alpha / (alpha + beta_p)) * dI1 + intercept * dI0


def _draw_cw_cell(ax, title, profiles, alpha, beta_p, average_val, weight_method, styles, t=None, xlim=None):
    """Draw one beta-weighted C_w subplot. Returns (weights, e_cw)."""

    # Cell size matches plot_ode_beta_snapshots's grid (figsize=(4.5*ncols, 3.5*nrows)).
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.5, 3.5)
    profile_list = list(profiles.items())
    _MAX_LABEL = 20

    # Beta-weighted average per profile
    avgs = {}
    for lbl, prof in profile_list:
        avgs[lbl] = sum(
            _beta_segment_integral(prof['segments'][j][0], prof['segments'][j][1],
                                   prof['breakpoints'][j], prof['breakpoints'][j+1],
                                   alpha, beta_p)
            for j in range(len(prof['segments'])))

    e_vals = np.array([avgs[lbl] for lbl, _ in profile_list])

    # Compute weights first so active profiles can be drawn thicker.
    weights = _compute_cw_weights(profile_list, e_vals, average_val, weight_method, species=title, t=t, silent=True)
    _active = np.where(weights > 1e-10)[0]

    # Horizontal lines at beta-weighted averages, with labels.
    # Active-pair profiles are drawn with a thicker line.
    for i, (lbl, _) in enumerate(profile_list):
        val = avgs[lbl]
        f_g = np.array([0.0, 1.0])
        lw = _lw * (2.5 / 1.5) if i in _active else _lw * (1.2 / 1.5)
        color, ls, mk = styles[i]
        display = lbl if len(lbl) <= _MAX_LABEL else lbl[:_MAX_LABEL - 3] + '...'
        ax.plot(f_g, [val, val], color=color, linestyle=ls, marker=mk, mfc=_face(color),
                ms=_ms, lw=lw, label=display)
    if len(profile_list) > 1:
        ax.legend(loc='best', frameon=False, fontsize=_leg_fs)
    _f_cw = _profile_f_grid(profile_list)
    # Draw the actual C(f) profiles for the bracketing pair (the active, non-zero
    # weight limits) so the weighted blend is seen to interpolate between them.
    for i in _active:
        _w_one = np.zeros(len(profile_list))
        _w_one[i] = 1.0
        color, ls, mk = styles[i]
        ax.plot(_f_cw, _eval_cw_on_grid(profile_list, _w_one, _f_cw),
                color=color, lw=_lw * (1.6 / 1.5), ls=ls, marker=mk, mfc=_face(color),
                markevery=max(1, len(_f_cw) // 15), ms=_ms + 1, alpha=0.9, zorder=4)
    y_weighted = _eval_cw_on_grid(profile_list, weights, _f_cw)

    ax.plot(_f_cw, y_weighted, color='black', lw=1.8, ls='--', zorder=5)
    mid = len(_f_cw) // 2
    ax.annotate('weighted', xy=(_f_cw[mid], y_weighted[mid]),
                fontsize=7, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none'), zorder=6)

    e_cw = float(weights @ e_vals)
    _bbox = dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, ec='none')
    ax.axhline(average_val, color='orange', lw=2.0, ls=':', zorder=4)
    ax.annotate(f'average={average_val:.4g}',
                xy=(0.97, average_val), xycoords=('axes fraction', 'data'),
                fontsize=6, ha='right', va='bottom', color='orange', bbox=_bbox, zorder=7)
    ax.axhline(e_cw, color='red', lw=2.0, ls=':', zorder=4)
    ax.annotate(f'E[C_w]={e_cw:.4g}',
                xy=(0.97, e_cw), xycoords=('axes fraction', 'data'),
                fontsize=6, ha='right', va='top', color='red', bbox=_bbox, zorder=7)

    # Beta PDF on twin axis
    f_dense = np.linspace(1e-4, 1.0 - 1e-4, 100)
    ax_pdf = ax.twinx()
    ax_pdf.plot(f_dense, _stats.beta.pdf(f_dense, alpha, beta_p), color='lightgrey', lw=1.0, zorder=0)
    ax_pdf.set_ylabel('β-PDF', fontsize=7, color='grey')
    ax_pdf.tick_params(axis='y', labelcolor='grey', labelsize=6)
    ax_pdf.set_ylim(bottom=0)
    ax.set_zorder(ax_pdf.get_zorder() + 1)
    ax.patch.set_visible(False)

    # Title with coloured weights
    ax.set_title(f'{title}  {weight_method}', fontsize=9, pad=20)
    _n_w = len(weights)
    _wy = 1.04
    ax.text(0.0, _wy, 'w=[', transform=ax.transAxes, fontsize=7, ha='left', va='bottom', clip_on=False)
    for _i, (_wv, _wc) in enumerate(zip(weights, colors)):
        _wx = 0.1 + _i * 0.8 / max(_n_w - 1, 1)
        _fw = 'bold' if _wv > 1e-10 else 'normal'
        ax.text(_wx, _wy, f'{_wv:.1f}', transform=ax.transAxes,
                fontsize=7, ha='center', va='bottom', color=_wc, fontweight=_fw, clip_on=False)
    ax.text(1.0, _wy, ']', transform=ax.transAxes, fontsize=7, ha='right', va='bottom', clip_on=False)

    ax.set_xlabel('f', fontsize=8)
    ax.set_ylabel('E[Y]', fontsize=8)
    if xlim is not None:
        ax.set_xlim(xlim)

    return weights, e_cw


def _profile_f_grid(profile_list, eps=1e-9):
    """Sorted f-grid covering all profile breakpoints with ±eps sentinel points."""
    bp_set = set()
    for _, prof in profile_list:
        for b in prof['breakpoints']:
            bp_set.add(b)
            if b > 0.0: bp_set.add(b - eps)
            if b < 1.0: bp_set.add(b + eps)
    return np.array(sorted(bp_set))


def _eval_cw_on_grid(profile_list, weights, f_grid):
    """Evaluate blended C_w(f) = Σ_n w_n * C_n(f) on f_grid."""
    y = np.zeros(len(f_grid))
    for w, (_lbl, prof) in zip(weights, profile_list):
        if abs(w) < 1e-15:
            continue
        bps = prof['breakpoints']
        segs = prof['segments']
        for j in range(len(segs)):
            lo, hi = bps[j], bps[j + 1]
            mask = (f_grid >= lo) & (f_grid < hi) if j < len(segs) - 1 else (f_grid >= lo) & (f_grid <= hi)
            y[mask] += w * (segs[j][0] * f_grid[mask] + segs[j][1])
    return y


def plot_blendfs_diagnostics(ode_result, save_stem=None):
    """For a blend_fs ODE run, plot the time-varying per-subset blend weights and
    the blended fs, with each blended subset's fs as a reference line."""
    import pathlib as _pathlib

    bf = ode_result.get('blendfs')
    if bf is None:
        return
    t = ode_result['t']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    subs, fs, misses, W = bf['subs'], bf['fs'], bf['misses'], bf['w']
    n = len(subs)
    _tab = _build_tab_palette()
    styles = [_tab[i % len(_tab)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    _ms, _lw, _leg_fs = _scaled_marker_lw(8, 5)
    _mk_every_bf = max(1, len(t) // 15)
    for k in range(n):
        color, ls, mk = styles[k]
        ax.plot(t, W[:, k], color=color, linestyle=ls, marker=mk, mfc=_face(color),
                markevery=_mk_every_bf, ms=_ms, lw=_lw,
                label=f"w (subset {subs[k]}, misses {misses[k]})")
    ax.set_xlabel('t (s)', fontsize=9)
    ax.set_ylabel('blend weight', fontsize=9)
    ax.set_ylim(-0.02, 1.02)

    ax2 = ax.twinx()
    ax2.plot(t, bf['fsb'], color='black', lw=2.0, label='fs_blend')
    for k in range(n):
        color, _, _ = styles[k]
        ax2.axhline(fs[k], color=color, ls=':', alpha=0.6)
    _annotate_fs_endpoints(ax2, t, bf['fsb'])
    _fs_ylim = _fs_axis_ylim(bf['fsb'])
    if _fs_ylim is not None:
        ax2.set_ylim(*_fs_ylim)
    ax2.set_ylabel('fs', fontsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _fit_legend_avoiding_curves(ax, h1 + h2, l1 + l2, _leg_fs, other_axes=(ax2,), loc='upper left')
    ax.set_title(f'blend_fs: {n} blended limits — weights & blended fs vs time  '
                 f'(ε={m_epsilon:.4g})', fontsize=10)
    fig.tight_layout()

    if save_stem is not None:
        _wm = ode_result.get('weight_method', 'blend_fs')
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_{_wm}_diag.png')
    else:
        plt.show()


def _annotate_fs_endpoints(ax, t, fsb):
    """Label the first and last finite fs value just above the two ends of the
    fs line (with a small marker at each end)."""
    fsb = np.asarray(fsb, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.where(np.isfinite(fsb))[0]
    if finite.size == 0:
        return
    ends = [(int(finite[0]), 'left')]
    if int(finite[-1]) != int(finite[0]):
        ends.append((int(finite[-1]), 'right'))
    for idx, ha in ends:
        ax.plot(t[idx], fsb[idx], marker='o', ms=4, color='black', zorder=6)
        ax.annotate(f"{fsb[idx]:.3f}", xy=(t[idx], fsb[idx]),
                    xytext=(3 if ha == 'left' else -3, 6),
                    textcoords='offset points', ha=ha, va='bottom',
                    fontsize=8, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.75), zorder=7)


def _fs_axis_ylim(fsb, pad=0.05):
    """y-limits for a twin fs axis: the full finite range of *fsb*, widened if
    needed so it always spans at least ±*pad* around the final finite value."""
    fsb = np.asarray(fsb, dtype=float)
    finite = fsb[np.isfinite(fsb)]
    if finite.size == 0:
        return None
    fs_final = float(finite[-1])
    lo = max(0.0, min(float(finite.min()), fs_final - pad))
    hi = max(float(finite.max()), fs_final + pad)
    return lo, hi


def plot_raylimit_diagnostics(ode_result, save_stem=None):
    """For a ray_limit ODE run, plot the time-varying selectivity ray
    d(t) = y(t) − y0 (one line per active species, left axis) and the
    resulting complete-reaction fs (right axis)."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None:
        return
    t = np.asarray(ode_result['t'])
    y_full = np.asarray(ode_result['y'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    if not active_indices:
        return

    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    d = y_full[:, active_indices] - y0_active[None, :]

    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}

    # Detect any species whose |d| dwarfs every OTHER SPECIES FED BY THE SAME
    # STREAM (e.g. a solvent fed in vast excess, like water in the acetal
    # case); scale it down to ~1000.  A stream-1 concentrate is never
    # compared against stream-2 species (or vice versa).
    gmax = {s: float(np.max(np.abs(d[:, k]))) for k, s in enumerate(active_indices)}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: Y1[s] > 0 for s in active_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr

    fig, ax = plt.subplots(figsize=(8, 5))
    _ms, _lw, _leg_fs = _scaled_marker_lw(8, 5)
    _mk_every = max(1, len(t) // 15)
    for k, s in enumerate(active_indices):
        ax.plot(t, _yv(s, d[:, k]), color=color_map[s], linestyle=ls_map[s], lw=_lw,
                marker=mk_map[s], markevery=_mk_every, mfc=_face(color_map[s]), ms=_ms, label=_lbl(s))
    ax.axhline(0.0, color='grey', lw=1.0, ls='--', zorder=1)
    ax.set_xlabel('t (s)', fontsize=9)
    ax.set_ylabel('d = y − y0  (consumed < 0, produced > 0)', fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(t, rl['fsb'], color='black', lw=2.0, label='fs (complete-reaction)')
    _annotate_fs_endpoints(ax2, t, rl['fsb'])
    _fs_ylim = _fs_axis_ylim(rl['fsb'])
    if _fs_ylim is not None:
        ax2.set_ylim(*_fs_ylim)
    ax2.set_ylabel('fs', fontsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _fit_legend_avoiding_curves(ax, h1 + h2, l1 + l2, _leg_fs, other_axes=(ax2,), loc='upper left')
    ax.set_title(f'ray_limit: selectivity ray d(t) & complete-reaction fs vs time  '
                 f'(ε={m_epsilon:.4g})', fontsize=10)
    fig.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_raylimit_diag.png')
    else:
        plt.show()


def plot_raylimit_rotation(ode_result, save_stem=None):
    """For a ray_limit ODE run, quantify and plot how much the selectivity ray
    d(t)=y(t)-y0 rotates over the course of the run.

    Top panel: the angle between d(t) and the FINAL ray d(t_end) — the
    closest available proxy for "the settled, asymptotically-dominant
    selectivity" this run reaches.  Starts high (early ray misaligned with
    where it ends up) and falls toward 0 as the ray settles; a late dip back
    up would be a direct signature of a genuine late-run selectivity shift.

    Bottom panel: the step-to-step rotation rate — the angle between
    consecutive d(t_k), d(t_{k+1}) — showing WHEN the ray is currently
    swinging fastest, rather than how much rotation is left overall.

    Also prints t_99 (first time the angle-to-final drops below ~8.1°, i.e.
    cosine similarity >= 0.99) alongside the mixing timescale tau_s, so you
    can compare how long the ray takes to settle against how long mixing
    itself takes."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None:
        return
    t = np.asarray(ode_result['t'])
    y_full = np.asarray(ode_result['y'])
    active_indices = list(ode_result['active_indices'])
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ode_result.get('m_lambda', DEFAULT_M_LAMBDA)
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    if not active_indices:
        return

    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    d = y_full[:, active_indices] - y0_active[None, :]
    norms = np.linalg.norm(d, axis=1)
    tiny = 1e-12 * max(float(np.max(norms)), 1.0)

    def _angle_deg(u, v, nu, nv):
        if nu < tiny or nv < tiny:
            return np.nan
        cos_th = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_th)))

    d_final, n_final = d[-1], norms[-1]
    angle_to_final = np.array([_angle_deg(d[k], d_final, norms[k], n_final)
                               for k in range(len(t))])
    step_angle = np.array([_angle_deg(d[k], d[k + 1], norms[k], norms[k + 1])
                           for k in range(len(t) - 1)])
    t_mid = 0.5 * (t[:-1] + t[1:])

    finite = np.isfinite(angle_to_final)
    settled = finite & (angle_to_final <= 8.1)   # ~8.1° <=> cosine similarity 0.99
    t99_idx = int(np.where(settled)[0][0]) if settled.any() else None
    t99 = float(t[t99_idx]) if t99_idx is not None else None
    tau_s = mixing_timescale(m_epsilon, m_lambda)
    if t99 is not None:
        # Conversion of the stream-1 limiting reactant at t_99, if identifiable.
        L_conv_str = ""
        L_name = ode_result.get('stream1_closure', {}).get('limiting_reactant')
        species_list = ode_result['species']
        if L_name is not None and L_name in species_list:
            L_global = species_list.index(L_name)
            if L_global in active_indices:
                y0_L = float(y0_active[active_indices.index(L_global)])
                if y0_L:
                    L_conv = 1.0 - float(y_full[t99_idx, L_global]) / y0_L
                    L_conv_str = f";  conversion of {L_name} at t_99 = {100.0 * L_conv:.4g}%"
        print(f"[ray_limit]   [rotation] settling time t_99 (angle-to-final <= 8.1°) "
              f"= {t99:.4g} s;  mixing timescale tau_s = {tau_s:.4g} s;  "
              f"t_99/tau_s = {t99 / tau_s:.3g}{L_conv_str}")
    else:
        print(f"[ray_limit]   [rotation] never settles to within 8.1° of the final ray "
              f"within this run (tau_s = {tau_s:.4g} s).")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(t[finite], angle_to_final[finite], color='steelblue', lw=1.8)
    ax1.axhline(8.1, color='grey', lw=1.0, ls='--', alpha=0.7, label='8.1° (cos = 0.99)')
    if t99 is not None:
        ax1.axvline(t99, color='k', lw=1.0, ls=':', alpha=0.7)
        ax1.annotate(f't_99={t99:.3g} s', xy=(t99, 8.1), xytext=(5, 8), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.75))
    ax1.set_ylabel('angle to final ray (°)', fontsize=9)
    ax1.legend(fontsize=7, loc='upper right', frameon=False)
    ax1.set_title(f'ray_limit: selectivity-ray rotation vs time  (ε={m_epsilon:.4g},  '
                 f'τ_s={tau_s:.4g} s)', fontsize=10)

    step_finite = np.isfinite(step_angle)
    ax2.plot(t_mid[step_finite], step_angle[step_finite], color='indianred', lw=1.2)
    ax2.set_ylabel('step-to-step\nrotation (°)', fontsize=9)
    ax2.set_xlabel('t (s)', fontsize=9)
    fig.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_raylimit_rotation.png')
    else:
        plt.show()


def plot_raylimit_selectivity(ode_result, save_stem=None):
    """For a ray_limit ODE run, plot every other active species' ray
    component d_i(t) against -d_A(t) (extent consumed of the dominant
    stream-1-only species), instead of against time.

    This is the standard reaction-engineering selectivity plot (yield vs.
    conversion rather than vs. time), and it makes the ray's rotation and lag
    visible geometrically rather than needing a reference direction: the
    secant slope from the origin to the current point is d_i/d_A — the
    ACCUMULATED ray's current direction, i.e. what the ray currently "thinks"
    the selectivity is.  The local tangent slope is the derivative
    d(d_i)/d(d_A) — the TRUE instantaneous rate ratio right now.  A straight
    line through the origin means the two always agree (no lag); curvature is
    rotation happening in real time, and the secant/tangent gap at any point
    is exactly the lag, made visible without needing the final ray as a
    reference."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None:
        return
    y_full = np.asarray(ode_result['y'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    if not active_indices:
        return

    stream_labels = identify_stream_feeds(Y1, Y2)
    s1_candidates = [s for s in active_indices if stream_labels[s] in (1, 12) and Y1[s] > 0]
    if not s1_candidates:
        print("[ray_limit]   [selectivity-plot] no pure-stream-1 species among "
              "active species; skipping.")
        return
    ref = max(s1_candidates, key=lambda s: Y1[s])
    ref_pos = active_indices.index(ref)

    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    d = y_full[:, active_indices] - y0_active[None, :]
    x = -d[:, ref_pos]

    # Detect any (non-reference) species whose |d| dwarfs every OTHER SPECIES
    # FED BY THE SAME STREAM (e.g. a solvent fed in vast excess, like water
    # in the acetal case); scale it down to ~1000 -- applied to d before the
    # slope/R² fits below, so the dashed best-fit lines and annotations stay
    # consistent with the curves.
    other_indices = [s for s in active_indices if s != ref]
    gmax = {s: float(np.max(np.abs(d[:, k]))) for k, s in enumerate(active_indices) if s != ref}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: Y1[s] > 0 for s in other_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    div = np.array([div_of.get(s, 1.0) for s in active_indices])
    d = d / div[None, :]

    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}

    fig, ax = plt.subplots(figsize=(7, 6))
    _ms, _lw, _leg_fs = _scaled_marker_lw(7, 6)
    other_cols = [k for k, s in enumerate(active_indices) if s != ref]
    label_info = []
    _mk_every = max(1, len(x) // 15)
    for k, s in enumerate(active_indices):
        if s == ref:
            continue
        ax.plot(x, d[:, k], color=color_map[s], linestyle=ls_map[s], lw=_lw,
                marker=mk_map[s], markevery=_mk_every, mfc=_face(color_map[s]), ms=_ms, label=_lbl(s))
        xx = float(np.sum(x * x))
        if xx > 0:
            slope = float(np.sum(x * d[:, k]) / xx)
            x_end = x.max() if abs(x.max()) >= abs(x.min()) else x.min()
            ax.plot([x.min(), x.max()], [slope * x.min(), slope * x.max()],
                    color=color_map[s], lw=1.2, ls='--', zorder=2)
            yy = float(np.sum(d[:, k] ** 2))
            if yy > 0:
                r2 = 1.0 - float(np.sum((d[:, k] - slope * x) ** 2)) / yy
                label_info.append((color_map[s], r2, x_end, slope * x_end))
    ax.axhline(0.0, color='grey', lw=1.0, ls='--', zorder=1)

    if label_info:
        y_range = float(np.nanmax(d[:, other_cols]) - np.nanmin(d[:, other_cols])) or 1.0
        x_span = float(x.max() - x.min()) or 1.0
        x_label = x.max() + 0.16 * x_span
        min_gap = 0.09 * y_range
        order = sorted(range(len(label_info)), key=lambda i: label_info[i][3])
        adj_y = [label_info[i][3] for i in order]
        for i in range(1, len(adj_y)):
            if adj_y[i] - adj_y[i - 1] < min_gap:
                adj_y[i] = adj_y[i - 1] + min_gap
        for pos, i in enumerate(order):
            color, r2, x_end, y_end = label_info[i]
            ax.annotate(f'R²={r2:.3f}', xy=(x_end, y_end), xytext=(x_label, adj_y[pos]),
                        color=color, fontsize=10, ha='left', va='center',
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.6, alpha=0.6))
        ax.set_xlim(x.min(), x_label + 0.22 * x_span)
        ymin, ymax = ax.get_ylim()
        if max(adj_y) > ymax:
            ymax = max(adj_y) + 0.05 * y_range
        if min(adj_y) < ymin:
            ymin = min(adj_y) - 0.05 * y_range
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(f'−d_{species_list[ref]}  (extent of {species_list[ref]} consumed)',
                 fontsize=9)
    ax.set_ylabel('d = y − y0  (other species)', fontsize=9)
    ax.legend(fontsize=_leg_fs, loc='best', frameon=False, ncol=min(3, len(active_indices) - 1))
    ax.set_title(f'ray_limit: selectivity vs. {species_list[ref]} consumed  '
                 f'(ε={m_epsilon:.4g})', fontsize=10)
    fig.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_raylimit_selectivity.png')
    else:
        plt.show()


def plot_raylimit_early_selectivity(ode_result, conversion_cutoff=0.20, save_stem=None):
    """Same secant/best-fit/R² selectivity view as `plot_raylimit_selectivity`,
    but restricted to the early part of the run — up to `conversion_cutoff`
    (fraction of the fed, mean_f-weighted amount) of the dominant stream-1
    species consumed.  fs is overlaid on a secondary axis so the early fs
    swing can be compared directly against how linear (or not) the ray's
    individual species components are over that same early window."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None:
        return
    y_full = np.asarray(ode_result['y'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    fsb = np.asarray(rl['fsb'])
    if not active_indices:
        return

    stream_labels = identify_stream_feeds(Y1, Y2)
    s1_candidates = [s for s in active_indices if stream_labels[s] in (1, 12) and Y1[s] > 0]
    if not s1_candidates:
        print("[ray_limit]   [early-selectivity-plot] no pure-stream-1 species among "
              "active species; skipping.")
        return
    ref = max(s1_candidates, key=lambda s: Y1[s])
    ref_pos = active_indices.index(ref)

    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    d_full = y_full[:, active_indices] - y0_active[None, :]
    x_full = -d_full[:, ref_pos]
    conv = x_full / y0_active[ref_pos]

    mask = conv <= conversion_cutoff
    if mask.sum() < 2:
        print(f"[ray_limit]   [early-selectivity-plot] fewer than 2 points below "
              f"{100.0 * conversion_cutoff:.4g}% conversion; skipping.")
        return
    d = d_full[mask]
    x = x_full[mask]
    fsb_early = fsb[mask]

    # Detect any (non-reference) species whose |d| dwarfs every OTHER SPECIES
    # FED BY THE SAME STREAM (e.g. a solvent fed in vast excess, like water
    # in the acetal case); scale it down to ~1000 -- applied to d before the
    # slope/R² fits below, so the dashed best-fit lines and annotations stay
    # consistent with the curves.
    other_indices = [s for s in active_indices if s != ref]
    gmax = {s: float(np.max(np.abs(d[:, k]))) for k, s in enumerate(active_indices) if s != ref}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: Y1[s] > 0 for s in other_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    div = np.array([div_of.get(s, 1.0) for s in active_indices])
    d = d / div[None, :]

    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}

    fig, ax = plt.subplots(figsize=(7, 6))
    _ms, _lw, _leg_fs = _scaled_marker_lw(7, 6)
    other_cols = [k for k, s in enumerate(active_indices) if s != ref]
    label_info = []
    _mk_every = max(1, len(x) // 15)
    for k, s in enumerate(active_indices):
        if s == ref:
            continue
        ax.plot(x, d[:, k], color=color_map[s], linestyle=ls_map[s], lw=_lw,
                marker=mk_map[s], markevery=_mk_every, mfc=_face(color_map[s]), ms=_ms, label=_lbl(s))
        xx = float(np.sum(x * x))
        if xx > 0:
            slope = float(np.sum(x * d[:, k]) / xx)
            x_end = x.max() if abs(x.max()) >= abs(x.min()) else x.min()
            ax.plot([x.min(), x.max()], [slope * x.min(), slope * x.max()],
                    color=color_map[s], lw=1.2, ls='--', zorder=2)
            yy = float(np.sum(d[:, k] ** 2))
            if yy > 0:
                r2 = 1.0 - float(np.sum((d[:, k] - slope * x) ** 2)) / yy
                label_info.append((color_map[s], r2, x_end, slope * x_end))
    ax.axhline(0.0, color='grey', lw=1.0, ls='--', zorder=1)

    if label_info:
        y_range = float(np.nanmax(d[:, other_cols]) - np.nanmin(d[:, other_cols])) or 1.0
        x_span = float(x.max() - x.min()) or 1.0
        x_label = x.max() + 0.16 * x_span
        min_gap = 0.09 * y_range
        order = sorted(range(len(label_info)), key=lambda i: label_info[i][3])
        adj_y = [label_info[i][3] for i in order]
        for i in range(1, len(adj_y)):
            if adj_y[i] - adj_y[i - 1] < min_gap:
                adj_y[i] = adj_y[i - 1] + min_gap
        for pos, i in enumerate(order):
            color, r2, x_end, y_end = label_info[i]
            ax.annotate(f'R²={r2:.3f}', xy=(x_end, y_end), xytext=(x_label, adj_y[pos]),
                        color=color, fontsize=10, ha='left', va='center',
                        arrowprops=dict(arrowstyle='-', color=color, lw=0.6, alpha=0.6))
        ax.set_xlim(x.min(), x_label + 0.22 * x_span)
        ymin, ymax = ax.get_ylim()
        if max(adj_y) > ymax:
            ymax = max(adj_y) + 0.05 * y_range
        if min(adj_y) < ymin:
            ymin = min(adj_y) - 0.05 * y_range
        ax.set_ylim(ymin, ymax)
    ax.set_xlabel(f'−d_{species_list[ref]}  (extent of {species_list[ref]} consumed)',
                 fontsize=9)
    ax.set_ylabel('d = y − y0  (other species)', fontsize=9)

    ax2 = ax.twinx()
    ax2.plot(x, fsb_early, color='black', lw=2.0, ls='-', marker='o', mfc=_face('black'), ms=_ms,
              label='fs', zorder=5)
    ax2.set_ylabel('fs', fontsize=9)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    _fit_legend_avoiding_curves(ax, h1 + h2, l1 + l2, _leg_fs, other_axes=(ax2,), loc='upper left')

    ax.set_title(f'ray_limit: early selectivity (≤{100.0 * conversion_cutoff:.4g}% '
                 f'conversion of {species_list[ref]})  (ε={m_epsilon:.4g})', fontsize=10)
    fig.tight_layout()

    if save_stem is not None:
        _save_fig(fig, save_stem, f'{_pathlib.Path(save_stem).name}_raylimit_early_selectivity.png')
    else:
        plt.show()


def plot_raylimit_limit_grid(ode_result, save_stem=None):
    """ray_limit only: the staged complete-reaction limit B(f), laid out like the
    `_species_by_subset_grid` plot — all species' coloured profiles overlaid on a
    single axes — but one subplot per snapshot time instead of per subset.

    Each subplot overlays B_i(f) for every active species at that time, using the
    SAME species→colour mapping, stream-split (stream-1 species on a twin axis),
    and `/20` scaling of a dominant species as the subset grid, so the two read
    alike.  The four times match the beta snapshots (first near 1/96 of t_end)."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None or rl.get('limit_bps') is None:
        return
    t = np.asarray(ode_result['t'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    if not active_indices:
        return
    y_full = np.asarray(ode_result['y'])
    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]

    # Snapshot times (match plot_ode_beta_snapshots): first near 1/96 of t_end.
    T = float(t[-1] - t[0])
    targets = np.linspace(t[0], t[-1], 4)
    targets[0] = t[0] + T / 96.0
    idxs = sorted(int(np.argmin(np.abs(t - tt))) for tt in targets)
    fg = np.linspace(0.0, 1.0, 400)

    # Species→colour: based on global species index so colour is consistent across plots.
    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}
    stream_labels = identify_stream_feeds(Y1, Y2)
    s1 = [s for s in active_indices if stream_labels[s] in (1, 12)]   # twin (right) axis
    s2 = [s for s in active_indices if stream_labels[s] not in (1, 12)]

    def _Bcurve(k, s):
        bps = rl['limit_bps'][k]
        if bps is None:
            return np.zeros_like(fg)
        return np.interp(fg, bps, rl['limit_B'][k][:, active_indices.index(s)])

    # Scale down any species whose max dwarfs every other species FED BY THE
    # SAME STREAM (e.g. solvent) to ~1000 -- stream-1 (twin axis) species are
    # never compared against stream-2 (main axis) species.
    gmax = {s: max(float(np.max(np.abs(_Bcurve(k, s)))) for k in idxs) for s in active_indices}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: s in s1 for s in active_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.5 * nrows),
                             squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(7.0, 4.5)
    for i, k in enumerate(idxs):
        ax = axes[i // ncols][i % ncols]
        for s in s2:
            ax.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                    marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                    ms=_ms, lw=_lw, label=_lbl(s))
        ax_r = None
        if s1:
            ax_r = ax.twinx()
            for s in s1:
                ax_r.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                          marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                          ms=_ms, lw=_lw, label=_lbl(s))
            ax_r.tick_params(axis='y', labelsize=6)
        _h, _l = ax.get_legend_handles_labels()
        if ax_r is not None:
            _h_r, _l_r = ax_r.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs)
        fsk = float(rl['fsb'][k]) if np.isfinite(rl['fsb'][k]) else None
        ttl = f't = {t[k]:.4g} s'
        if fsk is not None and 0.0 < fsk < 1.0:
            ttl += f'   (fs ≈ {fsk:.4f})'
            ax.axvline(fsk, color='k', linestyle=':', alpha=0.5)
        ray_eq = _ray_equation_str(y_full[k, active_indices] - y0_active,
                                    species_list, active_indices)
        if ray_eq is not None:
            ttl += f'\nd: {ray_eq}'
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('f', fontsize=8)
        ax.set_ylabel('B(f)  (mol/m³)', fontsize=8)
    for i in range(len(idxs), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle('ray_limit complete-reaction limit B(f) by snapshot time  '
                 f'(mean_f={mean_f:.4f},  ε={m_epsilon:.4g})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_raylimit_limit_grid.png')
    else:
        plt.show()


def _raylimit_field_grids(ode_result, f_grid=None):
    """Compute ray_limit's B(f, t) and C(f, t) on a fixed f grid for every
    active species over the run's full time history -- the shared data
    behind the C(f)/B(f) heatmap plots.  Returns (t, f_grid, B_grids,
    C_grids) where B_grids/C_grids are {species: array shape (n_t, len(f_grid))},
    or None if this run has no ray_limit data."""
    rl = ode_result.get('raylimit')
    if rl is None or rl.get('limit_bps') is None:
        return None
    if f_grid is None:
        f_grid = np.linspace(0.0, 1.0, 200)
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    prof = rl.get('prof', {})
    t = np.asarray(ode_result['t'])
    n_t = len(t)

    B_grids, C_grids = {}, {}
    for sp, sp_prof in prof.items():
        sg = species_list.index(sp)
        if sg not in active_indices:
            continue
        ai = active_indices.index(sg)
        Mi = Y1[sg] * f_grid + Y2[sg] * (1.0 - f_grid)
        B_arr = np.full((n_t, len(f_grid)), np.nan)
        C_arr = np.full((n_t, len(f_grid)), np.nan)
        for k in range(n_t):
            bps = rl['limit_bps'][k]
            if bps is None:
                continue
            bv = np.interp(f_grid, bps, rl['limit_B'][k][:, ai])
            B_arr[k] = bv
            lam = sp_prof[k, 3]
            if np.isfinite(lam):
                C_arr[k] = lam * Mi + (1.0 - lam) * bv
        B_grids[sp] = B_arr
        C_grids[sp] = C_arr
    return t, f_grid, B_grids, C_grids


def _raylimit_beta_weight_grid(ode_result, t, f_grid):
    """Beta-PDF weight on the same (t, f_grid) as :func:`_raylimit_field_grids`,
    for weighting a C(f) heatmap by how much each f actually contributes to
    the true mean concentration.  Uses the *binned* probability mass (CDF
    difference over each f cell, divided by cell width) rather than the raw
    pointwise Beta density: at the segregated (t=0) and near-fully-mixed
    (t=t_end) limits the shape parameters alpha, beta -> 0 or -> large, and
    the pointwise density diverges at f=0/1, while the binned mass stays
    finite everywhere (it's what the model actually integrates against)."""
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ode_result.get('m_lambda', DEFAULT_M_LAMBDA)
    m_nu = ode_result.get('m_nu', DEFAULT_M_NU)
    m_Sc = ode_result.get('m_Sc', DEFAULT_M_SC)
    max_var = mean_f * (1.0 - mean_f)

    edges = np.concatenate(([0.0], 0.5 * (f_grid[1:] + f_grid[:-1]), [1.0]))
    widths = np.diff(edges)

    W = np.empty((len(t), len(f_grid)))
    for k, t_k in enumerate(t):
        var_t = mixing_variance(t_k, mean_f, m_epsilon, m_lambda, m_nu, m_Sc)
        s_t = max_var / var_t - 1.0
        alpha_t, beta_t = mean_f * s_t, (1.0 - mean_f) * s_t
        cdf = _stats.beta.cdf(edges, alpha_t, beta_t)
        W[k] = (cdf[1:] - cdf[:-1]) / widths
    return W


def plot_raylimit_Cf_heatmap(ode_result, save_stem=None):
    """Heatmap of ray_limit's interpolated C(f) versus f (x) and time (y),
    one panel per species -- shows how the C(f) profile relaxes from the
    no-reaction limit M(f) (t=0) toward the complete-reaction limit B(f) over
    the course of the run.  Uses the colour-blind-safe, perceptually-uniform
    'viridis' colormap.

    Each panel's colour scale is fixed to a feed concentration rather than
    autoscaled to its own data, so scales are comparable across panels: a
    species fed by stream 1 scales to the highest stream-1 feed
    concentration; a species fed by stream 2, or fed by neither stream
    (a pure reaction product), scales to the highest stream-2 feed
    concentration.

    Red solid contours trace the 0.001/0.01/0.1/0.5/0.9-of-max levels of the
    local mixture-fraction density Beta(f;t) at each time (normalised per
    time row, since the density's absolute scale grows enormously as mixing
    narrows it) -- showing where the Beta-PDF mass actually sits, and how its
    core and wider spread both narrow in toward mean_f, without replacing the
    plotted quantity itself.  The lowest two levels (0.001, 0.01) are needed
    to see any activity at early times: in the segregated limit the peak is
    many orders of magnitude above the rest of the domain, so the 0.1/0.5/0.9
    levels alone stay collapsed against f=0/1 until mixing is well underway.
    A red dotted line (same colour, so it reads as part of the same overlay
    family against every part of the viridis colormap) traces fs(t), the
    complete-reaction limit's kink location.

    The C(f) colour scale itself is log (floored at 1e-3 of each panel's
    vmax, since C(f) can be exactly 0 for an unformed product), rather than
    linear -- this reveals structure in the near-zero region (e.g. a product
    species' early, still-tiny values) that a linear scale flattens out."""
    import pathlib as _pathlib
    import matplotlib.colors as mcolors

    grids = _raylimit_field_grids(ode_result)
    if grids is None:
        return
    t, f_grid, _B_grids, C_grids = grids
    active = list(C_grids.keys())
    if not active:
        return
    rl = ode_result['raylimit']
    species_list = ode_result['species']
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    W = _raylimit_beta_weight_grid(ode_result, t, f_grid)
    W_norm = W / np.where(W.max(axis=1, keepdims=True) > 0, W.max(axis=1, keepdims=True), 1.0)
    fsb = np.asarray(rl['fsb'], dtype=float)
    fs_mask = np.isfinite(fsb) & (fsb > 0.0) & (fsb < 1.0)

    vmax_of, div_of = _heatmap_vmax_divisors(species_list, active, Y1, Y2)

    ncols = min(3, max(1, int(np.ceil(np.sqrt(len(active))))))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sp in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        div = div_of.get(sp, 1.0)
        vmax = vmax_of[sp]
        floor = vmax * 1e-3
        im = ax.pcolormesh(f_grid, t, np.clip(C_grids[sp] / div, floor, None), shading='auto', cmap='viridis',
                           norm=mcolors.LogNorm(vmin=floor, vmax=vmax))
        ax.contour(f_grid, t, W_norm, levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
        ax.plot(fsb[fs_mask], t[fs_mask], color=CB_PALETTE[7], linestyle=':', linewidth=1.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'C(f)/{div:g}' if div != 1.0 else 'C(f)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        ax.set_title(f'{sp}/{div:g}' if div != 1.0 else sp, fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        ax.set_ylabel('t  (s)', fontsize=8)
        ax.set_xlim(0.0, 1.0)

    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('ray_limit interpolated C(f) vs f and time', fontsize=12, y=0.99)
    fig.text(0.5, 0.945, f'mean_f={mean_f:.4f},  ε={m_epsilon:.4g};  '
                        f'red solid: 0.001/0.01/0.1/0.5/0.9-max of β(f,t);  red dotted: fs(t)',
             ha='center', va='top', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_raylimit_Cf_heatmap.png')
    else:
        plt.show()


def plot_raylimit_Bf_heatmap(ode_result, save_stem=None):
    """Identical to :func:`plot_raylimit_Cf_heatmap`, but plots the staged
    complete-reaction limit B(f) instead of the interpolated C(f) -- an
    experimental side-by-side alternative; see that function for the full
    description of the colour scale and overlays."""
    import pathlib as _pathlib
    import matplotlib.colors as mcolors

    grids = _raylimit_field_grids(ode_result)
    if grids is None:
        return
    t, f_grid, B_grids, _C_grids = grids
    active = list(B_grids.keys())
    if not active:
        return
    rl = ode_result['raylimit']
    species_list = ode_result['species']
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    W = _raylimit_beta_weight_grid(ode_result, t, f_grid)
    W_norm = W / np.where(W.max(axis=1, keepdims=True) > 0, W.max(axis=1, keepdims=True), 1.0)
    fsb = np.asarray(rl['fsb'], dtype=float)
    fs_mask = np.isfinite(fsb) & (fsb > 0.0) & (fsb < 1.0)

    vmax_of, div_of = _heatmap_vmax_divisors(species_list, active, Y1, Y2)

    ncols = min(3, max(1, int(np.ceil(np.sqrt(len(active))))))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sp in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        div = div_of.get(sp, 1.0)
        vmax = vmax_of[sp]
        floor = vmax * 1e-3
        im = ax.pcolormesh(f_grid, t, np.clip(B_grids[sp] / div, floor, None), shading='auto', cmap='viridis',
                           norm=mcolors.LogNorm(vmin=floor, vmax=vmax))
        ax.contour(f_grid, t, W_norm, levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
        ax.plot(fsb[fs_mask], t[fs_mask], color=CB_PALETTE[7], linestyle=':', linewidth=1.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'B(f)/{div:g}' if div != 1.0 else 'B(f)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        ax.set_title(f'{sp}/{div:g}' if div != 1.0 else sp, fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        ax.set_ylabel('t  (s)', fontsize=8)
        ax.set_xlim(0.0, 1.0)

    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('ray_limit staged limit B(f) vs f and time', fontsize=12, y=0.99)
    fig.text(0.5, 0.945, f'mean_f={mean_f:.4f},  ε={m_epsilon:.4g};  '
                        f'red solid: 0.001/0.01/0.1/0.5/0.9-max of β(f,t);  red dotted: fs(t)',
             ha='center', va='top', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_raylimit_Bf_heatmap.png')
    else:
        plt.show()


def plot_raylimit_Cf_heatmap_epsilon_sweep_movie(ode_results, save_stem=None, fps=2):
    """Animate the :func:`plot_raylimit_Cf_heatmap` grid across an ε sweep --
    one frame per epsilon (sorted increasing), so the qualitative change in
    how C(f) relaxes over time can be compared across mixing rates.  Each
    panel's colour scale is fixed exactly as in :func:`plot_raylimit_Cf_heatmap`
    (stream-1 feed max for stream-1 species, stream-2 feed max otherwise), so
    colours stay comparable frame to frame; each run's own time axis is used,
    since the conversion-event termination gives each ε a different
    integration time.  As in :func:`plot_raylimit_Cf_heatmap`, red solid
    contours trace the (per-time-row-normalised) 0.001/0.01/0.1/0.5/0.9-of-max
    levels of the local mixture-fraction density Beta(f;t), and a red dotted
    line traces fs(t).  `ode_results` is the full per-ε ray_limit results
    list, as in :func:`plot_raylimit_epsilon_sweep_movie`."""
    import pathlib as _pathlib
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    runs = [r for r in ode_results if r.get('weight_method') == 'ray_limit'
            and r.get('raylimit', {}).get('limit_bps') is not None]
    if len(runs) < 2:
        return
    runs = sorted(runs, key=lambda r: r['m_epsilon'])

    species_list = runs[0]['species']
    mean_f = runs[0]['mean_f']
    rl0 = runs[0]['raylimit']
    Y1, Y2 = np.asarray(rl0['Y1']), np.asarray(rl0['Y2'])
    active = [sp for sp in species_list if sp in rl0.get('prof', {})]
    if not active:
        return
    vmax_of, div_of = _heatmap_vmax_divisors(species_list, active, Y1, Y2)
    floor_of = {sp: vmax_of[sp] * 1e-3 for sp in active}
    _lbl = lambda sp: f'{sp}/{div_of[sp]:g}' if sp in div_of else sp

    grids_by_run = [_raylimit_field_grids(res) for res in runs]
    Wnorm_by_run = []
    for res, (t_i, f_grid_i, _, _) in zip(runs, grids_by_run):
        W = _raylimit_beta_weight_grid(res, t_i, f_grid_i)
        Wmax = W.max(axis=1, keepdims=True)
        Wnorm_by_run.append(W / np.where(Wmax > 0, Wmax, 1.0))
    fsb_by_run = [np.asarray(res['raylimit']['fsb'], dtype=float) for res in runs]
    fs_mask_by_run = [np.isfinite(fsb) & (fsb > 0.0) & (fsb < 1.0) for fsb in fsb_by_run]

    def _style_time_axis(ax):
        # Fixed-width y-tick labels (a handful of '%.2e'-formatted ticks) so
        # the space matplotlib reserves for them never changes frame to
        # frame -- t_end spans orders of magnitude across the sweep, and
        # without this the axes visibly shift/judder between frames.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))

    ncols = min(3, max(1, int(np.ceil(np.sqrt(len(active))))))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    ax_of = {}
    t0, f_grid, _, C0 = grids_by_run[0]
    for idx, sp in enumerate(active):
        ax_of[sp] = axes[idx // ncols][idx % ncols]
        ax = ax_of[sp]
        im = ax.pcolormesh(f_grid, t0, np.clip(C0.get(sp, np.zeros((len(t0), len(f_grid)))) / div_of.get(sp, 1.0), floor_of[sp], None),
                           shading='auto', cmap='viridis',
                           norm=mcolors.LogNorm(vmin=floor_of[sp], vmax=vmax_of[sp]))
        ax.contour(f_grid, t0, Wnorm_by_run[0], levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
        ax.plot(fsb_by_run[0][fs_mask_by_run[0]], t0[fs_mask_by_run[0]], color=CB_PALETTE[7],
               linestyle=':', linewidth=1.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'C(f)/{div_of[sp]:g}' if sp in div_of else 'C(f)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        # Set titles/labels here too (not just in _draw) so the one-time
        # tight_layout() call below reserves room for them from the start.
        ax.set_title(_lbl(sp), fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        ax.set_ylabel('t  (s)', fontsize=8)
        ax.set_xlim(0.0, 1.0)
        _style_time_axis(ax)
    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _caption = fig.text(0.5, 0.945, '', ha='center', va='top', fontsize=9)

    def _draw(i):
        res = runs[i]
        t_i, f_grid_i, _, C_grids = grids_by_run[i]
        for sp in active:
            ax = ax_of[sp]
            ax.clear()
            ax.pcolormesh(f_grid_i, t_i, np.clip(C_grids.get(sp, np.zeros((len(t_i), len(f_grid_i)))) / div_of.get(sp, 1.0), floor_of[sp], None),
                         shading='auto', cmap='viridis',
                         norm=mcolors.LogNorm(vmin=floor_of[sp], vmax=vmax_of[sp]))
            ax.contour(f_grid_i, t_i, Wnorm_by_run[i], levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
            ax.plot(fsb_by_run[i][fs_mask_by_run[i]], t_i[fs_mask_by_run[i]], color=CB_PALETTE[7],
                   linestyle=':', linewidth=1.5)
            ax.set_title(_lbl(sp), fontsize=10)
            ax.set_xlabel('f', fontsize=8)
            ax.set_ylabel('t  (s)', fontsize=8)
            ax.set_xlim(0.0, 1.0)
            _style_time_axis(ax)
        fig.suptitle('ray_limit C(f) vs f and time across the ε sweep', fontsize=12, y=0.99)
        _caption.set_text(f"mean_f={mean_f:.4f},  ε = {res['m_epsilon']:.4g},  "
                          f"t_end = {res['t'][-1]:.4g} s;  "
                          f"red solid: 0.001/0.01/0.1/0.5/0.9-max of β(f,t);  red dotted: fs(t)")
        # NOTE: layout is fixed once, below, right after setup -- not
        # recomputed here. _style_time_axis gives every frame the same
        # fixed-width y-tick label format, so the space they need never
        # changes; a per-frame tight_layout() call is exactly what caused
        # the axes to visibly shift/judder between frames.
        return list(ax_of.values())

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    anim = animation.FuncAnimation(fig, _draw, frames=len(runs), interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_Cf_heatmap_epsilon_sweep_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def plot_raylimit_Bf_heatmap_epsilon_sweep_movie(ode_results, save_stem=None, fps=2):
    """Identical to :func:`plot_raylimit_Cf_heatmap_epsilon_sweep_movie`, but
    animates the staged complete-reaction limit B(f) instead of the
    interpolated C(f) -- an experimental side-by-side alternative; see that
    function for the full description of the colour scale and overlays."""
    import pathlib as _pathlib
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker

    runs = [r for r in ode_results if r.get('weight_method') == 'ray_limit'
            and r.get('raylimit', {}).get('limit_bps') is not None]
    if len(runs) < 2:
        return
    runs = sorted(runs, key=lambda r: r['m_epsilon'])

    species_list = runs[0]['species']
    mean_f = runs[0]['mean_f']
    rl0 = runs[0]['raylimit']
    Y1, Y2 = np.asarray(rl0['Y1']), np.asarray(rl0['Y2'])
    active = [sp for sp in species_list if sp in rl0.get('prof', {})]
    if not active:
        return
    vmax_of, div_of = _heatmap_vmax_divisors(species_list, active, Y1, Y2)
    floor_of = {sp: vmax_of[sp] * 1e-3 for sp in active}
    _lbl = lambda sp: f'{sp}/{div_of[sp]:g}' if sp in div_of else sp

    grids_by_run = [_raylimit_field_grids(res) for res in runs]
    Wnorm_by_run = []
    for res, (t_i, f_grid_i, _, _) in zip(runs, grids_by_run):
        W = _raylimit_beta_weight_grid(res, t_i, f_grid_i)
        Wmax = W.max(axis=1, keepdims=True)
        Wnorm_by_run.append(W / np.where(Wmax > 0, Wmax, 1.0))
    fsb_by_run = [np.asarray(res['raylimit']['fsb'], dtype=float) for res in runs]
    fs_mask_by_run = [np.isfinite(fsb) & (fsb > 0.0) & (fsb < 1.0) for fsb in fsb_by_run]

    def _style_time_axis(ax):
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2e'))

    ncols = min(3, max(1, int(np.ceil(np.sqrt(len(active))))))
    nrows = (len(active) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    ax_of = {}
    t0, f_grid, B0, _ = grids_by_run[0]
    for idx, sp in enumerate(active):
        ax_of[sp] = axes[idx // ncols][idx % ncols]
        ax = ax_of[sp]
        im = ax.pcolormesh(f_grid, t0, np.clip(B0.get(sp, np.zeros((len(t0), len(f_grid)))) / div_of.get(sp, 1.0), floor_of[sp], None),
                           shading='auto', cmap='viridis',
                           norm=mcolors.LogNorm(vmin=floor_of[sp], vmax=vmax_of[sp]))
        ax.contour(f_grid, t0, Wnorm_by_run[0], levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
        ax.plot(fsb_by_run[0][fs_mask_by_run[0]], t0[fs_mask_by_run[0]], color=CB_PALETTE[7],
               linestyle=':', linewidth=1.5)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'B(f)/{div_of[sp]:g}' if sp in div_of else 'B(f)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        # Set titles/labels here too (not just in _draw) so the one-time
        # tight_layout() call below reserves room for them from the start.
        ax.set_title(_lbl(sp), fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        ax.set_ylabel('t  (s)', fontsize=8)
        ax.set_xlim(0.0, 1.0)
        _style_time_axis(ax)
    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    _caption = fig.text(0.5, 0.945, '', ha='center', va='top', fontsize=9)

    def _draw(i):
        res = runs[i]
        t_i, f_grid_i, B_grids, _ = grids_by_run[i]
        for sp in active:
            ax = ax_of[sp]
            ax.clear()
            ax.pcolormesh(f_grid_i, t_i, np.clip(B_grids.get(sp, np.zeros((len(t_i), len(f_grid_i)))) / div_of.get(sp, 1.0), floor_of[sp], None),
                         shading='auto', cmap='viridis',
                         norm=mcolors.LogNorm(vmin=floor_of[sp], vmax=vmax_of[sp]))
            ax.contour(f_grid_i, t_i, Wnorm_by_run[i], levels=[0.001, 0.01, 0.1, 0.5, 0.9], colors=[CB_PALETTE[7]], linewidths=1.2)
            ax.plot(fsb_by_run[i][fs_mask_by_run[i]], t_i[fs_mask_by_run[i]], color=CB_PALETTE[7],
                   linestyle=':', linewidth=1.5)
            ax.set_title(_lbl(sp), fontsize=10)
            ax.set_xlabel('f', fontsize=8)
            ax.set_ylabel('t  (s)', fontsize=8)
            ax.set_xlim(0.0, 1.0)
            _style_time_axis(ax)
        fig.suptitle('ray_limit B(f) vs f and time across the ε sweep', fontsize=12, y=0.99)
        _caption.set_text(f"mean_f={mean_f:.4f},  ε = {res['m_epsilon']:.4g},  "
                          f"t_end = {res['t'][-1]:.4g} s;  "
                          f"red solid: 0.001/0.01/0.1/0.5/0.9-max of β(f,t);  red dotted: fs(t)")
        return list(ax_of.values())

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    anim = animation.FuncAnimation(fig, _draw, frames=len(runs), interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_Bf_heatmap_epsilon_sweep_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def plot_raylimit_limit_movie(ode_result, save_stem=None, fps=15):
    """Animate ray_limit's staged complete-reaction limit B(f) over every
    output time step of one run — the same content as one
    :func:`plot_raylimit_limit_grid` panel, stepped continuously instead of
    at 4 fixed snapshots, so the fs kink's rotation is visible directly."""
    import pathlib as _pathlib
    import matplotlib.animation as animation

    rl = ode_result.get('raylimit')
    if rl is None or rl.get('limit_bps') is None:
        return
    t = np.asarray(ode_result['t'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    if not active_indices:
        return
    y_full = np.asarray(ode_result['y'])
    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    n_t = len(t)

    fg = np.linspace(0.0, 1.0, 400)
    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}
    stream_labels = identify_stream_feeds(Y1, Y2)
    s1 = [s for s in active_indices if stream_labels[s] in (1, 12)]
    s2 = [s for s in active_indices if stream_labels[s] not in (1, 12)]

    def _Bcurve(k, s):
        bps = rl['limit_bps'][k]
        if bps is None:
            return np.zeros_like(fg)
        return np.interp(fg, bps, rl['limit_B'][k][:, active_indices.index(s)])

    gmax = {s: max(float(np.max(np.abs(_Bcurve(k, s)))) for k in range(n_t)) for s in active_indices}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: s in s1 for s in active_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr
    _sgmax = lambda s: gmax[s] / div_of[s] if s in div_of else gmax[s]
    # Fixed axis limits (global max across all frames) so the movie doesn't
    # jitter from per-frame autoscaling.
    y2max = max((_sgmax(s) for s in s2), default=1.0) * 1.1
    y1max = max((_sgmax(s) for s in s1), default=1.0) * 1.1

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax_r = ax.twinx() if s1 else None
    _ms, _lw, _leg_fs = _scaled_marker_lw(8, 5.5)

    def _draw(k):
        ax.clear()
        if ax_r is not None:
            ax_r.clear()
        for s in s2:
            ax.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                    marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                    ms=_ms, lw=_lw, label=_lbl(s))
        if ax_r is not None:
            for s in s1:
                ax_r.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                          marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                          ms=_ms, lw=_lw, label=_lbl(s))
            ax_r.tick_params(axis='y', labelsize=6)
            ax_r.set_ylim(-0.02 * y1max, y1max)
        _h, _l = ax.get_legend_handles_labels()
        if ax_r is not None:
            _h_r, _l_r = ax_r.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs)
        fsk = float(rl['fsb'][k]) if np.isfinite(rl['fsb'][k]) else None
        ttl = f't = {t[k]:.4g} s'
        if fsk is not None and 0.0 < fsk < 1.0:
            ttl += f'   (fs ≈ {fsk:.4f})'
            ax.axvline(fsk, color='k', linestyle=':', alpha=0.5)
        ray_eq = _ray_equation_str(y_full[k, active_indices] - y0_active,
                                   species_list, active_indices, rel_tol=0.02)
        if ray_eq is not None:
            ttl += f'\nd: {ray_eq}'
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('f', fontsize=9)
        ax.set_ylabel('B(f)  (mol/m³)', fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02 * y2max, y2max)

    fig.suptitle(f'ray_limit complete-reaction limit B(f) vs time  '
                f'(mean_f={mean_f:.4f},  ε={m_epsilon:.4g})', fontsize=11)
    anim = animation.FuncAnimation(fig, _draw, frames=n_t, interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_limit_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def plot_raylimit_epsilon_sweep_movie(ode_results, save_stem=None, fps=2):
    """Animate ray_limit's final-state complete-reaction limit B(f) across an
    ε sweep — one frame per ε (increasing order), showing how the converged
    fs and B(f) shift with the mixing rate.  `ode_results` is the full list of
    per-ε ray_limit results from the 51-point sweep
    (`sweep_epsilon_product_fractions`'s `['ray_limit']['ode_results']`), not
    just the small base-case epsilon list."""
    import pathlib as _pathlib
    import matplotlib.animation as animation

    runs = [r for r in ode_results if r.get('weight_method') == 'ray_limit'
            and r.get('raylimit', {}).get('limit_bps') is not None]
    if len(runs) < 2:
        return
    runs = sorted(runs, key=lambda r: r['m_epsilon'])

    species_list = runs[0]['species']
    active_indices = list(runs[0]['active_indices'])
    mean_f = runs[0]['mean_f']
    if not active_indices:
        return

    fg = np.linspace(0.0, 1.0, 400)
    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}
    Y1, Y2 = np.asarray(runs[0]['raylimit']['Y1']), np.asarray(runs[0]['raylimit']['Y2'])
    stream_labels = identify_stream_feeds(Y1, Y2)
    s1 = [s for s in active_indices if stream_labels[s] in (1, 12)]
    s2 = [s for s in active_indices if stream_labels[s] not in (1, 12)]

    def _final_Bcurve(res, s):
        rl = res['raylimit']
        bps = rl['limit_bps'][-1]
        if bps is None:
            return np.zeros_like(fg)
        ai = list(res['active_indices']).index(s)
        return np.interp(fg, bps, rl['limit_B'][-1][:, ai])

    gmax = {s: max(float(np.max(np.abs(_final_Bcurve(r, s)))) for r in runs) for s in active_indices}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: s in s1 for s in active_indices})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr
    _sgmax = lambda s: gmax[s] / div_of[s] if s in div_of else gmax[s]
    y2max = max((_sgmax(s) for s in s2), default=1.0) * 1.1
    y1max = max((_sgmax(s) for s in s1), default=1.0) * 1.1

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax_r = ax.twinx() if s1 else None
    _ms, _lw, _leg_fs = _scaled_marker_lw(8, 5.5)

    def _draw(i):
        res = runs[i]
        ax.clear()
        if ax_r is not None:
            ax_r.clear()
        for s in s2:
            ax.plot(fg, _yv(s, _final_Bcurve(res, s)), color=color_map[s], linestyle=ls_map[s],
                    marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                    ms=_ms, lw=_lw, label=_lbl(s))
        if ax_r is not None:
            for s in s1:
                ax_r.plot(fg, _yv(s, _final_Bcurve(res, s)), color=color_map[s], linestyle=ls_map[s],
                          marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                          ms=_ms, lw=_lw, label=_lbl(s))
            ax_r.tick_params(axis='y', labelsize=6)
            ax_r.set_ylim(-0.02 * y1max, y1max)
        _h, _l = ax.get_legend_handles_labels()
        if ax_r is not None:
            _h_r, _l_r = ax_r.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs)
        fsb = res['raylimit']['fsb']
        fsk = float(fsb[-1]) if np.isfinite(fsb[-1]) else None
        ttl = f"ε = {res['m_epsilon']:.4g}  (t_end = {res['t'][-1]:.4g} s)"
        if fsk is not None and 0.0 < fsk < 1.0:
            ttl += f'   (fs ≈ {fsk:.4f})'
            ax.axvline(fsk, color='k', linestyle=':', alpha=0.5)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel('f', fontsize=9)
        ax.set_ylabel('B(f)  (mol/m³)', fontsize=9)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02 * y2max, y2max)

    fig.suptitle(f'ray_limit final complete-reaction limit B(f) across the ε sweep  '
                f'(mean_f={mean_f:.4f})', fontsize=11)
    anim = animation.FuncAnimation(fig, _draw, frames=len(runs), interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_epsilon_sweep_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def _species_limits_grid_axes(unique_sp_data, include_cf=True):
    """Shared setup for the two ray_limit "species_limits" movies: the same
    per-species subplot grid and static per-subset profile lines as
    :func:`plot_unique_species_profiles`, plus one extra (initially empty)
    bold line and one dashed vertical line per subplot for the animated
    ray_limit B(f) and its fs kink.  When ``include_cf`` is True, also adds a
    colour-blind-safe red line for the interpolated C(f) (only meaningful
    when C(f) still visibly differs from B(f) frame to frame, e.g. the
    single-run time-evolution movie -- not the epsilon-sweep movie, whose
    frames are each run's converged final state, where C(f) and B(f) are
    indistinguishable by eye).  Returns
    (fig, axes, active, b_lines, c_lines, fs_lines, static_yrange) where
    static_yrange[sp] = (ymin, ymax) of that species' static curves alone,
    and c_lines is {} when include_cf is False."""
    meta = unique_sp_data['meta']
    species_data = unique_sp_data['species']
    active = [sp for sp in meta['species'] if sp in species_data]
    if not active:
        return None

    n_active = len(active)
    ncols = min(3, max(1, int(np.ceil(np.sqrt(n_active)))))
    nrows = (n_active + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.5, 3.5)
    _tab_palette = _build_tab_palette()

    # Same global label->style convention as plot_unique_species_profiles, so
    # a given subset/group label gets the same colour and legend position in
    # every subplot here too.
    _all_labels = []
    _seen_labels = set()
    for sp in active:
        for label in species_data[sp]:
            if label not in _seen_labels:
                _seen_labels.add(label)
                _all_labels.append(label)
    _label_style = {label: _tab_palette[i % len(_tab_palette)] for i, label in enumerate(_all_labels)}

    b_lines, c_lines, fs_lines, static_yrange = {}, {}, {}, {}
    _CB_RED = CB_PALETTE[7]   # colour-blind-safe red, for the interpolated C(f)
    _MAX_LABEL = 20
    for idx, sp in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        profiles = species_data[sp]

        y_all = []
        for label in _all_labels:
            if label not in profiles:
                continue
            profile = profiles[label]
            bps = profile['breakpoints']
            segs = profile['segments']
            if not bps or not segs:
                continue
            f_grid = np.array(bps)
            y_grid = np.array(
                [segs[0][0] * bps[0] + segs[0][1]] +
                [segs[j][0] * bps[j + 1] + segs[j][1] for j in range(len(segs))]
            )
            color, ls, mk = _label_style[label]
            display = label if len(label) <= _MAX_LABEL else label[:_MAX_LABEL - 3] + '...'
            ax.plot(f_grid, y_grid, color=color, linestyle=ls, marker=mk,
                    markevery=max(1, len(f_grid) // 15), mfc=_face(color), ms=_ms, lw=_lw, label=display)
            y_all.append(y_grid)
        static_yrange[sp] = ((min(float(np.min(y)) for y in y_all), max(float(np.max(y)) for y in y_all))
                             if y_all else (0.0, 1.0))

        (b_lines[sp],) = ax.plot([], [], color='black', lw=_lw * 1.6, ls='-', marker='o',
                                 markevery=20, ms=_ms, mfc='none', zorder=6, label='ray_limit B(f)')
        if include_cf:
            (c_lines[sp],) = ax.plot([], [], color=_CB_RED, lw=_lw * 1.6, ls='-', marker='s',
                                     markevery=20, ms=_ms, mfc='none', zorder=6.5, label='C(f)')
        fs_lines[sp] = ax.axvline(np.nan, color='k', linestyle=':', alpha=0.5, zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 1:
            ax.legend(handles, labels, loc='best', frameon=False, fontsize=_leg_fs)
        ax.set_title(sp, fontsize=10)
        ax.set_xlabel('f', fontsize=8)
        ax.set_xlim(0.0, 1.0)

    for idx in range(n_active, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    return fig, axes, active, b_lines, c_lines, fs_lines, static_yrange


def _set_species_limits_ylim(ax, static_range, b_vals_all):
    """y-limits covering both a species' static subset profiles and every
    frame of its animated ray_limit B(f) and C(f), so the movie never
    autoscales or clips."""
    ymin, ymax = static_range
    for bv in b_vals_all:
        ymin = min(ymin, float(np.min(bv)))
        ymax = max(ymax, float(np.max(bv)))
    pad = 0.05 * ((ymax - ymin) or 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)


def plot_raylimit_species_limits_movie(unique_sp_data, ode_result, save_stem=None, fps=15):
    """Animate ray_limit's B(f) for each species over time, overlaid on that
    species' static per-subset profile grid -- the same layout as
    :func:`plot_unique_species_profiles` (the "species_limits" plot), so the
    live ray_limit limit can be compared directly against the LP-derived
    subset references it interpolates between."""
    import pathlib as _pathlib
    import matplotlib.animation as animation

    rl = ode_result.get('raylimit')
    if rl is None or rl.get('limit_bps') is None:
        return
    t = np.asarray(ode_result['t'])
    n_t = len(t)
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    prof = rl.get('prof', {})

    _setup = _species_limits_grid_axes(unique_sp_data)
    if _setup is None:
        return
    fig, axes, active, b_lines, c_lines, fs_lines, static_yrange = _setup

    fg = np.linspace(0.0, 1.0, 400)

    def _Bcurve(sp_global, k):
        if sp_global not in active_indices:
            return None
        bps = rl['limit_bps'][k]
        if bps is None:
            return None
        ai = active_indices.index(sp_global)
        return np.interp(fg, bps, rl['limit_B'][k][:, ai])

    def _Ccurve(sp, sp_global, k, bv):
        if bv is None or sp not in prof:
            return None
        lam = prof[sp][k, 3]
        if not np.isfinite(lam):
            return None
        Mi = Y1[sp_global] * fg + Y2[sp_global] * (1.0 - fg)
        return lam * Mi + (1.0 - lam) * bv

    sp_global_of = {sp: species_list.index(sp) for sp in active}
    for idx, sp in enumerate(active):
        ax = axes[idx // len(axes[0])][idx % len(axes[0])]
        sg = sp_global_of[sp]
        b_vals_all = [bv for k in range(n_t) if (bv := _Bcurve(sg, k)) is not None]
        c_vals_all = [cv for k in range(n_t)
                      if (cv := _Ccurve(sp, sg, k, _Bcurve(sg, k))) is not None]
        _set_species_limits_ylim(ax, static_yrange[sp], b_vals_all + c_vals_all)

    def _draw(k):
        fsk = float(rl['fsb'][k]) if np.isfinite(rl['fsb'][k]) else None
        for sp in active:
            sg = sp_global_of[sp]
            bv = _Bcurve(sg, k)
            b_lines[sp].set_data(fg, bv if bv is not None else [])
            cv = _Ccurve(sp, sg, k, bv)
            c_lines[sp].set_data(fg, cv if cv is not None else [])
            fs_lines[sp].set_xdata([fsk, fsk] if fsk is not None and 0.0 < fsk < 1.0 else [np.nan, np.nan])
        _fs_str = f',  fs ≈ {fsk:.4f}' if fsk is not None and 0.0 < fsk < 1.0 else ''
        fig.suptitle(f'ray_limit B(f) vs time  (mean_f={mean_f:.4f},  ε={m_epsilon:.4g},  '
                    f't = {t[k]:.4g} s{_fs_str})', fontsize=12)
        return list(b_lines.values()) + list(c_lines.values()) + list(fs_lines.values())

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    anim = animation.FuncAnimation(fig, _draw, frames=n_t, interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_species_limits_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def plot_raylimit_species_limits_static(unique_sp_data, ode_result, save_stem=None,
                                        conversion_target=0.5):
    """Static snapshot of :func:`plot_raylimit_species_limits_movie`, frozen
    at the output step whose conversion of the stream-1 limiting reactant is
    closest to `conversion_target`, with C(f) omitted -- leaving just the
    LP's static per-subset profiles and ray_limit's B(f) for each species."""
    import pathlib as _pathlib

    rl = ode_result.get('raylimit')
    if rl is None or rl.get('limit_bps') is None:
        return
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    t = np.asarray(ode_result['t'])

    L_name = ode_result.get('stream1_closure', {}).get('limiting_reactant')
    if L_name is None:
        return
    L_idx = species_list.index(L_name)
    y0_L = float(ode_result['y'][0, L_idx])
    if not y0_L:
        return
    conv = 1.0 - ode_result['y'][:, L_idx] / y0_L
    k = int(np.argmin(np.abs(conv - conversion_target)))

    _setup = _species_limits_grid_axes(unique_sp_data, include_cf=False)
    if _setup is None:
        return
    fig, axes, active, b_lines, _c_lines, fs_lines, static_yrange = _setup
    ncols = len(axes[0])

    fg = np.linspace(0.0, 1.0, 400)

    def _Bcurve(sp_global):
        if sp_global not in active_indices:
            return None
        bps = rl['limit_bps'][k]
        if bps is None:
            return None
        ai = active_indices.index(sp_global)
        return np.interp(fg, bps, rl['limit_B'][k][:, ai])

    sp_global_of = {sp: species_list.index(sp) for sp in active}
    fsk = float(rl['fsb'][k]) if np.isfinite(rl['fsb'][k]) else None
    for idx, sp in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        sg = sp_global_of[sp]
        bv = _Bcurve(sg)
        b_lines[sp].set_data(fg, bv if bv is not None else [])
        fs_lines[sp].set_xdata([fsk, fsk] if fsk is not None and 0.0 < fsk < 1.0 else [np.nan, np.nan])
        _set_species_limits_ylim(ax, static_yrange[sp], [bv] if bv is not None else [])

    _fs_str = f',  fs ≈ {fsk:.4f}' if fsk is not None and 0.0 < fsk < 1.0 else ''
    fig.suptitle(f'ray_limit B(f)  (mean_f={mean_f:.4f},  ε={m_epsilon:.4g},  '
                f't = {t[k]:.4g} s,  conversion of {L_name} ≈ {100.0 * conv[k]:.1f}%{_fs_str})',
                fontsize=12, y=0.99)
    Y1, Y2 = np.asarray(rl['Y1']), np.asarray(rl['Y2'])
    y0_active = mean_f * Y1[active_indices] + (1.0 - mean_f) * Y2[active_indices]
    ray_eq = _ray_equation_str(ode_result['y'][k, active_indices] - y0_active,
                               species_list, active_indices)
    if ray_eq is not None:
        fig.text(0.5, 0.945, f'ray: {ray_eq}', ha='center', va='top', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    if save_stem is not None:
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_raylimit_species_limits_static.png')
    else:
        plt.show()


def plot_raylimit_species_limits_epsilon_sweep_movie(unique_sp_data, ode_results, save_stem=None, fps=2):
    """Animate ray_limit's final-state B(f) for each species across an ε
    sweep, overlaid on that species' static per-subset profile grid -- the
    ε-sweep counterpart of :func:`plot_raylimit_species_limits_movie`.
    `ode_results` is the full per-ε ray_limit results list, as in
    :func:`plot_raylimit_epsilon_sweep_movie`."""
    import pathlib as _pathlib
    import matplotlib.animation as animation

    runs = [r for r in ode_results if r.get('weight_method') == 'ray_limit'
            and r.get('raylimit', {}).get('limit_bps') is not None]
    if len(runs) < 2:
        return
    runs = sorted(runs, key=lambda r: r['m_epsilon'])
    species_list = runs[0]['species']
    mean_f = runs[0]['mean_f']

    _setup = _species_limits_grid_axes(unique_sp_data, include_cf=False)
    if _setup is None:
        return
    fig, axes, active, b_lines, c_lines, fs_lines, static_yrange = _setup

    fg = np.linspace(0.0, 1.0, 400)

    def _final_Bcurve(res, sp_global):
        ai_list = list(res['active_indices'])
        if sp_global not in ai_list:
            return None
        rl = res['raylimit']
        bps = rl['limit_bps'][-1]
        if bps is None:
            return None
        ai = ai_list.index(sp_global)
        return np.interp(fg, bps, rl['limit_B'][-1][:, ai])

    sp_global_of = {sp: species_list.index(sp) for sp in active}
    for idx, sp in enumerate(active):
        ax = axes[idx // len(axes[0])][idx % len(axes[0])]
        sg = sp_global_of[sp]
        b_vals_all = [bv for r in runs if (bv := _final_Bcurve(r, sg)) is not None]
        _set_species_limits_ylim(ax, static_yrange[sp], b_vals_all)

    def _draw(i):
        res = runs[i]
        fsb = res['raylimit']['fsb']
        fsk = float(fsb[-1]) if np.isfinite(fsb[-1]) else None
        for sp in active:
            sg = sp_global_of[sp]
            bv = _final_Bcurve(res, sg)
            b_lines[sp].set_data(fg, bv if bv is not None else [])
            fs_lines[sp].set_xdata([fsk, fsk] if fsk is not None and 0.0 < fsk < 1.0 else [np.nan, np.nan])
        _fs_str = f',  fs ≈ {fsk:.4f}' if fsk is not None and 0.0 < fsk < 1.0 else ''
        fig.suptitle(f'ray_limit final B(f) across the ε sweep  (mean_f={mean_f:.4f},  '
                    f"ε = {res['m_epsilon']:.4g}{_fs_str})", fontsize=12)
        return list(b_lines.values()) + list(fs_lines.values())

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    anim = animation.FuncAnimation(fig, _draw, frames=len(runs), interval=1000.0 / fps)

    if save_stem is not None:
        plots_dir = _pathlib.Path(save_stem).parent / 'plots'
        plots_dir.mkdir(exist_ok=True)
        out_path = plots_dir / f'{_pathlib.Path(save_stem).name}_raylimit_species_limits_epsilon_sweep_movie.mp4'
        anim.save(out_path, writer='ffmpeg', fps=fps)
        plt.close(fig)
        print(f"Saved to {out_path}")
    else:
        plt.show()


def plot_blendfs_limit_grid(ode_result, save_stem=None, blend_subsets=None):
    """blend_fs only: the blended complete-reaction limit B(f), laid out exactly
    like :func:`plot_raylimit_limit_grid` — all species' coloured profiles overlaid
    on a single axes, one subplot per snapshot time.

    blend_fs's per-species limit is piecewise-linear over the union of every
    blended subset's own breakpoints (``ode_result['blendfs']['bp']``), with
    the blended value at each recorded each step in
    ``ode_result['blendfs']['prof_full']`` -- not just a single-kink
    (v0, fs_blend, v1) approximation, so a catalytic reaction's interior kink
    (e.g. the near-f=0 step where a catalyst confined to one stream first
    becomes present) is preserved rather than smeared into a long diagonal.
    Uses the SAME species→colour mapping, stream-split (stream-1 species on a twin
    axis) and `/20` scaling as the ray_limit grid, so the two read alike.  The four
    times match the beta snapshots (first near 1/96 of t_end)."""
    import pathlib as _pathlib

    bf = ode_result.get('blendfs')
    if bf is None or not bf.get('prof'):
        return
    t = np.asarray(ode_result['t'])
    species_list = ode_result['species']
    active_indices = list(ode_result['active_indices'])
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    mean_f = ode_result['mean_f']
    Y1, Y2 = np.asarray(bf['Y1']), np.asarray(bf['Y2'])
    fsb = np.asarray(bf['fsb'], dtype=float)
    prof = bf['prof']
    bf_bp = np.asarray(bf['bp'], dtype=float)
    prof_full = bf['prof_full']
    # Only species that carry a blend profile have a B(f); plot those.
    plotted = [s for s in active_indices if species_list[s] in prof]
    if not plotted:
        return

    # Snapshot times (match plot_ode_beta_snapshots): first near 1/96 of t_end.
    T = float(t[-1] - t[0])
    targets = np.linspace(t[0], t[-1], 4)
    targets[0] = t[0] + T / 96.0
    idxs = sorted(int(np.argmin(np.abs(t - tt))) for tt in targets)
    fg = np.linspace(0.0, 1.0, 400)

    # Species→colour: based on global species index so colour is consistent across plots.
    color_map = {s: _sp_color(s, species_list) for s in active_indices}
    ls_map = {s: _sp_linestyle(s, species_list) for s in active_indices}
    mk_map = {s: _sp_marker(s, species_list) for s in active_indices}
    stream_labels = identify_stream_feeds(Y1, Y2)
    s1 = [s for s in plotted if stream_labels[s] in (1, 12)]   # twin (right) axis
    s2 = [s for s in plotted if stream_labels[s] not in (1, 12)]

    def _Bcurve(k, s):
        v_arr = prof_full[species_list[s]][k]
        if not np.all(np.isfinite(v_arr)):
            return np.zeros_like(fg)
        return np.interp(fg, bf_bp, v_arr)

    _fs_subs = bf['fs']

    # Scale down any species whose max dwarfs every other species FED BY THE
    # SAME STREAM (e.g. solvent) to ~1000 -- stream-1 (twin axis) species are
    # never compared against stream-2 (main axis) species.
    gmax = {s: max(float(np.max(np.abs(_Bcurve(k, s)))) for k in idxs) for s in plotted}
    div_of = _outlier_scale_divisors_by_stream(gmax, {s: s in s1 for s in plotted})
    _lbl = lambda s: f'{species_list[s]}/{div_of[s]:g}' if s in div_of else species_list[s]
    _yv = lambda s, arr: arr / div_of[s] if s in div_of else arr

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.5 * nrows),
                             squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(7.0, 4.5)
    for i, k in enumerate(idxs):
        ax = axes[i // ncols][i % ncols]
        for s in s2:
            ax.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                    marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                    ms=_ms, lw=_lw, label=_lbl(s))
        ax_r = None
        if s1:
            ax_r = ax.twinx()
            for s in s1:
                ax_r.plot(fg, _yv(s, _Bcurve(k, s)), color=color_map[s], linestyle=ls_map[s],
                          marker=mk_map[s], markevery=max(1, len(fg) // 15), mfc=_face(color_map[s]),
                          ms=_ms, lw=_lw, label=_lbl(s))
            ax_r.tick_params(axis='y', labelsize=6)
        _h, _l = ax.get_legend_handles_labels()
        if ax_r is not None:
            _h_r, _l_r = ax_r.get_legend_handles_labels()
            _h, _l = _h + _h_r, _l + _l_r
        if len(_l) > 1:
            ax.legend(_h, _l, loc='center right', frameon=False, fontsize=_leg_fs)
        # Individual subset kinks (thin grey); blended kink fsb (black).
        for fsj in _fs_subs:
            if np.isfinite(fsj) and 0.0 < fsj < 1.0:
                ax.axvline(fsj, color='0.72', linestyle=':', lw=0.8)
        fsk = float(fsb[k]) if np.isfinite(fsb[k]) else None
        ttl = f't = {t[k]:.4g} s'
        if fsk is not None and 0.0 < fsk < 1.0:
            ttl += f'   (fs_b ≈ {fsk:.4f})'
            ax.axvline(fsk, color='k', linestyle=':', alpha=0.5)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel('f', fontsize=8)
    for i in range(len(idxs), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    _bs_note = f'  blend_subsets={list(blend_subsets)}' if blend_subsets is not None else ''
    fig.suptitle('blend_fs blended complete-reaction limit B(f) by snapshot time  '
                 f'(mean_f={mean_f:.4f},  ε={m_epsilon:.4g}{_bs_note})\n'
                 'solid = control-point approx  ·  grey ·· = individual subset kinks  ·  black ·· = fsb',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_stem is not None:
        _wm = ode_result.get('weight_method', 'blend_fs')
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_{_wm}_limit_grid.png')
    else:
        plt.show()


def _eval_profile_array(bps, segs, f_arr):
    """Evaluate a piecewise-linear profile (breakpoints ``bps``, per-segment
    ``[slope, intercept]`` rows ``segs``) at an array of f values."""
    bps = np.asarray(bps, dtype=float)
    segs = np.asarray(segs, dtype=float)
    idx = np.clip(np.searchsorted(bps, f_arr, side='right') - 1, 0, len(segs) - 1)
    return segs[idx, 0] * f_arr + segs[idx, 1]


def plot_cprofile_per_reaction(unique_sp_data, ode_result, save_stem=None):
    """C(f) closure profiles for reactant species of each reaction.

    Layout: rows = reactions, columns = 4 snapshot times (same as
    plot_ode_beta_snapshots).  Each cell shows C_i(f) for every reactant of
    that reaction with the rate integrand overlaid.  Supported for blend_fs,
    ray_limit, and linear_interp; returns silently for other methods.
    """
    import pathlib as _pathlib

    weight_method = ode_result.get('weight_method', '')
    is_li = weight_method == 'linear_interp'
    if weight_method == 'blend_fs':
        diag = ode_result.get('blendfs')
    elif weight_method == 'ray_limit':
        diag = ode_result.get('raylimit')
    elif is_li:
        diag = ode_result.get('li_weights')
    else:
        return
    if diag is None:
        return
    sp_profiles = ode_result.get('sp_profiles', {}) if is_li else None

    meta = unique_sp_data['meta']
    species_list = meta['species']
    rxns = meta['reactions']
    rxn_labels = [r.split(':')[0].strip() for r in rxns]
    n_rxns = len(rxn_labels)

    active_indices = list(ode_result['active_indices'])
    active_species = [species_list[i] for i in active_indices]
    t_arr = ode_result['t']
    mean_f = ode_result['mean_f']
    m_epsilon = ode_result.get('m_epsilon', DEFAULT_M_EPSILON)
    m_lambda = ode_result.get('m_lambda', DEFAULT_M_LAMBDA)
    m_nu = ode_result.get('m_nu', DEFAULT_M_NU)
    m_Sc = ode_result.get('m_Sc', DEFAULT_M_SC)
    max_var = mean_f * (1.0 - mean_f)

    # Rate-law order per (species, reaction) — NOT the raw stoichiometric
    # coefficient; matches the order actually used by the ODE integration
    # (default 1, unless the reaction is `elementary` or the JSON supplies
    # explicit `orders`).
    rxn_reactants = ode_result['rxn_reactants']
    global_to_active = {int(i): pos for pos, i in enumerate(active_indices)}
    order_lookup = [
        {global_to_active[i]: order for i, order in rxn_reactants[j] if i in global_to_active}
        for j in range(n_rxns)
    ]

    # Per-species display scale for the C(f) curves (main axis only — the rate
    # integrand keeps the true, unscaled concentrations): a fed species is
    # scaled by its own (nonzero) feed concentration; an unfed product has no
    # feed of its own, so it's scaled by the smallest nonzero stream-2 feed.
    Y1_arr = np.asarray(ode_result['Y1'], dtype=float)
    Y2_arr = np.asarray(ode_result['Y2'], dtype=float)
    _y2_pos = Y2_arr[Y2_arr > 0]
    stream2_min = float(_y2_pos.min()) if _y2_pos.size else 1.0

    def _display_scale(sp_global):
        y1_i, y2_i = float(Y1_arr[sp_global]), float(Y2_arr[sp_global])
        if y1_i > 0:
            return y1_i
        if y2_i > 0:
            return y2_i
        return stream2_min

    # Snapshot times matching plot_ode_beta_snapshots
    _T = float(t_arr[-1] - t_arr[0])
    target_times = np.linspace(t_arr[0], t_arr[-1], 4)
    target_times[0] = t_arr[0] + _T / 96.0
    snap_idxs = sorted([int(np.argmin(np.abs(t_arr - tt))) for tt in target_times])
    snap_labels = [f't={t_arr[i]:.4g} s' for i in snap_idxs]
    n_cols = len(snap_idxs)

    f_grid = np.linspace(0.0, 1.0, 300)
    f_pdf  = np.linspace(1e-4, 1.0 - 1e-4, 300)

    fig, axes = plt.subplots(n_rxns, n_cols,
                             figsize=(4.0 * n_cols, 3.0 * n_rxns), squeeze=False)
    _ms, _lw, _leg_fs = _scaled_marker_lw(4.0, 3.0)

    for row, (j, lbl) in enumerate(zip(range(n_rxns), rxn_labels)):
        reactant_pos = np.array(sorted(order_lookup[j]), dtype=int)
        row_axpdf = []   # (ax_pdf, rig_max) pairs — shared secondary-axis scale across the row
        for col, (t_idx, t_label) in enumerate(zip(snap_idxs, snap_labels)):
            ax = axes[row][col]
            t_snap = float(t_arr[t_idx])

            # Beta PDF for this snapshot
            var_t   = mixing_variance(t_snap, mean_f, m_epsilon, m_lambda, m_nu, m_Sc)
            s_t     = max_var / var_t - 1.0
            alpha_t = mean_f * s_t
            beta_t  = (1.0 - mean_f) * s_t
            pdf_vals = _stats.beta.pdf(f_pdf, alpha_t, beta_t)

            if not is_li:
                fsb = float(diag['fsb'][t_idx])
                if not (np.isfinite(fsb) and 0.0 < fsb < 1.0):
                    fsb = 0.5

            reactant_Cf_pdf = {}  # sp_pos -> (Cf_on_f_pdf, rate_order)
            cf_max = None   # overall max C(f) in this subplot, for the log y-axis
            for sp_pos in reactant_pos:
                sp = active_species[sp_pos]
                sp_global = int(active_indices[sp_pos])
                color = _sp_color(sp_global, species_list)
                ls = _sp_linestyle(sp_global, species_list)
                mk = _sp_marker(sp_global, species_list)
                _mk_every_cp = max(1, len(f_grid) // 15)

                if is_li:
                    if sp not in sp_profiles or sp not in diag:
                        continue
                    w = diag[sp][t_idx]
                    profiles = sp_profiles[sp]
                    C_f = np.zeros_like(f_grid)
                    C_fp = np.zeros_like(f_pdf)
                    for n, (_lbl_n, prof_n) in enumerate(profiles):
                        wn = float(w[n]) if n < len(w) else 0.0
                        if not np.isfinite(wn) or wn == 0.0:
                            continue
                        bps_n, segs_n = prof_n['breakpoints'], prof_n['segments']
                        if not bps_n or not segs_n:
                            continue
                        C_f += wn * _eval_profile_array(bps_n, segs_n, f_grid)
                        C_fp += wn * _eval_profile_array(bps_n, segs_n, f_pdf)
                    scale = _display_scale(sp_global)
                    ax.plot(f_grid, C_f / scale, color=color, linestyle=ls, marker=mk,
                            markevery=_mk_every_cp, mfc=_face(color), ms=_ms, lw=_lw, label=sp)
                    _cf_max = float(np.nanmax(C_f)) / scale
                    cf_max = _cf_max if cf_max is None else max(cf_max, _cf_max)
                    reactant_Cf_pdf[sp_pos] = (C_fp, order_lookup[j][sp_pos])
                    continue

                if sp not in diag.get('prof', {}):
                    continue
                Y1_i = float(diag['Y1'][sp_global])
                Y2_i = float(diag['Y2'][sp_global])
                v0, vfs, v1, lam = diag['prof'][sp][t_idx]
                lam = float(lam)
                # blend_fs only: use the FULL blended B(f) over diag['bp'] (not
                # just the single-kink v0/vfs/v1) when available -- a
                # catalyzed reaction's near-boundary kink needs the fuller
                # resolution.  ray_limit (no 'bp' here) keeps the 2-segment form.
                _bp_full = diag.get('bp')
                _v_arr = diag.get('prof_full', {}).get(sp)
                _v_arr = _v_arr[t_idx] if _v_arr is not None else None
                _use_full = _bp_full is not None and _v_arr is not None and np.all(np.isfinite(_v_arr))

                # C(f) on f_grid for the main plot
                M_f = f_grid * Y1_i + (1.0 - f_grid) * Y2_i
                if _use_full:
                    B_f = np.interp(f_grid, _bp_full, _v_arr)
                else:
                    B_f = np.where(f_grid <= fsb,
                                   v0 + (vfs - v0) * f_grid / fsb,
                                   vfs + (v1 - vfs) * (f_grid - fsb) / (1.0 - fsb))
                C_f = B_f + (M_f - B_f) * lam
                scale = _display_scale(sp_global)
                ax.plot(f_grid, C_f / scale, color=color, linestyle=ls, marker=mk,
                        markevery=_mk_every_cp, mfc=_face(color), ms=6, lw=1.5, label=sp)
                _cf_max = float(np.nanmax(C_f)) / scale
                cf_max = _cf_max if cf_max is None else max(cf_max, _cf_max)
                # C(f) on f_pdf for the rate integrand
                M_fp = f_pdf * Y1_i + (1.0 - f_pdf) * Y2_i
                if _use_full:
                    B_fp = np.interp(f_pdf, _bp_full, _v_arr)
                else:
                    B_fp = np.where(f_pdf <= fsb,
                                    v0 + (vfs - v0) * f_pdf / fsb,
                                    vfs + (v1 - vfs) * (f_pdf - fsb) / (1.0 - fsb))
                reactant_Cf_pdf[sp_pos] = (B_fp + (M_fp - B_fp) * lam,
                                           order_lookup[j][sp_pos])

            # Rate integrand: β-PDF(f) × ∏ C_i(f)^order_i (no rescaling; row-shared axis)
            ax_pdf = ax.twinx()
            rig_max = 0.0
            if reactant_Cf_pdf:
                rate_ig = pdf_vals.copy()
                for Cf_p, order in reactant_Cf_pdf.values():
                    rate_ig *= np.maximum(Cf_p, 0.0) ** order
                ax_pdf.plot(f_pdf, rate_ig, color='tab:orange', lw=1.2, ls='--', zorder=0)
                rig_max = float(rate_ig.max())
            ax_pdf.set_ylabel('β-PDF·∏C', fontsize=7, color='grey')
            ax_pdf.tick_params(axis='y', labelcolor='grey', labelsize=6)
            ax_pdf.set_ylim(bottom=0)
            row_axpdf.append((ax_pdf, rig_max))
            ax.set_zorder(ax_pdf.get_zorder() + 1)
            ax.patch.set_visible(False)

            if cf_max is not None and cf_max > 0:
                ax.set_ylim(0, cf_max * 1.05)

            ax.axvline(mean_f, color='gray', lw=0.8, ls=':', alpha=0.7)
            ax.set_title(f'{lbl}  {t_label}', fontsize=9)
            ax.set_xlabel('f', fontsize=8)
            ax.set_ylabel('C(f) / feed scale', fontsize=8)
            ax.tick_params(labelsize=7)
            if reactant_pos.size > 0:
                ax.legend(fontsize=_leg_fs, loc='best', frameon=False)

        row_max = max((m for _, m in row_axpdf), default=0.0)
        if row_max > 0:
            for ax_pdf, _ in row_axpdf:
                ax_pdf.set_ylim(0, row_max * 1.05)

    fig.suptitle(
        f'C(f) reactant profiles per reaction  '
        f'(ε={m_epsilon:.4g},  mean_f={mean_f:.4f},  {weight_method})',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_stem is not None:
        _wm = ode_result.get('weight_method', 'unknown')
        _save_fig(fig, save_stem,
                  f'{_pathlib.Path(save_stem).name}_{_wm}_cprofile.png')
    else:
        plt.show()


def _clamp_total(res):
    """Return the total number of clamp steps across all species for one ODE result."""
    wm = res.get('weight_method', '')
    if wm == 'linear_interp':
        lc = res.get('li_clamps', {})
        return (sum(lc.get('clamp_below_min', {}).values())
                + sum(lc.get('clamp_above_max', {}).values()))
    if wm == 'blend_fs':
        bf = res.get('blendfs', {})
        return (sum(bf.get('clamp_below_M', {}).values())
                + sum(bf.get('clamp_above_B', {}).values()))
    if wm == 'ray_limit':
        rl = res.get('raylimit', {})
        return (sum(rl.get('clamp_below_M', {}).values())
                + sum(rl.get('clamp_above_B', {}).values()))
    return 0


def save_ios_csv(mean_f, m_epsilon, t_end, save_stem, n_pts=500,
                 m_lambda=DEFAULT_M_LAMBDA, m_nu=DEFAULT_M_NU, m_Sc=DEFAULT_M_SC):
    """Save the intensity-of-segregation curve I_s(t) to CSV.

    Columns: t (s), I_s (dimensionless).
    File is named <save_stem>_ios.csv."""
    import pathlib as _pathlib
    import csv as _csv
    import numpy as _np

    max_var = mean_f * (1.0 - mean_f)
    t_vals = _np.linspace(0.0, float(t_end), n_pts)
    ios_vals = _np.array([mixing_variance(t, mean_f, m_epsilon, m_lambda, m_nu, m_Sc) / max_var for t in t_vals])

    stem = _pathlib.Path(save_stem)
    plots_dir = stem.parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f'{stem.name}_ios.csv'
    with open(path, 'w', newline='') as fh:
        writer = _csv.writer(fh)
        writer.writerow(['t (s)', 'I_s'])
        for t, ios in zip(t_vals, ios_vals):
            writer.writerow([float(t), float(ios)])
    print(f"[IoS]   intensity of segregation saved to {path}")


def _method_summary(results_list, label=''):
    """Print per-method totals (CPU time, clamp steps) from a list of ODE result dicts."""
    from collections import defaultdict
    cpu_by_wm = defaultdict(float)
    clamp_by_wm = defaultdict(int)
    for r in results_list:
        wm = r.get('weight_method', 'unknown')
        cpu_by_wm[wm] += float(r.get('solve_cpu_s', 0.0))
        clamp_by_wm[wm] += _clamp_total(r)
    _col = 50
    _bar = '=' * _col
    print(f"\n{_bar}")
    if label:
        print(f"[summary] {label}")
    print(f"[summary] {'method':<20} {'CPU (s)':>10} {'clamp steps':>12}")
    print(f"[summary] {'-' * (_col - 10)}")
    for wm in sorted(cpu_by_wm):
        print(f"[summary] {wm:<20} {cpu_by_wm[wm]:>10.4g} {clamp_by_wm[wm]:>12d}")
    print(_bar)


if __name__ == '__main__':
    import json
    import pathlib
    import sys

    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    run_sweep = '--sweep' in sys.argv  # ε sweep is off by default
    make_movie = '--movie' in sys.argv  # ray_limit B(f) movies are off by default
    _SAVE_PDF = '--pdf' in sys.argv  # also save each figure as a vector PDF
    heatmaps_only = '--heatmaps-only' in sys.argv  # skip every plot except the Cf/Bf heatmaps (and their movies)

    config_path = pathlib.Path(args[0]) if args else pathlib.Path(__file__).parent / 'inputs' / 'input_example.json'
    # Outputs go next to the inputs/ folder, not inside it.
    base_dir = (config_path.parent.parent
                if config_path.parent.name == 'inputs'
                else config_path.parent)
    with open(config_path) as _f:
        _cfg = json.load(_f)

    species = _cfg['species']
    rxns = _cfg['reactions']
    Stream_1_Feed = np.array(_cfg['stream_feeds']['stream_1'])
    Stream_2_Feed = np.array(_cfg['stream_feeds']['stream_2'])
    rate_constants = _cfg.get('rate_constants', {})

    # If every reaction's rate constant is zero, nothing ever reacts: the
    # mixing-limited machinery downstream (selectivity rays, kappa1/kappa2
    # closed forms, f_conserved, etc.) all divide by some measure of reaction
    # progress, which is identically zero here -- rather than let that
    # surface as a divide-by-zero/NaN crash somewhere downstream, catch it
    # up front with a clear message.
    _rxn_labels_check = [r.split(':')[0].strip() for r in rxns]
    _k_vals_check = [float((rate_constants or {}).get(lbl, 1000.0)) for lbl in _rxn_labels_check]
    if _k_vals_check and not any(_k_vals_check):
        print(f"[error] every reaction's rate constant is zero ({config_path}); "
              f"nothing can react, so there is no mixing-limited closure to compute. Stopping.")
        sys.exit(1)

    nu_reactants, nu_products = parse_reactions(rxns, species)

    results = analyze_stream_limit_system(species, rxns, nu_reactants, nu_products,
                                          Stream_1_Feed, Stream_2_Feed)

    fs_extents = results.get('fs_extents', {})
    if fs_extents:
        N_red = results['reduced_matrix']
        print("\nNet species change at fs:")
        # pyrefly: ignore [missing-attribute]
        delta_Y = sum(xi * N_red[:, r] for r, xi in fs_extents.items())
        for s_idx, name in enumerate(species):
            # pyrefly: ignore [bad-index]
            if abs(delta_Y[s_idx]) > 1e-8:
                # pyrefly: ignore [bad-index]
                print(f"  - {name}: Δ = {delta_Y[s_idx]:.6f}")

    segments = generate_line_segments(species, Stream_1_Feed, Stream_2_Feed, results)
    print_line_segments(segments, species)

    report_subset_fs_residuals(species, rxns, nu_reactants, nu_products,
                               Stream_1_Feed, Stream_2_Feed)

    out_stem = config_path.stem.removeprefix('input_')
    lines_dir = base_dir / 'lines'
    lines_dir.mkdir(exist_ok=True)
    output_path = lines_dir / f'lines_{out_stem}.json'
    lines_out = export_all_subsets_json(species, rxns, nu_reactants, nu_products,
                                         Stream_1_Feed, Stream_2_Feed, output_path)
    print(f"\nLine segments saved to {output_path}")
    count_unique_profiles(lines_out['subsets'], species)

    # Optionally retain only some of the computed subset limits before deduping
    # into per-species profiles for the ODE integration.  Selection is by
    # integer subset_num via the config: 'keep_subsets' (whitelist) and/or
    # 'remove_subsets' (blacklist; 'discard_subsets' accepted as an alias).  These
    # affect ALL methods (they shrink the actual subset pool).  With neither field
    # present, all subsets are used (the default).  The saved lines_*.json above
    # still holds the full enumeration.  Note: 'blend_subsets' (read later) is a
    # separate, blend_fs-only selector that does NOT shrink this pool.
    _remove_cfg = _cfg.get('remove_subsets', _cfg.get('discard_subsets'))
    lines_for_species = filter_subsets_by_num(
        lines_out,
        keep=_cfg.get('keep_subsets'),
        discard=_remove_cfg)

    species_dir = base_dir / 'species'
    species_dir.mkdir(exist_ok=True)
    unique_sp_path = species_dir / f'species_{out_stem}.json'
    unique_sp_data = export_unique_species_json(lines_for_species, unique_sp_path)
    print(f"Unique species profiles saved to {unique_sp_path}")

    save_stem = base_dir / out_stem
    if not heatmaps_only:
        plot_concentration_profiles(species, results, segments, save_stem=save_stem)

        plot_all_subset_limits(species, rxns, nu_reactants, nu_products,
                               Stream_1_Feed, Stream_2_Feed, save_stem=save_stem,
                               keep_subsets=_cfg.get('keep_subsets'),
                               discard_subsets=_remove_cfg)

        plot_unique_species_profiles(unique_sp_data, save_stem=save_stem)

    BETA_MEAN_F = _cfg.get('mean_f', 0.2)
    # Turbulent dissipation rate ε: read from the config if given, else the default.
    # 'epsilon' may be a single value (the normal single set of results/plots) or a
    # list of values (the full set of results/plots is generated at each ε).
    _eps_cfg = _cfg.get('epsilon', DEFAULT_M_EPSILON)
    _multi_eps = isinstance(_eps_cfg, (list, tuple))
    BETA_EPSILONS = [float(e) for e in _eps_cfg] if _multi_eps else [float(_eps_cfg)]

    # Mixing-model constants: read from the config if given, else the code defaults.
    M_LAMBDA = _cfg.get('m_lambda', DEFAULT_M_LAMBDA)
    M_NU = _cfg.get('m_nu', DEFAULT_M_NU)
    M_SC = _cfg.get('m_Sc', DEFAULT_M_SC)

    # ray_limit only: uniform rate constant for the initial mass-action-rate
    # fallback ray, e.g. "initial_ray": 1.0.  Omit to use the real k_j.
    RAYLIMIT_INITIAL_RAY_K = _cfg.get('initial_ray')

    # Weighting methods to run.  The JSON key 'weight_methods' accepts a list of
    # integer codes; omit it (or set it to [2]) to get the ray_limit default.
    #   1 = blend_fs   2 = ray_limit   3 = linear_interp
    _WM_CODE = {1: 'blend_fs', 2: 'ray_limit', 3: 'linear_interp'}
    _wm_codes = _cfg.get('weight_methods', [2])
    RUN_METHODS = tuple(dict.fromkeys(_WM_CODE[int(c)] for c in _wm_codes if int(c) in _WM_CODE))
    if not RUN_METHODS:
        raise ValueError(f"'weight_methods' in config resolved to an empty list "
                         f"(codes {_wm_codes}; valid codes: {list(_WM_CODE)})")

    # When several ε are requested, give each its own plot filenames so they do not
    # overwrite one another; a single ε keeps the original (unsuffixed) names.
    def _eps_stem(eps):
        if not _multi_eps:
            return save_stem
        tag = f"{eps:g}".replace('+', '')
        return save_stem.parent / f'{out_stem}_eps{tag}'

    # No fixed integration time: CVODE terminates on the conversion event when
    # the stream-1 limiting reactant reaches near-full conversion.
    _all_base_results = []
    for _eps in BETA_EPSILONS:
        if _multi_eps:
            print(f"\n===== ε = {_eps:g} "
                  f"(τ_s = {mixing_timescale(_eps, M_LAMBDA):.4g} s) =====")
        _eps_save = _eps_stem(_eps)
        _blend_subsets_cfg = _cfg.get('blend_subsets')
        _blend_fs_ok = ('blend_fs' not in RUN_METHODS) or _blend_fs_check_runnable(
            unique_sp_data, Stream_1_Feed, Stream_2_Feed, _blend_subsets_cfg)
        if 'blend_fs' in RUN_METHODS and not _blend_fs_ok:
            print(f"[blend_fs]   not runnable (fewer than 2 products available "
                  f"for weighting); skipping.")
        ode_results = []
        for _wm in RUN_METHODS:
            if _wm == 'blend_fs' and not _blend_fs_ok:
                continue
            _res = integrate_species_odes(
                unique_sp_data, Stream_1_Feed, Stream_2_Feed,
                rate_constants=rate_constants,
                mean_f=BETA_MEAN_F,
                m_epsilon=_eps,
                weight_method=_wm,
                blend_subsets=_blend_subsets_cfg if _wm == 'blend_fs' else None,
                ode_rtol=_cfg.get('rtol'),
                ode_atol=_cfg.get('atol'),
                reaction_orders=_cfg.get('orders'),
                m_lambda=M_LAMBDA, m_nu=M_NU, m_Sc=M_SC,
                raylimit_initial_ray_k=RAYLIMIT_INITIAL_RAY_K)
            if _res.get('method_ran', True):
                ode_results.append(_res)
            else:
                print(f"[{_wm}]   method did not run (insufficient subsets); skipping plots.")

        # Compare the CPU time of the ODE solve across methods that ran.
        if ode_results:
            _cpus = [(r.get('weight_method', f'run{i}'), float(r.get('solve_cpu_s', float('nan'))))
                     for i, r in enumerate(ode_results)]
            print(f"\n[timing] ODE solve CPU time by method (ε={_eps:g}):")
            for _wm, _c in _cpus:
                print(f"[{_wm}]   [timing] {_c:.4g} s")
            if len(_cpus) > 1:
                _fast = min(_cpus, key=lambda kv: kv[1])
                _slow = max(_cpus, key=lambda kv: kv[1])
                print(f"[timing]   fastest: {_fast[0]} ({_fast[1]:.4g} s);  "
                      f"slowest: {_slow[0]} ({_slow[1]:.4g} s);  "
                      f"{_slow[1]/_fast[1]:.2f}× ratio")

        if not ode_results:
            continue
        if not heatmaps_only:
            plot_ode_trajectories(ode_results, save_stem=_eps_save)
            plot_product_fractions_vs_time(ode_results, Stream_1_Feed,
                                           nu_reactants, nu_products, save_stem=_eps_save,
                                           Y2=Stream_2_Feed)
            if abs(_eps - 100.0) < 1e-6 * 100.0:
                _t_end_ref = float(ode_results[0]['t'][-1])
                save_ios_csv(BETA_MEAN_F, _eps, _t_end_ref, _eps_save,
                            m_lambda=M_LAMBDA, m_nu=M_NU, m_Sc=M_SC)
        for _res in ode_results:
            if not heatmaps_only:
                plot_ode_limit_averages(_res, save_stem=_eps_save)
                plot_ode_beta_snapshots(unique_sp_data, _res, save_stem=_eps_save)
                plot_blendfs_diagnostics(_res, save_stem=_eps_save)
                _bsubs_for_plot = _cfg.get('blend_subsets') if _res.get('weight_method') == 'blend_fs' else None
                plot_blendfs_limit_grid(_res, save_stem=_eps_save,
                                        blend_subsets=_bsubs_for_plot)
                plot_raylimit_diagnostics(_res, save_stem=_eps_save)
                plot_raylimit_rotation(_res, save_stem=_eps_save)
                plot_raylimit_selectivity(_res, save_stem=_eps_save)
                plot_raylimit_early_selectivity(_res, save_stem=_eps_save)
                plot_raylimit_limit_grid(_res, save_stem=_eps_save)
            plot_raylimit_Cf_heatmap(_res, save_stem=_eps_save)
            plot_raylimit_Bf_heatmap(_res, save_stem=_eps_save)
            if not heatmaps_only:
                plot_raylimit_species_limits_static(unique_sp_data, _res, save_stem=_eps_save)
                if make_movie:
                    plot_raylimit_limit_movie(_res, save_stem=_eps_save)
                    plot_raylimit_species_limits_movie(unique_sp_data, _res, save_stem=_eps_save)
                plot_cprofile_per_reaction(unique_sp_data, _res, save_stem=_eps_save)
        _all_base_results.extend(ode_results)

    _method_summary(_all_base_results, label='Base-case runs — all ε values combined')
    if not heatmaps_only and _multi_eps and len(BETA_EPSILONS) > 2:
        _save_base_case_epsilon_csv(_all_base_results, RUN_METHODS, save_stem)

    # Sweep the turbulent dissipation rate ε and plot the product fractions
    # X_species (opt-in via --sweep).
    if run_sweep:
        _sweep_result = sweep_epsilon_product_fractions(
            unique_sp_data, Stream_1_Feed, Stream_2_Feed,
            epsilons=np.geomspace(1.0e-6, 1.0e6, 51),
            rate_constants=rate_constants, mean_f=BETA_MEAN_F,
            weight_methods=RUN_METHODS,
            save_stem=save_stem, blend_subsets=_cfg.get('blend_subsets'),
            ode_rtol=_cfg.get('rtol'), ode_atol=_cfg.get('atol'),
            reaction_orders=_cfg.get('orders'),
            m_lambda=M_LAMBDA, m_nu=M_NU, m_Sc=M_SC,
            raylimit_initial_ray_k=RAYLIMIT_INITIAL_RAY_K)
        if make_movie:
            _rl_sweep_results = _sweep_result.get('ray_limit', {}).get('ode_results')
            if _rl_sweep_results:
                if not heatmaps_only:
                    plot_raylimit_epsilon_sweep_movie(_rl_sweep_results, save_stem=save_stem)
                    plot_raylimit_species_limits_epsilon_sweep_movie(
                        unique_sp_data, _rl_sweep_results, save_stem=save_stem)
                plot_raylimit_Cf_heatmap_epsilon_sweep_movie(_rl_sweep_results, save_stem=save_stem)
                plot_raylimit_Bf_heatmap_epsilon_sweep_movie(_rl_sweep_results, save_stem=save_stem)
