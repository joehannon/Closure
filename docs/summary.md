The code models a two-stream reactor as a linear program: given feeds $Y^1$ and $Y^2$ mixed at fraction $f$, find reaction extents $\xi_r \geq 0$ that maximise total conversion subject to all species remaining non-negative.

Before solving, unfed intermediates are eliminated by algebraically fusing producer–consumer reaction pairs, reducing the LP dimension.

The stoichiometric mixing fraction $f_s$ — where cross-stream reactants are jointly exhausted — is found by a joint LP over both $f$ and the extents simultaneously, using the blended feed $Y^{\text{mix}}(f) = fY^1 + (1-f)Y^2$ as inequality constraints so that excess co-reactants remain as slack rather than forcing premature depletion.

At any fixed $f$, a two-phase sequential solve runs single-stream reactions first (then non-catalytic, then catalytic), which correctly accounts for catalyst depletion without needing a single large LP.

Because the LP solution is piecewise affine in $f$, concentration profiles are computed exactly by solving only at a small set of algebraically derived breakpoints and connecting them with straight lines — no numerical grid required.
