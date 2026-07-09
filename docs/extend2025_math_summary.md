# `Extend_2025`: closed-form stoichiometric mixture fraction and closure profile $C(f)$

*Mixing-limited reaction closure — `species_limits.py`, `integrate_species_odes`*

## Setup

Active species $i\in\mathcal{A}$, reactions $j=1,\dots,R$.  Net stoichiometry matrix

$$N\in\mathbb{R}^{|\mathcal{A}|\times R},\qquad N_{ij}=\nu^{\mathrm{prod}}_{ij}-\nu^{\mathrm{reac}}_{ij}.$$

Feed vectors $Y^{(1)},Y^{(2)}\in\mathbb{R}^{|\mathcal{A}|}$ (stream 1 at $f=1$, stream 2 at
$f=0$).  The **no-reaction line** at mixture fraction $f\in[0,1]$ is

$$M_i(f)=f\,Y^{(1)}_i+(1-f)\,Y^{(2)}_i.$$

Let $y$ be the current ODE state, $y^0$ its initial value, and $\langle\,\cdot\,\rangle$
the expectation under the step's $\mathrm{Beta}(\alpha_t,\beta_t)$ pdf.

`Extend_2025` shares its selectivity ray with `ray_limit` (Section 1) but replaces
`ray_limit`'s general breakpoint search with a **closed-form** stoichiometric
mixture-fraction formula (Section 2) — the classical two-stream "$f_s$" formula from
reactive-mixing theory, generalized to any number of reactants per stream.

---

## 1. Selectivity ray (shared with `ray_limit`)

The accumulated change $\Delta:=y-y^0$ lies in $\operatorname{range}(N)$.  Since
$\Delta = N\int_0^t r\,d\tau = N\,\xi(t)$ for the vector of per-reaction extents
$\xi_j(t)=\int_0^t r_j\,d\tau'\ge 0$, collapsing the whole network to **one** effective
reaction with $d=\Delta$ is exact: $\Delta$ *is* $\sum_j\xi_j N_j$, so no extent needs to
be tracked or recovered individually — the sum is already the ray.  `Extend_2025` reuses
`ray_limit`'s own `_selectivity_ray` unchanged:

$$d = \Delta = y - y^0,$$

falling back near $y^0$ (where $\Delta\approx 0$) to the mass-action projection
$d = N\,r(y)$.

The components of $d$ are the coefficients of the **one-line effective reaction**:
species with $d_i<0$ are net reactants with coefficient $\nu_i=-d_i$; species with
$d_i>0$ are net products with coefficient $d_i$.

---

## 2. Closed-form stoichiometric mixture fraction $f_s$

Rather than the general pairwise-crossing search over all consumed species
(`ray_limit`'s `_cmax_along`), `Extend_2025` exploits a structural fact about two-stream
feeds: if a net-reactant species is fed **purely** by one stream, its constraint curve
$M_i(f)/\nu_i$ is a single line through the origin at that stream's end.

**Reduction within a stream.** For reactants fed purely by stream 1 ($Y^{(2)}_i=0$),
$M_i(f)/\nu_i = f\cdot(Y^{(1)}_i/\nu_i)$ — a line through $f=0$.  Two such lines with
different slopes meet only at $f=0$, so for $f>0$ the one with the **smaller** slope is
the minimum everywhere on $(0,1]$: the whole group collapses to a single line with slope

$$\kappa_1 = \min_{i\;\text{pure stream 1}}\frac{Y^{(1)}_i}{\nu_i}.$$

Symmetrically, reactants fed purely by stream 2 ($Y^{(1)}_j=0$) collapse to

$$\kappa_2 = \min_{j\;\text{pure stream 2}}\frac{Y^{(2)}_j}{\nu_j}.$$

**The crossing.** The achievable extent is $c_{\max}(f)=\min\bigl(f\kappa_1,\,(1-f)\kappa_2\bigr)$
— one line increasing from $0$, one decreasing to $0$ — so they cross at exactly one
interior point:

$$\boxed{f_s = \frac{\kappa_2}{\kappa_1+\kappa_2}}\,,\qquad c^{\star}=f_s\,\kappa_1=(1-f_s)\,\kappa_2.$$

This is the standard two-stream stoichiometric mixture fraction, generalized beyond a
single reactant per stream by taking the binding (minimum-ratio) reactant on each side.
The complete-reaction limit is the same **two-segment tent** as `ray_limit`'s (only its
derivation differs):

$$B_i(0)=Y^{(2)}_i,\qquad B_i(f_s)=M_i(f_s)+c^{\star}d_i,\qquad B_i(1)=Y^{(1)}_i.$$

**Scope.** The reduction requires every net-reactant species to be fed by exactly one
stream. Species absent from both feeds ($y^0_i=0$) can never violate this — since
concentrations are non-negative, $d_i=y_i-y^0_i=y_i\ge 0$, so an unfed intermediate is
never a net reactant. If an *actual feed* species is present in both streams and turns
out to be a net reactant, the closed form does not apply; `Extend_2025` does not fall
back to `ray_limit`'s general search in that case — it prints a diagnostic message and
refuses to compute a limit for that step (rates set to $0$).

---

## 3. Closure profile $C_i(f)$ and scalar $\lambda$

Identical machinery to `ray_limit` (same boxed formula, same $\lambda$, same segment
integrals — only $(f_s,B)$'s construction differs):

$$\boxed{C_i(f) = B_i(f) + \bigl[M_i(f)-B_i(f)\bigr]\,\lambda_i}$$

$$\lambda_i = \operatorname{clip}_{[0,1]}\frac{y_i - \langle B_i\rangle}{\langle M_i\rangle - \langle B_i\rangle}.$$

Segment integrals use the regularised incomplete Beta function $I_x(\alpha,\beta)$:

$$\int_{f_a}^{f_b}(sf+b)\,p_\beta\,df
= s\,\frac{\alpha_t}{\alpha_t+\beta_t}
  \bigl[I_{f_b}(\alpha_t{+}1,\beta_t)-I_{f_a}(\alpha_t{+}1,\beta_t)\bigr]
+ b\,\bigl[I_{f_b}(\alpha_t,\beta_t)-I_{f_a}(\alpha_t,\beta_t)\bigr].$$

Because $B_i-M_i=c_{\max}(f)\,d_i$ throughout (same as `ray_limit`), the same **global
$\lambda$** identity holds: $\lambda_i = 1-1/\langle c_{\max}\rangle$ is the same scalar
for every species, and $\langle C_i\rangle=y_i$ exactly whenever the clamp is not
binding.

---

## 4. Worked illustration: `input_fuller_BC_azo_coupling_kinetics`, $\varepsilon=1$ W/kg

**Scheme:** R1: $A{+}B\!\to\!R$, R2: $A{+}R\!\to\!S$, R3: $A{+}B\!\to\!T$,
R4: $A{+}T\!\to\!S$, R5: $A{+}C\!\to\!Q$.
**Feeds:** $Y^{(1)}=15\,A$, $Y^{(2)}=1.2\,B+1.2\,C$; $\bar{f}=0.0625$.

Same output-step time-midpoint as `ray_limit`'s own worked example
($t=t_{\mathrm{end}}/2=0.07629$ s $\approx3.08\,\tau_s$, $\tau_s=0.02476$ s,
$(\alpha_t,\beta_t)=(0.372,\,5.575)$ — this instant is $\approx70\%$ $A$ conversion, not
$50\%$), the ray gives the effective reaction

$$0.6556\,A + 0.3314\,B + 0.2868\,C \;\longrightarrow\; 0.2832\,R + 0.0108\,T + 0.0374\,S + 0.2868\,Q.$$

**Closed-form $f_s$.** $A$ is pure stream 1 ($Y^{(1)}_A=15$); $B,C$ are pure stream 2
($Y^{(2)}_B=Y^{(2)}_C=1.2$):

$$\kappa_1 = \frac{15}{0.6556}=22.88,\qquad
\kappa_2 = \min\!\left(\frac{1.2}{0.3314},\,\frac{1.2}{0.2868}\right)=\min(3.622,\,4.184)=3.622\ (B\text{ binds}).$$

$$f_s = \frac{3.622}{22.88+3.622}=0.1366,\qquad c^{\star}=f_s\,\kappa_1 = 3.127.$$

Peak values $v^{\mathrm{fs}}_i=M_i(f_s)+c^{\star}d_i$ (both $A$ and $B$ vanish exactly at
$f_s$, by construction) and the resulting averages and closure scalars:

| Species | $\langle M_i\rangle=y^0_i$ | $v^{\mathrm{fs}}_i$ (mol/L) | $\langle B_i\rangle$ (mol/L) | $y_i$ (mol/L) | $\lambda_i$ | $\langle C_i\rangle$ |
|---|---|---|---|---|---|---|
| $A$ | 0.9375 | 0.000 | 0.2814 | 0.2819 | 0.000839 | 0.2819 |
| $B$ | 1.1250 | 0.000 | 0.7934 | 0.7936 | 0.000839 | 0.7936 |
| $C$ | 1.1250 | 0.139 | 0.8379 | 0.8382 | 0.000839 | 0.8382 |
| $R$ | 0.0000 | 0.886 | 0.2835 | 0.2832 | 0.000839 | 0.2832 |
| $T$ | 0.0000 | 0.034 | 0.01076 | 0.01075 | 0.000839 | 0.01075 |
| $S$ | 0.0000 | 0.117 | 0.03740 | 0.03737 | 0.000839 | 0.03737 |
| $Q$ | 0.0000 | 0.897 | 0.2871 | 0.2868 | 0.000839 | 0.2868 |

Every $\lambda_i$ equals the same global scalar and $\langle C_i\rangle=y_i$ to the
quoted digits — identical structure to `ray_limit`.  At this same instant `ray_limit`'s
own worked example gives $d_A=-0.6556$, $d_B=-0.3314$, $d_C=-0.2868$, $f_s^{\star}=0.1366$,
$c^{\star}=3.127$ — matching to the quoted digits, as expected: $A$, $B$, $C$ are each
fed by exactly one stream throughout this network (and $R,T,S,Q$, being unfed, can never
violate that condition), so the closed form applies at every step and `Extend_2025`
reproduces `ray_limit`'s trajectory numerically, at matching CPU cost, without ever
performing the general breakpoint search.

---

## Note: relation to `ray_limit`

`Extend_2025` and `ray_limit` share the selectivity ray $d$ and the entire closure-profile
machinery (Section 3); they differ only in how $(f_s, B)$ is constructed from $d$:
`ray_limit`'s search handles *any* feed distribution (including a reactant fed by both
streams) via an $O(|\mathcal{C}|^2)$ pairwise-crossing search; `Extend_2025`'s closed
form handles only the pure single-stream-per-reactant case, in closed form, and refuses
outright — rather than silently falling back — when that condition fails. Whenever the
condition holds (as it does for every network exercised so far), the two constructions
are mathematically identical, not merely similar: both reduce to the unique crossing of
one increasing and one decreasing affine function of $f$.
