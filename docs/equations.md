# Species Limits — Program Logic as Equations

## Notation

| Symbol | Meaning |
|---|---|
| $s = 1\ldots S$, $r = 1\ldots R$ | species, reactions |
| $\nu^-_{sr},\ \nu^+_{sr}$ | reactant / product stoichiometric coefficients |
| $N_{sr} = \nu^+_{sr} - \nu^-_{sr}$ | net stoichiometric matrix |
| $Y^1_s,\ Y^2_s$ | feed amounts in stream 1, stream 2 |
| $f \in [0,1]$ | mixture fraction (stream-1 fraction) |
| $\xi_r \geq 0$ | extent of reaction $r$ |

---

## 1. Pure mixing

$$Y^{\text{mix}}_s(f) = f\,Y^1_s + (1-f)\,Y^2_s$$

---

## 2. Reacted state

$$Y_s(f) = Y^{\text{mix}}_s(f) + \sum_r N_{sr}\,\xi_r, \qquad Y_s(f) \geq 0 \;\forall s$$

---

## 3. Intermediate elimination

If unfed species $I$ has a unique producer $r_p$ and unique consumer $r_c$, they are fused. Scale factors cancel $I$ exactly:

$$\sigma_p = \nu^-_{I,r_c}, \qquad \sigma_c = \nu^+_{I,r_p}$$

$$N^{\text{fused}} = \sigma_p\, N_{r_p} + \sigma_c\, N_{r_c}, \qquad N^{\text{fused}}_I = 0$$

(Skipped if $r_p$ has a competitor with identical fed-reactant stoichiometry.)

---

## 4. Effective feeds after single-stream pre-solve

Single-stream reactions (all reactants from one stream) run to completion on their own feed before any cross-stream balance:

$$Y^{1,\text{eff}} = Y^1 + N_{\mathcal{S}_1}\,\xi^*_{\mathcal{S}_1}, \qquad Y^{2,\text{eff}} = Y^2 + N_{\mathcal{S}_2}\,\xi^*_{\mathcal{S}_2}$$

where $\mathcal{S}_1, \mathcal{S}_2$ are the stream-1-only and stream-2-only reaction sets, and $\xi^*$ solves the extent LP on the respective pure feeds.

---

## 5. Extent LP (fixed $f$)

$$\max_{\xi \geq 0} \quad \mathbf{c}^\top \xi$$

$$\text{subject to} \quad -N_{\mathcal{A}}\,\xi \leq Y^{\text{mix}}(f)$$

$$\xi_i = \xi_j \quad \text{for competing primary pairs } (i,j)$$

$$\xi_r = 0 \quad \text{if catalyst } k \text{ of reaction } r \text{ is absent } (Y_k < \varepsilon)$$

Objective weights encode a tiebreaker that matches routing with the $f_s$ LP:

$$c_r = 1 + \varepsilon_0 \cdot \frac{\min_{s \in \text{req}(r)} Y_s \;\cdot\; \sum_{s\,\in\,\text{fed}} \nu^-_{sr}}{c_{\max}}$$

---

## 6. Stoichiometric mixing fraction $f_s$ LP

Variables: $f \in [0,1]$ and extents $\xi_r$ for all active reactions. Effective feeds $Y^{1,\text{eff}}, Y^{2,\text{eff}}$ from step 4 replace raw feeds.

$$\max_{f,\,\xi \geq 0} \quad \sum_r w_r\,\xi_r, \qquad w_r = \sum_{s\,\in\,\text{fed}} \nu^-_{sr}$$

$$\text{subject to} \quad -\!\left(Y^{1,\text{eff}}_s - Y^{2,\text{eff}}_s\right)f - \sum_r N_{sr}\,\xi_r \leq Y^{2,\text{eff}}_s \quad \forall s \text{ consumed}$$

$$f \in [0,1], \qquad \xi_i = \xi_j \quad \text{(competing primaries)}$$

This constraint written out is just $Y^{\text{mix}}_s(f) + N_s \cdot \xi \geq 0$ — species non-negativity as inequalities so that excess co-reactants from the same stream remain as slack.

---

## 7. Piecewise-linear profiles

The LP solution $\xi^*(f)$ is piecewise affine in $f$, so $Y_s(f)$ is piecewise linear. Breakpoints occur at:

$$\mathcal{F} = \{0,\; f_s,\; 1\} \;\cup\; \left\{f^* = \frac{-Y^{2,\text{eff}}_s}{Y^{1,\text{eff}}_s - Y^{2,\text{eff}}_s} : f^* \in (0,1)\right\} \;\cup\; \{\varepsilon,\; f_s \pm \varepsilon\}_{\text{catalytic}}$$

On each segment $[f_k, f_{k+1}]$:

$$Y_s(f) = m_{s,k}\,f + b_{s,k}, \qquad m_{s,k} = \frac{Y_s(f_{k+1}) - Y_s(f_k)}{f_{k+1} - f_k}$$

The LP is solved only at the breakpoints in $\mathcal{F}$ — not on a grid — giving exact profiles.
