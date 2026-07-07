# `ray_limit`: selectivity ray, complete-reaction limit, and closure profile $C(f)$

*Mixing-limited reaction closure — `species_limits.py`, `integrate_species_odes`*

## Setup

Active species $i\in\mathcal{A}$, reactions $j=1,\dots,R$.  Net stoichiometry matrix

$$N\in\mathbb{R}^{|\mathcal{A}|\times R},\qquad N_{ij}=\nu^{\mathrm{prod}}_{ij}-\nu^{\mathrm{reac}}_{ij}.$$

Feed vectors $Y^{(1)},Y^{(2)}\in\mathbb{R}^{|\mathcal{A}|}$ (stream 1 at $f=1$, stream 2 at
$f=0$).  The **no-reaction line** at mixture fraction $f\in[0,1]$ is

$$M_i(f)=f\,Y^{(1)}_i+(1-f)\,Y^{(2)}_i.$$

Let $y$ be the current ODE state, $y^0$ its initial value, and $\langle\,\cdot\,\rangle$
the expectation under the step's $\mathrm{Beta}(\alpha_t,\beta_t)$ pdf.

---

## 1. Selectivity ray

The accumulated change $\Delta:=y-y^0$ lies in $\operatorname{range}(N)$.  The
**selectivity ray** is taken directly as this accumulated extent:

$$d = \Delta = y - y^0.$$

This preserves the true reaction direction without distortion from a nonnegativity clip —
important when products form by more than one route (parallel reactions, intermediates).
Near $y^0$ (where $\Delta\approx 0$ and the direction is ill-conditioned) the code falls
back to the mass-action projection $d = N\,r(y)$, where $r(y)$ is the vector of
instantaneous mass-action rates.

The ray $d$ encodes the **current selectivity**: which species have been consumed or
produced and in what ratio.  It rotates as the ODE state evolves, tracking the shift
between competing pathways.

---

## 2. The complete-reaction limit $B(f)$

$B_i(f)$ is built by reacting the mixing line $M(f)$ to completion along the ray $d$
— advancing as far as possible without any species going negative.

**Maximum extent.** At each $f$ the largest non-negative scalar $c_{\max}(f)$ that
keeps $M(f)+c\,d\ge 0$ is

$$c_{\max}(f) = \min_{\substack{i:\,d_i<0}}\frac{M_i(f)}{-d_i}.$$

Only consumed species ($d_i<0$) constrain $c_{\max}$; products ($d_i>0$) are
unconstrained.  The limit is then

$$B_i(f) = M_i(f) + c_{\max}(f)\,d_i.$$

---

## 3. Closure profile $C_i(f)$ and scalar $\lambda$

The closure profile interpolates between the complete-reaction limit $B_i$ and the
no-reaction line $M_i$:

$$\boxed{C_i(f) = B_i(f) + \bigl[M_i(f)-B_i(f)\bigr]\,\lambda_i}$$

$\lambda_i=0$ gives the pure complete-reaction limit; $\lambda_i=1$ gives pure
no-reaction mixing.  $\lambda_i$ is set so that $\langle C_i\rangle = y_i$:

$$\lambda_i = \operatorname{clip}_{[0,1]}\frac{y_i - \langle B_i\rangle}{\langle M_i\rangle - \langle B_i\rangle}.$$

Segment integrals use the regularised incomplete Beta function $I_x(\alpha,\beta)$:

$$\int_{f_a}^{f_b}(sf+b)\,p_\beta\,df
= s\,\frac{\alpha_t}{\alpha_t+\beta_t}
  \bigl[I_{f_b}(\alpha_t{+}1,\beta_t)-I_{f_a}(\alpha_t{+}1,\beta_t)\bigr]
+ b\,\bigl[I_{f_b}(\alpha_t,\beta_t)-I_{f_a}(\alpha_t,\beta_t)\bigr].$$

**Global $\lambda$ for pure cross-stream.** Since $M_i$ is linear,
$\langle M_i\rangle = M_i(\bar{f}) = y^0_i$, and $B_i - M_i = c_{\max}\,d_i$
by construction, so

$$\lambda_i = \frac{y_i - \langle B_i\rangle}{y^0_i - \langle B_i\rangle}
= \frac{d_i(1-\langle c_{\max}\rangle)}{-\langle c_{\max}\rangle\,d_i}
= 1 - \frac{1}{\langle c_{\max}\rangle}$$

— **the same scalar for every species** — and $\langle C_i\rangle = y_i$ exactly
(no clamp needed).  Per-species $\lambda_i$ diverge from this only when the clamp
$[0,1]$ binds.

---

## 4. Worked illustration: `input_fuller_BC_azo_coupling_kinetics`, $\varepsilon=1$ W/kg

**Scheme:** R1: $A{+}B\!\to\!R$, R2: $A{+}R\!\to\!S$, R3: $A{+}B\!\to\!T$, R4: $A{+}T\!\to\!S$, R5: $A{+}C\!\to\!Q$.  
**Feeds:** $Y^{(1)}=15\,A$, $Y^{(2)}=1.2\,B+1.2\,C$; $\bar{f}=0.0625$.

**The effective single reaction.** The selectivity ray $d=\Delta$ records the net moles
of each species consumed or produced per unit of accumulated reaction progress. Its
components $d_i$ are the net stoichiometric coefficients of a single effective lumped
reaction encoding the current mixture of all five pathways:

$$|d_A|\,A + |d_B|\,B + |d_C|\,C \;\longrightarrow\; d_R\,R + d_S\,S + d_Q\,Q$$

(species with $d_i\approx 0$, here $T$, are in steady-state balance and do not appear
explicitly). The coefficients rotate as the ODE state evolves. At the half-way output
step ($\approx 50\%$ $A$ conversion):

| | $\varepsilon=10^{-6}$ | $\varepsilon=1$ | $\varepsilon=10^{6}$ |
|---|---|---|---|
| $(\alpha_t,\beta_t)$ | $(0.288,\,4.32)$ | $(0.379,\,5.68)$ | $(17.7,\,265)$ |
| kink $f_s^{\star}$ | $0.193$ | $0.135$ | $0.078$ |
| $\lambda$ (global) | $0.000^{\dagger}$ | $0.001$ | $0.054$ |
| $d_A$ | $-0.706$ | $-0.655$ | $-0.868$ |
| $d_B$ | $-0.237$ | $-0.337$ | $-0.819$ |
| $d_C$ | $-0.237$ | $-0.291$ | $-0.048$ |
| $d_R$ | $+0.004$ | $+0.309$ | $+0.818$ |
| $d_S$ | $+0.233$ | $+0.027$ | $+0.001$ |
| $d_Q$ | $+0.237$ | $+0.291$ | $+0.048$ |

$^{\dagger}$At $\varepsilon=10^{-6}$ the mean has reached the complete-reaction average, so $\lambda$ clamps to $0$ and $\langle C_i\rangle=\langle B_i\rangle$.

**Building the limit at $\varepsilon=1$.** The effective reaction at this step is

$$0.655\,A + 0.337\,B + 0.291\,C \;\longrightarrow\; 0.309\,R + 0.027\,S + 0.291\,Q.$$

The complete-reaction limit $B(f)$ is found by advancing $M(f)$ along this reaction as
far as possible without a reactant going negative:

$$c_{\max}(f) = \min\!\left(\frac{15f}{0.655},\;\frac{1.2(1-f)}{0.337},\;\frac{1.2(1-f)}{0.291}\right).$$

Since $B$ is tighter than $C$ (smaller denominator), the binding constraints are $A$ and
$B$. With $p_i=M_i(0)/|d_i|$, $q_i=(M_i(1)-M_i(0))/|d_i|$:

$$(p,q)_A=(0,\;22.89),\quad (p,q)_B=(3.565,\;{-3.565}),\quad (p,q)_C=(4.119,\;{-4.119}).$$

The $A$- and $B$-lines cross at the kink: $f_s^{\star}=3.565/26.46=0.135$,
$c^{\star}=3.083$.  Peak values $v^{\mathrm{fs}}_i=M_i(f_s^{\star})+c^{\star}d_i$
(e.g.\ $v^{\mathrm{fs}}_R = 3.083\times0.309=0.953$) and resulting averages and closure scalars:

| $\varepsilon=1$ | $\langle M_i\rangle=y^0_i$ | $v^{\mathrm{fs}}_i$ (mol/m³) | $\langle B_i\rangle$ (mol/m³) | $y_i$ (mol/m³) | $\lambda_i$ | $\langle C_i\rangle$ |
|---|---|---|---|---|---|---|
| $A$ | 0.9375 | 0.000 | 0.282 | 0.282 | 0.001 | 0.282 |
| $B$ | 1.1250 | 0.000 | 0.788 | 0.788 | 0.001 | 0.788 |
| $C$ | 1.1250 | 0.140 | 0.833 | 0.834 | 0.001 | 0.834 |
| $R$ | 0.0000 | 0.954 | 0.310 | 0.309 | 0.001 | 0.309 |
| $S$ | 0.0000 | 0.084 | 0.027 | 0.027 | 0.001 | 0.027 |
| $Q$ | 0.0000 | 0.899 | 0.292 | 0.291 | 0.001 | 0.291 |

Every $\lambda_i$ equals the global scalar and $\langle C_i\rangle=y_i$ to the quoted digits.

---

## Note: single interior kink

**Single interior kink.** For a pure cross-stream scheme the consumed species are absent
from their respective feed endpoints ($M_i(0)=0$ or $M_i(1)=0$), so $c_{\max}=0$ at
$f=0$ and $f=1$.  Between the endpoints $c_{\max}(f)$ is piecewise linear and concave;
its kink $f_s^{\star}$ is where the two tightest consumed-species constraints are
simultaneously binding:

$$f_s^{\star} = \frac{p_b - p_a}{q_a - q_b},
\qquad p_i = \frac{M_i(0)}{-d_i},\quad q_i = \frac{M_i(1)-M_i(0)}{-d_i}.$$

The peak extent at the kink is $c^{\star}=c_{\max}(f_s^{\star})$.  This gives the
**two-segment tent**:

$$B_i(f) = \begin{cases}
M_i(f) + \dfrac{f}{f_s^{\star}}\,c^{\star}\,d_i & f \le f_s^{\star}\\[8pt]
M_i(f) + \dfrac{1-f}{1-f_s^{\star}}\,c^{\star}\,d_i & f > f_s^{\star}
\end{cases}$$

with control-point peak value $v^{\mathrm{fs}}_i = M_i(f_s^{\star})+c^{\star}d_i$ and
pure-feed endpoints $B_i(0)=Y^{(2)}_i$, $B_i(1)=Y^{(1)}_i$.

This structure holds when every species with $d_i<0$ satisfies $Y^{(1)}_i=0$ or
$Y^{(2)}_i=0$.

**Intermediates are safe.** A species absent from both feeds has $y^0_i=0$, and since
concentrations are non-negative, $d_i = y_i - y^0_i = y_i \ge 0$.  So an intermediate
can never have $d_i<0$; the condition is vacuously satisfied for it.

**Condition failure does not cause collapse.** If a feed species appears in both streams
($Y^{(1)}_i>0$ and $Y^{(2)}_i>0$) and $d_i<0$, then $M_i(f)>0$ for all $f$ and
$c_{\max}$ remains positive.  The only consequence is that $B(f)$ loses its
endpoint-pinning ($B(0)\ne Y^{(2)}$ or $B(1)\ne Y^{(1)}$); the piecewise-linear
machinery still computes the limit correctly.
