# blend\_fs: dynamic reaction-limit closure for two-stream reactors

**System:** `input_fuller_BC_azo_coupling_kinetics` — species A, B, C, R, T, S, Q;
reactions R1–R5.  
**Example point:** $\varepsilon = 1$ W/kg, $t = 0.0576$ s ($\approx 50\%$ A conversion).

---

## 1. Overview

`blend_fs` constructs a piecewise-linear concentration profile $C_i(f)$ over mixture
fraction $f \in [0,1]$ that tracks the **reaction limit** as the ODE composition evolves.
Rather than committing to a single stoichiometric subset, it **blends** $K \geq 1$
user-specified "one-short" subset limits, weighting them by the concentrations of their
distinguishing products in the current ODE state.  The blend is converted to a single
kink position $f_{sb}$ and a peak height $v^{fs}_i$ for each species, and a scalar
$\lambda_i \in [0,1]$ interpolates between the no-reaction line and the limit profile so
that the Beta-weighted mean matches the ODE value.

The $K$ subsets are prescribed by the user via the `blend_subsets` key in the JSON
configuration file (e.g. `"blend_subsets": [20, 31]`).  Each entry is an integer index
into the enumeration of all reaction subsets; any number of subsets $K \geq 1$ may be
listed.

---

## 2. Background: mixture fraction and Beta distribution

Mixture fraction $f$ is the mass (or molar) fraction originating from stream 1.  The
feed concentrations are

$$Y_1: \quad A = 15 \text{ mol/L},\quad \text{others} = 0$$
$$Y_2: \quad B = 1.2,\quad C = 1.2 \text{ mol/L},\quad \text{others} = 0$$

with mean $\bar{f} = 0.0625$.  At time $t$ the mixture-fraction PDF is
$\text{Beta}(\alpha,\beta)$ with

$$s = \frac{\bar{f}(1-\bar{f})}{\sigma^2(t)} - 1, \qquad \alpha = \bar{f}\,s, \qquad \beta = (1-\bar{f})\,s$$

At the example point: $\sigma^2 = 0.01622$, $s = 2.613$, $\alpha = 0.1633$,
$\beta = 2.4498$, $\tau_s = 0.02476$ s.

---

## 3. Subset stoichiometric limits

For a reaction subset $k \in \{1, \ldots, K\}$, the **stoichiometric mixture fraction**
$f_{sk}$ is the value of $f$ at which the stream-2 reactant is exactly consumed if only
subset $k$'s reactions run.  At $f = f_{sk}$ each species $i$ attains a maximum
concentration $c^{fs}_{ki}$.

The **no-reaction line** (conservative mixing, no reaction) is

$$M_i(f) = Y_{2i} + (Y_{1i} - Y_{2i})\,f$$

The user prescribes the $K$ subset indices via `blend_subsets`; for this example $K = 2$
(subsets 20 and 31):

| Subset | Reactions | $f_{sk}$ | Distinguishing products | $c^{fs}_{k}$ (non-zero, mol/L) |
|--------|-----------|----------|-------------------------|-------------------------------|
| 20 | R1, R3, R5 | 0.1379 | R, T | R = T = 0.5172, Q = 1.0345 |
| 31 | R1–R5 | 0.1935 | S | S = Q = 0.9677 |

Distinguishing products for subset $k$ are those whose limit concentration is non-zero
in subset $k$'s limit but zero in all other subsets' limits.

---

## 4. Dynamic blend weights

Each subset $k$ has a set of **distinguishing products** $\mathcal{D}_k$.  The weight for
$k = 1, \ldots, K$ is

$$w_k = \frac{\sum_{i \in \mathcal{D}_k} y_i}{\sum_{k'=1}^{K} \sum_{i \in \mathcal{D}_{k'}} y_i}$$

where $y_i$ is the ODE concentration at the current time step.  The weights sum to unity
and shift dynamically as the composition evolves: subsets whose distinguishing products
are more abundant receive higher weight.

For this example ($K = 2$) at the example point:

$$w_{20} = \frac{y_R + y_T}{(y_R + y_T) + y_S} = \frac{0.2290 + 0.0083}{0.2290 + 0.0083 + 0.0187} = 0.9271$$

$$w_{31} = \frac{y_S}{(y_R + y_T) + y_S} = \frac{0.0187}{0.2560} = 0.0729$$

---

## 5. Blended kink position $f_{sb}$

The kink is placed using **odds ratios** to remain in the stoichiometric frame:

$$s_k = \frac{f_{sk}}{1 - f_{sk}}, \qquad
s_\text{blend} = \sum_{k=1}^{K} w_k\,s_k, \qquad
f_{sb} = \frac{s_\text{blend}}{1 + s_\text{blend}}$$

The odds ratio $s_k = f_{sk}/(1-f_{sk})$ is proportional to the A:B molar consumption
ratio in subset $k$.  Averaging in odds-ratio space therefore averages the blended
stoichiometry, which is the physically correct quantity to interpolate.

For this example ($K = 2$) at the example point:

$$s_{20} = \frac{0.1379}{0.8621} = 0.1600, \qquad s_{31} = \frac{0.1935}{0.8065} = 0.2400$$

$$s_\text{blend} = 0.9271 \times 0.1600 + 0.0729 \times 0.2400 = 0.16583$$

$$f_{sb} = \frac{0.16583}{1.16583} = 0.1422$$

---

## 6. Control-point limit profile $B_i(f)$

The limit profile is a **tent function** — piecewise linear with a single kink at $f_{sb}$:

$$B_i(f) = \begin{cases}
v^0_i + (v^{fs}_i - v^0_i)\,\dfrac{f}{f_{sb}} & f \le f_{sb}\\[6pt]
v^{fs}_i + (v^1_i - v^{fs}_i)\,\dfrac{f - f_{sb}}{1 - f_{sb}} & f > f_{sb}
\end{cases}$$

The control-point values are formed from the weighted subset limits.  For the **peak**,
the blend is evaluated at the stoichiometric yield at $f_{sb}$:

$$v^{fs}_i = (1 - f_{sb}) \sum_{k=1}^{K} w_k \frac{c^{fs}_{ki}}{1 - f_{sk}}$$

In each term of the sum, $c^{fs}_{ki}/(1 - f_{sk})$ is the stoichiometric yield of
species *i* per mol of stream-2 reactant consumed in subset *k* (independent of feed
concentrations); the prefactor $(1 - f_{sb})$ is the stream-2 reactant available at the
blended kink.  The formula holds for any $K \geq 1$ because the subsets are
stoichiometrically independent.  The endpoint values at $f = 0$ and $f = 1$ are
weighted averages of the subset limits; for unfed product species both are zero.

**Example ($K = 2$) — species R and S:**

$$v^{fs}_R = 0.8578 \times \left[0.9271 \times \frac{0.5172}{0.8621} + 0\right] = 0.4771 \text{ mol/L}$$

$$v^{fs}_S = 0.8578 \times \left[0 + 0.0729 \times \frac{0.9677}{0.8065}\right] = 0.0751 \text{ mol/L}$$

---

## 7. Species profile $C_i(f)$ and $\lambda_i$

The full species profile starts from the blended limit and adds a correction toward the
no-reaction line:

$$C_i(f) = B_i(f) + \bigl[M_i(f) - B_i(f)\bigr]\,\lambda_i$$

$\lambda_i$ is chosen so that the Beta-weighted mean matches the ODE value $y_i$:

$$\lambda_i = \frac{y_i - \mathbb{E}[B_i]}{\mathbb{E}[M_i] - \mathbb{E}[B_i]}, \qquad \lambda_i \in [0, 1]$$

$\lambda_i = 0$ gives the pure limit profile $B_i$; $\lambda_i = 1$ gives the pure no-reaction line $M_i$.

For unfed product species $M_i(f) = 0$, so $\mathbb{E}[M_i] = 0$ and:

$$\lambda_i = 1 - \frac{y_i}{\mathbb{E}[B_i]}$$

| Species | $y_i$ (mol/L) | $\mathbb{E}[B_i]$ (mol/L) | $\lambda_i$ (raw) | $\lambda_i$ (clamped) |
|---------|--------------|--------------------------|-------------------|-----------------------|
| R | 0.2290 | 0.1070 | −1.14 | **0.0** |
| S | 0.0187 | 0.01682 | −0.11 | **0.0** |

Both species are clamped to $\lambda = 0$: the ODE concentrations exceed the Beta-weighted
limit profile mean, so the closure uses the pure limit profile $C_i(f) = B_i(f)$.

---

## 8. Rate integrals

The turbulent reaction rate for reaction $j$ is

$$r_j = \int_0^1 k_j \prod_i C_i(f)^{n_{ij}}\; \text{Beta}(f;\alpha,\beta)\, df$$

Because each $C_i(f)$ is piecewise linear in $f$, the integrand on each segment
$[0, f_{sb}]$ and $[f_{sb}, 1]$ is a polynomial in $f$.  The integral over each segment
is evaluated analytically using the **regularised incomplete Beta function**
$I_x(a,b) = B(x;a,b)/B(a,b)$.

Rate constants (L mol$^{-1}$ s$^{-1}$): $k_1 = 12238$, $k_2 = 1.835$, $k_3 = 921$,
$k_4 = 22.25$, $k_5 = 124.5$.
