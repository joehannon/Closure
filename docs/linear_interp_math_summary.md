# `linear_interp`: nearest-bracket profile interpolation

**System:** `input_fuller_BC_azo_coupling_kinetics` — species A, B, C, R, T, S, Q;
reactions R1–R5.

---

## 1. Overview

`linear_interp` constructs a species concentration profile $C_w(f)$ as a convex
combination of two pre-computed piecewise-linear profiles whose Beta-weighted means
bracket the current ODE value $y_i$.  Unlike `blend_fs`, it does not blend subsets or
compute a dynamic kink position: it selects directly from the library of enumerated
reaction-subset limit profiles.

---

## 2. Profile library

For each active species $i$, the solver holds a library of $N$ piecewise-linear
profiles

$$C_n(f), \quad n = 1, \ldots, N$$

Each profile $C_n$ is derived from one reaction subset: it is the species concentration
across mixture fraction $f$ when only that subset's reactions run to their stoichiometric
limit.  The no-reaction (pure mixing) line is included as subset 0:

$$C_0(f) = M_i(f) = Y_{2i} + (Y_{1i} - Y_{2i})\,f$$

---

## 3. Beta-weighted profile averages

At each ODE time step the current Beta distribution parameters $(\alpha, \beta)$ are
computed from the variance $\sigma^2(t)$:

$$s = \frac{\bar{f}(1-\bar{f})}{\sigma^2} - 1, \qquad \alpha = \bar{f}\,s, \qquad \beta = (1-\bar{f})\,s$$

The Beta-weighted mean of each profile is

$$\mathbb{E}[C_n] = \int_0^1 C_n(f)\,\mathrm{Beta}(f;\alpha,\beta)\,df$$

evaluated analytically segment by segment using the regularised incomplete Beta function.

---

## 4. Bracketing pair selection

The $N$ profiles are sorted by their means:

$$\mathbb{E}[C_{\pi(1)}] \leq \mathbb{E}[C_{\pi(2)}] \leq \cdots \leq \mathbb{E}[C_{\pi(N)}]$$

The ODE value $y_i$ is first clamped to the achievable range
$[\mathbb{E}[C_{\pi(1)}],\, \mathbb{E}[C_{\pi(N)}]]$ if it falls outside (this fires a
diagnostic message).  The adjacent pair $(\mathrm{lo}, \mathrm{hi})$ is then the unique
bracket satisfying

$$\mathbb{E}[C_\mathrm{lo}] \leq y_i \leq \mathbb{E}[C_\mathrm{hi}]$$

where $\mathrm{lo} = \pi(k)$ and $\mathrm{hi} = \pi(k+1)$ for the smallest $k$ such
that $\mathbb{E}[C_{\pi(k+1)}] \geq y_i$.

---

## 5. Interpolation weights

The weight on the lower profile (analogous to $\lambda$ in the blend convention, where
0 = pure lower profile and 1 = pure upper) is

$$w_\mathrm{lo} = \frac{\mathbb{E}[C_\mathrm{hi}] - y_i}{\mathbb{E}[C_\mathrm{hi}] - \mathbb{E}[C_\mathrm{lo}]}$$

with $w_\mathrm{hi} = 1 - w_\mathrm{lo}$.  All other profiles receive weight zero.

The weights satisfy $\sum_n w_n = 1$ and $\sum_n w_n\,\mathbb{E}[C_n] = y_i$ by
construction, so the Beta-weighted mean of the blended profile matches the ODE value
exactly (within the clamped range).

---

## 6. Blended profile

The species profile used in the rate integral is

$$C_w(f) = w_\mathrm{lo}\,C_\mathrm{lo}(f) + w_\mathrm{hi}\,C_\mathrm{hi}(f)$$

Because both $C_\mathrm{lo}$ and $C_\mathrm{hi}$ are piecewise linear in $f$, $C_w$ is
also piecewise linear and the turbulent reaction rate

$$r_j = \int_0^1 k_j \prod_i C_{w,i}(f)^{n_{ij}}\,\mathrm{Beta}(f;\alpha,\beta)\,df$$

is evaluated analytically using the same regularised incomplete Beta machinery as
`blend_fs`.

---

## 7. Continuity

$C_w(f)$ is continuous in $y_i$ but only **$C^0$**: the bracketing pair
$(\mathrm{lo}, \mathrm{hi})$ jumps each time $y_i$ crosses a profile mean
$\mathbb{E}[C_n]$, producing a kink in the weight trajectory.  Within any one bracket
the profile shape varies smoothly with $y_i$.

---

## 8. Comparison with `blend_fs`

| Property | `linear_interp` | `blend_fs` |
|---|---|---|
| Profile basis | Enumerated subset limits | Dynamically blended limit |
| Number of active profiles | 2 (bracketing pair) | 1 (blended $B(f)$) + no-reaction line |
| Primary weight | $w_\mathrm{lo}$ (weight on lower profile) | $\lambda$ (weight on no-reaction line $M$) |
| Mean-matching | $\sum w_n \mathbb{E}[C_n] = y_i$ | $\mathbb{E}[B] + (\mathbb{E}[M]-\mathbb{E}[B])\lambda = y_i$ |
| Clamping | Clamps $y_i$ to profile range | Clamps $\lambda$ to $[0,1]$ |
| Kink in weight trajectory | At each profile-mean crossing | Never (single bracket) |
| Subset tracking | Implicit via bracketing pair | Explicit via $w_k$ blend weights |

Rate constants (L mol$^{-1}$ s$^{-1}$): $k_1 = 12238$, $k_2 = 1.835$, $k_3 = 921$,
$k_4 = 22.25$, $k_5 = 124.5$.
