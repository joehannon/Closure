"""Render the ray_limit selectivity / blended-limit summary to a PDF.

No LaTeX engine is available, so this lays the document out with matplotlib's
mathtext on letter-size pages via PdfPages.  Run:  python3 make_ray_limit_pdf.py
"""
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PAGE_W, PAGE_H = 8.5, 11.0          # inches (letter)
L_MARGIN, R_MARGIN = 0.95, 0.95
TOP, BOTTOM = 10.4, 0.85            # inches from page bottom
WRAP = 96                           # body wrap width (chars)

# Line advances, in inches.
ADV = {'title': 0.40, 'subtitle': 0.34, 'h': 0.40, 'body': 0.225,
       'eq': 0.46, 'gap': 0.16, 'small': 0.20}
SIZE = {'title': 17, 'subtitle': 10.5, 'h': 13, 'body': 10.5, 'eq': 13,
        'small': 9}


def main():
    blocks = build_blocks()
    pdf = PdfPages('ray_limit_math_summary_mpl.pdf')
    fig = _new_page()
    y = TOP
    for kind, text in blocks:
        lines = _expand(kind, text)
        for ln_kind, ln_text in lines:
            adv = ADV[ln_kind if ln_kind in ADV else 'body']
            if y - adv < BOTTOM:
                pdf.savefig(fig); plt.close(fig)
                fig = _new_page(); y = TOP
            _draw(fig, ln_kind, ln_text, y)
            y -= adv
    pdf.savefig(fig); plt.close(fig)
    pdf.close()
    print('wrote ray_limit_math_summary_mpl.pdf')


def _new_page():
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    return fig


def _x(frac_kind):
    return L_MARGIN / PAGE_W


def _draw(fig, kind, text, y_in):
    yf = y_in / PAGE_H
    if kind == 'eq':
        fig.text(0.5, yf, text, fontsize=SIZE['eq'], ha='center', va='top')
    elif kind == 'title':
        fig.text(0.5, yf, text, fontsize=SIZE['title'], ha='center', va='top',
                 fontweight='bold')
    elif kind == 'subtitle':
        fig.text(0.5, yf, text, fontsize=SIZE['subtitle'], ha='center', va='top',
                 color='0.3')
    elif kind == 'h':
        fig.text(_x('h'), yf, text, fontsize=SIZE['h'], ha='left', va='top',
                 fontweight='bold')
    elif kind == 'small':
        fig.text(_x('body'), yf, text, fontsize=SIZE['small'], ha='left', va='top',
                 color='0.35')
    else:
        fig.text(_x('body'), yf, text, fontsize=SIZE['body'], ha='left', va='top')


def _expand(kind, text):
    """Turn a logical block into a list of (line_kind, line_text)."""
    if kind == 'body':
        out = []
        for para in text.split('\n'):
            wrapped = textwrap.wrap(para, WRAP) or ['']
            out += [('body', w) for w in wrapped]
        return out
    if kind == 'gap':
        return [('gap', '')]
    return [(kind, text)]


def build_blocks():
    B = []
    B.append(('title', 'ray_limit: selectivity and the blended limit  C(f)'))
    B.append(('subtitle',
              'Mixing-limited reaction closure  -  species_limits.py, '
              'integrate_species_odes'))
    B.append(('gap', ''))

    # ---- Setup ----
    B.append(('h', 'Setup'))
    B.append(('body',
        'Restrict to the active species i in A and reactions j = 1..R.  Define:'))
    B.append(('eq', r'$N \in \mathbb{R}^{|A|\times R}:\ '
                    r'N_{ij}=\nu^{\mathrm{prod}}_{ij}-\nu^{\mathrm{reac}}_{ij}$'
                    r'$\qquad$ (net stoichiometry, nu_net_active)'))
    B.append(('eq', r'$Y^{(1)},\,Y^{(2)}:\ \mathrm{the}\ f=1\ \mathrm{and}\ '
                    r'f=0\ \mathrm{feeds}\ \ (\mathrm{bd\_Y1a,\ bd\_Y2a})$'))
    B.append(('body', 'Mixing line (the no-reaction limit) at mixture fraction f:'))
    B.append(('eq', r'$M_i(f)=f\,Y^{(1)}_i+(1-f)\,Y^{(2)}_i,'
                    r'\qquad f\in[0,1]$'))
    B.append(('body',
        'Let y be the current conditional-mean state (the ODE variable), y0 its '
        'initial value, and angle-brackets the average against the step\'s '
        'Beta(alpha_t, beta_t) pdf of f.'))
    B.append(('gap', ''))

    # ---- Section 1 ----
    B.append(('h', '1.  Selectivity = a single net direction d'))
    B.append(('body',
        'ray_limit collapses all reactions into ONE net-reaction ray d, chosen '
        '(in priority order, _raylimit_limit) as the accumulated-extent '
        'direction, with a mass-action fallback at startup:'))
    B.append(('eq', r'$d=\Delta:=y-y^0\quad\mathrm{if}\ '
                    r'\sum_i|\Delta_i|>10^{-6},\qquad '
                    r'\mathrm{else}\quad d=N\,r(y)$'))
    B.append(('eq', r'$r_j(y)=k_j\prod_i \max(y_i,0)^{\nu^{\mathrm{reac}}_{ij}}$'))
    B.append(('body',
        'The first is the accumulated-extent direction.  Since the ODE is '
        'y-dot = N r,'))
    B.append(('eq', r'$\Delta=N\!\int_0^t r\,d\tau=N\,\xi_{\mathrm{acc}},'
                    r'\qquad \xi_{\mathrm{acc}}\geq 0,$'))
    B.append(('body',
        'so Delta lies in range(N) - a valid net-stoichiometric direction.  Only '
        'the DIRECTION of d is used (a ray); its magnitude is discarded.  That '
        'ray is the selectivity: the pathway split among R1/R2/R3.'))
    B.append(('gap', ''))

    # ---- Section 2 ----
    B.append(('h', '2.  The blended limit B(f) = react M(f) to completion along d'))
    B.append(('body',
        'Let the consumed set be C = { i : d_i < -1e-12 } (require |C| >= 2).  '
        'Reacting the mixing line along the fixed direction d with extent c >= 0 '
        'keeps M(f) + c d >= 0 iff c <= M_i(f)/(-d_i) for every i in C.  The '
        'completion extent is therefore'))
    B.append(('eq', r'$c_{\max}(f)=\min_{i\in\mathcal{C}}\frac{M_i(f)}{-d_i}'
                    r'=\min_{i\in\mathcal{C}}\,(p_i+q_i f),$'))
    B.append(('eq', r'$p_i=\frac{Y^{(2)}_i}{-d_i},\qquad '
                    r'q_i=\frac{Y^{(1)}_i-Y^{(2)}_i}{-d_i},$'))
    B.append(('body',
        'a concave, piecewise-linear function of f (a min of affine pieces).'))
    B.append(('small', 'Kink  f_s*  (_complete_along): the peak of that lower envelope, '
                       'where two consumed species vanish together.'))
    B.append(('body',
        'Over pairs (a,b) in C with q_a != q_b, set f_ab = (p_b - p_a)/(q_a - q_b) '
        'and take'))
    B.append(('eq', r'$f_s^\star=\mathrm{argmax}_{\,f_{ab}\in(0,1)}\ '
                    r'\min_{i\in\mathcal{C}}\,(p_i+q_i f_{ab}),\qquad '
                    r'c^\star=c_{\max}(f_s^\star)>0.$'))
    B.append(('body',
        'Blended complete-reaction (infinite-Da) limit.  With pure-feed endpoints '
        '(the cross-stream reactant is absent at f = 0, 1, so no completion there):'))
    B.append(('eq', r'$B_i(0)=Y^{(2)}_i,\quad B_i(1)=Y^{(1)}_i,\quad '
                    r'B_i(f_s^\star)=M_i(f_s^\star)+c^\star d_i=:v^{\mathrm{fs}}_i,$'))
    B.append(('body',
        'and B_i is the two-segment (single-kink) linear interpolant of the points '
        '(0, Y2_i), (f_s*, v_i^fs), (1, Y1_i):'))
    B.append(('eq', r'$B_i(f)=Y^{(2)}_i+\frac{v^{\mathrm{fs}}_i-Y^{(2)}_i}'
                    r'{f_s^\star}\,f,\qquad 0\leq f\leq f_s^\star,$'))
    B.append(('eq', r'$B_i(f)=v^{\mathrm{fs}}_i+\frac{Y^{(1)}_i-v^{\mathrm{fs}}_i}'
                    r'{1-f_s^\star}\,(f-f_s^\star),\qquad f_s^\star\leq f\leq 1.$'))
    B.append(('body',
        'Stoichiometric exactness.  On each segment B(f) - M(f) = c_max(f) d = '
        'N ( c_max(f) xi ) with c_max(f) xi >= 0 - a valid nonnegative extent '
        'vector, so B creates/destroys no net moles.  It is the mixing line driven '
        'to the stoichiometric boundary along the selectivity ray.'))
    B.append(('body',
        'This B(f) IS the blended profile produced before any interpolation.  '
        '"Blend" here = one direction d fusing all reactions into a single '
        'completed limit, not a convex combination of enumerated subset limits.'))
    B.append(('gap', ''))

    # ---- Section 3 ----
    B.append(('h', '3.  (Next step) interpolation with the no-reaction limit'))
    B.append(('body',
        'Only afterward does ray_limit form the reported profile by mean-matching '
        'against the no-reaction limit M:'))
    B.append(('eq', r'$C_i(f)=(1-\lambda_i)\,M_i(f)+\lambda_i\,B_i(f),$'))
    B.append(('eq', r'$\lambda_i=\mathrm{clip}_{[0,1]}'
                    r'\frac{y_i-\langle M_i\rangle}'
                    r'{\langle B_i\rangle-\langle M_i\rangle},$'))
    B.append(('body',
        'so that <C_i> = y_i when lambda_i is interior.  Steps 1-2 are the '
        'selectivity-driven construction of B; step 3 is the finite-Da scaling.'))
    return B


if __name__ == '__main__':
    main()
