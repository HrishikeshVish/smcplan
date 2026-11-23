"""Appendix ablation and sensitivity tables of our planner (2026-10-07).

  table_abl_mstep.tex   M-step (score, pathwise) x cost (L2, PDE), every cell
                        and Cube (arms em, pde, l2fix, ours)
  table_abl_budget.tex  success of `ours` within K iterations, every cell; exact
                        because no setting of the planner depends on K
                        (writeup/run_ablations.sh header), so the run with
                        K=150 is the run with budget K up to its K-th iteration
  table_abl_smc.tex     particle count N, resampling threshold tau, and
                        uniform weights (no weighting or resampling)
  table_abl_sens.tex    alpha, eta, sigma_p, roadmap k and c, lambda_V scale,
                        and the operator that orders the roadmap vertices
                        (heat step vs hop count, Green's function, shortest
                        path; planning/gd.py MCMollifiedHeat._solve_operator)
Each ablation row: success with 95% Wilson interval, mean iterations of the
successful seeds, and the exact McNemar p against `ours` (the main setting) on
the paired seeds. Arms that have not finished all seeds print "--".

Usage: python make_ablation_tables.py <paper_dir>
"""
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

from equal_particle_stats import ENVS, HS, SCORE_ENVS, mcnemar, outcomes, paired, wilson

PAPER = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
CELLS = [("Point maze", "pointmaze"), ("Reacher", "reacher")]
N_SEEDS = 100
SMC = [("Particles $N$", [("2", "N2"), ("4", "N4"), ("8", "N8"),
                          ("12", None), ("16", "N16"), ("32", "N32")]),
       ("Resampling threshold $\\tau$", [("0 (never)", "tau0"), ("0.25", "tau025"),
                                          ("0.5", None), ("0.75", "tau075"),
                                          ("1 (always)", "tau1")]),
       ("Weights", [("SMC weights", None), ("uniform", "uniformw")])]
SENS = [("Temperature $\\alpha/\\operatorname{std}C$", [("0.25", "alpha025"), ("0.5", "alpha05"),
                                                        ("1", None), ("2", "alpha2"), ("4", "alpha4")]),
        ("Step size $\\eta$", [("0.09", "eta009"), ("0.18", None), ("0.36", "eta036")]),
        ("Reference width $\\sigma_p$", [("0.5", "sigp05"), ("1", None), ("2", "sigp2")]),
        ("Roadmap $k$", [("$k/2$", "khalf"), ("$k$", None), ("$2k$", "kdouble")]),
        ("Diffusion constant $c$", [("$c/3$", "cthird"), ("$c$", None), ("$3c$", "ctriple")]),
        ("PDE weight $\\lambda_V$", [("$\\times0.5$", "lamv05"), ("$\\times1$", None),
                                     ("$\\times2$", "lamv2")]),
        ("Operator on the roadmap", [("heat step", None), ("number of edges", "ophop"),
                                     ("Green's function", "opgreen"),
                                     ("shortest path", "opsp")])]


def rnd(x, d):
    """x to d decimals, halves rounded up (an exact p of 0.125 prints as 0.13)."""
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def cell(arm, env):
    """Formatted (success [CI], iterations, p vs ours) for one arm at H=10."""
    # the ablation arms ran from pdefix (pathwise M-step), so it is their
    # reference whatever OURS_PM selects for `ours`
    o = outcomes(f"abl_{arm}" if arm else "pdefix", env, 10)
    if len(o) < N_SEEDS:
        return ["--", "--", "--"]
    k = sum(ok for ok, _ in o.values())
    lo, hi = wilson(k, len(o))
    its = [it for ok, it in o.values() if ok and it is not None]
    mi = f"{sum(its) / len(its):.1f}" if its else "--"
    if arm is None:
        p = "--"
    else:
        b, c, pv = paired(o, outcomes("pdefix", env, 10))
        p = "$<$0.001" if pv < 1e-3 else rnd(pv, 3) if pv < 0.1 else rnd(pv, 2)
    return [f"{100 * k / len(o):.0f} [{lo:.0f}, {hi:.0f}]", mi, p]


def ablation_table(groups, label, caption):
    rows = []
    for gname, settings in groups:
        first = True
        for sname, arm in settings:
            vals = []
            for _, env in CELLS:
                vals += cell(arm, env)
            g = f"\\multirow{{{len(settings)}}}{{*}}{{{gname}}}" if first else ""
            mark = "$^{\\ast}$" if arm is None else ""
            rows.append(f"    {g} & {sname}{mark} & " + " & ".join(vals) + r" \\")
            first = False
        rows.append(r"    \midrule")
    rows[-1] = r"    \bottomrule"
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n} $H{{=}}10$}}" for n, _ in CELLS)
    return "\n".join([
        r"\begin{table}[!htbp]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        r"  \centering", r"  \scriptsize", r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{llcccccc}", r"    \toprule",
        f"    & & {head} \\\\",
        r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"    & Setting & Success (\%) & Iter. & $p$ & Success (\%) & Iter. & $p$ \\",
        r"    \midrule", *rows, r"  \end{tabular}", r"\end{table}", ""])


def budget_table():
    Ks = (10, 25, 50, 100, 150)
    rows = []
    for name, e in ENVS + [("Cube", "cube")]:
        for H in (HS if e != "cube" else (5,)):
            o = outcomes("ours", e, H)
            n = len(o)
            v = [100 * sum(1 for ok, it in o.values() if ok and it is not None and it <= K) / n
                 for K in Ks]
            rows.append(f"    {name} & ${H}$ & " + " & ".join(f"${x:.0f}$" for x in v) + r" \\")
    return "\n".join([
        r"\begin{table}[!htbp]",
        r"  \caption{Success rate (\%) of our planner within a budget of $K$ iterations. No setting of the planner depends on $K$, so the runs of \Cref{tab:planners} with $K=150$, which stop at the first success, give the success rate at every smaller budget; the column $K=150$ repeats \Cref{tab:planners} and, for Cube, \Cref{sec:exp:planning}.}",
        r"  \label{tab:abl-budget}", r"  \centering", r"  \small",
        r"  \begin{tabular}{llrrrrr}", r"    \toprule",
        r"    Environment & $H$ & " + " & ".join(f"$K={K}$" for K in Ks) + r" \\",
        r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])


def mstep_cost_table():
    """M-step (score, pathwise) x cost (L2, PDE), every cell of tab:planners and
    Cube; all four arms are Alg. 1 with the settings of tab:hyper otherwise.
    An arm with fewer than its full number of seeds (a run in progress) prints
    \red{[PENDING]}, as do the tests that use it, and is named in the caption;
    an arm that was not run (score form with the PDE cost on Cube) prints "--"."""
    arms = [("em", "Score", "$L_2$"), ("pde", "Score", "PDE"),
            ("l2fix", "Pathwise", "$L_2$"), ("pdefix", "Pathwise", "PDE")]
    tests = [("pde", "em"), ("pdefix", "l2fix"), ("l2fix", "em"), ("pdefix", "pde")]
    cells = [(n, e, H) for n, e in ENVS for H in HS] + [("Cube", "cube", 5)]
    rows, pending = [], []
    PEND = r"\red{[PENDING]}"
    for n, e, H in cells:
        P = set()
        full = 50 if e == "cube" else N_SEEDS
        O = {}
        for a, _, _ in arms:
            try:
                o = outcomes(a, e, H)
            except KeyError:                 # no score-M-step PDE run on Cube
                o = None
            if o is not None and 0 < len(o) < full:
                pending.append((n, H, a))
                P.add(a)
            O[a] = o if o is not None and len(o) == full else None
        t1 = "pde" if e in SCORE_ENVS else "pdefix"     # the cell of tab:planners
        vals = [(f"{100 * sum(ok for ok, _ in O[a].values()) / len(O[a]):.0f}"
                 + ("$^{\\ast}$" if a == t1 else ""))
                if O[a] is not None else (PEND if a in P else "--") for a, _, _ in arms]
        for x, y in tests:
            if O[x] is None or O[y] is None:
                vals.append(PEND if (x in P or y in P) else "--")
                continue
            _, _, pv = paired(O[x], O[y])
            vals.append("$<$0.001" if pv < 1e-3 else rnd(pv, 3) if pv < 0.1 else rnd(pv, 2))
        rows.append(f"    {n} & ${H}$ & " + " & ".join(vals) + r" \\")
    pend = "" if not pending else (
        r" \red{[PENDING: " + "; ".join(f"{n} $H{{=}}{H}$, {dict(em='score M-step, $L_2$ cost', pde='score M-step, PDE cost', l2fix='pathwise M-step, $L_2$ cost', pdefix='pathwise M-step, PDE cost')[a]}"
                                         for n, H, a in pending) + r"]}")
    caption = (
        r"Success rate (\%) of \Cref{alg:smc} with the score form of \Cref{eq:mstep} and the pathwise form of \Cref{eq:mstep-path} of the M-step, each with the $L_2$ cost ($\lambda_V=0$) and with the PDE cost of \Cref{eq:cost}, all other settings as in \Cref{tab:hyper}, on the $100$ seeds per cell of \Cref{tab:planners} and the $50$ seeds of Cube; the success rate marked $^{\ast}$ in each row repeats \Cref{tab:planners} and, for Cube, \Cref{sec:exp:planning}" + (" (the score form on " + " and ".join(n for n, e in ENVS if e in SCORE_ENVS) + ", the pathwise form elsewhere)" if SCORE_ENVS else "") + r", and Cube has no run with the score form and the PDE cost. The last four columns give the exact McNemar $p$-value on the paired seeds (\Cref{app:ci}) for the PDE cost against the $L_2$ cost under each form of the M-step, and for the pathwise form against the score form under each cost."
        + pend)
    return "\n".join([
        r"\begin{table}[!htbp]",
        f"  \\caption{{{caption}}}",
        r"  \label{tab:abl-mstep}", r"  \centering", r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{llcccccccc}", r"    \toprule",
        r"    & & \multicolumn{2}{c}{Score M-step} & \multicolumn{2}{c}{Pathwise M-step} & \multicolumn{2}{c}{$p$, PDE vs.\ $L_2$} & \multicolumn{2}{c}{$p$, pathwise vs.\ score} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"    Environment & $H$ & $L_2$ & PDE & $L_2$ & PDE & score & pathwise & $L_2$ & PDE \\",
        r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])


# The ablation arms ran from pdefix: on an environment where tab:planners uses
# the score form, they ablate the pathwise form instead, and the captions say so.
PM_NOTE = "".join(
    rf" On {n}, every row, the one marked $^{{\ast}}$ included, uses the pathwise form of \Cref{{eq:mstep-path}} of the M-step in place of the score form of \Cref{{eq:mstep}} of \Cref{{tab:hyper}}, so that its row marked $^{{\ast}}$ is not the {n} $H{{=}}10$ cell of \Cref{{tab:planners}}."
    for n, e in CELLS if e in SCORE_ENVS)


if __name__ == "__main__":
    open(f"{PAPER}/table_abl_mstep.tex", "w").write(mstep_cost_table())
    open(f"{PAPER}/table_abl_budget.tex", "w").write(budget_table())
    open(f"{PAPER}/table_abl_smc.tex", "w").write(ablation_table(
        SMC, "tab:abl-smc",
        r"Ablations of the population of our planner at $H{=}10$, one setting changed at a time from those of \Cref{tab:hyper} (marked $^{\ast}$), on the $100$ seeds of \Cref{tab:planners}. Success with $95\%$ Wilson interval, mean iterations to success of the successful seeds, and the exact McNemar $p$-value against the setting marked $^{\ast}$ on the paired seeds (\Cref{app:ci}). With uniform weights, the weights are reset to be equal before the resampling test and before the M-step: no resampling, and the M-step averages over the particles with equal weights; $\beta_k$ is still chosen from the change-of-target factor (\Cref{app:abl-smc})." + PM_NOTE))
    open(f"{PAPER}/table_abl_sens.tex", "w").write(ablation_table(
        SENS, "tab:abl-sens",
        r"Sensitivity of our planner at $H{=}10$ to the settings below, one changed at a time from \Cref{tab:hyper} (marked $^{\ast}$; $k$ and $c$ are the per-environment values of \Cref{tab:hyper}), on the $100$ seeds of \Cref{tab:planners}. Columns as in \Cref{tab:abl-smc}. The operator rows order the roadmap vertices, in place of $u$ of \Cref{eq:heat}, by the number of edges to the goal, the limit of $\log u/\log t$ as $t\to0$ \citep[Lemma~1]{crane2012geodesicsheat}; by the Green's function $L^{\dagger}e_g$, whose ordering $u$ takes as $t\to\infty$ (\Cref{app:operator}); or by the shortest-path distance to the goal with edge lengths $\lVert\xi_i-\xi_j\rVert$. The roadmap, the rank transform of \Cref{eq:rank} and the surrogate $\hat V$ are unchanged. As the setting marked $^{\ast}$ is close to $100\%$ success in both cells, these rows can show a loss of success but hardly a gain; \Cref{tab:calib-operators} compares the operators on the offline criterion used to choose $k$ and $c$." + PM_NOTE))
    print("wrote table_abl_mstep.tex, table_abl_budget.tex, table_abl_smc.tex, table_abl_sens.tex")
