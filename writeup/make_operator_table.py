"""Appendix table tab:calib-operators: the operators that order the roadmap
vertices, compared offline on the criterion used to choose k and c
(writeup/calib_operators.py): mean Spearman correlation between V and the
observed number of steps to the goal over 60 goal vertices, and the paired
difference heat - operator. By default the 60 goals are disjoint from those on
which k and c were chosen (--heldout runs); --calibration reads the runs on the
calibration goals instead.

Usage: python make_operator_table.py [--calibration] <paper_dir>
"""
import csv
import sys

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PAPER = ARGS[0] if ARGS else "/home/hviswan/Documents/grasp-paper"
HELDOUT = "--calibration" not in sys.argv
ROOT = "/home/hviswan/Documents/grasp/sweep_emsmc"
ENVS = [("Reacher", "reacher"), ("Point maze", "pointmaze"), ("Wall", "wall"),
        ("Push-T", "pusht"), ("Cube", "cube")]
OPS = [("heat", "Heat step$^{\\ast}$"), ("hop", "Number of edges"),
       ("green", "Green's function"), ("shortest_path", "Shortest path"),
       ("euclidean", "Euclidean distance")]   # p4/p8 of calib_operators.py not reported


def load(env):
    f = f"{ROOT}/calib_operators{'_heldout' * HELDOUT}_{env}.csv"
    return {r["operator"]: r for r in csv.DictReader(open(f))}


def num(x, s, sign=False):
    return f"${float(x):{'+' if sign else ''}.3f}\\pm{float(s):.3f}$"


def table():
    R = {e: load(e) for _, e in ENVS}
    head = " & ".join(n for n, _ in ENVS)
    rows = [r"    \multicolumn{6}{l}{\emph{Rank correlation with the number of steps to the goal}} \\"]
    for op, name in OPS:
        rows.append(f"    {name} & " + " & ".join(num(R[e][op]["rho"], R[e][op]["se"])
                                               for _, e in ENVS) + r" \\")
    rows += [r"    \midrule",
             r"    \multicolumn{6}{l}{\emph{Heat step minus operator, paired over goals}} \\"]
    for op, name in OPS[1:]:
        rows.append(f"    {name} & " + " & ".join(
            num(R[e][op]["diff_heat_minus_op"], R[e][op]["diff_se"], sign=True)
            for _, e in ENVS) + r" \\")
    goals = ("$60$ goal vertices, disjoint from the $60$ on which $k$ and $c$ were chosen"
             if HELDOUT else "the $60$ goal vertices on which $k$ and $c$ were chosen")
    # a goal where an ordering is constant over the earlier vertices has no
    # Spearman correlation and is left out of that ordering's mean and difference
    drop = [(name, en) for op, name in OPS for en, e in ENVS if int(R[e][op]["n_goals"]) < 60]
    names = sorted({nm for nm, _ in drop}, key=[n for _, n in OPS].index)
    dropped = "" if not drop else (
        " A goal at which an ordering is constant over the earlier vertices of its episode has no rank correlation and is left out of that ordering's mean and difference: "
        + "; ".join(f"{nm[0].lower() + nm[1:]} on " + " and ".join(en for n2, en in drop if n2 == nm)
                    for nm in names) + ".")
    caption = (
        r"Orderings of the roadmap vertices compared offline, on the criterion by which we choose $k$ and $c$ (\Cref{app:impl}): the Spearman correlation between the cost-to-go and the observed number of steps to the goal over the earlier vertices of its episode, averaged over "
        + goals + r" (mean $\pm$ standard error over goals). Each environment uses the roadmap of \Cref{tab:hyper}. The heat step$^{\ast}$ is \Cref{eq:heat} at the diffusion constant $c$ of \Cref{tab:hyper}; the number of edges is the limit of $\log u/\log t$ as $t\to0$, with ties left unbroken, and the Green's function $L^{\dagger}e_g$ is the ordering of $u$ as $t\to\infty$; the shortest path is the Dijkstra distance with edge lengths $\lVert\xi_i-\xi_j\rVert$; the Euclidean distance $\lVert\xi_i-\xi_g\rVert$ uses no edge. The lower block gives the difference to the heat step on the same goals, positive where the heat step agrees better with the observed number of steps."
        + dropped)
    return "\n".join([
        r"\begin{table}[!htbp]",
        f"  \\caption{{{caption}}}",
        r"  \label{tab:calib-operators}", r"  \centering", r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3.5pt}",
        r"  \begin{tabular}{lccccc}", r"    \toprule",
        f"    Ordering & {head} \\\\", r"    \midrule", *rows,
        r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])


if __name__ == "__main__":
    open(f"{PAPER}/table_calib_operators.tex", "w").write(table())
    print("wrote table_calib_operators.tex"
          + (" (held-out goals)" if HELDOUT else " (calibration goals)"))
