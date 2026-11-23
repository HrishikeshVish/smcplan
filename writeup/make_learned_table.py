"""Appendix table tab:learned (app:learned-planners), 2026-10-11: planners that
train additional networks on the offline dataset, against our planner, on the
LeWM environments: Reacher H=5,10,15 (100 seeds) and Cube H=5 (50 seeds).

  leflow12  LeFlow, 12 sampled paths (the particle count of tab:planners)
  leflow64  LeFlow, 64 sampled paths (its default)
  gcidm_ol  GC-IDM chained through the frozen LeWM, executed open-loop (our adaptation)
The closed-loop GC-IDM run (gcidm_cl) is not reported (user, 2026-10-11).
Runs: writeup/baselines/run_learned.sh. Each cell: success with 95% Wilson
interval, and the exact McNemar p of our planner against the row on the paired
seeds. Every number comes from equal_particle_stats.

Usage: python make_learned_table.py <paper_dir>
"""
import os
import sys

from equal_particle_stats import outcomes, paired, wilson
from make_ablation_tables import rnd

PAPER = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
CELLS = [("reacher", 5, 100), ("reacher", 10, 100), ("reacher", 15, 100), ("cube", 5, 50)]
ROWS = [("ours", r"Ours$^{\ast}$"), ("leflow12", "LeFlow, $12$ paths"),
        ("leflow64", "LeFlow, $64$ paths"), ("gcidm_ol", "GC-IDM")]


def fmt_p(pv):
    return "$<$0.001" if pv < 1e-3 else rnd(pv, 3) if pv < 0.1 else rnd(pv, 2)


def table():
    rows = []
    for arm, name in ROWS:
        vals = []
        for e, H, n in CELLS:
            o = outcomes(arm, e, H)
            if len(o) < n:
                vals += ["--", "--"]
                continue
            k = sum(ok for ok, _ in o.values())
            lo, hi = wilson(k, n)
            vals.append(f"{100 * k / n:.0f} [{lo:.0f}, {hi:.0f}]")
            vals.append("--" if arm == "ours" else fmt_p(paired(outcomes("ours", e, H), o)[2]))
        rows.append(f"    {name} & " + " & ".join(vals) + r" \\")
        if arm == "ours":
            rows.append(r"    \midrule")
    caption = (
        r"Planners that train additional networks on the offline dataset (\Cref{app:learned-planners}), on the LeWM environments, against our planner$^{\ast}$ with the settings of \Cref{tab:hyper} ($12$ particles, $150$ iterations): Reacher on the $100$ seeds per horizon of \Cref{tab:planners} and Cube on the $50$ seeds of \Cref{sec:exp:planning}. Success rate (\%) with $95\%$ Wilson interval, and the exact McNemar $p$-value of our planner against each row on the paired seeds (\Cref{app:ci}). Every row executes plans of $H$ world-model actions open-loop from the start state, our planner after each of its iterations (\Cref{app:protocol}) and LeFlow and GC-IDM once (\Cref{app:learned-planners}).")
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{'Reacher' if e == 'reacher' else 'Cube'} $H{{=}}{H}$}}"
                      for e, H, _ in CELLS)
    return "\n".join([
        r"\begin{table}[t]",
        f"  \\caption{{{caption}}}",
        r"  \label{tab:learned}", r"  \centering", r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \begin{tabular}{lcccccccc}", r"    \toprule",
        f"    & {head} \\\\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"    Planner & " + " & ".join(["Success (\\%)", "$p$"] * len(CELLS)) + r" \\",
        r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])


if __name__ == "__main__":
    open(f"{PAPER}/table_learned.tex", "w").write(table())
    print("wrote table_learned.tex")
