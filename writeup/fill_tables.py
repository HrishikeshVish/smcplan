"""Harvest the sweeps and emit the LaTeX results table + a console summary.

Reads two roots, because the baselines and the FK arm were run by different
scripts at different times:
  sweep/<env>/<planner>_H<h>/seed_<s>/result.txt        <- run_sweep.sh
  sweep_fk/table/<env>/fk_H<h>/seed_<s>/result.txt      <- run_fk_sweep.sh
Both hold "<success 0|1> <wallclock seconds>". The success flag is parsed
uniformly from run.log ("Success rate:  1.0" ever reported), because
planning/grasp.py never writes seed_success_log.txt while the other planners
do -- keying off that file would silently drop every GRASP run.

Cells that have no data are NOT left blank: an arm that is queued or running
prints \\queued, one that was never scheduled prints \\na. Keeping those
distinct is the point -- a blank cell in a draft reads as "the method failed
here", which is a different claim from "this has not run yet".

sweep/_pusht_random_state_invalid/ is deliberately NOT read. Those runs used
goal_source=random_state, whose start/goal pairs on Push-T are routinely
unconnectable and floored every planner at ~0/100; the valid Push-T protocol
is goal_source=dset (see run_sweep.sh).

Usage:  python writeup/fill_tables.py [sweep_root] [fk_root]
"""
import os
import sys
import glob
from collections import defaultdict

ENVS = [("reacher", "Reacher"), ("pointmaze", "Point maze"), ("pusht", "Push-T"),
        ("wall", "Wall"), ("cube", "Cube")]
PLANNERS = ["gd", "grasp", "cem", "fk"]
HEADS = {"gd": "GD", "grasp": "GRASP", "cem": "CEM", "fk": r"FK (ours)"}
HORIZONS = [5, 10, 15]
# Arms the 2026-09-09 FK sweep will fill in; anything else missing is \na.
# Emitted literally rather than as \providecommand macros: definitions sitting
# in the document body before \begin{table} are what Overleaf's syntax checker
# reports as "unclosed open group found at \end{table}", and the table should
# be paste-safe into any preamble.
NA_TXT = r"\textendash"
QUEUED_TXT = r"\textit{running}"
QUEUED = {("reacher", "fk", h) for h in HORIZONS} | \
         {("pointmaze", "fk", h) for h in HORIZONS} | \
         {("pusht", "fk", h) for h in HORIZONS}


# Cells with no sweep data yet, backed by an earlier standalone probe run at
# the IDENTICAL config (8 particles x 8 smoothing x 150 steps, random_state).
# These are provisional: the sweep re-runs seeds 1-100 into its own root and
# will overwrite them, so they always print starred.
PROVISIONAL = {
    ("reacher", "fk", 10): "sweep_fk/reacher_probe_H10_p8/fk_H10",
}


def load_probe(path):
    """Probe dirs predate result.txt; they hold seed_success_log.txt instead,
    whose first field is the success flag. No wallclock is recorded there."""
    rows = []
    for f in glob.glob(f"{path}/seed_*/seed_success_log.txt"):
        try:
            rows.append((int(float(open(f).read().split()[0])), None))
        except (ValueError, IndexError):
            continue
    return rows


def load(root, planners=None, depth_fk=False):
    """-> {(env, planner, H): [(success, wallclock), ...]}"""
    out = defaultdict(list)
    pattern = f"{root}/*/*_H*/seed_*/result.txt"
    for f in glob.glob(pattern):
        parts = f.split(os.sep)
        env, cfg = parts[-4], parts[-3]
        # Quarantined arms are prefixed with '_' (see sweep/_pusht_random_state_invalid).
        # Skip them everywhere, not just in the table, so they cannot leak into
        # the pooled wall-clock accounting either.
        if env.startswith("_"):
            continue
        planner, H = cfg.rsplit("_H", 1)
        if planners and planner not in planners:
            continue
        try:
            ok, dt = open(f).read().split()
            out[(env, planner, int(H))].append((int(ok), float(dt)))
        except ValueError:
            continue
    return out


def cell(rows, key):
    """LaTeX cell + mean wallclock.

    An incomplete arm carries its n as a superscript, and a star when the cell
    is still moving -- either the sweep is mid-arm, or the figure is a
    provisional probe value the sweep will overwrite. A bare number therefore
    always means "final, 100 seeds"."""
    if not rows:
        return (QUEUED_TXT if key in QUEUED else NA_TXT), None
    n = len(rows)
    succ = 100.0 * sum(r[0] for r in rows) / n
    walls = [r[1] for r in rows if r[1] is not None]
    wall = sum(walls) / len(walls) if walls else None
    mark = f"{n}*" if (n < 100 and key in QUEUED) else (f"{n}" if n < 100 else "")
    txt = rf"{succ:.0f}" + (rf"\textsuperscript{{{mark}}}" if mark else "")
    return txt, wall


def main(root, fk_root):
    data = load(root, planners=["gd", "grasp", "cem"])
    data.update(load(fk_root, planners=["fk"]))
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key, rel in PROVISIONAL.items():
        have = data.get(key, [])
        if len(have) >= 100:                        # sweep arm finished: it wins
            continue
        rows = load_probe(os.path.join(repo, rel))
        # While the sweep arm is partial, report whichever source has more
        # seeds. Otherwise the first completed sweep seed (n=1) would replace a
        # 20-seed probe figure with a 0% or 100% cell -- a strictly worse
        # estimate presented as more current.
        if rows and len(rows) > len(have):
            data[key] = rows
            print(f"[provisional] {key} <- {rel} (n={len(rows)}, "
                  f"sweep has {len(have)} so far)")
    if not data:
        print(f"no results under {root} or {fk_root}")
        return

    print(f"{'env':<11}{'planner':<7}{'H':>3}  {'n':>4} {'success':>8} {'mean s':>8}")
    print("-" * 46)
    body = []
    for key, name in ENVS:
        body.append(rf"    \multirow{{3}}{{*}}{{{name}}}")
        for h in HORIZONS:
            cells = []
            for p in PLANNERS:
                rows = data.get((key, p, h), [])
                c, w = cell(rows, (key, p, h))
                cells.append(c)
                if rows:
                    # w is None for probe-sourced cells: those logs record the
                    # success flag but no wallclock.
                    wtxt = f"{w:>8.0f}" if w is not None else f"{'--':>8}"
                    print(f"{key:<11}{p:<7}{h:>3}  {len(rows):>4} "
                          f"{100*sum(r[0] for r in rows)/len(rows):>7.1f}% {wtxt}")
            body.append(rf"     & {h:<2} & " + " & ".join(cells) + r" \\")
        body.append(r"    \midrule")
    body[-1] = r"    \bottomrule"

    tex = "\n".join([
        r"% Generated by writeup/fill_tables.py -- do not hand-edit.",
        r"% Needs only booktabs + multirow. No custom macros, no xcolor.",
        r"% An em-dash cell = not scheduled; 'running' = sweep has not filled it yet.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Planning success rate (\%) over 100 seeds per cell, at a matched",
        r"  150-iteration budget. Goals are drawn with \texttt{random\_state} on Reacher",
        r"  and Point maze and from the dataset on Push-T, whose random start/goal pairs",
        r"  are frequently unconnectable. Superscripts give $n$ where an arm is",
        r"  incomplete, and a star marks a figure that is still moving: the arm is",
        r"  either mid-sweep or a provisional probe value that the full run will",
        r"  replace. An unmarked number is final over 100 seeds.",
        r"  \textbf{The FK budget is not uniform across environments}:",
        r"  Reacher runs 8 particles $\times$ 8 smoothing samples, while Point maze and",
        r"  Push-T run 3 particles with smoothing disabled --- the DINO-WM encoder emits",
        r"  196 ViT patch tokens per frame and the rollout is retained for autograd, so",
        r"  8 particles exceeds 16\,GB of VRAM there. Those two rows are therefore",
        r"  reduced-budget for our method and favour the baselines.}",
        r"  \label{tab:planners}",
        r"  \begin{tabular}{ll" + "c" * len(PLANNERS) + r"}",
        r"    \toprule",
        r"    Environment & $H$ & " + " & ".join(HEADS[p] for p in PLANNERS) + r" \\",
        r"    \midrule",
    ] + body + [
        r"  \end{tabular}",
        r"\end{table}",
    ]) + "\n"

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_table.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    print(f"\nwrote {out}")

    missing = [(e, p, h) for e, _ in ENVS for p in PLANNERS for h in HORIZONS
               if not data.get((e, p, h))]
    if missing:
        print("\nempty cells:")
        for k in missing:
            print(f"  {k[0]:<10} {k[1]:<6} H{k[2]:<3} "
                  f"{'queued (FK sweep)' if k in QUEUED else 'NOT SCHEDULED'}")
    partial = {k: len(v) for k, v in data.items() if len(v) < 100}
    if partial:
        print("\npartial cells:")
        for k, n in sorted(partial.items()):
            print(f"  {k[0]:<10} {k[1]:<6} H{k[2]:<3} n={n}")

    print("\nmean wall-clock (s) by planner, pooled over env/H:")
    for p in PLANNERS:
        rows = [r for k, v in data.items() if k[1] == p for r in v if r[1] is not None]
        if rows:
            print(f"  {p:<6} n={len(rows):<5} {sum(r[1] for r in rows)/len(rows):.0f}s")


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "sweep"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo, "sweep_fk", "table"))
