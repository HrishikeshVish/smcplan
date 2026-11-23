"""Does the roadmap cost of Sec. 3.4 also help the baseline planners?

One bar per (environment, planner): the PAIRED change in success rate when the
planner's objective_fn is L2 + lambda_V * C_V instead of L2 alone, in
percentage points. Only the difference is plotted -- the two absolute rates are
in the CSV, and the difference is the quantity the reviewer's question is
about, so showing both arms would spend half the figure restating Table 1.

Both arms of a pair share the seed, hence the start and the goal, so the
difference is within-seed and the bootstrap CI (seed-level, paired) is the
honest error bar. Stars are the exact two-sided McNemar test on the discordant
pairs, not a test against the CI.

Numbers come from writeup/pde_baseline_transfer.csv; regenerate it with
    python writeup/pde_baseline_stats.py
Then
    python figs/fig_pde_transfer.py --out-dir ~/Documents/grasp-paper/figs
writes pde_transfer.{pdf,png}.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "writeup", "pde_baseline_transfer.csv")

# The paper's two colours only: pink is the roadmap cost, grey is L2. Here every
# bar is a baseline planner, so colour encodes the sign -- pink where swapping L2
# for the roadmap cost helps that planner, grey where it hurts -- and the planner
# itself is named under its bar. That keeps pink meaning what it means elsewhere.
ARM_ORDER = ["GD", "CEM", "MPPI"]
UP = (PINK_LIGHT, PINK_DARK)
DOWN = (GREY_LIGHT, GREY_DARK)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("delta_pp", "ci_lo_pp", "ci_hi_pp", "success_l2",
                  "success_pde", "p_mcnemar"):
            r[k] = float(r[k])
        for k in ("n_paired", "horizon", "wins_pde", "wins_l2"):
            r[k] = int(r[k])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--out-dir", default=None,
                    help="where the pdf/png go (default: $FIG_OUT)")
    ap.add_argument("--name", default="pde_transfer")
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--height", type=float, default=2.25)
    ap.add_argument("--show-n", action="store_true",
                    help="print the paired seed count under each group")
    args = ap.parse_args()
    if args.out_dir:
        os.environ["FIG_OUT"] = os.path.expanduser(args.out_dir)

    rows = load(args.csv)
    cells = []                                  # group order follows the CSV
    for r in rows:
        key = (r["env_label"], r["horizon"])
        if key not in cells:
            cells.append(key)

    use_style()
    fig, ax = plt.subplots(figsize=(args.width, args.height))

    width = 0.26
    offs = {a: (i - 1) * width for i, a in enumerate(ARM_ORDER)}
    ymin, ymax = 0.0, 0.0
    for r in rows:
        g = cells.index((r["env_label"], r["horizon"]))
        x = g + offs[r["planner_label"]]
        d = r["delta_pp"]
        fill, edge = UP if d > 0 else DOWN
        ax.bar(x, d, width=width * 0.92, color=fill, edgecolor=edge, lw=1.2,
               zorder=3)
        lo, hi = r["ci_lo_pp"], r["ci_hi_pp"]
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor=INK,
                        elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
            ymin, ymax = min(ymin, lo), max(ymax, hi)
        ymin, ymax = min(ymin, d), max(ymax, d)

    span = max(ymax - ymin, 1.0)
    for r in rows:
        s = stars(r["p_mcnemar"])
        if not s:
            continue
        g = cells.index((r["env_label"], r["horizon"]))
        x = g + offs[r["planner_label"]]
        d, hi, lo = r["delta_pp"], r["ci_hi_pp"], r["ci_lo_pp"]
        up = d >= 0
        y = (hi if np.isfinite(hi) else d) if up else (lo if np.isfinite(lo) else d)
        ax.text(x, y + (0.035 if up else -0.055) * span, s, ha="center",
                va="bottom" if up else "top", color=INK, fontsize=8)

    ax.axhline(0, color=INK, lw=1.0, zorder=2)
    ax.set_xticks([g + offs[a] for g in range(len(cells)) for a in ARM_ORDER])
    ax.set_xticklabels([a for _ in cells for a in ARM_ORDER], color=INK,
                       fontsize=plt.rcParams["font.size"] - 1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    # A second x-axis carries the environment names under the planner names. It
    # is a real axis rather than free text so tight_layout reserves room for it.
    glabels = []
    for g, (env, H) in enumerate(cells):
        lab = f"{env}, $H={H}$"
        if args.show_n:
            ns = [r["n_paired"] for r in rows
                  if (r["env_label"], r["horizon"]) == (env, H)]
            lab += f", $n={min(ns)}$" if min(ns) == max(ns) else f", $n={min(ns)}$-${max(ns)}$"
        glabels.append(lab)
        if g:                                   # hairline between the groups
            ax.axvline(g - 0.5, color=GREY_LIGHT, lw=0.8, zorder=1)
    gax = ax.secondary_xaxis("bottom")
    gax.set_xticks(range(len(cells)))
    gax.set_xticklabels(glabels, color=INK, fontweight="bold")
    gax.tick_params(axis="x", length=0, pad=14)
    gax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("change in success rate", color=INK)
    ax.set_ylim(ymin - 0.16 * span, ymax + 0.18 * span)
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(facecolor=UP[0], edgecolor=UP[1], lw=1.2,
                             label="roadmap cost helps"),
                       Patch(facecolor=DOWN[0], edgecolor=DOWN[1], lw=1.2,
                             label="roadmap cost hurts")],
              loc="upper left", ncol=2, handlelength=1.1, handleheight=0.9,
              columnspacing=1.0, borderaxespad=0.2)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, args.name)
    out = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
    print(f"wrote {os.path.join(out, args.name)}.{{pdf,png}}")
    for r in rows:
        print(f"  {r['env_label']:11s} H{r['horizon']:<3d} {r['planner_label']:<5s}"
              f" n={r['n_paired']:3d}  {r['delta_pp']:+6.1f} pp"
              f"  [{r['ci_lo_pp']:+6.1f}, {r['ci_hi_pp']:+6.1f}]"
              f"  p={r['p_mcnemar']:.3f} {stars(r['p_mcnemar'])}")


if __name__ == "__main__":
    main()
