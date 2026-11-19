"""One figure that says what Table 1 says: mean success rate per planner.

The table has 144 numbers, which is right for a paper and wrong for a README.
This averages each planner's success rate over the twelve cells (four
environments, three horizons) and draws them as one bar each, so the headline
ordering is visible at a glance. The spread of the twelve cells is drawn as a
thin range line, because a mean over cells this different would otherwise
suggest more uniformity than there is.

Numbers are parsed from the table the paper compiles, so the figure cannot
drift from it. Regenerate the table first if it is stale:
    python writeup/make_equal_particle_tables.py
Then
    python figs/fig_readme_summary.py --out assets/mean_success.png
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402
from iclr_style import (use_style, GREY, GREY_DARK, GREY_LIGHT,  # noqa: E402
                        PINK, PINK_DARK, INK)

LIGHTEST = "#E8E9EC"   # a fourth grey, so that no two baselines share a tone

TABLE = os.environ.get(
    "TABLE1_TEX", "/home/hviswan/Documents/grasp-paper/table1_iclr.tex")
# Row label in the tex -> (label on the figure, bar colour).
STYLE = {
    "Ours (PDE cost)": ("ours, PDE cost", PINK),
    "Ours ($L_2$ cost)": ("ours, $L_2$ cost", PINK_DARK),
    "MPPI": ("MPPI", GREY_DARK),
    "GD": ("GD", GREY),
    "CEM": ("CEM", GREY_LIGHT),
    "GRASP": ("GRASP", LIGHTEST),
}


def parse(path):
    """-> [(row label, array of the twelve success rates)] in tex order."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.endswith(r"\\") or line.startswith("%"):
                continue
            cols = [c.strip() for c in line[:-2].split("&")]
            if len(cols) != 13 or "multicolumn" in cols[0]:
                continue
            name = re.sub(r"\\textbf\{(.*)\}", r"\1", cols[0]).strip()
            if name.startswith("Planner"):
                continue
            values = []
            for c in cols[1:]:
                m = re.search(r"\\best\{([0-9]+)\}|(?<![0-9])([0-9]+)", c)
                if not m:
                    break
                values.append(int(m.group(1) or m.group(2)))
            if len(values) == 12:
                rows.append((name, np.array(values, dtype=float)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--out", default="assets/mean_success.png")
    args = ap.parse_args()

    rows = parse(args.table)
    missing = [n for n, _ in rows if n not in STYLE]
    if missing:
        sys.exit(f"unstyled rows in {args.table}: {missing}")
    rows.sort(key=lambda r: r[1].mean())

    use_style(base=9)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    for y, (name, values) in enumerate(rows):
        label, colour = STYLE[name]
        mean = values.mean()
        ax.barh(y, mean, height=0.66, color=colour, edgecolor=INK,
                linewidth=0.6, zorder=2)
        ax.plot([values.min(), values.max()], [y, y], color=INK, lw=0.8,
                alpha=0.55, zorder=3, solid_capstyle="butt")
        ax.text(values.max() + 2.5, y, f"{mean:.1f}", va="center", ha="left",
                fontsize=8, color=INK, zorder=4)
        ax.text(-2.0, y, label, va="center", ha="right", fontsize=8.5,
                color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("success rate (%), mean over 12 cells; line spans them")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"{out}: {len(rows)} planners, "
          + ", ".join(f"{n}={v.mean():.2f}" for n, v in rows))


if __name__ == "__main__":
    main()
