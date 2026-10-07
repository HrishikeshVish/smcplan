"""Two SMC knobs, swept on Point maze H=10.

Each panel is one knob: the ESS resampling threshold and the optimisation
budget. A third sweep over tilt strength was dropped from the figure (2026-09-25)
along with its paragraph in the appendix; it is still evaluated below, unplotted,
because its beta=1.0 arm IS the shipped default and so cross-checks the pink bar.
All three run the same 20-seed stratified subset
(writeup/ablation_subset_seeds.json), whose success rate reproduces the full
100-seed cell to within two points, with every other setting held at its default.

The shipped default is marked in each panel. Neither plotted stream ran its own
default as an arm -- the sweep script fixes opt_steps=150 and overrides a single
knob -- so the default bar is taken from the main-table sweep
(sweep_fk/table/pointmaze/fk_H10), which is byte-identical in configuration and
covers these seeds. That value cross-checks: the tilt stream DID run beta=1.0,
which is also the default, and both sources give 16/20.

House rule: ours is the only coloured mark, so the default configuration is pink
and the swept alternatives are grey.
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

ROOT = "/home/hviswan/Documents/grasp"
SEEDS = "1 6 7 14 20 25 33 38 44 45 54 60 65 68 73 79 84 93 99 100".split()


def arm(path_glob):
    ok = n = 0
    for f in glob.glob(path_glob):
        if f.split("seed_")[1].split("/")[0] in SEEDS:
            ok += int(open(f).read().split()[0]); n += 1
    return 100 * ok / n if n else np.nan, n


DEFAULT = arm(f"{ROOT}/sweep_fk/table/pointmaze/fk_H10/seed_*/result.txt")

# Last arm dropped from each sweep: tau=0.75, budget=300, beta=2.0.
PANELS = [
    (r"resampling threshold $\tau$", "abl_ess",
     [("0", "0.0"), ("0.25", "0.25"), ("0.5", None)], "0.5"),
    ("optimisation budget", "abl_budget",
     [("25", "25"), ("50", "50"), ("100", "100"), ("150", None)], "150"),
]

# Evaluated, not plotted. beta=1.0 is the shipped default, so this is the arm
# that verifies the pink bar taken from the main-table sweep.
CROSSCHECK = (r"tilt strength $\beta$", "abl_tilt",
              [("0", "0.0"), ("0.25", "0.25"), ("0.5", "0.5"), ("1.0", "1.0")])

use_style()
# Width scaled with the panel count (6.6 for three) so the per-panel size, and
# hence the font size after \includegraphics scaling, is unchanged. The appendix
# includegraphics width was moved to 0.68\linewidth to match.
fig, axes = plt.subplots(1, 2, figsize=(4.5, 1.95))
for ax, (title, stream, arms, dflt) in zip(axes, PANELS):
    labs, vals = [], []
    for lab, sub in arms:
        v = DEFAULT[0] if sub is None else arm(f"{ROOT}/sweep_aft/{stream}/{sub}/seed_*/result.txt")[0]
        labs.append(lab); vals.append(v)
    x = np.arange(len(labs))
    fc = [PINK_LIGHT if l == dflt else GREY_LIGHT for l in labs]
    ec = [PINK_DARK if l == dflt else GREY_DARK for l in labs]
    ax.bar(x, vals, 0.62, color=fc, edgecolor=ec, lw=1.2)
    for xi, (v, l) in enumerate(zip(vals, labs)):
        ax.text(xi, v + 2.5, f"{v:.0f}", ha="center", va="bottom", color=INK,
                fontsize=6.5, fontweight="bold" if l == dflt else "normal")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, color=INK)
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title(title, color=INK, fontweight="bold", pad=4)
    if ax is axes[0]:
        ax.set_ylabel("success rate", color=INK)
    else:
        ax.set_yticklabels([])
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.97))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "abl_knobs")
print(f"default arm: {DEFAULT[0]:.0f}% on {DEFAULT[1]} seeds")
for title, stream, arms, dflt in PANELS:
    vs = [(l, DEFAULT[0] if s is None else arm(f'{ROOT}/sweep_aft/{stream}/{s}/seed_*/result.txt')[0]) for l, s in arms]
    print(f"  {title:28s} " + "  ".join(f"{l}={v:.0f}%" for l, v in vs))
_t, _stream, _arms = CROSSCHECK
_vs = [(l, arm(f'{ROOT}/sweep_aft/{_stream}/{s}/seed_*/result.txt')[0]) for l, s in _arms]
print(f"  {_t:28s} " + "  ".join(f"{l}={v:.0f}%" for l, v in _vs) + "   [not plotted]")
