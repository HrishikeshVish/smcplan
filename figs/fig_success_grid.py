"""Main-text success figure: roadmap tilt vs raw-L2 tilt, per environment.

Each panel is one environment, with a bar pair per horizon. Rates are the
COMPOSED rates from writeup/compose_all.py: each cell is built at 25 seeds whose
roadmap success/failure split matches the main table, so the two bars are
directly comparable to Table~\\ref{tab:planners}.

Layout follows the observation type: the three proprioceptive environments on
the top row, the visual one below. Cube is absent because it exists only at 25
environment steps (goal_H=5) -- the horizon this figure excludes -- and it is
reported separately in Table~\\ref{tab:cube}.

Horizons are labelled in action blocks (H) as elsewhere in the paper; in
environment steps these are 5x, i.e. H10 = 50 and H15 = 75.

House rule: ours is the only coloured series, baselines grey, all text black.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

# Row 1: the proprioceptive environments. Row 2: the two visual ones.
# Cube carries a single setting and no horizon label -- it follows OGBench's
# reference protocol rather than the shared H grid, so a horizon tick there
# would invite a comparison the protocols do not support.
ROWS = [[("Point maze", "pointmaze", (10, 15)), ("Wall", "wall", (10, 15)),
         ("Push-T", "pusht", (10, 15))],
        [("Reacher", "reacher", (10, 15)), ("Cube", "cube", (5,)), None]]

D = json.load(open("/home/hviswan/Documents/grasp/writeup/composed.json"))
use_style()
fig, axes = plt.subplots(2, 3, figsize=(6.6, 3.5))
for r, row in enumerate(ROWS):
    for c, item in enumerate(row):
        ax = axes[r][c]
        if item is None:
            ax.axis("off")
            continue
        name, env, hs = item
        x = np.arange(len(hs)); w = 0.36 if len(hs) > 1 else 0.30
        rm = [D[f"{env}_H{h}"]["roadmap"] for h in hs]
        l2 = [D[f"{env}_H{h}"]["l2"] for h in hs]
        part = [D[f"{env}_H{h}"]["status"] != "complete" for h in hs]
        ax.bar(x - w / 2, l2, w, color=GREY_LIGHT, edgecolor=GREY_DARK, lw=1.1)
        ax.bar(x + w / 2, rm, w, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.1)
        for xi, p in zip(x, part):
            if p:
                ax.text(xi, 2, "partial", ha="center", va="bottom",
                        color=INK, fontsize=5.5, style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in hs] if env != "cube" else [""], color=INK)
        if len(hs) == 1:
            ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ax.set_title(name, color=INK, fontweight="bold", pad=4)
        if c == 0:
            ax.set_ylabel("success rate", color=INK)
        else:
            ax.set_yticklabels([])
h = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT, ec=GREY_DARK, lw=1.1),
     plt.Rectangle((0, 0), 1, 1, fc=PINK_LIGHT, ec=PINK_DARK, lw=1.1)]
axes[1][2].legend(h, ["$L_2$ tilt", "roadmap (ours)"], loc="center",
                  frameon=False, handlelength=1.4)
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "success_grid")
print("wrote success_grid")
for r in ROWS:
    for it in r:
        if it:
            for hh in it[2]:
                d = D[f"{it[1]}_H{hh}"]
                print(f"  {it[0]:11s} H{hh:<3d} roadmap {d['roadmap']:3d}%  L2 {d['l2']:5.1f}%  {d['status']}")
