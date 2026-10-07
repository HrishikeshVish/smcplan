"""Roadmap tilt vs raw-L2 tilt, one row per environment.

Horizons are pooled: each environment's two cells (H=10 and H=15) are composed
at 25 seeds apiece, so pooling them is a plain average of the two rates. H=5 is
excluded -- the world model rolls out only 5 steps there, the planning problem
is shallow, and the two costs agree almost everywhere. Cube carries its own
setting under OGBench's protocol and is not pooled.

A dumbbell, not paired bars: the reader needs BOTH the level (Push-T at 16% and
Point maze at 71% are different regimes) and the difference between the two
marks. Bars in separate panels make the second comparison a judgement about bar
heights across panels. Here the level is the dot position and the gap is the
line length -- and because a half-point gap is still invisible at this scale,
the gap is printed at the right so small effects are exact rather than guessed.

Rows are ordered by the gap.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK, PINK_DARK,  # noqa: E402
                        GREY_DARK, INK)

NAME = {"pointmaze": "Point maze", "wall": "Wall", "pusht": "Push-T",
        "reacher": "Reacher", "cube": "Cube"}
HS = (10, 15)
D = json.load(open("/home/hviswan/Documents/grasp/writeup/composed.json"))

rows = []
for env in ("pointmaze", "wall", "pusht", "reacher"):
    cells = [D[f"{env}_H{h}"] for h in HS]
    rm = sum(c["roadmap"] for c in cells) / len(cells)
    l2 = sum(c["l2"] for c in cells) / len(cells)
    part = any(c["status"] != "complete" for c in cells)
    rows.append((NAME[env], l2, rm, rm - l2, part))
c = D["cube_H5"]
rows.append((NAME["cube"], c["l2"], c["roadmap"], c["roadmap"] - c["l2"], False))
rows.sort(key=lambda x: x[3])

use_style()
fig, ax = plt.subplots(figsize=(3.3, 1.7))
y = np.arange(len(rows))
for yi, (lab, l2, rm, gap, part) in zip(y, rows):
    ax.plot([l2, rm], [yi, yi], color=GREY_DARK, lw=1.3, zorder=2,
            solid_capstyle="round")
    ax.plot([l2], [yi], "o", ms=6, mfc="white", mec=GREY_DARK, mew=1.5, zorder=3)
    ax.plot([rm], [yi], "o", ms=6, mfc=PINK, mec=PINK_DARK, mew=1.5, zorder=4)
    ax.text(102, yi, f"{gap:+.0f}" + ("*" if part else ""), va="center",
            ha="left", color=INK, fontsize=7, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.tick_params(left=False)
ax.set_ylim(-0.6, len(rows) - 0.15)
ax.set_xlim(0, 100)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_xlabel("success rate", color=INK)
ax.spines["left"].set_visible(False)
ax.grid(axis="y", visible=False)
# Direct labelling instead of a legend: with five rows a legend box has nowhere
# to sit that does not land on the data or the axis label, and naming the two
# marks once on the top row is read faster than a key.
top = len(rows) - 1
ax.annotate("$L_2$", (rows[top][1], top), textcoords="offset points",
            xytext=(0, 9), ha="center", color=GREY_DARK, fontsize=7)
ax.annotate("ours", (rows[top][2], top), textcoords="offset points",
            xytext=(0, 9), ha="center", color=PINK_DARK, fontsize=7,
            fontweight="bold")
fig.tight_layout(rect=(0.02, 0.02, 0.94, 0.97))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "tilt_dumbbell")
for lab, l2, rm, gap, part in rows:
    print(f"  {lab:11s} L2 {l2:5.1f}%  ours {rm:5.1f}%  gap {gap:+5.1f}"
          + ("  (partial)" if part else ""))
