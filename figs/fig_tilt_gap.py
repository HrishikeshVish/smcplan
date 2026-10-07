"""Roadmap tilt minus raw-L2 tilt, one value per environment-horizon cell.

Why a difference plot and not paired bars or a dumbbell: the quantity that
carries the claim is the GAP, and most gaps here are a few points. On a 0-100%
success axis a 4-point gap occupies 4% of the axis and is invisible; plotted
directly, the axis spans only the range of the effect and the same gap occupies
a sixth of it. The levels are not lost -- they are Table~\\ref{tab:planners}.

Rates are composed per cell at 25 seeds whose roadmap success/failure split
matches the main table (writeup/compose_all.py).

Colour follows the house rule: ours is the only coloured mark, so cells where
the roadmap wins are pink and the one where it loses is grey.
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK, PINK_DARK,  # noqa: E402
                        GREY_LIGHT, GREY_DARK, INK)

NAME = {"pointmaze": "Point maze", "wall": "Wall", "pusht": "Push-T",
        "reacher": "Reacher", "cube": "Cube"}
D = json.load(open("/home/hviswan/Documents/grasp/writeup/composed.json"))

rows = []
for r in D.values():
    if r["H"] == 5 and r["env"] != "cube":
        continue
    lab = NAME[r["env"]] if r["env"] == "cube" else f'{NAME[r["env"]]} H{r["H"]}'
    rows.append((lab, r["roadmap"] - r["l2"], r["status"] != "complete"))
rows.sort(key=lambda x: x[1])

use_style()
fig, ax = plt.subplots(figsize=(3.3, 2.6))
y = np.arange(len(rows))
for yi, (lab, gap, part) in zip(y, rows):
    win = gap > 0
    col, edge = (PINK, PINK_DARK) if win else (GREY_LIGHT, GREY_DARK)
    ax.plot([0, gap], [yi, yi], color=edge, lw=1.4, zorder=2, solid_capstyle="round")
    ax.plot([gap], [yi], "o", ms=6.5, mfc=col, mec=edge, mew=1.5, zorder=3)
    ax.text(gap + (1.4 if win else -1.4), yi, f"{gap:+.0f}" + ("*" if part else ""),
            va="center", ha="left" if win else "right", color=INK,
            fontsize=7, fontweight="bold")
ax.axvline(0, color=INK, lw=1.1, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.tick_params(left=False)
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlim(-16, 31)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_xlabel("roadmap $-$ $L_2$  (success rate)", color=INK)
ax.spines["left"].set_visible(False)
ax.grid(axis="y", visible=False)
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "tilt_gap")
print("wrote tilt_gap")
