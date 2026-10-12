"""Paired outcomes against each baseline, aggregated over environments.

One panel per baseline, one diverging bar per horizon. Bars to the right count
seeds our planner solves and the baseline does not; bars to the left count the
reverse. Counts are summed over all four environments at that horizon, so each
bar aggregates 400 paired seeds.

This is the main-text summary of the per-environment panels in
Appendix~\\ref{app:paired}: the asymmetry is what matters here, and the twelve
separate panels are needed only to see which environment drives it.

House rule: ours is the only coloured series, baselines grey, all text black.
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

ROOT = "/home/hviswan/Documents/grasp"
ENVS = ["reacher", "pointmaze", "wall", "pusht"]
# 2026-10-07: the equal-particle comparison of Table 1 -- every planner holds
# 12 particles for 150 iterations (gds12, grasp12, cem12; ours = fixed JALA-EM,
# pdefix). The 1-particle / 30x5 arms of the previous version are appendix tables.
BASE = [("GD", "sweep/%s/gds12_H%d"), ("GRASP", "sweep/%s/grasp12_H%d"),
        ("CEM", "sweep/%s/cem12_H%d"), ("MPPI", "sweep/%s/mppi12_H%d")]
sys.path.insert(0, "/home/hviswan/Documents/grasp/writeup")
from equal_particle_stats import arm_dir  # noqa: E402  (`ours` per OURS_PM)


class _Ours:
    """OURS % (env, H) -> the directory of `ours` in that cell."""
    def __mod__(self, eh):
        return arm_dir("ours", *eh)


OURS = _Ours()
HS = (5, 10, 15)


def outcomes(pat):
    d = {}
    for sd in glob.glob(os.path.join(ROOT, pat) + "/seed_*"):
        try:
            v = open(sd + "/result.txt").read().split()
        except OSError:
            continue
        if v:
            d[int(sd.split("_")[-1])] = (v[0] == "1")
    return d


use_style()
fig, axes = plt.subplots(1, len(BASE), figsize=(6.9, 1.85))
counts = {}
for ax, (bname, bpat) in zip(axes, BASE):
    w, l, n = [], [], []
    for H in HS:
        ww = ll = nn = 0
        for e in ENVS:
            o, x = outcomes(OURS % (e, H)), outcomes(bpat % (e, H))
            common = sorted(set(o) & set(x))
            ww += sum(o[s] and not x[s] for s in common)
            ll += sum(x[s] and not o[s] for s in common)
            nn += len(common)
        w.append(ww); l.append(ll); n.append(nn)
    counts[bname] = (w, l, n)
    y = np.arange(len(HS))[::-1]
    ax.barh(y, w, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.2, height=0.55)
    ax.barh(y, [-v for v in l], color=GREY_LIGHT, edgecolor=GREY_DARK, lw=1.2, height=0.55)
    for yi, (a, b) in zip(y, zip(w, l)):
        ax.text(a + 6, yi, str(a), va="center", ha="left", color=INK,
                fontsize=6.5, fontweight="bold")
        ax.text(-b - 6, yi, str(b), va="center", ha="right", color=INK, fontsize=6.5)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"$H{{=}}{h}$" for h in HS], color=INK)
    ax.tick_params(left=False)
    ax.set_xlim(-90, 310)              # equal-particle counts reach 244 (vs GRASP, H=5)
    ax.set_xticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.grid(False)
    ax.set_title(f"vs {bname}", color=INK, fontweight="bold", pad=4)
fig.text(0.5, 0.02, "seeds solved by one method only "
                    "(left: baseline, right: ours)", ha="center", color=INK, fontsize=7)
fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.97))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "paired_summary")
for b, (w, l, n) in counts.items():
    for i, H in enumerate(HS):
        print(f"  vs {b:6s} H{H:<3d} ours-only {w[i]:4d}  theirs-only {l[i]:4d}  of {n[i]} paired")
