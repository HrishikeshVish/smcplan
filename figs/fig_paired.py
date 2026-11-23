"""Per-seed paired outcomes against each baseline, as diverging bars.

ONE FIGURE PER (ENVIRONMENT, BASELINE) PAIR -- 12 in all, each three rows deep
(one per horizon) inside a rounded frame. Earlier versions packed 12 rows across
3 panels in a single strip, which left every label too small to read at print
size.

Bars to the right count seeds we solve and the baseline does not; bars to the
left count the reverse. The asymmetry is the point: a method that merely trades
instances would show mirrored bars, and a method that solves a superset shows
one-sided ones.

Colour follows the house rule: ours is the only coloured series, drawn in the
soft pink fill with a dark pink edge because the bars are the dominant mark;
baselines are neutral grey; all text is black.
"""
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,  # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2026-10-07: the equal-particle comparison of Table 1 -- every planner holds
# 12 particles for 150 iterations (gds12, grasp12, cem12; ours = fixed JALA-EM,
# pdefix). The 1-particle / 30x5 arms of the previous version are appendix tables.
BASE = [("GD", "sweep/%s/gds12_H%d"), ("GRASP", "sweep/%s/grasp12_H%d"),
        ("CEM", "sweep/%s/cem12_H%d"), ("MPPI", "sweep/%s/mppi12_H%d")]
sys.path.insert(0, os.path.join(ROOT, "writeup"))
from equal_particle_stats import arm_dir  # noqa: E402  (`ours` per OURS_PM)


class _Ours:
    """OURS % (env, H) -> the directory of `ours` in that cell."""
    def __mod__(self, eh):
        return arm_dir("ours", *eh)


OURS = _Ours()
ENVS = [("Reacher", "reacher"), ("Point maze", "pointmaze"),
        ("Wall", "wall"), ("Push-T", "pusht")]
HS = (5, 10, 15)


def outcomes(pat):
    d = {}
    for sd in glob.glob(os.path.join(ROOT, pat) + "/seed_*"):
        f = sd + "/result.txt"
        if not os.path.exists(f):
            continue
        v = open(f).read().split()
        if v:
            d[int(sd.split("_")[-1])] = (v[0] == "1")
    return d



use_style()
for name, env in ENVS:
    o_by_H = {H: outcomes(OURS % (env, H)) for H in HS}
    for b, pat in BASE:
        w, l = [], []
        for H in HS:
            o, x = o_by_H[H], outcomes(pat % (env, H))
            common = sorted(set(o) & set(x))
            w.append(sum(o[s] and not x[s] for s in common))
            l.append(sum(x[s] and not o[s] for s in common))
        w, l = np.array(w), np.array(l)

        lim = max(w.max(), l.max()) * 1.42
        y = np.arange(len(HS))[::-1]
        # Panels are drawn at 3.05in and included at 0.32\textwidth (~2.05in),
        # so print scales them by ~0.67. Type and line widths below are
        # pre-multiplied so the PRINTED result lands near 9-10pt.
        fig, ax = plt.subplots(figsize=(3.05, 1.95))
        ax.barh(y, w, color=PINK_LIGHT, edgecolor=PINK_DARK, linewidth=1.3, height=0.5)
        ax.barh(y, -l, color=GREY_LIGHT, edgecolor=GREY_DARK, linewidth=1.3, height=0.5)
        for yi, (wi, li) in enumerate(zip(w, l)):
            ax.text(wi + lim * 0.06, y[yi], str(wi), va="center", ha="left",
                    fontsize=14, color=INK, fontweight="bold" if wi > li else "normal")
            ax.text(-li - lim * 0.06, y[yi], str(li), va="center", ha="right",
                    fontsize=14, color=INK, fontweight="bold" if li > wi else "normal")
        ax.axvline(0, color=INK, linewidth=1.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(HS) - 0.4)
        ax.set_title(f"{name}  vs {b}", fontsize=15.5, color=INK, pad=7)
        ax.set_xticks([])
        ax.set_yticks(y)
        ax.set_yticklabels([f"$H{{=}}{H}$" for H in HS], fontsize=14, color=INK)
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(False)
        fig.subplots_adjust(left=0.30, right=0.945, top=0.825, bottom=0.07)
        rounded_frame(fig, 0.012, 0.015, 0.976, 0.97, lw=1.1, r=0.035)
        save(fig, f"fig_paired_{env}_{b.lower()}")
        plt.close(fig)
    print(f"{name}: done")
