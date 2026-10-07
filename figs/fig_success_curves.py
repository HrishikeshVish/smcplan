"""Cumulative success against planner iteration, in the style of an RL learning curve.

At iteration k a seed counts as solved if it has succeeded within its first k
iterations, so each curve is the fraction of the 100 seeds solved by that point.
Curves are monotone and top off at the arm's final success rate at iteration
150 -- the number reported in the main table -- so the table is the right-hand
endpoint of this figure, and the shape in between says how quickly each planner
gets there.

2026-10-07: the arms are the EQUAL-PARTICLE comparison of Table 1 -- every
planner holds 12 particles for 150 iterations (GD-12 = gds12, GRASP-12 =
grasp12, CEM-12 = cem12, ours = the fixed JALA-EM arm pdefix). Iterations are
counted as in writeup/equal_particle_stats.py: 1 + the index of the first
successful evaluation, every planner evaluating once per iteration after its
update. The 1-particle / 30x5 arms of the previous version are appendix tables.

Source: the per-evaluation 'success' flags in each run's run.log, which every arm
writes. The planner evaluates and steps the environment inside its optimisation
loop, so the evaluation index is the iteration index.

Colour: ours is the only coloured series; baselines are neutral and separated by
dash pattern, per the house rule.
"""
import os
import re
import sys
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_DARK,  # noqa: E402
                        GREY_DARK, GREY, INK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "figs", ".success_curves_cache_eq12.json")
MAXIT = 150
SUC = re.compile(r"'success': array\(\[\s*(True|False)")

# Lines take the DARK variants of the palette; ours is the only coloured series
# and the baselines are dark neutrals separated by dash pattern.
# Panels are drawn at 3.05in and included at ~0.315\textwidth (~2.05in), so
# everything is scaled by ~0.67 in print. Font sizes and line widths below are
# pre-multiplied by ~1.5 so the PRINTED result lands at ~9-10pt rather than ~6.
ARMS = [("GD",    "gds12",   GREY_DARK, (0, (5, 2)),       1.9),
        ("GRASP", "grasp12", GREY_DARK, (0, (1, 1.8)),     1.9),
        ("CEM",   "cem12",   INK,       (0, (3, 1, 1, 1)), 1.9),
        ("Ours",  "ours",    PINK_DARK, "solid",           3.1)]

# (label, env key, horizon). Cube exists at H=5 only, under its own protocol.
CELLS = [(f"{t} $H{{=}}{H}$", e, H) for t, e in
         [("Reacher", "reacher"), ("Point maze", "pointmaze"),
          ("Wall", "wall"), ("Push-T", "pusht")] for H in (5, 10, 15)]
CELLS.append(("Cube $H{=}5$", "cube", 5))


def pattern(arm_key, env, H):
    if env == "cube":
        return {"gds12": "sweep_cube_leflow/gds12",
                "grasp12": "sweep_cube_leflow/grasp12",
                "cem12": "sweep_cube_leflow/cem12",
                "ours": "sweep_cube_leflow/pdefix"}[arm_key]
    if arm_key == "ours":
        return f"sweep_emsmc/table/{env}/pdefix_H{H}"
    return f"sweep/{env}/{arm_key}_H{H}"


def first_success(log):
    try:
        seq = SUC.findall(open(log, errors="ignore").read())
    except OSError:
        return None
    for i, v in enumerate(seq):
        if v == "True":
            return i
    return None


def curve(pattern, cache):
    """Fraction of seeds solved at or before each iteration."""
    if pattern in cache:
        firsts = cache[pattern]
    else:
        firsts = []
        for sd in sorted(glob.glob(os.path.join(ROOT, pattern) + "/seed_*")):
            if not os.path.exists(sd + "/result.txt"):
                continue
            firsts.append(first_success(sd + "/run.log"))
        cache[pattern] = firsts
    n = len(firsts)
    if not n:
        return None, 0
    # iterations to success = 1 + index of the first successful evaluation
    hit = np.array([f + 1 for f in firsts if f is not None], dtype=float)
    y = np.array([(hit <= k).sum() for k in range(MAXIT + 1)]) / n * 100.0
    return y, n


cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
use_style()
x = np.arange(MAXIT + 1)

summary = {}
for label, env, H in CELLS:
    fig, ax = plt.subplots(figsize=(3.05, 2.30))
    ends = []
    for name, key, colour, dash, lw in ARMS:
        y, n = curve(pattern(key, env, H), cache)
        if y is None:
            ends.append(None)
            continue
        ends.append(y[-1])
        ax.plot(x, y, color=colour, linestyle=dash, linewidth=lw, label=name,
                zorder=3 if name == "Ours" else 2)
    ax.set_xlim(0, MAXIT)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0, 50, 100])
    ax.set_ylim(0, 100)
    ax.set_title(label, fontsize=16, color=INK, pad=6)
    ax.set_xlabel("iteration", fontsize=14.5, color=INK)
    ax.set_ylabel("solved (%)", fontsize=14.5, color=INK)
    ax.tick_params(labelsize=13, colors=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.245, right=0.945, top=0.855, bottom=0.235)
    rounded_frame(fig, 0.012, 0.015, 0.976, 0.97, lw=1.1, r=0.035)
    save(fig, f"fig_success_{env}_h{H}")
    plt.close(fig)
    margin = (ends[3] - max(e for e in ends[:3] if e is not None)) if ends[3] is not None else None
    summary[(env, H)] = (ends, margin)
    print(f"{label:22s} " + "/".join("--" if e is None else f"{e:.0f}" for e in ends)
          + f"   margin={margin:+.0f}")

json.dump(cache, open(CACHE, "w"))

# legend-only strip, so the 13 panels need not each carry a key
figL, axL = plt.subplots(figsize=(6.2, 0.42))
for name, key, colour, dash, lw in ARMS:
    axL.plot([], [], color=colour, linestyle=dash, linewidth=lw, label=name)
axL.axis("off")
axL.legend(frameon=False, fontsize=14, ncol=4, loc="center", handlelength=3.0)
save(figL, "fig_success_legend")
