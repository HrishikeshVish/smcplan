"""Kernel ablation (appendix): success by E-step propagation kernel. 2026-10-07.

Replaces the stratified FK-era version (sweep_aft/ablation*, now in git history).
Everything but the E-step kernel is the main-table setting of JALA-EM (pathwise
M-step, calibrated PDE cost, 12 particles, adaptive beta <= 1, 150 iterations,
stop at first success), on the SAME seeds as Table 1: 100 paired seeds at H=10
on the four grid environments, and Cube under the LeFlow protocol (50 seeds).

  JALA  ULA + Jarzynski weights (ours; = the pdefix arm of Table 1)
  MALA  Metropolis-adjusted Langevin, dual averaging to acceptance 0.574
  RWM   random-walk Metropolis, dual averaging to acceptance 0.234
  HMC   preconditioned HMC, leapfrog length uniform on {1..10}, acceptance 0.651

Runs: sweep_emsmc/table/<env>/kfix_<k>_H10 and sweep_cube_leflow/kfix_<k>
(writeup/run_emsmc_final.sh, writeup/run_emsmc_cube.sh with KERNEL=<k>).
Summary table: writeup/results/kernel_ablation_2026-10-07.md. Error bars are 95%
Wilson intervals (writeup/equal_particle_stats.py).

House rule: ours is the only coloured bar; the alternatives are separated by grey
level, all text black.
"""
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "writeup"))
from equal_particle_stats import wilson  # noqa: E402
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,  # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY, GREY_DARK, INK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNELS = [("pdefix", "JALA (ours)", PINK_LIGHT, PINK_DARK),
           ("kfix_mala", "MALA", GREY_LIGHT, GREY_DARK),
           ("kfix_rwm", "RWM", GREY, GREY_DARK),
           ("kfix_hmc", "HMC", GREY_DARK, INK)]
CELLS = [("Reacher", "reacher"), ("Point maze", "pointmaze"),
         ("Wall", "wall"), ("Push-T", "pusht"), ("Cube", "cube")]


def pattern(arm, env):
    if env == "cube":
        return f"sweep_cube_leflow/{arm}"
    return f"sweep_emsmc/table/{env}/{arm}_H10"


def rate(arm, env):
    ok = n = 0
    for f in glob.glob(f"{ROOT}/{pattern(arm, env)}/seed_*/result.txt"):
        v = open(f).read().split()
        if v:
            n += 1
            ok += (v[0] == "1")
    return (100.0 * ok / n, n) if n else (None, 0)


def interval(arm, env):
    """95% Wilson interval (%) of the success rate; error bars below."""
    r, n = rate(arm, env)
    return wilson(round(r * n / 100), n)


use_style()
for title, env in CELLS:
    fig, ax = plt.subplots(figsize=(3.05, 2.30))
    xs = np.arange(len(KERNELS))
    vals = []
    for x, (arm, label, fill, edge) in zip(xs, KERNELS):
        r, n = rate(arm, env)
        vals.append((r, n))
        lo, hi = interval(arm, env)
        ax.bar(x, max(r, 1.2), 0.68, color=fill, edgecolor=edge, linewidth=1.4,
               zorder=3)
        ax.errorbar(x, r, yerr=[[r - lo], [hi - r]], fmt="none", ecolor=INK,
                    elinewidth=1.3, capsize=4, capthick=1.3, zorder=4)
        ax.text(x, hi + 2.5, f"{r:.0f}", ha="center", va="bottom", fontsize=13,
                color=INK, fontweight="bold" if arm == "pdefix" else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([k[1].split()[0] for k in KERNELS], fontsize=12.5, color=INK)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="y", labelsize=13, colors=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("solved (%)", fontsize=14.5, color=INK)
    sub = "$H{=}5$" if env == "cube" else "$H{=}10$"
    ax.set_title(f"{title} {sub}", fontsize=16, color=INK, pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.subplots_adjust(left=0.245, right=0.945, top=0.855, bottom=0.15)
    rounded_frame(fig, 0.012, 0.015, 0.976, 0.97, lw=1.1, r=0.035)
    save(fig, f"fig_kernels_{env}")
    plt.close(fig)
    print(f"{title:11s} " + "  ".join(f"{k[1].split()[0]} {r:.0f} (n={n})"
                                       for k, (r, n) in zip(KERNELS, vals)))

from matplotlib.patches import Patch  # noqa: E402
figL, axL = plt.subplots(figsize=(6.2, 0.42))
axL.axis("off")
axL.legend(handles=[Patch(facecolor=f, edgecolor=e, linewidth=1.1, label=l)
                    for _, l, f, e in KERNELS],
           frameon=False, fontsize=14, ncol=4, loc="center", handlelength=1.8)
save(figL, "fig_kernels_legend")

# ---- all five cells on one axis (fig_kernels_success), same data ----
fig, ax = plt.subplots(figsize=(6.9, 2.0))
xs = np.arange(len(CELLS))
w = 0.8 / len(KERNELS)
for ki, (arm, label, fill, edge) in enumerate(KERNELS):
    off = (ki - (len(KERNELS) - 1) / 2) * w
    ys = [rate(arm, env)[0] for _, env in CELLS]
    ivs = [interval(arm, env) for _, env in CELLS]
    ax.bar(xs + off, [max(y, 1.2) for y in ys], w * 0.92, color=fill,
           edgecolor=edge, linewidth=1.0, label=label, zorder=3)
    ax.errorbar(xs + off, ys, yerr=[[y - lo for y, (lo, hi) in zip(ys, ivs)],
                                    [hi - y for y, (lo, hi) in zip(ys, ivs)]],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
    for x, y, (lo, hi) in zip(xs + off, ys, ivs):
        ax.text(x, hi + 1.5, f"{y:.0f}", ha="center", va="bottom", fontsize=6,
                color=INK, fontweight="bold" if arm == "pdefix" else "normal")
ax.set_xticks(xs)
ax.set_xticklabels([f"{t} " + ("$H{=}5$" if e == "cube" else "$H{=}10$")
                    for t, e in CELLS], color=INK)
ax.set_ylim(0, 116)
ax.set_yticks([0, 50, 100])
ax.set_ylabel("solved (%)", color=INK)
ax.tick_params(axis="x", length=0)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          handlelength=1.2, columnspacing=1.4)
fig.tight_layout(rect=(0.01, 0.02, 0.99, 0.98))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "fig_kernels_success")
