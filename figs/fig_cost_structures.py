"""Improvement from each cost structure, over the same SMC planner. 2026-10-07.

Both arms run the identical planner -- JALA-EM (Alg. 1: ULA move with Jarzynski
weights, 12 particles, adaptive beta <= 1, the score-function M-step of
Eq. (mstep), 150 iterations) on the same 100 seeds per cell -- and differ ONLY in
the cost that defines the target:

  L2   sweep_emsmc/table/<env>/em_H<H>   lambda_v = 0
  PDE  sweep_emsmc/table/<env>/pde_H<H>  calibrated roadmap cost-to-go
                                          (writeup/run_emsmc_pde.sh)

This replaces the earlier version, which drew both bars from writeup/composed.csv
(writeup/compose_all.py), a 25-seed composition that kept every L2 failure in the
roadmap-success stratum and so overstated the PDE gain. Cube is omitted: it has
no PDE run with this planner.
"""
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

NAME = {"pointmaze": "Point maze", "wall": "Wall", "pusht": "Push-T",
        "reacher": "Reacher", "cube": "Cube"}
ORDER = ["pointmaze", "wall", "pusht", "reacher"]
import glob

ROOT = "/home/hviswan/Documents/grasp"
sys.path.insert(0, f"{ROOT}/writeup")
from equal_particle_stats import wilson  # noqa: E402  (95% intervals for the error bars)


def _rate(pat):
    ok = n = 0
    for f in glob.glob(f"{ROOT}/{pat}/seed_*/result.txt"):
        v = open(f).read().split()
        if v:
            n += 1
            ok += (v[0] == "1")
    return 100.0 * ok / n, n


def _load():
    out = {}
    for e in ORDER:
        for h in (5, 10, 15):
            l2, n1 = _rate(f"sweep_emsmc/table/{e}/em_H{h}")
            rm, n2 = _rate(f"sweep_emsmc/table/{e}/pde_H{h}")
            assert n1 == n2 == 100, (e, h, n1, n2)
            out[f"{e}_H{h}"] = dict(env=e, H=h, roadmap=rm, l2=l2, status="complete",
                                    l2_ci=wilson(round(l2), 100), rm_ci=wilson(round(rm), 100))
    return out


D = _load()


def pooled(env, hs=(10, 15)):
    cs = [D[f"{env}_H{h}"] for h in hs if f"{env}_H{h}" in D]
    if not cs:
        return None
    return (sum(c["l2"] for c in cs) / len(cs),
            sum(c["roadmap"] for c in cs) / len(cs),
            any(c["status"] != "complete" for c in cs))


def bars(ax, labels, l2, rm, part, l2_ci=None, rm_ci=None):
    # `part` marks cells whose fail stratum is incomplete. No longer drawn on the
    # bars; the status column of writeup/composed.csv still records it.
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, l2, w, color=GREY_LIGHT, edgecolor=GREY_DARK, lw=1.1)
    ax.bar(x + w / 2, rm, w, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.1)
    tops = [max(a, b) for a, b in zip(l2, rm)]
    if l2_ci is not None:
        for off, ys, ci in ((-w / 2, l2, l2_ci), (w / 2, rm, rm_ci)):
            ax.errorbar(x + off, ys, yerr=[[y - lo for y, (lo, hi) in zip(ys, ci)],
                                           [hi - y for y, (lo, hi) in zip(ys, ci)]],
                        fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, capthick=0.7)
        tops = [max(a[1], b[1]) for a, b in zip(l2_ci, rm_ci)]
    for xi, (a, b, p, t) in enumerate(zip(l2, rm, part, tops)):
        ax.text(xi, t + 3, f"{b - a:+.0f}",
                ha="center", va="bottom", color=INK, fontsize=6.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, color=INK)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])          # pin: short panels otherwise drop the midpoint
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))


if __name__ == "__main__":
    use_style()

    # ---- main text: pooled over horizons, one pair per environment ----
    labs, l2, rm, part = [], [], [], []
    for e in ORDER:
        r = pooled(e)
        if r is None:
            continue
        labs.append(NAME[e]); l2.append(r[0]); rm.append(r[1]); part.append(r[2])
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    bars(ax, labs, l2, rm, part)
    ax.set_ylabel("success rate", color=INK)
    h = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT, ec=GREY_DARK, lw=1.1),
         plt.Rectangle((0, 0), 1, 1, fc=PINK_LIGHT, ec=PINK_DARK, lw=1.1)]
    ax.legend(h, ["$L_2$ objective", "latent cost structure"], loc="upper center",
              bbox_to_anchor=(0.5, -0.26), frameon=False, ncol=2,
              handlelength=1.1, columnspacing=1.4, borderpad=0.2)
    fig.tight_layout(rect=(0.02, 0.10, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, "cost_structures")

    # ---- main text: one panel per environment, in a single row ----
    # Split out of the single five-environment chart, which crowded five bar
    # pairs onto one axis and left no room for the horizon breakdown. One row of
    # five keeps every panel on the same success-rate axis, so bar heights are
    # comparable across environments, while the horizon split stays visible.
    fig, axes = plt.subplots(1, len(ORDER), figsize=(6.9, 1.62))
    for ax, e in zip(axes, ORDER):
        hs = [h for h in (5, 10, 15) if f"{e}_H{h}" in D]
        labs = [f"{h}" for h in hs]
        l2 = [D[f"{e}_H{h}"]["l2"] for h in hs]
        rm = [D[f"{e}_H{h}"]["roadmap"] for h in hs]
        part = [D[f"{e}_H{h}"]["status"] != "complete" for h in hs]
        bars(ax, labs, l2, rm, part, [D[f"{e}_H{h}"]["l2_ci"] for h in hs],
             [D[f"{e}_H{h}"]["rm_ci"] for h in hs])
        ax.set_title(NAME[e], color=INK, fontweight="bold", pad=4)
        if ax is axes[0]:
            ax.set_ylabel("success rate", color=INK)
        else:
            ax.set_yticklabels([])
        if len(labs) == 1:                      # Cube: one setting, no horizon axis
            ax.set_xlim(-0.7, 0.7)
            ax.set_xticklabels([""])
    h = [plt.Rectangle((0, 0), 1, 1, fc=GREY_LIGHT, ec=GREY_DARK, lw=1.1),
         plt.Rectangle((0, 0), 1, 1, fc=PINK_LIGHT, ec=PINK_DARK, lw=1.1)]
    fig.legend(h, ["$L_2$ objective", "latent cost structure"], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), frameon=False, ncol=2,
               handlelength=1.1, columnspacing=1.6)
    # No figure-level x label: the per-panel ticks are the horizons and the
    # caption says so, and a supxlabel collides with the legend at this height.
    fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.97))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, "cost_structures_row")

    # ---- appendix: resolved by horizon ----
    for H in (5, 10, 15):
        labs, l2, rm, part = [], [], [], []
        for e in ORDER:
            k = f"{e}_H{H}"
            if k not in D:
                continue
            labs.append(NAME[e]); l2.append(D[k]["l2"]); rm.append(D[k]["roadmap"])
            part.append(D[k]["status"] != "complete")
        if not labs:
            continue
        fig, ax = plt.subplots(figsize=(3.05, 1.9))
        bars(ax, labs, l2, rm, part)
        ax.set_ylabel("success rate", color=INK)
        ax.set_title(f"$H{{=}}{H}$", color=INK, fontweight="bold", pad=4)
        for t in ax.get_xticklabels():
            t.set_rotation(20); t.set_ha("right")
        fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.97))
        rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
        save(fig, f"cost_structures_h{H}")

    print("wrote cost_structures + cost_structures_h{5,10,15}")
    for e in ORDER:
        r = pooled(e)
        if r:
            print(f"  {NAME[e]:11s} L2 {r[0]:5.1f}%  latent {r[1]:5.1f}%  {r[1]-r[0]:+5.1f}")
