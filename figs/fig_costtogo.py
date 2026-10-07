"""Main-text figures for the cost-to-go claim.

The claim: a good path is one along which the cost-to-go keeps falling. We
measure it on the ENVIRONMENT's own cost-to-go -- the true geodesic through the
point maze and around the wall, and the block's distance to its goal in Push-T
-- never on the roadmap's own V, which would be circular. See costtogo.py for
the definition of the progress fraction and of each environment's geometry.

Two figures, one per claim:
  profiles   how the cost-to-go actually evolves (the mechanism)
  winloss    per-pair wins on the progress fraction, split by outcome (the count)

House rule: ours is the only coloured series; baselines grey; all text black.
Panels are drawn at 3.05in and included near 0.32\textwidth, so print scales
them by ~0.67 and type is pre-multiplied to land near 9-10pt.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costtogo as C                                                # noqa: E402
from iclr_style import (use_style, save, rounded_frame, PINK, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY, GREY_LIGHT, GREY_DARK, INK)

GRID = np.linspace(0, 1, 60)


def resample(c):
    """Cost-to-go on a common normalised time axis, scaled by its start value."""
    c = np.asarray(c, float)
    if not np.isfinite(c[0]) or c[0] <= 0:
        return None
    return np.interp(GRID, np.linspace(0, 1, len(c)), c / c[0])


def profiles(name, env, data):
    fig, ax = plt.subplots(figsize=(3.05, 2.15))
    for arm, col, edge, lab in (("rawl2", GREY_LIGHT, GREY_DARK, "$L_2$ tilt"),
                                ("roadmap", PINK_LIGHT, PINK_DARK, "roadmap (ours)")):
        cur = [resample(r["c_" + arm]) for r in data]
        cur = np.array([x for x in cur if x is not None])
        lo, md, hi = np.percentile(cur, [25, 50, 75], axis=0)
        ax.fill_between(GRID, lo, hi, color=col, alpha=0.75, lw=0, zorder=1)
        ax.plot(GRID, md, color=edge, lw=2.0, zorder=3, label=lab)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("episode progress", color=INK)
    ax.set_ylabel("cost-to-go / start", color=INK)
    ax.set_title(f"{name}   n={len(data)}", color=INK, fontweight="bold", pad=6)
    ax.legend(loc="upper right", handlelength=1.4, borderpad=0.2)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, f"costtogo_profile_{env}")


def winloss(rows):
    """Diverging bars: right = roadmap makes more consistent progress, left = L2."""
    labs, W, L = [], [], []
    for name, env, data in rows:
        for st, tag in (("both", "both solve"), ("l2_fails", "$L_2$ fails")):
            s = [r for r in data if r["stratum"] == st]
            labs.append(f"{name} - {tag}")
            W.append(sum(1 for r in s if r["desc_roadmap"] > r["desc_rawl2"]))
            L.append(sum(1 for r in s if r["desc_roadmap"] < r["desc_rawl2"]))
    W, L = np.array(W), np.array(L)
    y = np.arange(len(labs))[::-1]
    fig, ax = plt.subplots(figsize=(3.05, 2.55))
    ax.barh(y, W, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.3, height=0.55)
    ax.barh(y, -L, color=GREY_LIGHT, edgecolor=GREY_DARK, lw=1.3, height=0.55)
    for yi, (w, l) in zip(y, zip(W, L)):
        if w:
            ax.text(w + 0.6, yi, str(w), va="center", ha="left", color=INK, fontweight="bold")
        if l:
            ax.text(-l - 0.6, yi, str(l), va="center", ha="right", color=INK, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, color=INK)
    ax.tick_params(left=False)
    ax.set_xlim(-max(L.max(), 1) * 1.9, W.max() * 1.42)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(False)
    # Title and caption are figure-level: the axes is pushed right by the long
    # row labels, so an axes-level title would sit off-centre over the page.
    fig.text(0.5, 0.955, "cost-to-go progress", ha="center", va="top",
             color=INK, fontweight="bold")
    fig.text(0.5, 0.035, "pairs won  ($L_2$ | ours)", ha="center", va="bottom",
             color=INK)
    fig.tight_layout(rect=(0.02, 0.085, 0.98, 0.915))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, "costtogo_winloss")


if __name__ == "__main__":
    use_style()
    rows = [(n, e, C.load(e)) for n, e in C.ENVS]
    for n, e, d in rows:
        profiles(n, e, d)
    winloss(rows)
    print("wrote costtogo_profile_{pointmaze,pusht,wall} and costtogo_winloss")
