"""Per-step cost-to-go advantage over the raw-L2 tilt, paired within seed.

Both arms of a pair run the same seed, so they share the start and the goal and
therefore share c_0 exactly (verified: 0 mismatches in 100 pairs). The paired
difference

    delta(t) = 100 * [ c_L2(t) - c_roadmap(t) ] / c_0

therefore cancels the between-seed geometry that dominates the marginal curves,
and is the low-variance way to ask whether the roadmap cost produces a better
path. delta > 0 means the roadmap is closer to the goal at that point of the
episode. No split by outcome: all informative pairs, as the population.

Shaded band is a bootstrap 95% CI of the MEDIAN across pairs (4000 resamples),
which is the uncertainty the claim is about. Where the band covers zero there is
no per-step advantage, and the figure is drawn to make that readable rather than
hide it.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costtogo as C                                                # noqa: E402
from iclr_style import (use_style, save, rounded_frame, PINK, PINK_LIGHT,  # noqa: E402
                        PINK_DARK, GREY_DARK, INK)

GRID = np.linspace(0, 1, 60)
BOOT = 4000


def paired(data, rng):
    D = np.array([(np.interp(GRID, np.linspace(0, 1, len(r["c_rawl2"])), r["c_rawl2"])
                   - np.interp(GRID, np.linspace(0, 1, len(r["c_roadmap"])), r["c_roadmap"]))
                  * 100.0 / r["c_roadmap"][0]
                  for r in data if r["c_roadmap"][0] > 0])
    med = np.median(D, axis=0)
    bs = np.array([np.median(D[rng.integers(0, len(D), len(D))], axis=0) for _ in range(BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return D, med, lo, hi


def panel(name, env, data, rng):
    D, med, lo, hi = paired(data, rng)
    fig, ax = plt.subplots(figsize=(3.05, 2.15))
    ax.axhline(0, color=GREY_DARK, lw=1.2, zorder=2)
    ax.fill_between(GRID, lo, hi, color=PINK_LIGHT, alpha=0.85, lw=0, zorder=1)
    ax.plot(GRID, med, color=PINK_DARK, lw=2.0, zorder=3)
    sig = (lo > 0) | (hi < 0)
    if sig.any():
        ax.plot(GRID[sig], np.full(sig.sum(), ax.get_ylim()[0]), color=PINK_DARK,
                lw=3.0, solid_capstyle="butt", zorder=4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("episode progress", color=INK)
    ax.set_ylabel("cost-to-go advantage", color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title(f"{name}   n={len(D)}", color=INK, fontweight="bold", pad=6)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, f"costtogo_paired_{env}")
    return name, len(D), float(np.median(D.mean(1))), sig.mean()


if __name__ == "__main__":
    use_style()
    rng = np.random.default_rng(0)
    for name, env in C.ENVS:
        print("%-12s n=%3d  integrated %+.2f%%  significant timesteps %3.0f%%"
              % panel(name, env, C.load(env), rng))
