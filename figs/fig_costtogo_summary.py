"""Integrated cost-to-go advantage over the raw-L2 tilt, one number per env.

Parameterising on episode progress spends most of the axis on the early phase,
where the two arms have not yet diverged and the paired difference is ~0 with an
arbitrary sign -- readable as "negative regions" that are only noise. This figure
integrates the paired difference over the episode instead, giving one effect per
environment with a bootstrap 95% CI of the median (4000 resamples).

    advantage = mean_t 100 * [ c_L2(t) - c_roadmap(t) ] / c_0

positive = the roadmap arm is closer to the goal through the episode. Both arms
of a pair share the seed and hence c_0 exactly, so the geometry cancels.

The CI is shown because it is the result: it spans zero in every environment.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costtogo as C                                                # noqa: E402
from iclr_style import (use_style, save, rounded_frame, PINK_DARK,  # noqa: E402
                        PINK_LIGHT, GREY_DARK, INK)

GRID = np.linspace(0, 1, 60)
BOOT = 4000


def advantage(data, rng):
    A = np.array([np.mean(np.interp(GRID, np.linspace(0, 1, len(r["c_rawl2"])), r["c_rawl2"])
                          - np.interp(GRID, np.linspace(0, 1, len(r["c_roadmap"])), r["c_roadmap"]))
                  * 100.0 / r["c_roadmap"][0]
                  for r in data if r["c_roadmap"][0] > 0])
    bs = np.array([np.median(A[rng.integers(0, len(A), len(A))]) for _ in range(BOOT)])
    return A, float(np.median(A)), *np.percentile(bs, [2.5, 97.5])


if __name__ == "__main__":
    use_style()
    rng = np.random.default_rng(0)
    rows = [(n,) + advantage(C.load(e), rng) for n, e in C.ENVS]

    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=GREY_DARK, lw=1.2, zorder=1)
    for yi, (name, A, med, lo, hi) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=PINK_DARK, lw=2.4, solid_capstyle="round", zorder=3)
        ax.plot([med], [yi], "o", ms=6, mfc=PINK_LIGHT, mec=PINK_DARK, mew=1.6, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}  n={len(r[1])}" for r in rows], color=INK)
    ax.tick_params(left=False)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_xlabel("cost-to-go advantage over $L_2$", color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, "costtogo_summary")
    for name, A, med, lo, hi in rows:
        print(f"{name:12s} n={len(A):3d}  {med:+.2f}%  95% CI [{lo:+.2f}%, {hi:+.2f}%]"
              f"  {'excludes 0' if lo > 0 or hi < 0 else 'spans 0'}")
