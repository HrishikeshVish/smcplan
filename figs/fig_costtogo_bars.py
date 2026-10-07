"""Average cost-to-go improvement over the raw-L2 tilt, as two bar charts.

Both arms of a pair share the seed, hence the same start, goal and c_0, so the
paired difference cancels the geometry:

    improvement = 100 * [ c_L2 - c_roadmap ] / c_0

Two charts, because the two readings answer different questions:
  episode   mean over the whole episode -- was the path better throughout
  terminal  at the final state -- did it end closer

Bars are the MEAN (the "average improvement"); whiskers are a bootstrap 95% CI
of the mean, 4000 resamples. The whiskers are not decoration: the episode-mean
CI covers zero in all three environments, and only the terminal bars clear it.
The median is printed on each bar because the mean is pulled by a heavy tail --
pairs where one arm fails outright contribute improvements near +80%, so mean
and median disagree in sign for Push-T.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import costtogo as C                                                # noqa: E402
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,  # noqa: E402
                        PINK_DARK, GREY_DARK, INK)

GRID = np.linspace(0, 1, 60)
BOOT = 4000


def _rs(c):
    return np.interp(GRID, np.linspace(0, 1, len(c)), np.asarray(c, float))


QTY = {"episode": lambda r: np.mean(_rs(r["c_rawl2"]) - _rs(r["c_roadmap"])),
       "terminal": lambda r: r["c_rawl2"][-1] - r["c_roadmap"][-1]}


def chart(which, title, rng):
    names, mean, med, lo, hi, ns = [], [], [], [], [], []
    for name, env in C.ENVS:
        d = [r for r in C.load(env) if r["c_roadmap"][0] > 0]
        A = np.array([100 * QTY[which](r) / r["c_roadmap"][0] for r in d])
        bs = np.array([A[rng.integers(0, len(A), len(A))].mean() for _ in range(BOOT)])
        l, h = np.percentile(bs, [2.5, 97.5])
        names.append(name); ns.append(len(A))
        mean.append(A.mean()); med.append(np.median(A)); lo.append(l); hi.append(h)
    mean, med, lo, hi = map(np.array, (mean, med, lo, hi))

    fig, ax = plt.subplots(figsize=(3.05, 2.15))
    x = np.arange(len(names))
    ax.axhline(0, color=GREY_DARK, lw=1.2, zorder=2)
    ax.bar(x, mean, width=0.55, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.3, zorder=3)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor=INK,
                elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=4)
    for xi, (m, h, l) in zip(x, zip(med, hi, lo)):
        ax.text(xi, max(h, 0) + (hi.max() - min(lo.min(), 0)) * 0.04,
                f"med {m:+.1f}%", ha="center", va="bottom", color=INK, fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn={k}" for n, k in zip(names, ns)], color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("improvement over $L_2$", color=INK)
    ax.set_title(title, color=INK, fontweight="bold", pad=6)
    ax.margins(y=0.22)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, f"costtogo_bars_{which}")
    for n, k, m, md, l, h in zip(names, ns, mean, med, lo, hi):
        print(f"  {which:9s} {n:11s} n={k:3d} mean {m:+7.2f}% CI [{l:+7.2f}%, {h:+7.2f}%]"
              f" median {md:+7.2f}%  {'excludes 0' if l > 0 or h < 0 else 'spans 0'}")


if __name__ == "__main__":
    use_style()
    rng = np.random.default_rng(0)
    chart("episode", "cost-to-go, episode mean", rng)
    chart("terminal", "cost-to-go, final state", rng)
