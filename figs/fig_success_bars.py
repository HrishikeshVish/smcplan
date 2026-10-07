"""Success of the roadmap tilt vs the raw-L2 tilt, all five environments.

Paired: both arms of every pair run the same seed, interleaved, so start and
goal are bit-identical. Pairs whose two arms selected the SAME action sequence
are excluded -- the tilt reweighted the population but the argmin picked the
same particle, so they are uninformative and would dilute both bars equally.

Cube and Reacher appear here even though they are excluded from the cost-to-go
analysis: success is well defined for them, a configuration-distance cost-to-go
is not.
"""
import glob
import os
import sys
from math import comb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK_LIGHT,   # noqa: E402
                        PINK_DARK, GREY_LIGHT, GREY_DARK, INK)

SWEEP = "/home/hviswan/Documents/grasp/sweep_aft/rawl2"
ENVS = [("Point maze", "pointmaze"), ("Push-T", "pusht"), ("Wall", "wall"),
        ("Reacher", "reacher"), ("Cube", "cube")]


def pairs(env):
    got = {}
    for arm in ("roadmap", "rawl2"):
        for f in glob.glob(f"{SWEEP}/{arm}/{env}/H*/seed_*/result.txt"):
            p = f.split("/")
            got.setdefault((p[-3], p[-2]), {})[arm] = (f.rsplit("/", 1)[0],
                                                       int(open(f).read().split()[0]))
    out = []
    for k, v in got.items():
        if len(v) != 2:
            continue
        a = [np.load(v[x][0] + "/traj.npz", allow_pickle=True)["actions"] for x in ("roadmap", "rawl2")]
        if np.array_equal(a[0], a[1]):
            continue
        out.append((v["roadmap"][1], v["rawl2"][1]))
    return out


def sign_p(w, l):
    n = w + l
    return sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** (n - 1) if n else 1.0


if __name__ == "__main__":
    use_style()
    names, R, L, ns, ps = [], [], [], [], []
    for name, env in ENVS:
        pr = pairs(env)
        if not pr:
            continue
        r = np.mean([a for a, _ in pr]) * 100
        l = np.mean([b for _, b in pr]) * 100
        w = sum(1 for a, b in pr if a > b); x = sum(1 for a, b in pr if b > a)
        names.append(name); R.append(r); L.append(l); ns.append(len(pr)); ps.append(sign_p(w, x))
        print(f"  {name:11s} n={len(pr):3d}  roadmap {r:5.1f}%  L2 {l:5.1f}%  discordant {w}-{x}  p={sign_p(w,x):.4f}")

    x = np.arange(len(names)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    ax.bar(x - wdt / 2, L, wdt, color=GREY_LIGHT, edgecolor=GREY_DARK, lw=1.3, label="$L_2$ tilt")
    ax.bar(x + wdt / 2, R, wdt, color=PINK_LIGHT, edgecolor=PINK_DARK, lw=1.3, label="roadmap (ours)")
    for xi, (r, p) in enumerate(zip(R, ps)):
        if p < 0.05:
            ax.text(xi + wdt / 2, r + 2.5, "*", ha="center", va="bottom",
                    color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn={k}" for n, k in zip(names, ns)], color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("success rate", color=INK)
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", handlelength=1.2, borderpad=0.2, ncol=1)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
    save(fig, "success_bars")
