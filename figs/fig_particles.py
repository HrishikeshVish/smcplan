"""Particle count sweep: success and convergence against population size.

Reacher $H{=}10$, the 20-seed stratified subset, $N\\in\\{1,2,4,8,16,32\\}$ with
every other setting at its default ($N{=}8$). Run on Reacher rather than Point
maze because DINO-WM encodes a frame as 196 ViT patch tokens and caps the
population at three particles, so $N{=}16$ and $N{=}32$ are not expressible
there; Reacher is the LeWM path, one 192-d token per frame, and is also the
environment the kernel block was tuned on.

$N{=}1$ is the degenerate case in which no resampling is possible, so the gap to
$N>1$ is attributable to the population rather than to the tilt.

Only success is plotted. Mean iterations to success over the same sweep runs
59, 75, 55, 79, 58, 64 for the six settings -- no trend in $N$, and a sawtooth
that would invite a reader to find one. The convergence-speed claim is not
supported by this sweep and is not made.

Log-scaled $N$ axis: the sweep doubles, so a linear axis would compress the
small-$N$ region where the effect lives.
"""
import glob
import os
import sys
import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, save, rounded_frame, PINK, PINK_DARK,  # noqa: E402
                        GREY_DARK, INK)

ROOT = "/home/hviswan/Documents/grasp/sweep_aft/abl_particles"
DEFAULT_N = 8


def first_pass(sd):
    try:
        lines = [l for l in open(sd + "/run.log", errors="ignore") if "Success rate:" in l]
    except OSError:
        return None
    for i, l in enumerate(lines):
        if "1.0" in l.split("Success rate:")[1]:
            return i
    return None


g = collections.defaultdict(lambda: {"ok": [], "it": []})
for f in glob.glob(ROOT + "/*/seed_*/result.txt"):
    n = int(f.split("/")[-3])
    d = f.rsplit("/", 1)[0]
    good = open(f).read().split()[0] == "1"
    g[n]["ok"].append(good)
    if good:
        v = first_pass(d)
        if v is not None:
            g[n]["it"].append(v)

Ns = sorted(g)
succ = [100 * sum(g[n]["ok"]) / len(g[n]["ok"]) for n in Ns]
iters = [np.mean(g[n]["it"]) if g[n]["it"] else np.nan for n in Ns]

use_style()
fig, ax = plt.subplots(figsize=(3.05, 1.85))
ax.plot(Ns, succ, "-o", color=PINK_DARK, mfc=PINK, mec=PINK_DARK, mew=1.3,
        ms=5, lw=1.8, zorder=3)
ax.axvline(DEFAULT_N, color=GREY_DARK, lw=1.0, ls=":", zorder=1)
ax.annotate("default", (DEFAULT_N, 6), textcoords="offset points",
            xytext=(4, 0), ha="left", va="bottom", color=GREY_DARK, fontsize=6)
ax.set_xscale("log", base=2)
ax.set_xticks(Ns)
ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
ax.set_xlabel("particles $N$", color=INK)
ax.set_ylabel("success rate", color=INK)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.set_ylim(0, 108)
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
rounded_frame(fig, 0.008, 0.008, 0.984, 0.984)
save(fig, "abl_particles")
for n, s, i in zip(Ns, succ, iters):
    print(f"  N={n:<3d} success {s:5.1f}%  mean iters {i:5.1f}")
