"""Figure 5: does the roadmap cost-to-go add anything over the L2 objective?

Two panels from the Point maze H=10 ablation over the 38 seeds where gradient
descent fails and our method succeeds. Those seeds are the right test set
because gradient descent IS the L2 objective, so the objective alone provably
solves none of them.

  left   success rate, roadmap tilt vs L2 tilt
  right  the mechanism: cumulative selection KL (how hard the tilt actually
         pulls) and the number of seeds on which the tilt never acted at all

The right panel matters because the left one alone cannot distinguish "V carries
information" from "V happens to help here".
"""
import os
import re
import sys
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, rounded_frame, save, PINK, PINK_DARK, PINK_LIGHT,  # noqa: E402
                        GREY, GREY_DARK, GREY_LIGHT, INK)
from matplotlib.patches import FancyBboxPatch  # noqa: E402
OURS, L2 = PINK, GREY_DARK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = [("none", "roadmap V (ours)", OURS), ("objective_raw", "L2 tilt", L2)]


def load(arm):
    base = os.path.join(ROOT, "sweep_fk/ablate_np3", f"{arm}_H10")
    ok, kl = {}, {}
    for d in sorted(glob.glob(f"{base}/seed_*")):
        s = int(os.path.basename(d).replace("seed_", ""))
        r = os.path.join(d, "result.txt")
        if os.path.exists(r):
            ok[s] = int(open(r).read().split()[0])
        txt = open(os.path.join(d, "run.log"), errors="ignore").read()
        m = re.findall(r"cumKL=([\d.]+) nats", txt)
        if m:
            kl[s] = float(m[-1])
    return ok, kl


def main():
    use_style(7)
    data = {a: load(a) for a, _, _ in ARMS}
    seeds = sorted(set.intersection(*[set(data[a][0]) for a, _, _ in ARMS]))
    fig = plt.figure(figsize=(5.4, 1.8))

    # (left) one cell per seed, filled = solved
    ax = fig.add_axes([0.02, 0.10, 0.56, 0.70])
    ax.set_xlim(-0.2, 23.6); ax.set_ylim(-0.6, 6.6); ax.set_aspect("equal"); ax.set_axis_off()
    ncol = 19
    for r, (a_, lab, c) in enumerate(ARMS):
        ok = data[a_][0]
        n = sum(ok[s_] for s_ in seeds)
        y0 = 5.2 - r * 3.3
        ax.text(0.08, y0 + 0.2, lab, fontsize=7, color=PINK_DARK if c == PINK else GREY_DARK,
                va="bottom")
        ax.text(ncol + 2.3, y0 - 1.05, f"{n}/{len(seeds)}", fontsize=9, ha="center",
                va="center", color=PINK_DARK if c == PINK else GREY_DARK)
        for k, s_ in enumerate(seeds):
            i, j = divmod(k, ncol)
            solved = ok[s_]
            ax.add_patch(FancyBboxPatch((j + 0.08, y0 - i - 0.08 - 0.84), 0.84, 0.84,
                                        boxstyle="round,pad=0,rounding_size=0.18",
                                        fc=c if solved else "white",
                                        ec=INK if solved else GREY, lw=0.5,
                                        ls="-" if solved else (0, (1.5, 1.2))))
    fig.text(0.29, 0.93, f"point-maze H=10: {len(seeds)} paired seeds", ha="center",
             va="center", fontsize=8)

    # (right) cumulative selection KL
    ax = fig.add_axes([0.72, 0.2, 0.26, 0.62])
    for i, (a_, lab, c) in enumerate(ARMS):
        kl = np.array([data[a_][1][s_] for s_ in seeds if s_ in data[a_][1]])
        parts = ax.violinplot([kl], positions=[i], widths=0.75, showextrema=False)
        for b_ in parts["bodies"]:
            b_.set_facecolor(c); b_.set_alpha(0.3); b_.set_edgecolor(c)
        rng = np.random.default_rng(i)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(kl)), kl, s=5, color=c, lw=0, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [kl.mean()] * 2, color=INK, lw=1.2, zorder=4)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(["ours", "L2 tilt"])
    ax.get_xticklabels()[0].set_color(PINK_DARK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("selection KL (nats)")
    ax.set_ylim(bottom=-0.05)
    fig.text(0.85, 0.93, "tilt strength", ha="center", va="center", fontsize=8)
    rounded_frame(fig, 0.004, 0.01, 0.992, 0.98, r=0.04)
    save(fig, "fig5_ablation")
    print(f"wrote fig5_ablation ({len(seeds)} paired seeds)")
    for a_, lab, _ in ARMS:
        ok, kl = data[a_]
        v = [kl[s_] for s_ in seeds if s_ in kl]
        print(f"  {lab:<22} {sum(ok[s_] for s_ in seeds)}/{len(seeds)}   "
              f"mean cumKL {np.mean(v):.2f}   inert {sum(1 for s_ in seeds if kl.get(s_,0)<0.01)}")


if __name__ == "__main__":
    main()
