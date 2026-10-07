"""Sampler diagnostics: roadmap tilt versus objective tilt, per iteration.

Three rows per environment:
  (1) realised inverse temperature beta_k -- ESS-adaptive, so it is an output of
      the run rather than a setting;
  (2) effective sample size against the tau*N resampling threshold;
  (3) E_w[C] - E[C], the weighted-minus-uniform population mean of the ROADMAP
      cost. Both arms are scored on the same C, which is why the dump records
      V_roadmap before any substitution: otherwise the L2 arm would be scored on
      L2 and the two panels would not be comparable.

Two arms, eight seeds each, drawn as individual traces rather than a median --
following Kim et al. (arXiv:2503.15704, Figs. 26-28), whose diagnostics show the
realised schedules of a subset of runs. Seeds are stratified so the subset's
success rate matches the full 100-seed cell, so these are a fair sample.

  pde : tilt on the roadmap cost C (ours)
  l2  : tilt on the planning objective, rescaled to C's per-step mean and
        standard deviation, so only the ORDERING of particles differs.

x axis: the planner evaluates and steps the environment inside its optimisation
loop, so iteration index == environment step; a trace ends at success or the cap.
"""
import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import use_style, save, PINK_DARK, GREY_DARK, NAVY, INK  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "sweep_aft/fk_dumps")
TAU = 0.5
ARMS = [("pde", "roadmap tilt (ours)", PINK_DARK),
        ("l2", "objective tilt", GREY_DARK)]
ENVS = [("Point maze", "pointmaze"), ("Push-T", "pusht"),
        ("Reacher", "reacher"), ("Cube", "cube")]


def traces(arm, env):
    out = []
    for f in sorted(glob.glob(f"{ROOT}/{arm}/{env}/seed_*/trace_call000.npz")):
        d = np.load(f)
        if "V_roadmap" not in d:
            continue
        w, Vr = d["w"], d["V_roadmap"]
        out.append(dict(beta=d["beta"], ess=1.0 / np.sum(w ** 2, axis=1),
                        dEV=np.sum(w * Vr, axis=1) - Vr.mean(axis=1),
                        N=w.shape[1]))
    return out


have = [(t, e) for t, e in ENVS if traces("pde", e)]
if not have:
    raise SystemExit("no traces yet")

use_style()
fig, axes = plt.subplots(3, len(have), figsize=(3.1 * len(have), 5.0),
                         squeeze=False)
for col, (title, env) in enumerate(have):
    N = None
    for arm, label, colour in ARMS:
        for k, tr in enumerate(traces(arm, env)):
            N = tr["N"]
            x = np.arange(tr["beta"].size)
            for row, key in enumerate(("beta", "ess", "dEV")):
                axes[row][col].plot(x, tr[key], color=colour, linewidth=0.9,
                                    alpha=0.55,
                                    label=label if (k == 0 and row == 0) else None)
    axes[0][col].set_title(f"{title}  ($N{{=}}{N}$)", fontsize=9.5, color=INK, pad=4)
    axes[1][col].axhline(TAU * N, color=NAVY, linestyle=(0, (3, 2)), linewidth=1.0)
    axes[1][col].set_ylim(0, N * 1.05)
    axes[2][col].axhline(0.0, color=INK, linewidth=0.8)
    axes[2][col].set_xlabel("iteration", fontsize=8.5, color=INK)
    for row in range(3):
        for s in ("top", "right"):
            axes[row][col].spines[s].set_visible(False)

axes[0][0].set_ylabel(r"$\beta_k$", fontsize=9.5, color=INK)
axes[1][0].set_ylabel("ESS", fontsize=9.5, color=INK)
axes[2][0].set_ylabel(r"$E_w[C]-E[C]$", fontsize=9.5, color=INK)
axes[0][0].legend(frameon=False, fontsize=7.8, loc="best")
fig.subplots_adjust(left=0.10, right=0.99, top=0.93, bottom=0.085,
                    wspace=0.30, hspace=0.35)
save(fig, "fig_diagnostics")

for title, env in have:
    for arm, label, _ in ARMS:
        tr = traces(arm, env)
        if tr:
            fin = np.array([t["dEV"][-1] for t in tr])
            print(f"{title:11s} {arm:4s} n={len(tr)}  final E_w[C]-E[C]: "
                  f"mean={fin.mean():+.4f}  median={np.median(fin):+.4f}")
