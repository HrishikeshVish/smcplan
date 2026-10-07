"""Figure 4: cost to first success, per environment.

Metric is optimiser iterations. Particle rollouts are NOT counted separately:
they are batched into a single GPU call, so a particle-weighted count does not
correspond to any cost the user pays -- wall clock per seed is comparable across
planners (~2 min).

Cube excludes TRIVIAL seeds (block already inside the 0.04 m success radius at
t=0): 24 of our 44 successes there exit at iteration 0 purely because the goal
was already met, which would drag every median to zero.

Box plots over SUCCESSFUL runs only -- a failed run has no convergence time, and
counting it at the iteration cap would mix "slow" with "never", which are
different claims. The count of successes behind each box is printed under it, so
a narrow box built on few runs cannot be mistaken for a strong result.

Exit step is read from the planner's own record: seed_success_log.txt for our
arms, and the exit_step field of logs.json for the baselines (via the Hydra run
directory named in each run.log).
"""
import os
import re
import sys
import glob
import json

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, rounded_frame, save, PINK, PINK_DARK, GREY,  # noqa: E402
                        GREY_DARK, GREY_LIGHT, INK)
METHOD_COLOR = {"gd": GREY_DARK, "grasp": GREY, "cem": "#B4B7C0", "ours": PINK}
METHOD_LABEL = {"gd": "GD", "grasp": "GRASP", "cem": "CEM", "ours": "ours"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def exit_from_logsjson(run_log):
    if not os.path.exists(run_log):
        return None
    m = re.search(r"Planning result saved dir: (\S+)", open(run_log, errors="ignore").read())
    if not m:
        return None
    lj = os.path.join(m.group(1), "logs.json")
    if not os.path.exists(lj):
        return None
    last = None
    for line in open(lj):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if "exit_step" in e:
            last = e
    return (int(last["exit_step"]), float(last.get("final_eval/success_rate", 0.0))) if last else None


def cube_nontrivial():
    """Seeds where the block must actually move more than the success radius."""
    keep = set()
    for p in glob.glob(f"{ROOT}/sweep_cube_leflow/fk/seed_*/traj.npz"):
        s = int(os.path.basename(os.path.dirname(p)).replace("seed_", ""))
        z = np.load(p, allow_pickle=True)
        a = np.asarray(z["state_0"]).ravel()
        g = np.asarray(z["state_g"]).ravel()
        if a.size >= 17 and np.linalg.norm(g[14:17] - a[14:17]) > 0.04:
            keep.add(s)
    return keep


def collect(pattern, use_seedlog, keep_seeds=None):
    """-> list of exit steps on successful runs."""
    out = []
    for d in sorted(glob.glob(pattern)):
        if keep_seeds is not None:
            sd = int(os.path.basename(d).replace("seed_", ""))
            if sd not in keep_seeds:
                continue
        if use_seedlog:
            p = os.path.join(d, "seed_success_log.txt")
            if not os.path.exists(p):
                continue
            f = open(p).read().split()
            if len(f) >= 5 and float(f[0]) > 0.5:
                out.append(int(f[4]))
        else:
            r = exit_from_logsjson(os.path.join(d, "run.log"))
            if r and r[1] > 0.5:
                out.append(r[0])
    return out


# rollouts per optimiser iteration, from the configs used in each sweep
RPI = {
    ("reacher H=10", "gd"): 1, ("reacher H=10", "grasp"): 1,
    ("reacher H=10", "cem"): 30, ("reacher H=10", "ours"): 64,
    ("point-maze H=10", "gd"): 1, ("point-maze H=10", "grasp"): 1,
    ("point-maze H=10", "cem"): 30, ("point-maze H=10", "ours"): 3,
    ("Cube", "gd"): 8, ("Cube", "grasp"): 1,
    ("Cube", "cem"): 300, ("Cube", "ours"): 256,
}

PANELS = [
    ("reacher H=10", [
        ("gd",    f"{ROOT}/sweep/reacher/gd_H10/seed_*", False),
        ("grasp", f"{ROOT}/sweep/reacher/grasp_H10/seed_*", False),
        ("cem",   f"{ROOT}/sweep/reacher/cem_H10/seed_*", False),
        ("ours",  f"{ROOT}/sweep_fk/table/reacher/fk_H10/seed_*", True),
    ]),
    ("point-maze H=10", [
        ("gd",    f"{ROOT}/sweep/pointmaze/gd_H10/seed_*", False),
        ("grasp", f"{ROOT}/sweep/pointmaze/grasp_H10/seed_*", False),
        ("cem",   f"{ROOT}/sweep/pointmaze/cem_H10/seed_*", False),
        ("ours",  f"{ROOT}/sweep_fk/table/pointmaze/fk_H10/seed_*", True),
    ]),
    ("cube", [
        ("gd",    f"{ROOT}/sweep_cube_leflow/gd/seed_*", True),
        ("grasp", f"{ROOT}/sweep_cube_leflow/grasp/seed_*", True),
        ("cem",   f"{ROOT}/sweep_cube_leflow/mpc_cem/seed_*", True),
        ("ours",  f"{ROOT}/sweep_cube_leflow/fk/seed_*", True),
    ]),
]


def main(metric="iterations"):
    use_style(7)
    keep = cube_nontrivial()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.2, 2.1))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.2, wspace=0.25)
    rng = np.random.default_rng(0)
    for ax, (title, arms) in zip(np.atleast_1d(axes), PANELS):
        data, labels, colors, counts = [], [], [], []
        for key, pat, seedlog in arms:
            v = collect(pat, seedlog, keep if title == "cube" else None)
            if metric == "rollouts":
                v = [x * RPI[({"reacher H=10": "Reacher $H{=}10$", "point-maze H=10": "Point maze $H{=}10$", "cube": "Cube"}[title], key)] for x in v]
            data.append(v if v else [np.nan])
            labels.append(METHOD_LABEL[key])
            colors.append(METHOD_COLOR[key])
            counts.append(len(v))
        for k, (d, c) in enumerate(zip(data, colors)):
            d = np.asarray(d, float)
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            ax.scatter(k + 1 + rng.uniform(-0.2, 0.2, len(d)), d, s=4, color=c,
                       alpha=0.55, lw=0, zorder=2)
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color=INK, lw=1.3),
                        whiskerprops=dict(color=INK, lw=0.7), capprops=dict(color=INK, lw=0.7),
                        boxprops=dict(lw=0.8))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
            patch.set_edgecolor(INK)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_title(title, pad=3)
        ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labels, counts)], fontsize=6)
        for t, c in zip(ax.get_xticklabels(), colors):
            if c == PINK:
                t.set_color(PINK_DARK)
        ax.tick_params(axis="x", length=0)
        print(f"{title}: " + "  ".join(
            f"{l}={np.nanmedian(d):.0f} (n={n})" if n else f"{l}=-"
            for l, d, n in zip(labels, data, counts)))
    lab = ("world-model rollouts to first success" if metric == "rollouts"
           else "iterations to success")
    np.atleast_1d(axes)[0].set_ylabel(lab)
    if metric == "rollouts":
        for ax in np.atleast_1d(axes):
            ax.set_yscale("symlog", linthresh=10)
    rounded_frame(fig, 0.004, 0.01, 0.992, 0.98, r=0.035)
    fig.text(0.5, 0.95, "planner iterations to first success (solved seeds)", ha="center",
             va="center", fontsize=8.5)
    name = "fig4_rollouts" if metric == "rollouts" else "fig4_iterations"
    save(fig, name)
    print("wrote", name)


if __name__ == "__main__":
    main("iterations")
