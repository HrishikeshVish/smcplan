"""Paired PDE-roadmap-cost vs L2 comparison for the baseline planners.

Reviewer WT5H asked whether the cost of Sec. 3.4 also helps the competing
planners. For each (environment, horizon, planner) we have two arms that share
the seed, hence the same start and goal, so the comparison is paired:

    sweep/<env>/<arm>_H<H>        objective_fn = L2 only
    sweep/<env>/<arm>_pde_H<H>    objective_fn = L2 + lambda_V * C_V

The paired statistic is the difference in success rate over the seeds that both
arms completed, in percentage points. Its CI is a seed-level bootstrap (paired:
a resample draws seeds, not arms) and its p-value is the exact two-sided
McNemar test on the discordant pairs -- the binomial tail on b and c, where

    b = seeds the roadmap cost wins (PDE success, L2 failure)
    c = seeds it loses

GRASP is absent on purpose and not for lack of compute: `GRASPPlanner` never
calls `objective_fn` (its `goal_loss` is identically zero, planning/grasp.py),
reaching the goal by pinning a terminal virtual state at it and penalising
dynamics inconsistency. A cost-to-go evaluated at that pinned state is zero by
construction, so there is nowhere for C_V to enter without changing the
planner into a different one.

Writes writeup/pde_baseline_transfer.csv, which figs/fig_pde_transfer.py reads.
"""
import argparse
import csv
import glob
import os
from math import comb

import numpy as np

# (env, horizon) cells, in the order they should appear on the figure.
CELLS = [("pointmaze", 10, "Point maze"),
         ("pusht", 5, "Push-T"),
         ("reacher", 15, "Reacher")]
# (arm directory prefix, label on the figure)
ARMS = [("gds12", "GD"), ("cem12", "CEM"), ("mppi12", "MPPI")]

SWEEP = "sweep"
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pde_baseline_transfer.csv")
BOOT = 10000


def load_arm(d):
    """{seed: 0/1} from <d>/seed_*/result.txt, whose first field is the success."""
    out = {}
    for f in glob.glob(os.path.join(d, "seed_*/result.txt")):
        seed = os.path.basename(os.path.dirname(f)).split("_")[1]
        try:
            out[int(seed)] = int(open(f).read().split()[0])
        except (ValueError, IndexError):
            continue                      # a truncated result.txt: treat as missing
    return out


def mcnemar_exact(b, c):
    """Two-sided exact McNemar: binomial(n, 1/2) tail on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2 * tail))


def bootstrap_ci(l2, pde, rng, n_boot=BOOT):
    """Percentile CI of the paired difference in success rate, in points."""
    d = np.asarray(pde, float) - np.asarray(l2, float)
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * float(lo), 100 * float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=SWEEP)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="skip a cell with fewer paired seeds than this")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for env, H, env_label in CELLS:
        for arm, arm_label in ARMS:
            l2 = load_arm(os.path.join(args.sweep, env, f"{arm}_H{H}"))
            pde = load_arm(os.path.join(args.sweep, env, f"{arm}_pde_H{H}"))
            seeds = sorted(set(l2) & set(pde))
            if len(seeds) < args.min_seeds:
                print(f"  skip {env_label:11s} {arm_label:5s} "
                      f"{len(seeds)} paired seeds")
                continue
            a_l2 = [l2[s] for s in seeds]
            a_pde = [pde[s] for s in seeds]
            n = len(seeds)
            b = sum(1 for s in seeds if pde[s] == 1 and l2[s] == 0)
            c = sum(1 for s in seeds if pde[s] == 0 and l2[s] == 1)
            lo, hi = bootstrap_ci(a_l2, a_pde, rng)
            rows.append(dict(
                env=env, env_label=env_label, horizon=H,
                planner=arm, planner_label=arm_label,
                n_paired=n,
                n_l2_total=len(l2), n_pde_total=len(pde),
                success_l2=round(100 * sum(a_l2) / n, 4),
                success_pde=round(100 * sum(a_pde) / n, 4),
                delta_pp=round(100 * (sum(a_pde) - sum(a_l2)) / n, 4),
                ci_lo_pp=round(lo, 4), ci_hi_pp=round(hi, 4),
                wins_pde=b, wins_l2=c,
                p_mcnemar=round(mcnemar_exact(b, c), 6),
            ))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} cells)\n")

    hdr = (f"{'cell':<16}{'arm':<6}{'n':>4}{'L2':>7}{'PDE':>7}{'delta':>8}"
           f"{'95% CI':>18}{'b':>4}{'c':>4}{'p':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['env_label']+' H'+str(r['horizon']):<16}{r['planner_label']:<6}"
              f"{r['n_paired']:>4}{r['success_l2']:>6.0f}%{r['success_pde']:>6.0f}%"
              f"{r['delta_pp']:>+8.1f}"
              f"{'[' + format(r['ci_lo_pp'], '+.1f') + ', ' + format(r['ci_hi_pp'], '+.1f') + ']':>18}"
              f"{r['wins_pde']:>4}{r['wins_l2']:>4}{r['p_mcnemar']:>9.3f}")


if __name__ == "__main__":
    main()
