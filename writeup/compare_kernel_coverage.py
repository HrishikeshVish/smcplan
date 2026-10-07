"""Rank kernels by latent-space coverage, then check whether it transmits.

AFT (Arbel et al. 2021) reports the variance of the normalizing-constant
estimator, and -alpha*log Z genuinely is a soft-minimum of the cost over the
proposal's region. But the PATH-ACCUMULATED log Zhat is unusable for this
planner: mu_k is fit to the particles by the M-step, so each mu-increment is
positive by construction and the estimate is wildly biased -- measured ~130
against a hard bound of ~1.8 (C >= 0 implies Z <= 1, so log[Z_K/Z_0] <=
|log Z_0| ~ mean(C)/alpha). Adaptive-target SMC has no unbiased log Z estimator.

RANK ON Cmin (or on a COLD soft-min), not on the soft-min at the planner's own
alpha. alpha = std(C) is a sampling temperature; at it the soft-min is dominated
by the population bulk and behaves like a mean, which is the wrong functional
because the planner executes ONE sequence -- the argmin. Measured on seed 19:
MALA had the better softminC at every step and FAILED, while HMC had the worse
softminC and SUCCEEDED, its single best particle reaching 0.53 against MALA's
1.13 at the deciding step.

Coverage is measured PATH-FREE from the current population:
    softminC = -alpha * log( (1/P) sum_i exp(-C_i/alpha) )   (lower is better)
which is bounded by [min_i C_i, mean_i C_i], tends to min_i C_i as alpha -> 0,
and cannot drift with the target. Cmin = min_i C_i is reported alongside.

Reported per kernel: softminC and Cmin (coverage), their spread across seeds
(AFT's reliability idea, applied to a valid estimator), final ESS, acceptance,
and success -- so a kernel that wins coverage WITHOUT winning success is visible
as such. That gap is the p(s|z) channel, since coverage is a latent-space
quantity and success is not.

Usage: python writeup/compare_kernel_logz.py <dir-of-kernel-arms>
       (expects <dir>/<kernel>/seed_*/run.log)
"""
import os
import re
import sys

import numpy as np


def scrape(log):
    t = open(log, errors="ignore").read()
    sm = re.findall(r"softminC=([0-9.e+-]+)", t)
    sk = re.findall(r"softminCold=([0-9.e+-]+)", t)
    cm = re.findall(r"Cmin=([0-9.e+-]+)", t)
    ess = re.findall(r"ESS=([0-9.]+)/([0-9]+)", t)
    acc = re.findall(r"(?:mala|rwm|hmc|ula) acc=([0-9.]+)", t)
    ok = "Success rate:  1.0" in t
    return (float(sk[-1]) if sk else (float(sm[-1]) if sm else np.nan),
            float(cm[-1]) if cm else np.nan,
            (float(ess[-1][0]) / float(ess[-1][1])) if ess else np.nan,
            float(acc[-1]) if acc else np.nan, ok)


def main(root):
    print(f"{'kernel':>7} {'n':>4} {'success':>9} {'softminC':>10} {'sd':>7} "
          f"{'Cmin':>8} {'ESS frac':>9} {'accept':>7}")
    for k in sorted(os.listdir(root)):
        d = os.path.join(root, k)
        if not os.path.isdir(d):
            continue
        rows = [scrape(os.path.join(d, s, "run.log"))
                for s in sorted(os.listdir(d))
                if os.path.exists(os.path.join(d, s, "run.log"))]
        if not rows:
            continue
        sm = np.array([r[0] for r in rows], float)
        cm = np.array([r[1] for r in rows], float)
        es = np.array([r[2] for r in rows], float)
        ac = np.array([r[3] for r in rows], float)
        sc = np.array([r[4] for r in rows], float)
        if np.all(np.isnan(sm)):
            print(f"{k:>7} {len(rows):>4} {sc.sum():>4.0f}/{len(rows):<4} "
                  f"{'(no softminC -- run predates the metric)':>44}")
            continue
        print(f"{k:>7} {len(rows):>4} {sc.sum():>4.0f}/{len(rows):<4} "
              f"{np.nanmean(sm):>10.4f} {np.nanstd(sm):>7.4f} "
              f"{np.nanmean(cm):>8.4f} {np.nanmean(es):>9.2f} "
              f"{np.nanmean(ac):>7.3f}")
    print("\nsoftminC is LATENT-space coverage (lower is better). A kernel that wins")
    print("here without winning success has not been shown to help planning -- that")
    print("gap is p(s|z), measured by writeup/analyze_emsmc_oracle.py.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "/home/hviswan/Documents/grasp/sweep_emsmc/failed_kernels")
