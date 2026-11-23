"""Per-seed outcomes for the equal-particle comparison (main Table 1) and the
appendix ablations, read from the sweep directories. 2026-10-07.

Every number in the paper's planner comparison is computed here and nowhere
else, so the tables, figures and text read one source:

  success    result.txt field 1 (the planner's own write; any evaluation solved)
  iters      number of optimisation iterations before the first successful
             evaluation, i.e. 1 + the index of the first "Success rate: 1.0"
             line in run.log; every planner evaluates once per iteration, after
             its update (eval_every=1)

Arms:
  ours     JALA-EM, calibrated PDE cost, 12 particles: pdefix, except pde
           (score M-step) on Point maze under OURS_PM=score (default)
  gds12    GD, 12 particles (one sequence, gradient averaged over 12 perturbed copies)
  grasp12  GRASP, 12 parallel trajectories (num_parallel_trajectories=12)
  cem12    CEM, 12 samples per iteration, top-3 elites, 150 iterations
  gd/grasp 1 particle, 150 iterations (appendix ablation)
  cem      30 samples x 5 iterations, top-3 elites (appendix ablation)
  l2fix    JALA-EM, fixed setting, L2 cost (lambda_v=0): PDE-vs-L2 figure
  em, pde  JALA-EM with the score M-step, L2 and PDE cost (the 2026-09-30
           table, writeup/run_emsmc_table.sh, run_emsmc_pde.sh): M-step x cost
           appendix table
  mppi12   MPPI (stable_worldmodel solver), 12 samples, 150 iterations
  pdefix   JALA-EM, pathwise M-step, PDE cost: `ours` outside Point maze, and
           the reference of the appendix ablations, which ran from it
  ours_l2  the L2-cost counterpart of `ours` (same M-step per environment)

Form of the M-step of `ours` per environment (2026-10-11), ONE switch read by
every generator that uses `ours`/`ours_l2` (tables via outcomes(), figures via
arm_dir()): OURS_PM=score (default, the paper: score form on Point maze, arms
pde/em there; pathwise, pdefix/l2fix, elsewhere) or OURS_PM=pathwise
(pathwise everywhere).
  leflow12/leflow64, gcidm_ol/gcidm_cl  learned baselines (Reacher, Cube)
  abl_*    appendix ablations/sensitivity (ABLATE)
"""
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS = [("Reacher", "reacher"), ("Point maze", "pointmaze"),
        ("Wall", "wall"), ("Push-T", "pusht")]
HS = (5, 10, 15)
GRID = {"pdefix": "sweep_emsmc/table/{e}/pdefix_H{H}",
        "l2fix": "sweep_emsmc/table/{e}/l2fix_H{H}",
        "em": "sweep_emsmc/table/{e}/em_H{H}", "pde": "sweep_emsmc/table/{e}/pde_H{H}",
        "gds12": "sweep/{e}/gds12_H{H}", "grasp12": "sweep/{e}/grasp12_H{H}",
        "cem12": "sweep/{e}/cem12_H{H}",
        "gd": "sweep/{e}/gd_H{H}", "grasp": "sweep/{e}/grasp_H{H}",
        "cem": "sweep/{e}/cem_H{H}",
        "mppi12": "sweep/{e}/mppi12_H{H}",
        "leflow12": "sweep/{e}/leflow12_H{H}", "leflow64": "sweep/{e}/leflow64_H{H}",
        "gcidm_ol": "sweep/{e}/gcidm_ol_H{H}", "gcidm_cl": "sweep/{e}/gcidm_cl_H{H}"}
CUBE = {"pdefix": "sweep_cube_leflow/pdefix", "l2fix": "sweep_cube_leflow/l2fix",
        "em": "sweep_cube_leflow/em",   # score M-step, L2; no score-M-step PDE run on Cube
        "gds12": "sweep_cube_leflow/gds12", "grasp12": "sweep_cube_leflow/grasp12",
        "cem12": "sweep_cube_leflow/cem12", "gd": "sweep_cube_leflow/gd",
        "grasp": "sweep_cube_leflow/grasp", "cem": "sweep_cube_leflow/mpc_cem",
        "mppi12": "sweep_cube_leflow/mppi12",
        "leflow12": "sweep_cube_leflow/leflow12", "leflow64": "sweep_cube_leflow/leflow64",
        "gcidm_ol": "sweep_cube_leflow/gcidm_ol", "gcidm_cl": "sweep_cube_leflow/gcidm_cl"}
# Appendix ablations and sensitivity (writeup/run_ablations.sh): one setting of
# pdefix changed per arm, H=10, Point maze and Reacher.
ABLATE = "sweep_emsmc/ablate/{e}/abl_{name}_H10"

OURS_PM = os.environ.get("OURS_PM", "score")
assert OURS_PM in ("pathwise", "score"), OURS_PM
# environments whose `ours` uses the score form of the M-step
SCORE_ENVS = {"pointmaze"} if OURS_PM == "score" else set()


def resolve(arm, env):
    """The run arm behind `ours`/`ours_l2` in one environment; other arms as is."""
    if arm == "ours":
        return "pde" if env in SCORE_ENVS else "pdefix"
    if arm == "ours_l2":
        return "em" if env in SCORE_ENVS else "l2fix"
    return arm


def arm_dir(arm, env, H=5):
    """Directory (relative to ROOT) holding the seed_* runs of one cell."""
    if arm.startswith("abl_"):
        return ABLATE.format(e=env, name=arm[4:])
    arm = resolve(arm, env)
    return CUBE[arm] if env == "cube" else GRID[arm].format(e=env, H=H)


def first_pass(sd):
    """0-based index of the first successful evaluation in run.log, or None."""
    try:
        lines = [l for l in open(sd + "/run.log", errors="ignore")
                 if "Success rate:" in l]
    except OSError:
        return None
    for i, l in enumerate(lines):
        if "1.0" in l.split("Success rate:")[1]:
            return i
    return None


def outcomes(arm, env, H=5):
    """{seed: (success, iters or None)} over the finished seeds of one cell."""
    pat = arm_dir(arm, env, H)
    out = {}
    for sd in glob.glob(os.path.join(ROOT, pat, "seed_*")):
        rf = sd + "/result.txt"
        if not os.path.exists(rf):
            continue
        r = open(rf).read().split()
        if not r:
            continue
        ok = r[0] == "1"
        fp = first_pass(sd) if ok else None
        out[int(sd.rsplit("_", 1)[1])] = (ok, None if fp is None else fp + 1)
    return out


def cube_nontrivial():
    """Cube seeds whose block starts outside the 0.04 m success radius
    (writeup/make_cube_table.py: displacement from the fk arm's traj dumps)."""
    import numpy as np
    nt = set()
    for p in glob.glob(f"{ROOT}/sweep_cube_leflow/fk/seed_*/traj.npz"):
        s = int(os.path.basename(os.path.dirname(p)).replace("seed_", ""))
        z = np.load(p, allow_pickle=True)
        a = np.asarray(z["state_0"]).ravel()
        g = np.asarray(z["state_g"]).ravel()
        if a.size >= 17 and float(np.linalg.norm(g[14:17] - a[14:17])) > 0.04:
            nt.add(s)
    return nt


# ---------------------------------------------------------------- uncertainty
# 2026-10-07: 95% intervals for every success rate in the paper, and paired tests.
#   wilson(k, n)        Wilson score interval for a binomial proportion; unlike the
#                       Wald interval it stays informative at 0% and 100%.
#   mcnemar(b, c)       exact two-sided McNemar test on paired seeds: b = seeds
#                       only arm A solves, c = seeds only arm B solves; under H0
#                       b ~ Binomial(b + c, 1/2).
def wilson(k, n, z=1.959963984540054):
    import math
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(oa, ob):
    """(only-A, only-B, exact McNemar p) over the seeds both arms finished."""
    common = sorted(set(oa) & set(ob))
    b = sum(oa[s][0] and not ob[s][0] for s in common)
    c = sum(ob[s][0] and not oa[s][0] for s in common)
    return b, c, mcnemar(b, c)
