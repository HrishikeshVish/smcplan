"""Offline comparison of the operator that orders the roadmap vertices, on the
criterion used to choose k and c (writeup/calib_pde.py): the mean over 60 goal
vertices of Spearman(V(i), tau_g - tau_i) across the earlier vertices i of the
goal's episode, i.e. agreement of V with the observed number of steps to the
goal. Same goals as calib_pde.py; each environment at its calibrated (k, c);
no planning seed, world-model rollout or success signal enters.

Operators (all on the same roadmap; planning/gd.py MCMollifiedHeat):
  euclidean       distance to the goal in roadmap coordinates (the L2 analogue)
  heat            u of (I + tL)u = e_g, t = c / median D_ii (the planner's)
  hop             number of edges to the goal (log u / log t as t -> 0)
  green           Green's function L^+ e_g (the ordering of u as t -> infinity)
  shortest_path   Dijkstra distance with edge lengths |xi_i - xi_j|
  p<p>            p-torsion: sum_j w_ij |u_i-u_j|^(p-2) (u_i-u_j) = 1 at i != g,
                  u_g = 0, the point-source counterpart of the p-Laplacian on
                  the roadmap weights; p = 2 orders as green (L u = 1 - N e_g),
                  and p -> infinity tends to the hop count (w^(1/p) -> 1)
Paired differences heat - operator are over the same goals.

With --heldout the 60 goals are drawn from the remaining candidates, disjoint
from those on which k and c were chosen.

Usage: CUDA_VISIBLE_DEVICES= python writeup/calib_operators.py [--heldout] [env ...]
Writes sweep_emsmc/calib_operators[_heldout]_<env>.csv
"""
import csv
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

sys.path.insert(0, "/home/hviswan/Documents/grasp")
sys.path.insert(0, "/home/hviswan/Documents/grasp/writeup")
from calib_pde import ENVS, MIN_PAIRS, N_GOALS, episodes, solve_V  # noqa: E402
from planning.gd import MCMollifiedHeat, _load_replay_buffer  # noqa: E402

CALIB = {"pointmaze": (4, 1.0), "wall": (32, 30.0), "pusht": (64, 10.0),
         "reacher": (16, 3.0), "cube": (16, 1.0)}      # run_emsmc_final.sh, cube config
PS = (4.0, 8.0)
HELDOUT = "--heldout" in sys.argv


def ptorsion(I, J, w, lab, g, p, N):
    """Minimizer of (1/p) sum_e w_e |u_i - u_j|^p - sum_i u_i with u_g = 0 on
    the goal's component (strictly convex); +inf elsewhere."""
    C = lab == lab[g]
    m = C[I] & C[J]
    Ii, Jj, ww = I[m], J[m], w[m]
    free = C.copy()
    free[g] = False

    def f(x):
        u = np.zeros(N)
        u[free] = x
        d = u[Ii] - u[Jj]
        ad = np.abs(d)
        gr_e = ww * ad ** (p - 2) * d
        gr = np.bincount(Ii, gr_e, N) - np.bincount(Jj, gr_e, N)
        return (ww * ad ** p).sum() / p - x.sum(), gr[free] - 1.0

    r = minimize(f, np.zeros(free.sum()), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=20000, maxfun=40000, gtol=1e-9, ftol=1e-15))
    u = np.full(N, np.inf)
    u[C] = 0.0
    u[free] = r.x
    return u, r.nit, bool(r.success)


def run(env):
    path, n = ENVS[env]
    k, c = CALIB[env]
    z, zn, _ = _load_replay_buffer(path, "cpu")
    ep, tt = episodes(z, zn)
    h = MCMollifiedHeat()
    h.compile_environment(z_buffer=z, z_predicted_next=zn, n_nodes=n, K_neighbors=k,
                          solver_mode="heat")
    N = h.L_sym.shape[0]
    L = h.L_sym.tocsr()
    t = c / float(np.median(h.degree))
    W = (-L).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    _, lab = connected_components(W, directed=False)
    Wu = sp.triu(W).tocoo()
    I, J, w = Wu.row, Wu.col, Wu.data.astype(np.float64)

    # the goals of calib_pde.py (node_idx does not depend on k)
    e_n, t_n = ep[h.node_idx], tt[h.node_idx]
    cand = []
    for g in range(len(e_n)):
        pre = np.where((e_n == e_n[g]) & (t_n < t_n[g]))[0]
        if len(pre) >= MIN_PAIRS:
            cand.append((g, pre))
    sel = np.random.RandomState(0).choice(len(cand), min(N_GOALS, len(cand)), replace=False)
    if HELDOUT:   # goals not used to choose k and c
        rest = np.setdiff1d(np.arange(len(cand)), sel)
        sel = np.random.RandomState(1).choice(rest, min(N_GOALS, len(rest)), replace=False)
    goals = [cand[i] for i in sel]
    print(f"\n== {env}{' (held out)' * HELDOUT}: N={N} k={k} c={c} components={lab.max() + 1} "
          f"goals={len(goals)}",
          flush=True)

    ops = ["euclidean", "heat", "hop", "green", "shortest_path"] + [f"p{p:g}" for p in PS]
    rho = {o: [] for o in ops}
    secs = {o: [] for o in ops}
    nits, fails = {}, {}
    for gi, (g, pre) in enumerate(goals):
        h.goal_idx = g
        dtau = tt[h.node_idx[g]] - tt[h.node_idx[pre]]
        for o in ops:
            t0 = time.perf_counter()
            if o == "euclidean":
                V = np.linalg.norm(h.z_buffer_np - h.z_buffer_np[g], axis=1)
            elif o == "heat":
                V = solve_V(L, g, t)[0]
            elif o.startswith("p") and o[1:].replace(".", "").isdigit():
                V, nit, ok = ptorsion(I, J, w, lab, g, float(o[1:]), N)
                nits.setdefault(o, []).append(nit)
                fails[o] = fails.get(o, 0) + (not ok)
            else:
                V = h._solve_operator(o)
            secs[o].append(time.perf_counter() - t0)
            rho[o].append(spearmanr(V[pre], dtau).correlation)
        if gi % 10 == 9:
            print(f"   {gi + 1} goals | " + " ".join(
                f"{o}={np.nanmean(rho[o]):.3f}" for o in ops), flush=True)

    out = open(f"/home/hviswan/Documents/grasp/sweep_emsmc/calib_operators"
               f"{'_heldout' * HELDOUT}_{env}.csv", "w", newline="")
    wr = csv.writer(out)
    wr.writerow(["env", "k", "c", "operator", "n_goals", "rho", "se", "diff_heat_minus_op",
                 "diff_se", "sec_per_goal", "mean_iters", "not_converged"])
    hv = np.array(rho["heat"])
    for o in ops:
        r = np.array(rho[o])
        ok = np.isfinite(r) & np.isfinite(hv)
        d = hv[ok] - r[ok]
        se = r[ok].std() / np.sqrt(ok.sum())
        dse = d.std() / np.sqrt(ok.sum())
        it = np.mean(nits[o]) if o in nits else ""
        print(f"   {o:14s} rho={r[ok].mean():.3f}+-{se:.3f} | heat-op={d.mean():+.3f}+-{dse:.3f}"
              f" | {np.mean(secs[o]):.2f}s/goal" + (f" | iters {it:.0f} fails {fails[o]}"
                                                   if o in nits else ""), flush=True)
        wr.writerow([env, k, c, o, int(ok.sum()), r[ok].mean(), se, d.mean(), dse,
                     np.mean(secs[o]), it, fails.get(o, "")])
    out.close()


if __name__ == "__main__":
    for env in ([a for a in sys.argv[1:] if not a.startswith("--")] or list(ENVS)):
        run(env)
