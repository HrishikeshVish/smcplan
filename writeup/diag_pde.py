"""Diagnose the roadmap heat solve per environment (no planning, no world model).

For each emsmc roadmap, build the graph exactly as GDPlanner does, with and
without the duplicate-neighbour fix, then solve (I + tL) u = e_g for fixed goal
vertices under two diffusion-time rules:
  bandwidth  t = (median_i sigma_i)^2           (current code / paper)
  degree     t = 1 / median_i D_ii               (tL dimensionless)
and report what the planner's cost-to-go V = rank(-log u) can see:
  u<=0      fraction of vertices the CG iterate leaves at or below zero; these
            all tie at the top rank, so V is flat there
  rho_hop   Spearman(V, hop distance to g). Crane et al. Lemma 1: as t -> 0 the
            ordering of u tends to hop distance, so this is the sanity check that
            the solve resolves the graph at all
  ecc       eccentricity of g (max hop distance)
Usage: CUDA_VISIBLE_DEVICES=1 python writeup/diag_pde.py [env ...]
"""
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import breadth_first_order, shortest_path
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, "/home/hviswan/Documents/grasp")
from planning.gd import MCMollifiedHeat, _load_replay_buffer

B = "/home/hviswan/Documents/grasp/latent_buffers/"
ENVS = {"pointmaze": (B + "pointmaze_buffer.pt", 3800),
        "wall": (B + "wall_buffer.pt", 12600),
        "pusht": (B + "pusht_offline_replay_buffer.pt", 55000),
        "reacher": (B + "lewm_reacher_fixed.pt", 55000),
        "cube": (B + "lewm_cube_buffer.pt", 55000)}


def solve(L, g, t):
    N = L.shape[0]
    b = np.zeros(N); b[g] = 1.0
    A = sp.eye(N) + t * L
    M = sp.diags(1.0 / (A.diagonal() + 1e-8))
    u, info = spla.cg(A, b, rtol=1e-8, maxiter=5000, M=M)
    return u, info


def report(env, dedup):
    path, n = ENVS[env]
    z, zn, _ = _load_replay_buffer(path, "cuda")
    h = MCMollifiedHeat()
    t0 = time.time()
    h.compile_environment(z_buffer=z, z_predicted_next=zn, n_nodes=n,
                          K_neighbors=64, solver_mode="heat", knn_dedup=dedup)
    idx = h.knn_indices
    distinct = np.array([len(np.unique(r)) for r in idx])
    L = h.L_sym.tocsr()
    D = h.degree
    W = (sp.diags(D) - L).tocsr()
    rng = np.random.RandomState(0)
    goals = rng.choice(L.shape[0], 6, replace=False)
    hop = shortest_path(W, unweighted=True, directed=False, indices=goals)
    rules = {"bandwidth": h.sigma_perturb ** 2, "degree": 1.0 / np.median(D)}
    print(f"\n== {env} dedup={dedup} N={L.shape[0]} build={time.time()-t0:.0f}s "
          f"distinct/64 mean={distinct.mean():.1f} min={distinct.min()} "
          f"medianD={np.median(D):.2f} sigma={h.sigma_perturb:.4g}", flush=True)
    for name, t in rules.items():
        zs, rh, ties = [], [], []
        for gi, g in enumerate(goals):
            u, info = solve(L, g, t)
            u = np.maximum(u, 0.0)
            V = rankdata(-np.log(np.maximum(u, 1e-300)))
            d = hop[gi]; ok = np.isfinite(d)
            zs.append((u <= 0).mean())
            ties.append(1.0 - len(np.unique(V)) / len(V))
            rh.append(spearmanr(V[ok], d[ok]).correlation)
        ecc = np.nanmax(np.where(np.isfinite(hop), hop, np.nan), axis=1)
        print(f"  {name:9s} t={t:.4g} t*medD={t*np.median(D):.3g} | u<=0 {np.mean(zs):.3f} "
              f"| tied {np.mean(ties):.3f} | rho_hop {np.mean(rh):.3f} "
              f"(min {np.min(rh):.3f}) | ecc {ecc.mean():.1f}", flush=True)


if __name__ == "__main__":
    for env in (sys.argv[1:] or list(ENVS)):
        for dedup in (False, True):
            report(env, dedup)
