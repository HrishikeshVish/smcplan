"""Per-environment calibration of the roadmap cost-to-go against offline data only.

Ground truth: the latent buffers are whole episodes concatenated in order
(build_latent_buffer.py), so every roadmap vertex has an episode and a time
index. For a goal vertex g at time tau_g, an earlier vertex i of the same episode
reached g in tau_g - tau_i steps: an observed cost-to-go. A correct V ranks those
vertices the same way. The score of a configuration is the mean over goals of
Spearman(V(i), tau_g - tau_i) across the earlier same-episode vertices i. No
planning seed, no world-model rollout and no success signal enters this.

Grid: k (neighbours per successor) x c, with diffusion time t = c / median(D_ii)
(tL dimensionless). The configuration in use before 2026-09-28 (k=64,
t = (median sigma)^2, no dedupe) is scored as a reference row.

Usage: CUDA_VISIBLE_DEVICES=0 python writeup/calib_pde.py [env ...]
Writes sweep_emsmc/calib_pde_<env>.csv
"""
import sys, csv, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import rankdata, spearmanr
sys.path.insert(0, "/home/hviswan/Documents/grasp")
from planning.gd import MCMollifiedHeat, _load_replay_buffer

B = "/home/hviswan/Documents/grasp/latent_buffers/"
ENVS = {"pointmaze": (B + "pointmaze_buffer.pt", 3800),
        "wall": (B + "wall_buffer.pt", 12600),
        "pusht": (B + "pusht_offline_replay_buffer.pt", 55000),
        "reacher": (B + "lewm_reacher_fixed.pt", 55000),
        "cube": (B + "lewm_cube_buffer.pt", 55000)}
KS = (4, 8, 16, 32, 64)
CS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
N_GOALS, MIN_PAIRS = 60, 4


def episodes(z, zn):
    """Episode id and within-episode time of every buffer row."""
    cont = np.all(np.isclose(zn[:-1], z[1:], atol=1e-5), axis=1)   # row i -> i+1 same episode
    ep = np.concatenate([[0], np.cumsum(~cont)])
    t = np.zeros(len(z), dtype=np.int64)
    for i in range(1, len(z)):
        t[i] = t[i - 1] + 1 if cont[i - 1] else 0
    return ep, t


def solve_V(L, g, t):
    N = L.shape[0]
    b = np.zeros(N); b[g] = 1.0
    A = sp.eye(N) + t * L
    u, _ = spla.cg(A, b, rtol=1e-10, maxiter=20000, M=sp.diags(1.0 / A.diagonal()))
    u = np.maximum(u, 0.0)
    return rankdata(-np.log(np.maximum(u, 1e-300))), (u <= 0).mean()


def score(h, ep, tt, goals_pairs, t):
    rh, zs = [], []
    L = h.L_sym.tocsr()
    for g, pre in goals_pairs:
        V, z0 = solve_V(L, g, t)
        dtau = tt[h.node_idx[g]] - tt[h.node_idx[pre]]
        r = spearmanr(V[pre], dtau).correlation
        if np.isfinite(r):
            rh.append(r)
        zs.append(z0)
    return float(np.mean(rh)), float(np.std(rh) / np.sqrt(len(rh))), float(np.mean(zs))


def run(env):
    path, n = ENVS[env]
    z, zn, _ = _load_replay_buffer(path, "cuda")
    ep, tt = episodes(z, zn)
    rows = []
    out = open(f"/home/hviswan/Documents/grasp/sweep_emsmc/calib_pde_{env}.csv", "w", newline="")
    w = csv.writer(out); w.writerow(["env", "k", "rule", "c", "t", "t_medD", "rho", "se", "u0"])
    print(f"\n== {env}: {len(z)} rows, {ep.max()+1} episodes", flush=True)
    goals_pairs = None
    for k, dedup in [(64, False)] + [(k, True) for k in KS]:
        h = MCMollifiedHeat()
        h.compile_environment(z_buffer=z, z_predicted_next=zn, n_nodes=n, K_neighbors=k,
                              solver_mode="heat", knn_dedup=dedup)
        if goals_pairs is None:            # same goals for every configuration
            e_n, t_n = ep[h.node_idx], tt[h.node_idx]
            cand = []
            for g in range(len(e_n)):
                pre = np.where((e_n == e_n[g]) & (t_n < t_n[g]))[0]
                if len(pre) >= MIN_PAIRS:
                    cand.append((g, pre))
            rng = np.random.RandomState(0)
            sel = rng.choice(len(cand), min(N_GOALS, len(cand)), replace=False)
            goals_pairs = [cand[i] for i in sel]
            print(f"   {len(cand)} eligible goals, using {len(goals_pairs)}; "
                  f"mean pairs/goal {np.mean([len(p) for _, p in goals_pairs]):.1f}", flush=True)
        if not dedup:   # Euclidean distance to the goal in roadmap coordinates: the L2 analogue
            rh = []
            for g, pre in goals_pairs:
                d = np.linalg.norm(h.z_buffer_np[pre] - h.z_buffer_np[g], axis=1)
                dtau = tt[h.node_idx[g]] - tt[h.node_idx[pre]]
                r = spearmanr(d, dtau).correlation
                if np.isfinite(r): rh.append(r)
            print(f"   EUCLIDEAN (L2 analogue) rho={np.mean(rh):.3f}±{np.std(rh)/np.sqrt(len(rh)):.3f}", flush=True)
            w.writerow([env, "-", "euclidean", "-", "-", "-", np.mean(rh), np.std(rh)/np.sqrt(len(rh)), 0])
        medD = float(np.median(h.degree))
        grid = ([("bandwidth", None, h.sigma_perturb ** 2)] if not dedup else
                [("degree", c, c / medD) for c in CS])
        for rule, c, t in grid:
            rho, se, u0 = score(h, ep, tt, goals_pairs, t)
            tag = "REFERENCE (old)" if not dedup else ""
            print(f"   k={k:3d} {rule:9s} c={c} t={t:.4g} t*medD={t*medD:.3g} | rho={rho:.3f}±{se:.3f} "
                  f"| u<=0 {u0:.3f} {tag}", flush=True)
            w.writerow([env, k, rule, c, t, t * medD, rho, se, u0]); out.flush()
    out.close()


if __name__ == "__main__":
    for env in (sys.argv[1:] or list(ENVS)):
        run(env)
