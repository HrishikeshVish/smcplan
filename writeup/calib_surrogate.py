"""Check a calibrated roadmap setting THROUGH the surrogate the planner queries.

The planner never reads V on roadmap vertices: it evaluates the surrogate Vhat at
rollout latents. So for a chosen (k, t-rule, c), fit Vhat exactly as
solve_for_goal does and score it on the same offline ground truth as
calib_pde.py -- Spearman(Vhat(x), tau_g - tau_x) over earlier states x of the
goal's episode -- using ALL buffer rows of that episode, most of which are not
roadmap vertices on the subsampled buffers (off-roadmap, like a rollout).
Reported next to the same score for V on vertices and for Euclidean distance.

Usage: CUDA_VISIBLE_DEVICES=0 python writeup/calib_surrogate.py env k rule c [n_goals]
"""
import sys
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, "/home/hviswan/Documents/grasp")
sys.path.insert(0, "/home/hviswan/Documents/grasp/writeup")
from planning.gd import MCMollifiedHeat, _load_replay_buffer
from calib_pde import ENVS, episodes

env, k, rule, c = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
NG = int(sys.argv[5]) if len(sys.argv) > 5 else 30
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 250
path, n = ENVS[env]
z, zn, _ = _load_replay_buffer(path, "cuda")
ep, tt = episodes(z, zn)
h = MCMollifiedHeat()
h.heat_t_rule, h.heat_c, h.surrogate_steps = rule, c, STEPS
h.compile_environment(z_buffer=z, z_predicted_next=zn, n_nodes=n, K_neighbors=k,
                      solver_mode="heat", knn_dedup=True)
is_node = np.zeros(len(z), bool); is_node[h.node_idx] = True
rng = np.random.RandomState(1)
order = rng.permutation(len(h.node_idx))
res = {"vhat_all": [], "vhat_off": [], "v_nodes": [], "euclid_all": []}
used = 0
for gi in order:
    if used >= NG:
        break
    row = h.node_idx[gi]
    pre = np.where((ep == ep[row]) & (tt < tt[row]))[0]
    if len(pre) < 4:
        continue
    used += 1
    h.v_surrogate = None            # fresh fit per goal, as in a planning run
    torch.manual_seed(0)
    h.solve_for_goal(z[row])
    dtau = tt[row] - tt[pre]
    with torch.no_grad():
        vh = h.evaluate_rollout(torch.tensor(z[pre], dtype=torch.float32, device="cuda")).cpu().numpy()
    res["vhat_all"].append(spearmanr(vh, dtau).correlation)
    off = ~is_node[pre]
    if off.sum() >= 4:
        res["vhat_off"].append(spearmanr(vh[off], dtau[off]).correlation)
    on = np.searchsorted(np.sort(h.node_idx), pre)  # vertex ids of on-roadmap rows
    pos = {r: i for i, r in enumerate(h.node_idx)}
    vn = [(h.V_graph[pos[r]].item(), d) for r, d in zip(pre, dtau) if r in pos]
    if len(vn) >= 4:
        a = np.array(vn); res["v_nodes"].append(spearmanr(a[:, 0], a[:, 1]).correlation)
    zs = h.scaler_obj.transform(z[pre]); zg = h.scaler_obj.transform(z[row:row + 1])
    res["euclid_all"].append(spearmanr(np.linalg.norm(zs - zg, axis=1), dtau).correlation)
print(f"\n== {env} k={k} {rule} c={c} steps={STEPS} t={getattr(h, 'heat_t', float('nan')):.4g} goals={used}")
for key, v in res.items():
    v = np.array([x for x in v if np.isfinite(x)])
    if len(v):
        print(f"   {key:11s} rho={v.mean():.3f}±{v.std()/np.sqrt(len(v)):.3f} (n={len(v)})")
