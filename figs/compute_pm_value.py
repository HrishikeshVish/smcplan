"""Heat-equation cost-to-go V over the point-maze offline data, for the teaser.

Builds the latent graph exactly as the planner does (fk_pointmaze.yaml graph
settings), solves for the goal of the seed-24 trace in figs/renders/pm_fk, and
evaluates V at every buffer latent. Each buffer latent carries its env state
(`targets`), so V can be painted over (x, y). Output: figs/renders/pm_value.npz
"""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from planning.gd import MCMollifiedHeat, _load_replay_buffer

# The planner's graph buffer (3800 transitions, no env states) and the full
# offline buffer (same encoder -- its latents contain the planner's exactly --
# plus the env state of every latent).
BUF = "latent_buffers/pointmaze_buffer.pt"
OFF = "latent_buffers/pointmaze_offline_replay_buffer.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"
z_curr, z_next, _ = _load_replay_buffer(BUF, dev)
raw = torch.load(OFF, map_location="cpu", weights_only=True)
states = raw["targets"].reshape(-1, raw["targets"].shape[-1]).numpy()
Z = raw["z_buffer"].reshape(-1, raw["z_buffer"].shape[-1]).numpy()

hs = MCMollifiedHeat()
hs.compile_environment(z_buffer=z_curr, z_predicted_next=z_next, n_nodes=3800,
                       K_neighbors=64, use_isomap=False, solver_mode="heat", rw_steps=20)

TRACE = sys.argv[1] if len(sys.argv) > 1 else "figs/renders/pm_fk/traj.npz"
OUTF = sys.argv[2] if len(sys.argv) > 2 else "figs/renders/pm_value.npz"
tr = np.load(TRACE, allow_pickle=True)
sg = tr["state_g"][0].copy()
# env frame -> offline-buffer frame (constant shift fit to the data support; see fig_teaser.OFFSET)
sg[:2] = sg[:2] - np.array([1.83, 1.90])
gi = int(np.argmin(np.linalg.norm(states[:, :2] - sg[:2], axis=1) + 0.05 * np.linalg.norm(states[:, 2:], axis=1)))  # position match, prefer slow
print("goal state", sg, "nearest buffer state", states[gi])
hs.solve_for_goal(Z[gi])

V = []
with torch.no_grad():
    for s in range(0, len(Z), 8192):
        V.append(hs.evaluate_rollout(torch.tensor(Z[s:s + 8192], device=dev, dtype=torch.float32)).cpu().numpy())
V = np.concatenate(V)
np.savez(OUTF, xy=states[:, :2], V=V, goal=sg, goal_buf=states[gi])
print("V range", V.min(), V.max(), "n", len(V))

# --- the sparse roadmap itself: graph nodes (planner buffer latents), their
# heat-solve V, and a few kNN edges, placed in (x, y) by exact latent match to
# the offline buffer (which carries env states). ---
from scipy.spatial import cKDTree
zc = z_curr if isinstance(z_curr, np.ndarray) else z_curr.cpu().numpy()
dist, j = cKDTree(Z).query(zc)
print("roadmap node match: max latent dist", dist.max())
node_xy = states[j, :2]
node_V = hs.V_graph.detach().cpu().numpy()
knn = hs.knn_indices.cpu().numpy() if torch.is_tensor(hs.knn_indices) else np.asarray(hs.knn_indices)
np.savez(OUTF.replace(".npz", "_roadmap.npz"), node_xy=node_xy, node_V=node_V, knn=knn[:, :6])
print("roadmap", node_xy.shape, None if node_V is None else node_V.shape, knn.shape)
