"""Re-evaluate Euclidean, ISOMAP, and Spectral graphs using GEODESIC distance
as ground truth instead of Euclidean distance.

Ground truth geodesic: shortest path through transition graph (dynamics-aware).
Fallback geodesic: shortest path through spatial K-NN graph (ISOMAP distance).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.stats import spearmanr
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#fafafa',
    'figure.dpi': 150,
})

SAVE_DIR = "/tmp/geodesic_eval_plots"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 55000
N_GOALS = 10
K_NEIGHBORS = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_knn_graph(z_scaled, K=32, mc_candidates=4096, chunk_size=8192):
    N = z_scaled.shape[0]
    graph_nodes = torch.tensor(z_scaled, dtype=torch.float32, device=device)
    all_k_dists = torch.full((N, K), float('inf'), device=device)
    all_k_indices = torch.zeros((N, K), dtype=torch.long, device=device)
    n_rounds = min(50, max(1, N // mc_candidates))
    rng = np.random.RandomState(42)
    with torch.no_grad():
        for r in range(n_rounds):
            cand_idx = torch.tensor(
                rng.choice(N, min(mc_candidates, N), replace=False), device=device)
            cand_nodes = graph_nodes[cand_idx]
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                dists = torch.cdist(graph_nodes[start:end], cand_nodes)
                ck = min(K, dists.shape[1])
                cd, cl = torch.topk(-dists, k=ck, dim=1)
                cd, ci = -cd, cand_idx[cl]
                md = torch.cat([all_k_dists[start:end], cd], dim=1)
                mi = torch.cat([all_k_indices[start:end], ci], dim=1)
                bd, bp = torch.topk(-md, k=K, dim=1)
                all_k_dists[start:end] = -bd
                all_k_indices[start:end] = mi.gather(1, bp)
    sigma = (all_k_dists[:, 1].unsqueeze(1) * 2.0).median().item()
    return all_k_dists.cpu().numpy(), all_k_indices.cpu().numpy(), sigma, graph_nodes


def build_laplacian(knn_dists, knn_indices, N):
    local_sigma = np.clip(knn_dists[:, 1:2] * 2.0, 1e-3, None)
    W_local = np.exp(-(knn_dists ** 2) / (2 * local_sigma ** 2))
    K = knn_dists.shape[1]
    row = np.repeat(np.arange(N), K)
    col = knn_indices.flatten()
    W_0 = sp.csr_matrix((W_local.flatten(), (row, col)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym
    return L, W_sym, d


def build_transition_graph(z_scaled, z_next_scaled, K=8):
    """Build transition graph: for each i, edge to K nearest nodes to z_next[i]."""
    N = z_scaled.shape[0]
    z_t = torch.tensor(z_scaled, dtype=torch.float32, device=device)
    zn_t = torch.tensor(z_next_scaled, dtype=torch.float32, device=device)

    all_k_dists = torch.full((N, K), float('inf'), device=device)
    all_k_indices = torch.zeros((N, K), dtype=torch.long, device=device)

    chunk = 8192
    with torch.no_grad():
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            dists = torch.cdist(zn_t[start:end], z_t)
            # Exclude self
            for i in range(start, end):
                dists[i - start, i] = float('inf')
            cd, ci = torch.topk(-dists, k=K, dim=1)
            all_k_dists[start:end] = -cd
            all_k_indices[start:end] = ci

    knn_dists = all_k_dists.cpu().numpy()
    knn_indices = all_k_indices.cpu().numpy()

    row = np.repeat(np.arange(N), K)
    col = knn_indices.flatten()
    data = knn_dists.flatten()
    W = sp.csr_matrix((data, (row, col)), shape=(N, N))
    W.setdiag(0)
    W_sym = (W + W.T) * 0.5
    W_sym.eliminate_zeros()
    return W_sym


def compute_spectral_embedding(L, d, n_dims):
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    t0 = time.perf_counter()
    eigenvalues, eigenvectors = spla.eigsh(L_norm, k=n_dims + 1, which='SM', tol=1e-6)
    t1 = time.perf_counter()
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx][1:]
    eigenvectors = eigenvectors[:, idx][:, 1:]
    spectral_coords = D_inv_sqrt @ eigenvectors
    print(f"  Eigendecomposition ({n_dims} dims): {t1-t0:.2f}s")
    return spectral_coords, eigenvalues


def build_spectral_knn_graph(spectral_coords, K=32, chunk_size=8192):
    N = spectral_coords.shape[0]
    spec_t = torch.tensor(spectral_coords, dtype=torch.float32, device=device)
    all_k_dists = np.zeros((N, K))
    all_k_indices = np.zeros((N, K), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dists = torch.cdist(spec_t[start:end], spec_t)
            for i in range(start, end):
                dists[i - start, i] = float('inf')
            cd, ci = torch.topk(-dists, k=K, dim=1)
            all_k_dists[start:end] = (-cd).cpu().numpy()
            all_k_indices[start:end] = ci.cpu().numpy()

    local_sigma = np.clip(all_k_dists[:, 1:2] * 2.0, 1e-3, None)
    W_local = np.exp(-(all_k_dists ** 2) / (2 * local_sigma ** 2))
    row = np.repeat(np.arange(N), K)
    col = all_k_indices.flatten()
    W_0 = sp.csr_matrix((W_local.flatten(), (row, col)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym
    sigma = np.median(all_k_dists[:, 1]) * 2.0
    return L, W_sym, sigma


def solve_heat(L, goal_idx, sigma_perturb):
    N = L.shape[0]
    t = (sigma_perturb ** 2) * 500
    b = np.zeros(N)
    b[goal_idx] = 1.0
    A = sp.eye(N) + t * L
    M = sp.diags(1.0 / (A.diagonal() + 1e-8))
    u, info = spla.cg(A, b, rtol=1e-8, maxiter=5000, M=M)
    u = np.maximum(u, 1e-300)
    V = -np.log(u)
    V = V - V.min()
    V = V * 10.0 / (V.mean() + 1e-8)
    return V


# ── Main ─────────────────────────────────────────────────────────────

print("=" * 60)
print("Geodesic Evaluation of Graph Methods")
print(f"Device: {device}")
print("=" * 60)

print("\nLoading buffer...")
loaded = torch.load(BUF_PATH, map_location='cpu', weights_only=True)
N_total = loaded['z_curr'].shape[0]

rng = np.random.RandomState(42)
idx = rng.choice(N_total, N_SUBSAMPLE, replace=False)
z_curr = loaded['z_curr'][idx].numpy()
z_next = loaded['z_next'][idx].numpy()

scaler = StandardScaler()
scaler.fit(np.vstack([z_curr, z_next]))
z_scaled = scaler.transform(z_curr)
z_next_scaled = scaler.transform(z_next)
N = z_scaled.shape[0]

# ── Build ground-truth geodesic distances ────────────────────────────

# 1) Transition graph (dynamics-aware geodesic)
print("\n[Ground Truth] Building transition graph (K=8)...")
t0 = time.perf_counter()
W_trans = build_transition_graph(z_scaled, z_next_scaled, K=8)
n_comp, comp_labels = connected_components(W_trans, directed=False)
comp_sizes = np.bincount(comp_labels)
largest_comp = np.argmax(comp_sizes)
largest_mask = comp_labels == largest_comp
n_largest = comp_sizes[largest_comp]
print(f"  Components: {n_comp}, largest: {n_largest} ({n_largest/N*100:.1f}%)")
print(f"  Build: {time.perf_counter()-t0:.2f}s")

# 2) Spatial K-NN graph (manifold geodesic via ISOMAP)
print("\n[Ground Truth] Building spatial K-NN graph...")
knn_dists, knn_indices, sigma_euclid, graph_nodes = build_knn_graph(z_scaled, K=K_NEIGHBORS)

# K-NN graph with Euclidean edge weights for geodesic computation
row = np.repeat(np.arange(N), K_NEIGHBORS)
col = knn_indices.flatten()
W_knn_dist = sp.csr_matrix((knn_dists.flatten(), (row, col)), shape=(N, N))
W_knn_dist.setdiag(0)
W_knn_dist_sym = (W_knn_dist + W_knn_dist.T) * 0.5
W_knn_dist_sym.eliminate_zeros()

# ── Goals ────────────────────────────────────────────────────────────

rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N, N_GOALS, replace=False)

# Compute geodesic distances from each goal
print("\n[Ground Truth] Computing geodesic distances from goals...")
t0 = time.perf_counter()
# Transition geodesic (from goals)
trans_geo = dijkstra(W_trans, directed=False, indices=goal_indices)  # (N_GOALS, N)
# KNN geodesic (ISOMAP distance)
knn_geo = dijkstra(W_knn_dist_sym, directed=False, indices=goal_indices)  # (N_GOALS, N)
print(f"  Dijkstra: {time.perf_counter()-t0:.2f}s")

for gi in range(N_GOALS):
    trans_finite = np.isfinite(trans_geo[gi]).sum()
    knn_finite = np.isfinite(knn_geo[gi]).sum()
    print(f"  Goal {gi}: trans reachable={trans_finite}/{N} ({trans_finite/N*100:.1f}%), "
          f"knn reachable={knn_finite}/{N}")

# ── Build graph variants ─────────────────────────────────────────────

print("\n[Graphs] Building Euclidean Laplacian...")
L_euclid, W_euclid, d_euclid = build_laplacian(knn_dists, knn_indices, N)

print("\n[Graphs] Building Spectral embeddings...")
spectral_dims = [20, 50, 100]
spectral_graphs = {}
for sd in spectral_dims:
    print(f"\n  Spectral dim={sd}")
    spec_coords, _ = compute_spectral_embedding(L_euclid, d_euclid, sd)
    L_s, W_s, sigma_s = build_spectral_knn_graph(spec_coords, K=K_NEIGHBORS)
    spectral_graphs[sd] = (L_s, sigma_s)

# ── Evaluate all methods against all distance metrics ────────────────

print(f"\n{'=' * 60}")
print("EVALUATION: V vs {Euclidean, KNN-geodesic, Transition-geodesic}")
print(f"{'=' * 60}")

methods = {
    'Euclidean': (L_euclid, sigma_euclid),
}
for sd in spectral_dims:
    methods[f'Spectral-{sd}'] = spectral_graphs[sd]

# For each method, evaluate against all three distance metrics
all_results = {}
for method_name, (L_method, sigma_method) in methods.items():
    print(f"\n  [{method_name}]")
    corrs_euclid = []
    corrs_knn_geo = []
    corrs_trans_geo = []

    for gi in range(N_GOALS):
        goal_idx = goal_indices[gi]
        V = solve_heat(L_method, goal_idx, sigma_method)

        # Euclidean distance
        euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)
        r_euclid, _ = spearmanr(V, euclid_d)
        corrs_euclid.append(r_euclid)

        # KNN geodesic distance
        knn_d = knn_geo[gi]
        finite_knn = np.isfinite(knn_d)
        if finite_knn.sum() > 100:
            r_knn, _ = spearmanr(V[finite_knn], knn_d[finite_knn])
        else:
            r_knn = float('nan')
        corrs_knn_geo.append(r_knn)

        # Transition geodesic distance
        trans_d = trans_geo[gi]
        finite_trans = np.isfinite(trans_d)
        if finite_trans.sum() > 100:
            r_trans, _ = spearmanr(V[finite_trans], trans_d[finite_trans])
        else:
            r_trans = float('nan')
        corrs_trans_geo.append(r_trans)

        print(f"    Goal {gi}: Euclid={r_euclid:.4f}  KNN-geo={r_knn:.4f}  Trans-geo={r_trans:.4f}")

    m_e = np.nanmean(corrs_euclid)
    m_k = np.nanmean(corrs_knn_geo)
    m_t = np.nanmean(corrs_trans_geo)
    print(f"    MEAN:   Euclid={m_e:.4f}  KNN-geo={m_k:.4f}  Trans-geo={m_t:.4f}")

    all_results[method_name] = {
        'euclid': corrs_euclid,
        'knn_geo': corrs_knn_geo,
        'trans_geo': corrs_trans_geo,
        'mean_euclid': m_e,
        'mean_knn_geo': m_k,
        'mean_trans_geo': m_t,
    }

# ── Plots ────────────────────────────────────────────────────────────

method_colors = {
    'Euclidean': '#e8a0bf',
    'Spectral-20': '#a0c4e8',
    'Spectral-50': '#7fb7d4',
    'Spectral-100': '#5faac0',
}

# Plot 1: Mean Spearman across metrics for each method
fig, ax = plt.subplots(figsize=(10, 6))
method_names = list(all_results.keys())
x = np.arange(len(method_names))
width = 0.25
metrics = ['euclid', 'knn_geo', 'trans_geo']
metric_labels = ['vs Euclidean', 'vs KNN Geodesic', 'vs Transition Geodesic']
metric_colors = ['#e8a0bf', '#a0c4e8', '#a0e8bf']

for i, (metric, mlabel, mcol) in enumerate(zip(metrics, metric_labels, metric_colors)):
    means = [all_results[m][f'mean_{metric}'] for m in method_names]
    ax.bar(x + i * width, means, width, label=mlabel, color=mcol,
           edgecolor='white', linewidth=0.8)

ax.set_xlabel('Graph Construction Method')
ax.set_ylabel('Mean Spearman Correlation')
ax.set_title('V Quality Measured Against Different Distance Metrics')
ax.set_xticks(x + width)
ax.set_xticklabels(method_names, rotation=15, ha='right')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_mean_by_metric.png")
plt.close(fig)
print(f"\nSaved 01_mean_by_metric.png")

# Plot 2: Per-goal comparison for each metric
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax_i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
    ax = axes[ax_i]
    xg = np.arange(N_GOALS)
    n_m = len(method_names)
    w = 0.8 / n_m
    for mi, mname in enumerate(method_names):
        vals = all_results[mname][metric]
        color = method_colors.get(mname, '#cccccc')
        ax.bar(xg + mi * w, vals, w, label=mname, color=color,
               edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Goal')
    ax.set_ylabel('Spearman')
    ax.set_title(f'V {mlabel}')
    ax.set_xticks(xg + w * (n_m - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(N_GOALS)])
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.legend(fontsize=7, frameon=True, facecolor='white', edgecolor='#cccccc')

plt.suptitle('Per-Goal V Quality by Distance Metric', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_per_goal_all_metrics.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 02_per_goal_all_metrics.png")

# Plot 3: Correlation between distance metrics themselves
fig, axes = plt.subplots(1, N_GOALS, figsize=(4 * N_GOALS, 4))
scatter_rng = np.random.RandomState(42)
si = scatter_rng.choice(N, 2000, replace=False)

for gi in range(N_GOALS):
    ax = axes[gi] if N_GOALS > 1 else axes
    euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_indices[gi]], axis=1)
    knn_d = knn_geo[gi]
    finite = np.isfinite(knn_d)
    si_f = si[finite[si]]

    if len(si_f) > 10:
        r, _ = spearmanr(euclid_d[si_f], knn_d[si_f])
        ax.scatter(euclid_d[si_f], knn_d[si_f], s=2, alpha=0.3, color='#7fb7d4', rasterized=True)
        ax.set_title(f'Goal {gi}\nr(E,G)={r:.3f}', fontsize=9)
    ax.set_xlabel('Euclid', fontsize=8)
    if gi == 0:
        ax.set_ylabel('KNN Geodesic', fontsize=8)

plt.suptitle('Euclidean vs KNN-Geodesic Distance per Goal', fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/03_euclid_vs_knn_geodesic.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 03_euclid_vs_knn_geodesic.png")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"{'Method':15s} | {'vs Euclid':>10s} | {'vs KNN-geo':>10s} | {'vs Trans-geo':>12s}")
print("-" * 55)
for mname in method_names:
    r = all_results[mname]
    print(f"{mname:15s} | {r['mean_euclid']:10.4f} | {r['mean_knn_geo']:10.4f} | {r['mean_trans_geo']:12.4f}")

print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
