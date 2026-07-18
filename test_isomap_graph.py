"""Test ISOMAP-based graph construction for heat equation.

Hypothesis: using geodesic distances (shortest paths through K-NN graph)
instead of Euclidean distances for edge weights will improve V quality
for goals where the manifold is folded.

Approach:
1. Build spatial K-NN graph with Euclidean edge weights (baseline)
2. Compute geodesic distances via landmark Dijkstra
3. Build new Laplacian using geodesic-distance kernel
4. Solve heat equation on both graphs, compare Spearman correlations
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import shortest_path, dijkstra
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

SAVE_DIR = "/tmp/isomap_plots"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 55000
N_GOALS = 10
K_NEIGHBORS = 32
N_LANDMARKS = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Helpers ──────────────────────────────────────────────────────────────

def build_knn_graph(z_scaled, K=32, mc_candidates=4096, chunk_size=8192):
    """Build spatial K-NN graph. Returns (all_k_dists, all_k_indices, sigma_perturb)."""
    N = z_scaled.shape[0]
    graph_nodes = torch.tensor(z_scaled, dtype=torch.float32, device=device)
    all_k_dists = torch.full((N, K), float('inf'), device=device)
    all_k_indices = torch.zeros((N, K), dtype=torch.long, device=device)
    n_rounds = min(50, max(1, N // mc_candidates))
    rng = np.random.RandomState(42)

    with torch.no_grad():
        for r in range(n_rounds):
            cand_idx = torch.tensor(
                rng.choice(N, min(mc_candidates, N), replace=False),
                device=device
            )
            cand_nodes = graph_nodes[cand_idx]
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                dists = torch.cdist(graph_nodes[start:end], cand_nodes)
                ck = min(K, dists.shape[1])
                chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
                chunk_k_dists = -chunk_k_dists
                chunk_k_indices = cand_idx[chunk_k_local]
                merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
                merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
                best_dists, best_pos = torch.topk(-merged_dists, k=K, dim=1)
                all_k_dists[start:end] = -best_dists
                all_k_indices[start:end] = merged_indices.gather(1, best_pos)

    sigma_perturb = (all_k_dists[:, 1].unsqueeze(1) * 2.0).median().item()
    return all_k_dists.cpu().numpy(), all_k_indices.cpu().numpy(), sigma_perturb, graph_nodes


def build_laplacian_euclidean(all_k_dists, all_k_indices, N):
    """Standard Laplacian with Gaussian kernel on Euclidean distances."""
    local_sigma = all_k_dists[:, 1:2] * 2.0
    local_sigma = np.clip(local_sigma, 1e-3, None)
    W_local = np.exp(-(all_k_dists ** 2) / (2 * local_sigma ** 2))

    K = all_k_dists.shape[1]
    row_idx = np.repeat(np.arange(N), K)
    col_idx = all_k_indices.flatten()
    data = W_local.flatten()
    W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym
    return L, W_sym


def compute_landmark_geodesics(W_sym_dist, N, n_landmarks=500):
    """Compute geodesic distances from landmarks using Dijkstra.

    W_sym_dist: sparse matrix with edge weights = Euclidean distances (not kernel weights).
    Returns: (n_landmarks, N) array of geodesic distances.
    """
    rng = np.random.RandomState(42)
    landmarks = rng.choice(N, n_landmarks, replace=False)

    t0 = time.perf_counter()
    # Dijkstra from each landmark
    geo_dists = dijkstra(W_sym_dist, directed=False, indices=landmarks)
    t1 = time.perf_counter()
    print(f"  Landmark Dijkstra ({n_landmarks} landmarks): {t1-t0:.2f}s")

    # Check reachability
    n_inf = np.isinf(geo_dists).sum()
    total = geo_dists.size
    print(f"  Unreachable pairs: {n_inf}/{total} ({n_inf/total*100:.1f}%)")

    return geo_dists, landmarks


def build_laplacian_geodesic(z_scaled, geo_dists, landmarks, K=32, chunk_size=1000):
    """Build K-NN graph using approximate geodesic distances from landmarks.

    For each node i, approximate d_geo(i, j) = min_l (geo_dists[l, i] + geo_dists[l, j]).
    Then find K nearest neighbors by geodesic distance and use geodesic kernel.
    """
    N = z_scaled.shape[0]
    L_landmarks = len(landmarks)

    t0 = time.perf_counter()

    # geo_dists shape: (L, N)
    # For node i: d_approx[j] = min_l (geo_dists[l, i] + geo_dists[l, j])
    # Process in chunks to avoid N*N memory

    all_geo_knn_dists = np.full((N, K), np.inf)
    all_geo_knn_indices = np.zeros((N, K), dtype=np.int64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_n = end - start

        # (L, chunk_n) + (L, N) -> min over L -> (chunk_n, N)
        # geo_dists[:, start:end] is (L, chunk_n)
        # We want min_l (geo_dists[l, i] + geo_dists[l, j]) for i in chunk, j in all
        d_from_chunk = geo_dists[:, start:end]  # (L, chunk_n)

        # Compute in sub-chunks to manage memory
        # (L, chunk_n, 1) + (L, 1, N) -> (L, chunk_n, N) -- too big if chunk_n large
        # Instead: accumulate min across landmarks in batches
        d_approx = np.full((chunk_n, N), np.inf)
        lm_batch = 50
        for lb in range(0, L_landmarks, lm_batch):
            le = min(lb + lm_batch, L_landmarks)
            # (le-lb, chunk_n, 1) + (le-lb, 1, N)
            d_batch = d_from_chunk[lb:le, :, np.newaxis] + geo_dists[lb:le, np.newaxis, :]
            d_batch_min = d_batch.min(axis=0)  # (chunk_n, N)
            d_approx = np.minimum(d_approx, d_batch_min)

        # Set self-distance to inf for K-NN
        for ci in range(chunk_n):
            d_approx[ci, start + ci] = np.inf

        # Find K nearest
        knn_idx = np.argpartition(d_approx, K, axis=1)[:, :K]
        knn_dists = np.take_along_axis(d_approx, knn_idx, axis=1)

        # Sort within K
        sort_idx = np.argsort(knn_dists, axis=1)
        all_geo_knn_indices[start:end] = np.take_along_axis(knn_idx, sort_idx, axis=1)
        all_geo_knn_dists[start:end] = np.take_along_axis(knn_dists, sort_idx, axis=1)

        if (start // chunk_size) % 10 == 0:
            print(f"    Geodesic K-NN: {end}/{N}")

    t1 = time.perf_counter()
    print(f"  Geodesic K-NN: {t1-t0:.2f}s")

    # Build Laplacian with geodesic kernel
    geo_sigma = np.median(all_geo_knn_dists[:, 1]) * 2.0
    geo_sigma = max(geo_sigma, 1e-3)
    print(f"  Geodesic sigma: {geo_sigma:.4f}")

    W_local = np.exp(-(all_geo_knn_dists ** 2) / (2 * geo_sigma ** 2))

    row_idx = np.repeat(np.arange(N), K)
    col_idx = all_geo_knn_indices.flatten()
    data = W_local.flatten()
    W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym

    degrees = np.array(W_sym.astype(bool).sum(axis=1)).flatten()
    print(f"  Geodesic graph degree: min={degrees.min()}, mean={degrees.mean():.1f}, max={degrees.max()}")

    return L, W_sym, all_geo_knn_dists, all_geo_knn_indices


def solve_heat(L, z_buffer_np, goal_idx, sigma_perturb):
    """Solve heat equation and return V."""
    N = L.shape[0]
    h2 = sigma_perturb ** 2
    t = h2 * 500

    b = np.zeros(N)
    b[goal_idx] = 1.0
    A_heat = sp.eye(N) + t * L
    M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
    u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
    if info != 0:
        print(f"  [Warning] CG exited with code {info}")

    u = np.maximum(u, 1e-300)
    V_raw = -np.log(u)
    V_raw = V_raw - V_raw.min()
    scale = 10.0 / (V_raw.mean() + 1e-8)
    return V_raw * scale


# ── Main ─────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"ISOMAP Graph Construction Test")
print(f"Device: {device}")
print("=" * 60)

print("\nLoading buffer...")
loaded = torch.load(BUF_PATH, map_location='cpu', weights_only=True)
N_total = loaded['z_curr'].shape[0]
print(f"Total: {N_total}, dim: {loaded['z_curr'].shape[1]}")

rng = np.random.RandomState(42)
idx = rng.choice(N_total, N_SUBSAMPLE, replace=False)
z_curr = loaded['z_curr'][idx].numpy()
z_next = loaded['z_next'][idx].numpy()

scaler = StandardScaler()
z_combined = np.vstack([z_curr, z_next])
scaler.fit(z_combined)
z_scaled = scaler.transform(z_curr)
N = z_scaled.shape[0]

# ── Step 1: Build spatial K-NN graph ─────────────────────────────────

print("\n[1] Building spatial K-NN graph...")
knn_dists, knn_indices, sigma_perturb, graph_nodes = build_knn_graph(z_scaled, K=K_NEIGHBORS)
print(f"  sigma_perturb: {sigma_perturb:.6f}")

# Build Euclidean Laplacian (baseline)
L_euclid, W_euclid = build_laplacian_euclidean(knn_dists, knn_indices, N)

# Build distance-weighted K-NN graph for Dijkstra (edge weight = Euclidean dist)
row_idx = np.repeat(np.arange(N), K_NEIGHBORS)
col_idx = knn_indices.flatten()
data_dist = knn_dists.flatten()
W_dist = sp.csr_matrix((data_dist, (row_idx, col_idx)), shape=(N, N))
W_dist.setdiag(0)
W_dist_sym = (W_dist + W_dist.T) * 0.5
# Replace 0 with inf for proper shortest paths, but keep as sparse
W_dist_sym.eliminate_zeros()

# ── Step 2: Compute geodesic distances via landmarks ─────────────────

print(f"\n[2] Computing geodesic distances ({N_LANDMARKS} landmarks)...")
geo_dists, landmarks = compute_landmark_geodesics(W_dist_sym, N, n_landmarks=N_LANDMARKS)

# ── Step 3: Build geodesic K-NN graph ────────────────────────────────

print(f"\n[3] Building geodesic K-NN graph (K={K_NEIGHBORS})...")
L_geodesic, W_geodesic, geo_knn_dists, geo_knn_indices = build_laplacian_geodesic(
    z_scaled, geo_dists, landmarks, K=K_NEIGHBORS, chunk_size=500
)

# Use geodesic median distance as sigma for the geodesic graph
geo_sigma = np.median(geo_knn_dists[:, 1]) * 2.0

# ── Step 4: Compare on goals ─────────────────────────────────────────

print(f"\n[4] Solving heat equation for {N_GOALS} goals...")
rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N, N_GOALS, replace=False)

results_euclid = []
results_geodesic = []

for gi, goal_idx in enumerate(goal_indices):
    euclid_dists = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    # Euclidean graph
    V_e = solve_heat(L_euclid, z_scaled, goal_idx, sigma_perturb)
    corr_e, _ = spearmanr(V_e, euclid_dists)
    results_euclid.append(corr_e)

    # Geodesic graph
    V_g = solve_heat(L_geodesic, z_scaled, goal_idx, geo_sigma)
    corr_g, _ = spearmanr(V_g, euclid_dists)
    results_geodesic.append(corr_g)

    print(f"  Goal {gi} (idx={goal_idx}): Euclid r={corr_e:.4f}, Geodesic r={corr_g:.4f}, "
          f"diff={corr_g - corr_e:+.4f}")

mean_e = np.mean(results_euclid)
mean_g = np.mean(results_geodesic)
print(f"\n  Mean Spearman: Euclid={mean_e:.4f}, Geodesic={mean_g:.4f}, diff={mean_g - mean_e:+.4f}")

# ── Step 5: Analyze geodesic vs Euclidean neighbor overlap ───────────

print(f"\n[5] Neighbor overlap analysis...")
overlap_fracs = []
for i in range(0, N, N // 100):
    euclid_nn = set(knn_indices[i].tolist())
    geo_nn = set(geo_knn_indices[i].tolist())
    overlap = len(euclid_nn & geo_nn) / K_NEIGHBORS
    overlap_fracs.append(overlap)
mean_overlap = np.mean(overlap_fracs)
print(f"  Mean K-NN overlap (Euclid vs Geodesic): {mean_overlap*100:.1f}%")

# ── Step 6: Geodesic distance analysis for good vs bad goals ────────

print(f"\n[6] Geodesic distance analysis per goal...")
# Identify good/bad goals
sorted_by_euclid = sorted(range(N_GOALS), key=lambda i: results_euclid[i])
bad_goals = sorted_by_euclid[:3]
good_goals = sorted_by_euclid[-3:]

for gi in bad_goals + good_goals:
    goal_idx = goal_indices[gi]
    euclid_dists = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    # Approximate geodesic distances to this goal
    geo_to_goal = np.full(N, np.inf)
    for l in range(len(landmarks)):
        d_via_l = geo_dists[l, :] + geo_dists[l, goal_idx]
        geo_to_goal = np.minimum(geo_to_goal, d_via_l)
    geo_to_goal[goal_idx] = 0

    finite_mask = np.isfinite(geo_to_goal)
    n_finite = finite_mask.sum()

    if n_finite > 100:
        corr_eg, _ = spearmanr(euclid_dists[finite_mask], geo_to_goal[finite_mask])
    else:
        corr_eg = float('nan')

    label = "GOOD" if gi in good_goals else "BAD"
    print(f"  Goal {gi} ({label}): Euclid-vs-Geodesic Spearman={corr_eg:.4f}, "
          f"reachable={n_finite}/{N} ({n_finite/N*100:.1f}%)")

# ── Plots ────────────────────────────────────────────────────────────

mode_colors = {'Euclidean': '#e8a0bf', 'Geodesic': '#a0c4e8'}

# Plot 1: Per-goal Spearman comparison
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(N_GOALS)
width = 0.35
ax.bar(x, results_euclid, width, label='Euclidean K-NN', color=mode_colors['Euclidean'],
       edgecolor='white', linewidth=0.8)
ax.bar(x + width, results_geodesic, width, label='Geodesic K-NN (ISOMAP)',
       color=mode_colors['Geodesic'], edgecolor='white', linewidth=0.8)
ax.set_xlabel('Goal Index')
ax.set_ylabel('Spearman(V, Euclidean dist)')
ax.set_title('Heat V Quality: Euclidean vs Geodesic Graph')
ax.set_xticks(x + width/2)
ax.set_xticklabels([str(i) for i in range(N_GOALS)])
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axhline(mean_e, color=mode_colors['Euclidean'], ls='--', lw=1.5, alpha=0.7)
ax.axhline(mean_g, color=mode_colors['Geodesic'], ls='--', lw=1.5, alpha=0.7)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_spearman_comparison.png")
plt.close(fig)
print(f"\nSaved 01_spearman_comparison.png")

# Plot 2: Geodesic vs Euclidean distance scatter for best/worst goals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
show_goals = bad_goals[:2] + good_goals[:2]
scatter_rng = np.random.RandomState(42)
scatter_idx = scatter_rng.choice(N, 3000, replace=False)

for ax_i, gi in enumerate(show_goals):
    row, col = ax_i // 2, ax_i % 2
    goal_idx = goal_indices[gi]
    euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    geo_to_goal = np.full(N, np.inf)
    for l in range(len(landmarks)):
        d_via_l = geo_dists[l, :] + geo_dists[l, goal_idx]
        geo_to_goal = np.minimum(geo_to_goal, d_via_l)
    geo_to_goal[goal_idx] = 0

    finite = np.isfinite(geo_to_goal)
    si = scatter_idx[finite[scatter_idx]]

    label = "GOOD" if gi in good_goals else "BAD"
    color = '#7fb7d4' if gi in good_goals else '#d47f9e'
    axes[row, col].scatter(euclid_d[si], geo_to_goal[si], s=3, alpha=0.3,
                           color=color, rasterized=True)
    corr_eg, _ = spearmanr(euclid_d[finite], geo_to_goal[finite])
    axes[row, col].set_xlabel('Euclidean Distance')
    axes[row, col].set_ylabel('Geodesic Distance')
    axes[row, col].set_title(f'Goal {gi} ({label})\nEuclid r={results_euclid[gi]:.3f}, '
                              f'Euclid-Geodesic r={corr_eg:.3f}')

plt.suptitle('Euclidean vs Geodesic Distance to Goal', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_euclid_vs_geodesic.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 02_euclid_vs_geodesic.png")

# Plot 3: V comparison for worst goals
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax_i, gi in enumerate(bad_goals[:2] + good_goals[:2]):
    row, col = ax_i // 2, ax_i % 2
    goal_idx = goal_indices[gi]
    euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    V_e = solve_heat(L_euclid, z_scaled, goal_idx, sigma_perturb)
    V_g = solve_heat(L_geodesic, z_scaled, goal_idx, geo_sigma)

    si = scatter_idx
    label = "GOOD" if gi in good_goals else "BAD"
    axes[row, col].scatter(euclid_d[si], V_e[si], s=3, alpha=0.3,
                           color=mode_colors['Euclidean'], label='Euclidean', rasterized=True)
    axes[row, col].scatter(euclid_d[si], V_g[si], s=3, alpha=0.3,
                           color=mode_colors['Geodesic'], label='Geodesic', rasterized=True)
    axes[row, col].set_xlabel('Euclidean Distance')
    axes[row, col].set_ylabel('V')
    axes[row, col].set_title(f'Goal {gi} ({label})\nEuclid r={results_euclid[gi]:.3f}, '
                              f'Geodesic r={results_geodesic[gi]:.3f}')
    axes[row, col].legend(markerscale=5, frameon=True, facecolor='white', edgecolor='#cccccc')

plt.suptitle('V vs Euclidean Distance: Both Graph Types', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/03_V_comparison.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 03_V_comparison.png")

# Plot 4: Per-goal improvement
fig, ax = plt.subplots(figsize=(10, 5))
improvements = [g - e for e, g in zip(results_euclid, results_geodesic)]
colors = ['#7fb7d4' if imp > 0 else '#d47f9e' for imp in improvements]
ax.bar(range(N_GOALS), improvements, color=colors, edgecolor='white', linewidth=0.8)
ax.set_xlabel('Goal Index')
ax.set_ylabel('Spearman Improvement (Geodesic - Euclidean)')
ax.set_title('Per-Goal Improvement from ISOMAP Graph')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axhline(np.mean(improvements), color='#555555', ls='--', lw=1.5,
           label=f'Mean improvement: {np.mean(improvements):+.4f}')
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/04_improvement.png")
plt.close(fig)
print(f"Saved 04_improvement.png")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"  Euclidean K-NN:  mean Spearman = {mean_e:.4f} (+/- {np.std(results_euclid):.4f})")
print(f"  Geodesic K-NN:   mean Spearman = {mean_g:.4f} (+/- {np.std(results_geodesic):.4f})")
print(f"  K-NN overlap:    {mean_overlap*100:.1f}%")
print(f"  Mean improvement: {mean_g - mean_e:+.4f}")
print(f"\nPer-goal (Euclid -> Geodesic):")
for gi in range(N_GOALS):
    label = ""
    if gi in bad_goals: label = " [BAD]"
    elif gi in good_goals: label = " [GOOD]"
    print(f"  Goal {gi}: {results_euclid[gi]:.4f} -> {results_geodesic[gi]:.4f} "
          f"({results_geodesic[gi] - results_euclid[gi]:+.4f}){label}")

print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
