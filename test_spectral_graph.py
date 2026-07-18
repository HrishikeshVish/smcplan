"""Test spectral embedding + K-NN graph for heat equation.

Approach:
1. Build spatial K-NN graph (baseline)
2. Compute top-m eigenvectors of normalized Laplacian
3. Embed nodes in spectral coordinates
4. Rebuild K-NN in spectral space
5. Solve heat equation on new graph, compare Spearman
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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

SAVE_DIR = "/tmp/spectral_plots"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 55000
N_GOALS = 10
K_NEIGHBORS = 32
SPECTRAL_DIMS = [10, 20, 50, 100]

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
                chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
                chunk_k_dists = -chunk_k_dists
                chunk_k_indices = cand_idx[chunk_k_local]
                merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
                merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
                best_dists, best_pos = torch.topk(-merged_dists, k=K, dim=1)
                all_k_dists[start:end] = -best_dists
                all_k_indices[start:end] = merged_indices.gather(1, best_pos)

    sigma_perturb = (all_k_dists[:, 1].unsqueeze(1) * 2.0).median().item()
    return all_k_dists.cpu().numpy(), all_k_indices.cpu().numpy(), sigma_perturb


def build_laplacian(knn_dists, knn_indices, N):
    local_sigma = knn_dists[:, 1:2] * 2.0
    local_sigma = np.clip(local_sigma, 1e-3, None)
    W_local = np.exp(-(knn_dists ** 2) / (2 * local_sigma ** 2))
    K = knn_dists.shape[1]
    row_idx = np.repeat(np.arange(N), K)
    col_idx = knn_indices.flatten()
    data = W_local.flatten()
    W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym
    return L, W_sym, d


def compute_spectral_embedding(L, d, n_dims):
    """Compute spectral embedding from normalized Laplacian eigenvectors."""
    N = L.shape[0]
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt  # normalized Laplacian

    t0 = time.perf_counter()
    # Compute smallest eigenvalues (skip the trivial zero eigenvalue)
    eigenvalues, eigenvectors = spla.eigsh(L_norm, k=n_dims + 1, which='SM', tol=1e-6)
    t1 = time.perf_counter()

    # Sort by eigenvalue
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip first eigenvector (constant, eigenvalue ~0)
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    # Transform back: phi_unnorm = D^{-1/2} @ phi_norm
    spectral_coords = D_inv_sqrt @ eigenvectors

    print(f"  Eigendecomposition ({n_dims} dims): {t1-t0:.2f}s")
    print(f"  Eigenvalues: [{eigenvalues[0]:.6f}, ..., {eigenvalues[-1]:.6f}]")
    print(f"  Spectral coord range: [{spectral_coords.min():.4f}, {spectral_coords.max():.4f}]")

    return spectral_coords, eigenvalues


def build_spectral_knn_graph(spectral_coords, K=32, chunk_size=8192):
    """Build K-NN graph in spectral embedding space."""
    N = spectral_coords.shape[0]
    spec_tensor = torch.tensor(spectral_coords, dtype=torch.float32, device=device)
    all_k_dists = torch.full((N, K), float('inf'), device=device)
    all_k_indices = torch.zeros((N, K), dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dists = torch.cdist(spec_tensor[start:end], spec_tensor)
            # Set self-distance to inf
            for i in range(start, end):
                dists[i - start, i] = float('inf')
            chunk_k_dists, chunk_k_indices = torch.topk(-dists, k=K, dim=1)
            all_k_dists[start:end] = -chunk_k_dists
            all_k_indices[start:end] = chunk_k_indices
    t1 = time.perf_counter()
    print(f"  Spectral K-NN: {t1-t0:.2f}s")

    knn_dists = all_k_dists.cpu().numpy()
    knn_indices = all_k_indices.cpu().numpy()

    # Build Laplacian with Gaussian kernel in spectral space
    local_sigma = knn_dists[:, 1:2] * 2.0
    local_sigma = np.clip(local_sigma, 1e-3, None)
    W_local = np.exp(-(knn_dists ** 2) / (2 * local_sigma ** 2))

    row_idx = np.repeat(np.arange(N), K)
    col_idx = knn_indices.flatten()
    data = W_local.flatten()
    W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
    W_0.setdiag(0)
    W_sym = (W_0 + W_0.T) * 0.5
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym

    sigma_perturb = np.median(knn_dists[:, 1]) * 2.0
    degrees = np.array(W_sym.astype(bool).sum(axis=1)).flatten()
    print(f"  sigma_perturb: {sigma_perturb:.6f}")
    print(f"  Degree: min={degrees.min()}, mean={degrees.mean():.1f}, max={degrees.max()}")

    return L, W_sym, sigma_perturb


def solve_heat(L, goal_idx, sigma_perturb):
    N = L.shape[0]
    h2 = sigma_perturb ** 2
    t = h2 * 500
    b = np.zeros(N)
    b[goal_idx] = 1.0
    A_heat = sp.eye(N) + t * L
    M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
    u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
    if info != 0:
        print(f"  [Warning] CG code {info}")
    u = np.maximum(u, 1e-300)
    V_raw = -np.log(u)
    V_raw = V_raw - V_raw.min()
    scale = 10.0 / (V_raw.mean() + 1e-8)
    return V_raw * scale


def eval_goals(L, sigma_perturb, z_scaled, goal_indices):
    results = []
    for gi, goal_idx in enumerate(goal_indices):
        euclid_dists = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)
        V = solve_heat(L, goal_idx, sigma_perturb)
        corr, _ = spearmanr(V, euclid_dists)
        results.append(corr)
    return results


# ── Main ─────────────────────────────────────────────────────────────

print("=" * 60)
print("Spectral Embedding Graph Test")
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

# Goals
rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N, N_GOALS, replace=False)

# ── Baseline: Euclidean K-NN ─────────────────────────────────────────

print("\n[Baseline] Building Euclidean K-NN graph...")
knn_dists, knn_indices, sigma_euclid = build_knn_graph(z_scaled, K=K_NEIGHBORS)
L_euclid, W_euclid, d_euclid = build_laplacian(knn_dists, knn_indices, N)
print(f"  sigma_perturb: {sigma_euclid:.6f}")

print("\n[Baseline] Evaluating goals...")
results_euclid = eval_goals(L_euclid, sigma_euclid, z_scaled, goal_indices)
mean_euclid = np.mean(results_euclid)
print(f"  Mean Spearman: {mean_euclid:.4f}")
for gi in range(N_GOALS):
    print(f"    Goal {gi}: {results_euclid[gi]:.4f}")

# ── Spectral embedding at different dims ──────────────────────────────

all_results = {'Euclidean': results_euclid}

for n_dims in SPECTRAL_DIMS:
    print(f"\n{'=' * 60}")
    print(f"[Spectral dim={n_dims}]")

    spectral_coords, eigenvalues = compute_spectral_embedding(L_euclid, d_euclid, n_dims)

    print(f"  Building K-NN in spectral space...")
    L_spec, W_spec, sigma_spec = build_spectral_knn_graph(spectral_coords, K=K_NEIGHBORS)

    print(f"  Evaluating goals...")
    results_spec = eval_goals(L_spec, sigma_spec, z_scaled, goal_indices)
    mean_spec = np.mean(results_spec)
    print(f"  Mean Spearman: {mean_spec:.4f} (diff={mean_spec - mean_euclid:+.4f})")

    all_results[f'Spectral-{n_dims}'] = results_spec

    # K-NN overlap with Euclidean
    # Quick check: sample 500 nodes
    overlap_total = 0
    n_check = 500
    check_idx = rng.choice(N, n_check, replace=False)
    spec_tensor = torch.tensor(spectral_coords, dtype=torch.float32, device=device)
    with torch.no_grad():
        for ci in check_idx:
            dists = torch.cdist(spec_tensor[ci:ci+1], spec_tensor).squeeze(0)
            dists[ci] = float('inf')
            _, spec_nn = torch.topk(-dists, k=K_NEIGHBORS)
            spec_nn_set = set(spec_nn.cpu().numpy().tolist())
            euclid_nn_set = set(knn_indices[ci].tolist())
            overlap_total += len(spec_nn_set & euclid_nn_set) / K_NEIGHBORS
    mean_overlap = overlap_total / n_check
    print(f"  K-NN overlap with Euclidean: {mean_overlap*100:.1f}%")

    for gi in range(N_GOALS):
        diff = results_spec[gi] - results_euclid[gi]
        print(f"    Goal {gi}: {results_euclid[gi]:.4f} -> {results_spec[gi]:.4f} ({diff:+.4f})")

# ── Plots ────────────────────────────────────────────────────────────

colors = {
    'Euclidean': '#e8a0bf',
    'Spectral-10': '#a0c4e8',
    'Spectral-20': '#7fb7d4',
    'Spectral-50': '#5faac0',
    'Spectral-100': '#3f9dac',
}

# Plot 1: Per-goal Spearman for all methods
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(N_GOALS)
n_methods = len(all_results)
width = 0.8 / n_methods
for i, (label, results) in enumerate(all_results.items()):
    ax.bar(x + i * width, results, width, label=label,
           color=colors.get(label, '#cccccc'), edgecolor='white', linewidth=0.5)
ax.set_xlabel('Goal Index')
ax.set_ylabel('Spearman(V, Euclidean dist)')
ax.set_title('Heat V Quality: Euclidean vs Spectral Embedding Graphs')
ax.set_xticks(x + width * (n_methods - 1) / 2)
ax.set_xticklabels([str(i) for i in range(N_GOALS)])
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_spearman_all.png")
plt.close(fig)
print(f"\nSaved 01_spearman_all.png")

# Plot 2: Mean Spearman vs spectral dimension
fig, ax = plt.subplots(figsize=(8, 5))
dims_list = [0] + SPECTRAL_DIMS
means = [np.mean(all_results['Euclidean'])]
stds = [np.std(all_results['Euclidean'])]
for d in SPECTRAL_DIMS:
    r = all_results[f'Spectral-{d}']
    means.append(np.mean(r))
    stds.append(np.std(r))
ax.errorbar(dims_list, means, yerr=stds, fmt='o-', color='#5faac0',
            capsize=5, markersize=8, linewidth=2)
ax.set_xlabel('Spectral Embedding Dimension (0 = Euclidean)')
ax.set_ylabel('Mean Spearman')
ax.set_title('Mean V Quality vs Spectral Dimension')
ax.axhline(means[0], color='#e8a0bf', ls='--', lw=1.5, alpha=0.7, label='Euclidean baseline')
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_mean_vs_dim.png")
plt.close(fig)
print(f"Saved 02_mean_vs_dim.png")

# Plot 3: Per-goal improvement for best spectral dim
best_dim_key = max([k for k in all_results if k != 'Euclidean'],
                   key=lambda k: np.mean(all_results[k]))
best_results = all_results[best_dim_key]
improvements = [s - e for e, s in zip(results_euclid, best_results)]

fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ['#7fb7d4' if imp > 0 else '#d47f9e' for imp in improvements]
ax.bar(range(N_GOALS), improvements, color=bar_colors, edgecolor='white', linewidth=0.8)
ax.set_xlabel('Goal Index')
ax.set_ylabel(f'Spearman Improvement ({best_dim_key} - Euclidean)')
ax.set_title(f'Per-Goal Improvement: {best_dim_key}')
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.axhline(np.mean(improvements), color='#555555', ls='--', lw=1.5,
           label=f'Mean: {np.mean(improvements):+.4f}')
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/03_improvement.png")
plt.close(fig)
print(f"Saved 03_improvement.png")

# Plot 4: Eigenvalue spectrum
print("\nComputing full eigenspectrum for plot...")
_, eigenvalues_full = compute_spectral_embedding(L_euclid, d_euclid, max(SPECTRAL_DIMS))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(range(1, len(eigenvalues_full)+1), eigenvalues_full, 'o-',
             color='#5faac0', markersize=3)
axes[0].set_xlabel('Eigenvalue Index')
axes[0].set_ylabel('Eigenvalue')
axes[0].set_title('Laplacian Eigenvalue Spectrum')
for d in SPECTRAL_DIMS:
    axes[0].axvline(d, color='#d47f9e', ls=':', alpha=0.5, lw=1)
    axes[0].text(d, eigenvalues_full.max() * 0.95, f'd={d}', fontsize=8, ha='center')

# Cumulative "variance" (sum of inverse eigenvalues)
cum_inv = np.cumsum(1.0 / eigenvalues_full) / np.sum(1.0 / eigenvalues_full)
axes[1].plot(range(1, len(cum_inv)+1), cum_inv, 'o-', color='#5faac0', markersize=3)
axes[1].set_xlabel('Number of Eigenvectors')
axes[1].set_ylabel('Cumulative Weight (1/lambda)')
axes[1].set_title('Spectral Information Captured')
for d in SPECTRAL_DIMS:
    axes[1].axvline(d, color='#d47f9e', ls=':', alpha=0.5, lw=1)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/04_eigenspectrum.png")
plt.close(fig)
print(f"Saved 04_eigenspectrum.png")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
for label, results in all_results.items():
    m = np.mean(results)
    s = np.std(results)
    diff = m - mean_euclid
    print(f"  {label:15s}: mean={m:.4f} (+/- {s:.4f})  diff={diff:+.4f}")

print(f"\nBest: {best_dim_key} (mean={np.mean(best_results):.4f})")
print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
