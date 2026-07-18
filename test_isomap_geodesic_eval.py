"""ISOMAP graph evaluation with geodesic ground truth.

Goal: Show that ISOMAP distances (shortest paths through spatial K-NN graph)
capture Riemannian/geodesic structure by correlating with transition-geodesic
distances (dynamics-aware ground truth).

Evaluation:
1. Do ISOMAP distances correlate with transition-geodesic distances?
2. Does V from ISOMAP graph correlate with transition-geodesic distances?
3. Comparison: Euclidean graph V vs ISOMAP graph V, both against trans-geodesic.
"""

import torch
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

SAVE_DIR = "/tmp/isomap_geodesic_eval"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 30000  # reduced for speed
N_GOALS = 10
K_NEIGHBORS = 32
N_LANDMARKS = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mc_knn(query, database, K, mc_candidates=4096, chunk_size=8192, n_rounds=None):
    """MC K-NN search. query and database are torch tensors on device."""
    N_q = query.shape[0]
    N_d = database.shape[0]
    if n_rounds is None:
        n_rounds = min(50, max(1, N_d // mc_candidates))
    all_k_dists = torch.full((N_q, K), float('inf'), device=device)
    all_k_indices = torch.zeros((N_q, K), dtype=torch.long, device=device)
    rng = np.random.RandomState(42)
    with torch.no_grad():
        for r in range(n_rounds):
            cand_idx = torch.tensor(
                rng.choice(N_d, min(mc_candidates, N_d), replace=False), device=device)
            cand = database[cand_idx]
            for start in range(0, N_q, chunk_size):
                end = min(start + chunk_size, N_q)
                dists = torch.cdist(query[start:end], cand)
                ck = min(K, dists.shape[1])
                cd, cl = torch.topk(-dists, k=ck, dim=1)
                cd, ci = -cd, cand_idx[cl]
                md = torch.cat([all_k_dists[start:end], cd], dim=1)
                mi = torch.cat([all_k_indices[start:end], ci], dim=1)
                bd, bp = torch.topk(-md, k=K, dim=1)
                all_k_dists[start:end] = -bd
                all_k_indices[start:end] = mi.gather(1, bp)
    return all_k_dists.cpu().numpy(), all_k_indices.cpu().numpy()


def build_weighted_sparse(knn_dists, knn_indices, N, kernel='gaussian'):
    """Build symmetric sparse matrix from K-NN."""
    K = knn_dists.shape[1]
    if kernel == 'gaussian':
        local_sigma = np.clip(knn_dists[:, 1:2] * 2.0, 1e-3, None)
        data = np.exp(-(knn_dists ** 2) / (2 * local_sigma ** 2)).flatten()
    elif kernel == 'distance':
        data = knn_dists.flatten()
    else:
        data = np.ones(N * K)

    row = np.repeat(np.arange(N), K)
    col = knn_indices.flatten()
    W = sp.csr_matrix((data, (row, col)), shape=(N, N))
    W.setdiag(0)
    W_sym = (W + W.T) * 0.5
    return W_sym


def build_laplacian_from_W(W_sym):
    d = np.array(W_sym.sum(axis=1)).flatten()
    L = sp.diags(d) - W_sym
    return L, d


def solve_heat(L, goal_idx, sigma_perturb):
    N = L.shape[0]
    t = (sigma_perturb ** 2) * 500
    b = np.zeros(N)
    b[goal_idx] = 1.0
    A = sp.eye(N) + t * L
    M = sp.diags(1.0 / (A.diagonal() + 1e-8))
    u, info = spla.cg(A, b, rtol=1e-8, maxiter=5000, M=M)
    if info != 0:
        print(f"    [CG warn] code {info}")
    u = np.maximum(u, 1e-300)
    V = -np.log(u)
    V = V - V.min()
    V = V * 10.0 / (V.mean() + 1e-8)
    return V


def geodesic_knn_gpu(geo_dists_landmarks, K, chunk_size=500):
    """Find K nearest neighbors using landmark-approximated geodesic distances.
    geo_dists_landmarks: (L, N) array of distances from landmarks to all nodes.
    Uses GPU for the min-over-landmarks computation.
    """
    L_lm, N = geo_dists_landmarks.shape
    geo_t = torch.tensor(geo_dists_landmarks, dtype=torch.float32, device=device)  # (L, N)

    all_knn_dists = np.full((N, K), np.inf)
    all_knn_indices = np.zeros((N, K), dtype=np.int64)

    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_n = end - start

            # (L, chunk) distances from landmarks to chunk nodes
            d_chunk = geo_t[:, start:end]  # (L, chunk_n)

            # For each node i in chunk, d_approx[i,j] = min_l (d_chunk[l,i] + geo_t[l,j])
            # Process landmarks in batches to limit GPU memory
            d_approx = torch.full((chunk_n, N), float('inf'), device=device)
            lm_batch = 100
            for lb in range(0, L_lm, lm_batch):
                le = min(lb + lm_batch, L_lm)
                # (le-lb, chunk_n, 1) + (le-lb, 1, N) -> min over dim 0
                d_batch = d_chunk[lb:le, :, None] + geo_t[lb:le, None, :]  # (batch, chunk, N)
                d_batch_min = d_batch.min(dim=0).values  # (chunk, N)
                d_approx = torch.minimum(d_approx, d_batch_min)

            # Set self-distance to inf
            for ci in range(chunk_n):
                d_approx[ci, start + ci] = float('inf')

            # Top-K nearest
            topk_dists, topk_indices = torch.topk(-d_approx, k=K, dim=1)
            all_knn_dists[start:end] = (-topk_dists).cpu().numpy()
            all_knn_indices[start:end] = topk_indices.cpu().numpy()

            if (start // chunk_size) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    Geodesic K-NN: {end}/{N} ({elapsed:.1f}s)")

    print(f"    Geodesic K-NN total: {time.perf_counter()-t0:.1f}s")
    return all_knn_dists, all_knn_indices


# ── Main ─────────────────────────────────────────────────────────────

print("=" * 60)
print("ISOMAP Geodesic Evaluation")
print(f"Device: {device}, N={N_SUBSAMPLE}")
print("=" * 60)

# Load and subsample
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
z_tensor = torch.tensor(z_scaled, dtype=torch.float32, device=device)
zn_tensor = torch.tensor(z_next_scaled, dtype=torch.float32, device=device)

# ── 1. Ground truth: transition-geodesic (K=8) ──────────────────────

print("\n[1] Building transition graph (K=8, ground truth)...")
t0 = time.perf_counter()
# Exact K-NN from z_next to z_curr
trans_dists = np.zeros((N, 8))
trans_indices = np.zeros((N, 8), dtype=np.int64)
with torch.no_grad():
    for start in range(0, N, 8192):
        end = min(start + 8192, N)
        dists = torch.cdist(zn_tensor[start:end], z_tensor)
        for i in range(start, end):
            dists[i - start, i] = float('inf')
        cd, ci = torch.topk(-dists, k=8, dim=1)
        trans_dists[start:end] = (-cd).cpu().numpy()
        trans_indices[start:end] = ci.cpu().numpy()

W_trans = build_weighted_sparse(trans_dists, trans_indices, N, kernel='distance')
W_trans.eliminate_zeros()
n_comp, comp_labels = connected_components(W_trans, directed=False)
print(f"  Components: {n_comp}")
print(f"  Build: {time.perf_counter()-t0:.1f}s")

# Compute transition-geodesic from each goal
rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N, N_GOALS, replace=False)

print("  Computing transition-geodesic from goals...")
t0 = time.perf_counter()
trans_geo = dijkstra(W_trans, directed=False, indices=goal_indices)  # (N_GOALS, N)
print(f"  Dijkstra: {time.perf_counter()-t0:.1f}s")

for gi in range(N_GOALS):
    n_reach = np.isfinite(trans_geo[gi]).sum()
    print(f"    Goal {gi}: reachable {n_reach}/{N} ({n_reach/N*100:.1f}%)")

# ── 2. Euclidean K-NN graph (baseline) ───────────────────────────────

print("\n[2] Building Euclidean K-NN graph...")
knn_dists, knn_indices = mc_knn(z_tensor, z_tensor, K=K_NEIGHBORS)
# Zero out self
for i in range(N):
    mask = knn_indices[i] == i
    knn_dists[i, mask] = 0.0

sigma_euclid = np.median(knn_dists[:, 1]) * 2.0
W_euclid = build_weighted_sparse(knn_dists, knn_indices, N, kernel='gaussian')
L_euclid, d_euclid = build_laplacian_from_W(W_euclid)
print(f"  sigma_perturb: {sigma_euclid:.6f}")

# Also build distance-weighted version for ISOMAP Dijkstra
W_euclid_dist = build_weighted_sparse(knn_dists, knn_indices, N, kernel='distance')
W_euclid_dist.eliminate_zeros()

# ── 3. ISOMAP: geodesic distances through K-NN graph ────────────────

print(f"\n[3] ISOMAP: computing geodesic distances ({N_LANDMARKS} landmarks)...")
rng_lm = np.random.RandomState(42)
landmarks = rng_lm.choice(N, N_LANDMARKS, replace=False)

t0 = time.perf_counter()
isomap_dists = dijkstra(W_euclid_dist, directed=False, indices=landmarks)  # (L, N)
print(f"  Landmark Dijkstra: {time.perf_counter()-t0:.1f}s")
n_inf = np.isinf(isomap_dists).sum()
print(f"  Unreachable: {n_inf}/{isomap_dists.size} ({n_inf/isomap_dists.size*100:.1f}%)")

# ── 4. Key test: do ISOMAP distances correlate with transition-geodesic? ──

print(f"\n[4] ISOMAP distance vs Transition-geodesic distance (per goal)...")
print("    This is the core test: does ISOMAP capture Riemannian structure?")

isomap_vs_trans = []
euclid_vs_trans = []

for gi in range(N_GOALS):
    goal_idx = goal_indices[gi]

    # Euclidean distance from goal
    euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    # ISOMAP distance from goal (approximate via landmarks)
    isomap_d = np.full(N, np.inf)
    for l in range(N_LANDMARKS):
        d_via = isomap_dists[l, :] + isomap_dists[l, goal_idx]
        isomap_d = np.minimum(isomap_d, d_via)
    isomap_d[goal_idx] = 0

    # Transition-geodesic from goal
    trans_d = trans_geo[gi]

    finite = np.isfinite(trans_d) & np.isfinite(isomap_d)
    n_finite = finite.sum()

    if n_finite > 100:
        r_iso_trans, _ = spearmanr(isomap_d[finite], trans_d[finite])
        r_euc_trans, _ = spearmanr(euclid_d[finite], trans_d[finite])
    else:
        r_iso_trans = r_euc_trans = float('nan')

    isomap_vs_trans.append(r_iso_trans)
    euclid_vs_trans.append(r_euc_trans)

    print(f"    Goal {gi}: ISOMAP-vs-Trans={r_iso_trans:.4f}  Euclid-vs-Trans={r_euc_trans:.4f}  "
          f"(finite={n_finite}/{N})")

m_iso = np.nanmean(isomap_vs_trans)
m_euc = np.nanmean(euclid_vs_trans)
print(f"\n    MEAN: ISOMAP-vs-Trans={m_iso:.4f}  Euclid-vs-Trans={m_euc:.4f}")
print(f"    ISOMAP improvement over Euclidean: {m_iso - m_euc:+.4f}")

# ── 5. Build ISOMAP K-NN graph ──────────────────────────────────────

print(f"\n[5] Building ISOMAP K-NN graph (K={K_NEIGHBORS})...")
geo_knn_dists, geo_knn_indices = geodesic_knn_gpu(isomap_dists, K=K_NEIGHBORS, chunk_size=500)

sigma_isomap = np.median(geo_knn_dists[:, 1]) * 2.0
print(f"  sigma_perturb (ISOMAP): {sigma_isomap:.6f}")

W_isomap = build_weighted_sparse(geo_knn_dists, geo_knn_indices, N, kernel='gaussian')
L_isomap, d_isomap = build_laplacian_from_W(W_isomap)

degrees_iso = np.array(W_isomap.astype(bool).sum(axis=1)).flatten()
print(f"  Degree: min={degrees_iso.min()}, mean={degrees_iso.mean():.1f}, max={degrees_iso.max()}")

# K-NN overlap
overlap = 0
for i in range(0, N, N // 200):
    e_nn = set(knn_indices[i].tolist())
    g_nn = set(geo_knn_indices[i].tolist())
    overlap += len(e_nn & g_nn) / K_NEIGHBORS
overlap /= len(range(0, N, N // 200))
print(f"  K-NN overlap (Euclid vs ISOMAP): {overlap*100:.1f}%")

# ── 6. Evaluate V from both graphs against transition-geodesic ───────

print(f"\n[6] Evaluating V against transition-geodesic...")

results = {'Euclidean': [], 'ISOMAP': []}

for gi in range(N_GOALS):
    goal_idx = goal_indices[gi]
    trans_d = trans_geo[gi]
    finite = np.isfinite(trans_d)

    # Euclidean graph V
    V_e = solve_heat(L_euclid, goal_idx, sigma_euclid)
    if finite.sum() > 100:
        r_e, _ = spearmanr(V_e[finite], trans_d[finite])
    else:
        r_e = float('nan')
    results['Euclidean'].append(r_e)

    # ISOMAP graph V
    V_i = solve_heat(L_isomap, goal_idx, sigma_isomap)
    if finite.sum() > 100:
        r_i, _ = spearmanr(V_i[finite], trans_d[finite])
    else:
        r_i = float('nan')
    results['ISOMAP'].append(r_i)

    print(f"    Goal {gi}: Euclid V-vs-TransGeo={r_e:.4f}  ISOMAP V-vs-TransGeo={r_i:.4f}  "
          f"diff={r_i - r_e:+.4f}")

mean_e = np.nanmean(results['Euclidean'])
mean_i = np.nanmean(results['ISOMAP'])
print(f"\n    MEAN: Euclid={mean_e:.4f}  ISOMAP={mean_i:.4f}  diff={mean_i - mean_e:+.4f}")

# ── Plots ────────────────────────────────────────────────────────────

# Plot 1: ISOMAP distance vs Transition-geodesic (the core result)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
scatter_rng = np.random.RandomState(42)
si = scatter_rng.choice(N, 3000, replace=False)

for gi in range(N_GOALS):
    ax = axes[gi // 5, gi % 5]
    goal_idx = goal_indices[gi]
    trans_d = trans_geo[gi]
    euclid_d = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

    isomap_d = np.full(N, np.inf)
    for l in range(N_LANDMARKS):
        isomap_d = np.minimum(isomap_d, isomap_dists[l, :] + isomap_dists[l, goal_idx])
    isomap_d[goal_idx] = 0

    finite = np.isfinite(trans_d) & np.isfinite(isomap_d)
    si_f = si[finite[si]]

    ax.scatter(trans_d[si_f], isomap_d[si_f], s=3, alpha=0.3, color='#5faac0', rasterized=True,
               label='ISOMAP')
    ax.scatter(trans_d[si_f], euclid_d[si_f], s=3, alpha=0.3, color='#e8a0bf', rasterized=True,
               label='Euclidean')

    r_iso = isomap_vs_trans[gi]
    r_euc = euclid_vs_trans[gi]
    ax.set_title(f'Goal {gi}\nISO r={r_iso:.3f}, Euc r={r_euc:.3f}', fontsize=10)
    ax.set_xlabel('Trans-Geodesic', fontsize=8)
    if gi % 5 == 0:
        ax.set_ylabel('Distance to Goal', fontsize=8)
    if gi == 0:
        ax.legend(markerscale=5, fontsize=8, frameon=True, facecolor='white', edgecolor='#ccc')

plt.suptitle('Distance Metrics vs Transition-Geodesic Ground Truth', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_distance_correlation.png", bbox_inches='tight')
plt.close(fig)
print(f"\nSaved 01_distance_correlation.png")

# Plot 2: V vs transition-geodesic for both methods
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
for gi in range(N_GOALS):
    ax = axes[gi // 5, gi % 5]
    goal_idx = goal_indices[gi]
    trans_d = trans_geo[gi]
    finite = np.isfinite(trans_d)
    si_f = si[finite[si]]

    V_e = solve_heat(L_euclid, goal_idx, sigma_euclid)
    V_i = solve_heat(L_isomap, goal_idx, sigma_isomap)

    ax.scatter(trans_d[si_f], V_e[si_f], s=3, alpha=0.3, color='#e8a0bf',
               label='Euclidean', rasterized=True)
    ax.scatter(trans_d[si_f], V_i[si_f], s=3, alpha=0.3, color='#5faac0',
               label='ISOMAP', rasterized=True)

    r_e = results['Euclidean'][gi]
    r_i = results['ISOMAP'][gi]
    ax.set_title(f'Goal {gi}\nEuc r={r_e:.3f}, ISO r={r_i:.3f}', fontsize=10)
    ax.set_xlabel('Trans-Geodesic', fontsize=8)
    if gi % 5 == 0:
        ax.set_ylabel('V (heat)', fontsize=8)
    if gi == 0:
        ax.legend(markerscale=5, fontsize=8, frameon=True, facecolor='white', edgecolor='#ccc')

plt.suptitle('Heat V vs Transition-Geodesic Ground Truth', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_V_vs_trans_geodesic.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 02_V_vs_trans_geodesic.png")

# Plot 3: Per-goal comparison bars
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 3a: Distance correlation with trans-geodesic
ax = axes[0]
x = np.arange(N_GOALS)
width = 0.35
ax.bar(x, euclid_vs_trans, width, label='Euclidean', color='#e8a0bf',
       edgecolor='white', linewidth=0.8)
ax.bar(x + width, isomap_vs_trans, width, label='ISOMAP', color='#5faac0',
       edgecolor='white', linewidth=0.8)
ax.axhline(m_euc, color='#e8a0bf', ls='--', lw=1.5, alpha=0.7)
ax.axhline(m_iso, color='#5faac0', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('Goal')
ax.set_ylabel('Spearman(distance, trans-geodesic)')
ax.set_title(f'Distance Metric Quality\nEuclid mean={m_euc:.3f}, ISOMAP mean={m_iso:.3f}')
ax.set_xticks(x + width/2)
ax.set_xticklabels([str(i) for i in range(N_GOALS)])
ax.legend(frameon=True, facecolor='white', edgecolor='#ccc')

# 3b: V correlation with trans-geodesic
ax = axes[1]
ax.bar(x, results['Euclidean'], width, label='Euclidean V', color='#e8a0bf',
       edgecolor='white', linewidth=0.8)
ax.bar(x + width, results['ISOMAP'], width, label='ISOMAP V', color='#5faac0',
       edgecolor='white', linewidth=0.8)
ax.axhline(mean_e, color='#e8a0bf', ls='--', lw=1.5, alpha=0.7)
ax.axhline(mean_i, color='#5faac0', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('Goal')
ax.set_ylabel('Spearman(V, trans-geodesic)')
ax.set_title(f'V Quality (vs Trans-Geodesic)\nEuclid mean={mean_e:.3f}, ISOMAP mean={mean_i:.3f}')
ax.set_xticks(x + width/2)
ax.set_xticklabels([str(i) for i in range(N_GOALS)])
ax.legend(frameon=True, facecolor='white', edgecolor='#ccc')

plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/03_per_goal_bars.png")
plt.close(fig)
print(f"Saved 03_per_goal_bars.png")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"\n  DISTANCE METRIC vs TRANSITION-GEODESIC:")
print(f"    Euclidean:  mean r = {m_euc:.4f}")
print(f"    ISOMAP:     mean r = {m_iso:.4f}  ({m_iso - m_euc:+.4f})")
print(f"\n  HEAT V vs TRANSITION-GEODESIC:")
print(f"    Euclidean:  mean r = {mean_e:.4f}")
print(f"    ISOMAP:     mean r = {mean_i:.4f}  ({mean_i - mean_e:+.4f})")
print(f"\n  K-NN overlap: {overlap*100:.1f}%")
print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
