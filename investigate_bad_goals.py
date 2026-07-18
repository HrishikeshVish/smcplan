"""Investigate why goals 3 and 5 have bad Spearman correlation.
Compare graph shortest-path distance vs Euclidean distance for
good goals (1, 8) and bad goals (3, 5)."""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#fafafa',
    'figure.dpi': 150,
})

SAVE_DIR = "/tmp/reacher_goal_investigation"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 55000

# ── Load and subsample ────────────────────────────────────────────────

print("Loading buffer...")
loaded = torch.load(BUF_PATH, map_location='cpu', weights_only=True)
N_total = loaded['z_curr'].shape[0]

rng = np.random.RandomState(42)
idx = rng.choice(N_total, N_SUBSAMPLE, replace=False)
z_curr = loaded['z_curr'][idx].numpy()
z_next = loaded['z_next'][idx].numpy()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}, N={N_SUBSAMPLE}")

# Scale
scaler = StandardScaler()
z_combined = np.vstack([z_curr, z_next])
scaler.fit(z_combined)
z_scaled = scaler.transform(z_curr)
z_next_scaled = scaler.transform(z_next)

# ── Build transition graph (each node -> nn(z_next)) ──────────────────
# This gives us the dynamical reachability structure

print("Building transition graph...")
z_t = torch.tensor(z_scaled, dtype=torch.float32, device=device)
z_nt = torch.tensor(z_next_scaled, dtype=torch.float32, device=device)

# Find nearest node to each z_next (exact, chunked)
chunk_size = 4096
nn_indices = torch.zeros(N_SUBSAMPLE, dtype=torch.long, device=device)
nn_dists = torch.zeros(N_SUBSAMPLE, device=device)

with torch.no_grad():
    for start in range(0, N_SUBSAMPLE, chunk_size):
        end = min(start + chunk_size, N_SUBSAMPLE)
        dists = torch.cdist(z_nt[start:end], z_t)
        min_d, min_i = dists.min(dim=1)
        nn_indices[start:end] = min_i
        nn_dists[start:end] = min_d

nn_indices_np = nn_indices.cpu().numpy()
nn_dists_np = nn_dists.cpu().numpy()

# Build sparse transition adjacency (directed: i -> nn(z_next[i]))
rows = np.arange(N_SUBSAMPLE)
cols = nn_indices_np
W_trans = sp.csr_matrix((np.ones(N_SUBSAMPLE), (rows, cols)), shape=(N_SUBSAMPLE, N_SUBSAMPLE))
W_trans.setdiag(0)
W_trans_sym = (W_trans + W_trans.T)
W_trans_sym[W_trans_sym > 0] = 1  # binary

n_comp, comp_labels = connected_components(W_trans_sym, directed=False)
print(f"Transition graph: {n_comp} components")

# ── Build spatial K-NN graph (same as heat solver) ────────────────────

print("Building spatial K-NN graph...")
K = 32
mc_candidates = 4096
all_k_dists = torch.full((N_SUBSAMPLE, K), float('inf'), device=device)
all_k_indices = torch.zeros((N_SUBSAMPLE, K), dtype=torch.long, device=device)
n_rounds = min(50, max(1, N_SUBSAMPLE // mc_candidates))
rng_knn = np.random.RandomState(42)

with torch.no_grad():
    for r in range(n_rounds):
        cand_idx = torch.tensor(
            rng_knn.choice(N_SUBSAMPLE, min(mc_candidates, N_SUBSAMPLE), replace=False),
            device=device
        )
        cand_nodes = z_t[cand_idx]
        for start in range(0, N_SUBSAMPLE, chunk_size):
            end = min(start + chunk_size, N_SUBSAMPLE)
            dists = torch.cdist(z_nt[start:end], cand_nodes)
            ck = min(K, dists.shape[1])
            chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
            chunk_k_dists = -chunk_k_dists
            chunk_k_indices = cand_idx[chunk_k_local]
            merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
            merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
            best_dists, best_pos = torch.topk(-merged_dists, k=K, dim=1)
            all_k_dists[start:end] = -best_dists
            all_k_indices[start:end] = merged_indices.gather(1, best_pos)

    local_sigma = all_k_dists[:, 1].unsqueeze(1) * 2.0
    local_sigma = torch.clamp(local_sigma, min=1e-3)
    sigma_perturb = local_sigma.median().item()
    W_local = torch.exp(-(all_k_dists ** 2) / (2 * local_sigma ** 2))

W_local_np = W_local.cpu().numpy()
row_idx = np.repeat(np.arange(N_SUBSAMPLE), K)
col_idx = all_k_indices.cpu().numpy().flatten()
data = W_local_np.flatten()
W_knn = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N_SUBSAMPLE, N_SUBSAMPLE))
W_knn.setdiag(0)
W_knn_sym = (W_knn + W_knn.T) * 0.5

# Laplacian for heat solve
d_knn = np.array(W_knn_sym.sum(axis=1)).flatten()
L_knn = sp.diags(d_knn) - W_knn_sym

print(f"K-NN graph: sigma_perturb={sigma_perturb:.6f}")

# ── Goal indices (same as previous runs) ──────────────────────────────

rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N_SUBSAMPLE, 10, replace=False)

# Goals: 1 (good, 0.679), 8 (good, 0.670), 3 (bad, -0.092), 5 (bad, -0.027)
study_goals = {
    'Goal 1 (GOOD)': goal_indices[1],
    'Goal 8 (GOOD)': goal_indices[8],
    'Goal 3 (BAD)':  goal_indices[3],
    'Goal 5 (BAD)':  goal_indices[5],
}

print(f"\nStudy goals: {study_goals}")

# ── Compute graph shortest-path distance from each study goal ─────────
# Use the K-NN graph with unit weights (hop count = topological distance)

print("\nComputing graph shortest paths (K-NN, hop distance)...")
W_knn_binary = W_knn_sym.copy()
W_knn_binary[W_knn_binary > 0] = 1

# Also compute on transition graph
print("Computing graph shortest paths (transition, hop distance)...")

# For each goal, compute shortest path to all nodes
# This is O(N log N) per goal with Dijkstra
goal_sp_knn = {}
goal_sp_trans = {}

for label, goal_idx in study_goals.items():
    # K-NN graph shortest paths (hops)
    sp_dists = shortest_path(W_knn_binary, directed=False, indices=goal_idx,
                              method='D', unweighted=True)
    goal_sp_knn[label] = sp_dists

    # Transition graph shortest paths (hops)
    sp_dists_t = shortest_path(W_trans_sym, directed=False, indices=goal_idx,
                                method='D', unweighted=True)
    goal_sp_trans[label] = sp_dists_t

    # Stats
    reachable_knn = np.isfinite(sp_dists).sum()
    reachable_trans = np.isfinite(sp_dists_t).sum()
    print(f"  {label} (idx={goal_idx}):")
    print(f"    K-NN reachable: {reachable_knn}/{N_SUBSAMPLE} ({reachable_knn/N_SUBSAMPLE*100:.1f}%)")
    print(f"    Trans reachable: {reachable_trans}/{N_SUBSAMPLE} ({reachable_trans/N_SUBSAMPLE*100:.1f}%)")

# ── Euclidean distance from each goal ─────────────────────────────────

goal_euclid = {}
for label, goal_idx in study_goals.items():
    goal_euclid[label] = np.linalg.norm(z_scaled - z_scaled[goal_idx], axis=1)

# ── Heat solve V for each goal ────────────────────────────────────────

print("\nSolving heat equation for each study goal...")
goal_V = {}
h2 = sigma_perturb ** 2
t = h2 * 500

for label, goal_idx in study_goals.items():
    b = np.zeros(N_SUBSAMPLE)
    b[goal_idx] = 1.0
    A_heat = sp.eye(N_SUBSAMPLE) + t * L_knn
    M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
    u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
    u = np.maximum(u, 1e-300)
    V_raw = -np.log(u)
    V_raw = V_raw - V_raw.min()
    V_np = V_raw * (10.0 / (V_raw.mean() + 1e-8))
    goal_V[label] = V_np

    corr_euclid, _ = spearmanr(V_np, goal_euclid[label])
    corr_sp_knn, _ = spearmanr(V_np, goal_sp_knn[label][np.isfinite(goal_sp_knn[label])])
    corr_sp_trans, _ = spearmanr(V_np[np.isfinite(goal_sp_trans[label])],
                                  goal_sp_trans[label][np.isfinite(goal_sp_trans[label])])
    print(f"  {label}: V vs euclid={corr_euclid:.4f}, V vs knn_hops={corr_sp_knn:.4f}, "
          f"V vs trans_hops={corr_sp_trans:.4f}")

# ── Key diagnostic: Euclidean vs graph distance ───────────────────────

print("\n" + "=" * 60)
print("EUCLIDEAN vs GRAPH DISTANCE CORRELATION")
print("=" * 60)
for label in study_goals:
    euclid = goal_euclid[label]
    sp_knn = goal_sp_knn[label]
    sp_trans = goal_sp_trans[label]

    # Euclid vs K-NN hops
    finite_knn = np.isfinite(sp_knn)
    corr_ek, _ = spearmanr(euclid[finite_knn], sp_knn[finite_knn])

    # Euclid vs transition hops
    finite_trans = np.isfinite(sp_trans)
    corr_et, _ = spearmanr(euclid[finite_trans], sp_trans[finite_trans])

    print(f"  {label}:")
    print(f"    Euclid vs KNN-hops:   Spearman={corr_ek:.4f}")
    print(f"    Euclid vs Trans-hops: Spearman={corr_et:.4f}")

    # Check for manifold folding: nodes close in Euclid but far in graph
    # (or vice versa)
    euclid_pct = np.percentile(euclid, [10, 25, 50, 75, 90])
    close_euclid = euclid < euclid_pct[0]  # closest 10% by Euclidean
    far_euclid = euclid > euclid_pct[3]    # farthest 25% by Euclidean

    if finite_knn[close_euclid].any():
        hops_for_close = sp_knn[close_euclid & finite_knn]
        hops_for_far = sp_knn[far_euclid & finite_knn]
        print(f"    Closest 10% (Euclid): mean KNN-hops={hops_for_close.mean():.1f}, "
              f"max={hops_for_close.max():.0f}")
        print(f"    Farthest 25% (Euclid): mean KNN-hops={hops_for_far.mean():.1f}, "
              f"min={hops_for_far.min():.0f}")

        # Folding metric: fraction of Euclidean-close nodes that are graph-far
        median_hops = np.median(sp_knn[finite_knn])
        fold_frac = (hops_for_close > median_hops).mean()
        print(f"    Folding: {fold_frac*100:.1f}% of Euclid-close nodes are graph-far (>{median_hops:.0f} hops)")

# ── Transition chain analysis ─────────────────────────────────────────

print("\n" + "=" * 60)
print("TRANSITION CHAIN ANALYSIS")
print("=" * 60)

# For each goal, trace forward transitions from nearby nodes
# and see if they converge toward or away from the goal
for label, goal_idx in study_goals.items():
    euclid = goal_euclid[label]

    # Pick 100 nodes closest to goal (by Euclidean)
    close_nodes = np.argsort(euclid)[:100]

    # For each close node, follow the transition chain for 10 steps
    # and track distance to goal
    chain_dists = []
    for node in close_nodes:
        dists_along_chain = [euclid[node]]
        current = node
        for step in range(10):
            next_node = nn_indices_np[current]
            if next_node == current:
                break
            dists_along_chain.append(euclid[next_node])
            current = next_node
        chain_dists.append(dists_along_chain)

    # How many chains move toward goal vs away?
    converging = 0
    diverging = 0
    for chain in chain_dists:
        if len(chain) >= 3:
            if chain[-1] < chain[0]:
                converging += 1
            else:
                diverging += 1

    avg_start = np.mean([c[0] for c in chain_dists])
    avg_end = np.mean([c[-1] for c in chain_dists if len(c) > 1])

    print(f"\n  {label} (idx={goal_idx}):")
    print(f"    100 closest nodes, 10-step chains:")
    print(f"    Converging toward goal: {converging}, Diverging: {diverging}")
    print(f"    Avg distance: start={avg_start:.4f} -> end={avg_end:.4f}")

    # Also check: from RANDOM nodes, do transitions lead toward this goal?
    random_nodes = rng.choice(N_SUBSAMPLE, 200, replace=False)
    rand_converge = 0
    rand_diverge = 0
    for node in random_nodes:
        d_start = euclid[node]
        current = node
        for step in range(10):
            next_node = nn_indices_np[current]
            if next_node == current:
                break
            current = next_node
        d_end = euclid[current]
        if d_end < d_start:
            rand_converge += 1
        else:
            rand_diverge += 1
    print(f"    200 random nodes, 10-step chains:")
    print(f"    Converging: {rand_converge}, Diverging: {rand_diverge}")

# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

soft_colors = sns.color_palette("pastel")
accent_colors = sns.color_palette("Set2")

# ── t-SNE (compute once) ─────────────────────────────────────────────

print("\nComputing t-SNE...")
tsne_n = 8000
tsne_idx = rng.choice(N_SUBSAMPLE, tsne_n, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
z_2d = tsne.fit_transform(z_scaled[tsne_idx])

# ── Plot 1: Euclid vs graph-hops scatter for each goal ────────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 10))

for gi, label in enumerate(study_goals):
    euclid = goal_euclid[label]
    sp_knn = goal_sp_knn[label]
    sp_trans = goal_sp_trans[label]
    is_good = 'GOOD' in label

    scatter_idx = rng.choice(N_SUBSAMPLE, 4000, replace=False)
    color = accent_colors[0] if is_good else accent_colors[3]

    # Top row: Euclid vs KNN hops
    ax = axes[0, gi]
    finite = np.isfinite(sp_knn[scatter_idx])
    ax.scatter(euclid[scatter_idx][finite], sp_knn[scatter_idx][finite],
               s=2, alpha=0.3, color=color, rasterized=True)
    corr, _ = spearmanr(euclid[scatter_idx][finite], sp_knn[scatter_idx][finite])
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('KNN Graph Hops')
    ax.set_title(f'{label}\nEuclid vs KNN-hops (r={corr:.3f})', fontsize=10)

    # Bottom row: Euclid vs transition hops
    ax = axes[1, gi]
    finite = np.isfinite(sp_trans[scatter_idx])
    if finite.sum() > 10:
        ax.scatter(euclid[scatter_idx][finite], sp_trans[scatter_idx][finite],
                   s=2, alpha=0.3, color=color, rasterized=True)
        corr, _ = spearmanr(euclid[scatter_idx][finite], sp_trans[scatter_idx][finite])
        ax.set_title(f'Euclid vs Trans-hops (r={corr:.3f})', fontsize=10)
    else:
        ax.set_title('Too few reachable nodes', fontsize=10)
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Transition Graph Hops')

plt.suptitle('Euclidean Distance vs Graph Shortest-Path Distance', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_euclid_vs_graph_dist.png", bbox_inches='tight')
plt.close(fig)
print("Saved 01_euclid_vs_graph_dist.png")

# ── Plot 2: V vs graph-hops (should be better than V vs Euclid) ──────

fig, axes = plt.subplots(2, 4, figsize=(24, 10))

for gi, label in enumerate(study_goals):
    V_np = goal_V[label]
    sp_knn = goal_sp_knn[label]
    sp_trans = goal_sp_trans[label]
    is_good = 'GOOD' in label
    color = accent_colors[0] if is_good else accent_colors[3]
    scatter_idx = rng.choice(N_SUBSAMPLE, 4000, replace=False)

    # Top: V vs KNN hops
    ax = axes[0, gi]
    finite = np.isfinite(sp_knn[scatter_idx])
    ax.scatter(sp_knn[scatter_idx][finite], V_np[scatter_idx][finite],
               s=2, alpha=0.3, color=color, rasterized=True)
    corr, _ = spearmanr(sp_knn[scatter_idx][finite], V_np[scatter_idx][finite])
    ax.set_xlabel('KNN Graph Hops')
    ax.set_ylabel('V (heat)')
    ax.set_title(f'{label}\nV vs KNN-hops (r={corr:.3f})', fontsize=10)

    # Bottom: V vs transition hops
    ax = axes[1, gi]
    finite = np.isfinite(sp_trans[scatter_idx])
    if finite.sum() > 10:
        ax.scatter(sp_trans[scatter_idx][finite], V_np[scatter_idx][finite],
                   s=2, alpha=0.3, color=color, rasterized=True)
        corr, _ = spearmanr(sp_trans[scatter_idx][finite], V_np[scatter_idx][finite])
        ax.set_title(f'V vs Trans-hops (r={corr:.3f})', fontsize=10)
    else:
        ax.set_title('Too few reachable nodes', fontsize=10)
    ax.set_xlabel('Transition Graph Hops')
    ax.set_ylabel('V (heat)')

plt.suptitle('Heat V vs Graph Shortest-Path Distance', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_V_vs_graph_dist.png", bbox_inches='tight')
plt.close(fig)
print("Saved 02_V_vs_graph_dist.png")

# ── Plot 3: t-SNE colored by graph hops from each goal ───────────────

fig, axes = plt.subplots(2, 4, figsize=(24, 10))

for gi, label in enumerate(study_goals):
    is_good = 'GOOD' in label

    # Top: KNN hops
    ax = axes[0, gi]
    sp_sub = goal_sp_knn[label][tsne_idx]
    sp_sub[~np.isfinite(sp_sub)] = sp_sub[np.isfinite(sp_sub)].max() + 1
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=sp_sub, cmap='viridis',
                    s=4, alpha=0.7, rasterized=True,
                    vmin=0, vmax=np.percentile(sp_sub, 95))
    plt.colorbar(sc, ax=ax, label='Hops', shrink=0.8)
    ax.set_title(f'{label}\nKNN Graph Hops', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # Bottom: Euclidean distance
    ax = axes[1, gi]
    ed_sub = goal_euclid[label][tsne_idx]
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=ed_sub, cmap='viridis',
                    s=4, alpha=0.7, rasterized=True,
                    vmin=0, vmax=np.percentile(ed_sub, 95))
    plt.colorbar(sc, ax=ax, label='Euclid Dist', shrink=0.8)
    ax.set_title(f'Euclidean Distance', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Graph Hops vs Euclidean Distance in t-SNE', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/03_tsne_hops_vs_euclid.png", bbox_inches='tight')
plt.close(fig)
print("Saved 03_tsne_hops_vs_euclid.png")

# ── Plot 4: Transition chain trajectories in t-SNE ───────────────────

from sklearn.neighbors import NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=1).fit(z_scaled[tsne_idx])

fig, axes = plt.subplots(1, 4, figsize=(24, 6))

for gi, (label, goal_idx) in enumerate(study_goals.items()):
    ax = axes[gi]
    is_good = 'GOOD' in label

    # Background
    ax.scatter(z_2d[:, 0], z_2d[:, 1], c='#e8e8e8', s=2, alpha=0.3, rasterized=True)

    # Mark goal
    goal_in_tsne = (tsne_idx == goal_idx)
    if goal_in_tsne.any():
        gx, gy = z_2d[goal_in_tsne][0]
        ax.scatter([gx], [gy], c='lime', s=200, marker='*', edgecolors='black',
                   zorder=10, linewidth=1)

    # Trace 20 chains from closest nodes
    euclid = goal_euclid[label]
    close_nodes = np.argsort(euclid)[:20]

    chain_color = accent_colors[0] if is_good else accent_colors[3]
    for node in close_nodes:
        chain = [node]
        current = node
        for step in range(15):
            next_node = nn_indices_np[current]
            if next_node == current:
                break
            chain.append(next_node)
            current = next_node

        # Map chain to t-SNE coords
        chain_z = z_scaled[chain]
        _, tsne_nn = nn_model.kneighbors(chain_z)
        tsne_chain = z_2d[tsne_nn.flatten()]

        ax.plot(tsne_chain[:, 0], tsne_chain[:, 1], color=chain_color,
                alpha=0.5, lw=1.0, zorder=5)
        ax.scatter(tsne_chain[0, 0], tsne_chain[0, 1], c=chain_color,
                   s=20, zorder=6, edgecolors='white', linewidth=0.5)

    ax.set_title(f'{label}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Transition Chains from 20 Closest Nodes (15 steps)', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/04_transition_chains.png", bbox_inches='tight')
plt.close(fig)
print("Saved 04_transition_chains.png")

# ── Plot 5: Folding heatmap in t-SNE ─────────────────────────────────
# For each node, compute |rank(euclid) - rank(graph_hops)| as folding measure

fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))

for gi, label in enumerate(study_goals):
    ax = axes[gi]
    is_good = 'GOOD' in label

    euclid = goal_euclid[label][tsne_idx]
    hops = goal_sp_knn[label][tsne_idx]
    finite = np.isfinite(hops)

    # Rank difference as folding metric
    euclid_rank = np.argsort(np.argsort(euclid))
    hops_rank = np.full_like(euclid_rank, tsne_n)  # max rank for unreachable
    hops_rank[finite] = np.argsort(np.argsort(hops[finite]))

    fold_metric = np.abs(euclid_rank - hops_rank).astype(float)
    fold_metric = fold_metric / fold_metric.max()  # normalize to [0, 1]

    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=fold_metric, cmap='RdYlBu_r',
                    s=4, alpha=0.7, rasterized=True, vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax, label='Fold metric', shrink=0.8)
    ax.set_title(f'{label}\nMean fold={fold_metric.mean():.3f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.suptitle('Manifold Folding: |rank(Euclid) - rank(Graph Hops)|', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/05_folding_metric.png", bbox_inches='tight')
plt.close(fig)
print("Saved 05_folding_metric.png")

print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
