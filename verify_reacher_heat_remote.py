"""Graph diagnostics and heat solver analysis for reacher buffer.
Self-contained: no dependency on planning module or world model."""

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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#fafafa',
    'figure.dpi': 150,
})

SAVE_DIR = "/tmp/reacher_heat_plots"
import os
os.makedirs(SAVE_DIR, exist_ok=True)

BUF_PATH = "/tmp/lewm_reacher_buffer.pt"
N_SUBSAMPLE = 55000
N_GOALS = 10

# ── MCMollifiedHeat (inlined) ──────────────────────────────────────────

class MCMollifiedHeat:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.scaler_obj = StandardScaler()
        self.graph_nodes = None
        self.V_graph = None
        self.z_buffer_np = None
        self.v_surrogate = None
        self.proj_mean = None
        self.proj_std = None

    def compile_environment(self, z_buffer, z_predicted_next=None, n_nodes=None,
                             K_neighbors=32, mc_candidates=4096, chunk_size=8192,
                             graph_mode='knn'):
        """
        graph_mode:
          'knn'          - spatial K-NN from z_next to z_curr (original)
          'transition'   - edges from actual transitions: i -> nn(z_next[i])
          'hierarchical' - transition edges first, then bridge disconnected
                           components via nearest spatial neighbors
        All modes produce undirected (symmetrized) graphs.
        """
        z_combined = np.vstack([z_buffer, z_predicted_next]) if z_predicted_next is not None else z_buffer
        self.scaler_obj.fit(z_combined)
        z_scaled = self.scaler_obj.transform(z_buffer)
        z_next_scaled = self.scaler_obj.transform(z_predicted_next) if z_predicted_next is not None else z_scaled.copy()

        if n_nodes is not None and z_scaled.shape[0] > n_nodes:
            rng = np.random.RandomState(42)
            idx = rng.choice(z_scaled.shape[0], n_nodes, replace=False)
            z_scaled, z_next_scaled = z_scaled[idx], z_next_scaled[idx]

        self.z_buffer_np = z_scaled
        self.graph_nodes = torch.tensor(z_scaled, dtype=torch.float32, device=self.device)
        N = z_scaled.shape[0]

        t0 = time.perf_counter()
        z_next_tensor = torch.tensor(z_next_scaled, dtype=torch.float32, device=self.device)

        if graph_mode == 'hierarchical':
            # Hierarchical graph:
            # 1) Transition edges: for each i, find K nearest graph nodes
            #    to z_next[i]. These encode reachability from actual dynamics.
            # 2) Bridge: find connected components, then connect each component
            #    to its nearest spatial neighbor in another component.
            # Result: transition-based backbone with spatial bridges.
            from scipy.sparse.csgraph import connected_components as _cc

            K_trans = K_neighbors  # same K for transition edges
            all_k_dists = torch.full((N, K_trans), float('inf'), device=self.device)
            all_k_indices = torch.zeros((N, K_trans), dtype=torch.long, device=self.device)
            n_rounds = min(50, max(1, N // mc_candidates))
            rng = np.random.RandomState(42)

            with torch.no_grad():
                for r in range(n_rounds):
                    cand_idx = torch.tensor(
                        rng.choice(N, min(mc_candidates, N), replace=False),
                        device=self.device
                    )
                    cand_nodes = self.graph_nodes[cand_idx]
                    for start in range(0, N, chunk_size):
                        end = min(start + chunk_size, N)
                        dists = torch.cdist(z_next_tensor[start:end], cand_nodes)
                        ck = min(K_trans, dists.shape[1])
                        chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
                        chunk_k_dists = -chunk_k_dists
                        chunk_k_indices = cand_idx[chunk_k_local]
                        merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
                        merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
                        best_dists, best_pos = torch.topk(-merged_dists, k=K_trans, dim=1)
                        all_k_dists[start:end] = -best_dists
                        all_k_indices[start:end] = merged_indices.gather(1, best_pos)

            t_knn = time.perf_counter() - t0

            # Sigma from transition K-NN distances (not spatial NN)
            self.sigma_perturb = all_k_dists[:, 0].median().item()

            # Build transition adjacency with unit weights
            t1 = time.perf_counter()
            row_idx_t = np.repeat(np.arange(N), K_trans)
            col_idx_t = all_k_indices.cpu().numpy().flatten()
            data_t = np.ones(len(row_idx_t), dtype=np.float64)

            W_trans = sp.csr_matrix((data_t, (row_idx_t, col_idx_t)), shape=(N, N))
            W_trans.setdiag(0)
            W_trans_sym = (W_trans + W_trans.T) * 0.5
            n_trans_edges = W_trans_sym.nnz

            # Bridge disconnected components
            n_bridge_edges = 0
            W_full = W_trans.copy()
            for bridge_round in range(200):
                W_sym_tmp = (W_full + W_full.T) * 0.5
                n_comp, comp_labels = _cc(W_sym_tmp, directed=False)
                if n_comp == 1:
                    break

                comp_sizes = np.bincount(comp_labels)
                sorted_comps = np.argsort(comp_sizes)

                bridge_rows, bridge_cols = [], []
                connected_set = set()
                for ci in sorted_comps:
                    if ci in connected_set:
                        continue
                    nodes_ci = np.where(comp_labels == ci)[0]
                    other_nodes = np.where(comp_labels != ci)[0]
                    if len(other_nodes) == 0:
                        continue

                    n_s_ci = min(len(nodes_ci), 100)
                    n_s_other = min(len(other_nodes), 4096)
                    rng_b = np.random.RandomState(42 + bridge_round)
                    s_ci = nodes_ci[rng_b.choice(len(nodes_ci), n_s_ci, replace=False)]
                    s_other = other_nodes[rng_b.choice(len(other_nodes), n_s_other, replace=False)]

                    with torch.no_grad():
                        d_mat = torch.cdist(self.graph_nodes[s_ci], self.graph_nodes[s_other])
                        min_val, min_idx = d_mat.min(dim=1)
                        best = min_val.argmin().item()

                    bridge_rows.append(s_ci[best])
                    bridge_cols.append(s_other[min_idx[best].item()])
                    connected_set.add(ci)
                    connected_set.add(comp_labels[s_other[min_idx[best].item()]])

                if not bridge_rows:
                    break
                n_bridge_edges += len(bridge_rows)
                W_new = sp.csr_matrix(
                    (np.ones(len(bridge_rows)), (bridge_rows, bridge_cols)), shape=(N, N))
                W_full = W_full + W_new

            # Final symmetrize
            W_sym = (W_full + W_full.T) * 0.5
            W_sym.setdiag(0)
            d = np.array(W_sym.sum(axis=1)).flatten()
            self.L_sym = sp.diags(d) - W_sym
            t_graph = time.perf_counter() - t1

            n_comp_final, _ = _cc(W_sym, directed=False)
            degrees = np.array(W_sym.astype(bool).sum(axis=1)).flatten()

            print(f"[Hierarchical] N={N}, K_trans={K_trans}, rounds={n_rounds}")
            print(f"[Hierarchical] sigma_perturb={self.sigma_perturb:.6f}")
            print(f"[Hierarchical] Trans edges: {n_trans_edges}, Bridge edges: {n_bridge_edges}")
            print(f"[Hierarchical] Components: {n_comp} -> {n_comp_final}")
            print(f"[Hierarchical] Build: {t_knn:.2f}s (KNN) + {t_graph:.2f}s (sparse)")
            print(f"[Hierarchical] Degree: min={degrees.min()}, max={degrees.max()}, "
                  f"mean={degrees.mean():.1f}, median={np.median(degrees):.0f}")

        else:  # 'knn' (original)
            all_k_dists = torch.full((N, K_neighbors), float('inf'), device=self.device)
            all_k_indices = torch.zeros((N, K_neighbors), dtype=torch.long, device=self.device)
            n_rounds = min(50, max(1, N // mc_candidates))
            rng = np.random.RandomState(42)

            with torch.no_grad():
                for r in range(n_rounds):
                    cand_idx = torch.tensor(
                        rng.choice(N, min(mc_candidates, N), replace=False),
                        device=self.device
                    )
                    cand_nodes = self.graph_nodes[cand_idx]
                    for start in range(0, N, chunk_size):
                        end = min(start + chunk_size, N)
                        dists = torch.cdist(z_next_tensor[start:end], cand_nodes)
                        ck = min(K_neighbors, dists.shape[1])
                        chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
                        chunk_k_dists = -chunk_k_dists
                        chunk_k_indices = cand_idx[chunk_k_local]
                        merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
                        merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
                        best_dists, best_pos = torch.topk(-merged_dists, k=K_neighbors, dim=1)
                        all_k_dists[start:end] = -best_dists
                        all_k_indices[start:end] = merged_indices.gather(1, best_pos)

                local_sigma = all_k_dists[:, 1].unsqueeze(1) * 2.0
                local_sigma = torch.clamp(local_sigma, min=1e-3)
                self.sigma_perturb = local_sigma.median().item()
                W_local = torch.exp(-(all_k_dists ** 2) / (2 * local_sigma ** 2))

            t_knn = time.perf_counter() - t0

            t1 = time.perf_counter()
            W_local_np = W_local.cpu().numpy()
            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = all_k_indices.cpu().numpy().flatten()
            data = W_local_np.flatten()
            W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
            W_0.setdiag(0)
            W_sym = (W_0 + W_0.T) * 0.5
            d = np.array(W_sym.sum(axis=1)).flatten()
            self.L_sym = sp.diags(d) - W_sym
            t_graph = time.perf_counter() - t1

            print(f"[KNN Graph] N={N}, K={K_neighbors}, mc_cand={mc_candidates}, rounds={n_rounds}")
            print(f"[KNN Graph] sigma_perturb={self.sigma_perturb:.6f} | K-NN: {t_knn:.2f}s, Sparse: {t_graph:.2f}s")

        self.proj_mean = torch.tensor(self.scaler_obj.mean_, dtype=torch.float32, device=self.device)
        self.proj_std = torch.tensor(np.sqrt(self.scaler_obj.var_), dtype=torch.float32, device=self.device)

    def solve_for_goal(self, z_goal, phys_v_mean=10.0):
        z_goal_proj = self.scaler_obj.transform(z_goal.reshape(1, -1))
        goal_dists = np.linalg.norm(self.z_buffer_np - z_goal_proj, axis=1)
        self.goal_idx = np.argmin(goal_dists)

        N = self.L_sym.shape[0]
        h2 = self.sigma_perturb ** 2
        t = h2 * 500

        t0 = time.perf_counter()
        b = np.zeros(N)
        b[self.goal_idx] = 1.0
        A_heat = sp.eye(N) + t * self.L_sym
        M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
        u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
        if info != 0:
            print(f"[Warning] CG exited with code {info}")
        t_solve = time.perf_counter() - t0

        u = np.maximum(u, 1e-300)
        V_raw = -np.log(u)
        V_raw = V_raw - V_raw.min()
        scale_factor = phys_v_mean / (V_raw.mean() + 1e-8)
        V_graph_np = V_raw * scale_factor
        self.V_graph = torch.tensor(V_graph_np, dtype=torch.float32, device=self.device)

        euclid_dists = np.linalg.norm(self.z_buffer_np - self.z_buffer_np[self.goal_idx], axis=1)
        corr, pval = spearmanr(V_graph_np, euclid_dists)
        print(f"[Heat CG] Solve: {t_solve:.2f}s | V: [{V_graph_np.min():.2f}, {V_graph_np.max():.2f}] "
              f"mean={V_graph_np.mean():.2f} std={V_graph_np.std():.2f} | Spearman={corr:.4f}")

        # Train surrogate MLP
        t0 = time.perf_counter()
        D = self.graph_nodes.shape[1]
        if self.v_surrogate is None:
            self.v_surrogate = nn.Sequential(
                nn.Linear(D, 256), nn.Mish(),
                nn.Linear(256, 256), nn.Mish(),
                nn.Linear(256, 1), nn.Softplus(),
            ).to(self.device)

        optimizer = torch.optim.Adam(self.v_surrogate.parameters(), lr=5e-3)
        self.v_surrogate.train()
        for step in range(250):
            optimizer.zero_grad()
            pred = self.v_surrogate(self.graph_nodes).squeeze()
            loss = nn.functional.mse_loss(pred, self.V_graph)
            loss.backward()
            optimizer.step()

        self.v_surrogate.eval()
        for p in self.v_surrogate.parameters():
            p.requires_grad = False
        t_surr = time.perf_counter() - t0
        print(f"[Surrogate] Fit: {t_surr:.2f}s | final MSE={loss.item():.4f}")

        return self.V_graph


# ── Main ────────────────────────────────────────────────────────────────

from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

print("=" * 60)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

print("Loading buffer...")
loaded = torch.load(BUF_PATH, map_location='cpu', weights_only=True)
N_total = loaded['z_curr'].shape[0]
dim = loaded['z_curr'].shape[1]
print(f"Total transitions: {N_total}, dim: {dim}")

# Subsample (same indices for both graph modes)
rng = np.random.RandomState(42)
idx = rng.choice(N_total, N_SUBSAMPLE, replace=False)
z_curr = loaded['z_curr'][idx].numpy()
z_next = loaded['z_next'][idx].numpy()

# t-SNE embedding (shared, compute once)
print("\nComputing t-SNE (on 8K subsample)...")
# Use a temporary solver just for scaling
tmp_solver = MCMollifiedHeat()
tmp_scaler = StandardScaler()
z_combined = np.vstack([z_curr, z_next])
tmp_scaler.fit(z_combined)
z_scaled_all = tmp_scaler.transform(z_curr)

tsne_n = 8000
tsne_idx = rng.choice(N_SUBSAMPLE, tsne_n, replace=False)
tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
z_2d = tsne.fit_transform(z_scaled_all[tsne_idx])
print("  t-SNE done.")

# Shared goal indices
rng_goals = np.random.RandomState(123)
goal_indices = rng_goals.choice(N_SUBSAMPLE, N_GOALS, replace=False)

soft_colors = sns.color_palette("pastel")
accent_colors = sns.color_palette("Set2")

# ── Run both graph modes ──────────────────────────────────────────────

results_by_mode = {}

for mode in ['knn', 'hierarchical']:
    print(f"\n{'=' * 60}")
    print(f"GRAPH MODE: {mode.upper()}")
    print(f"{'=' * 60}")

    solver = MCMollifiedHeat()
    solver.compile_environment(z_buffer=z_curr, z_predicted_next=z_next, graph_mode=mode)

    # Connected components
    W_sym = (solver.L_sym - sp.diags(np.array(solver.L_sym.diagonal()))) * -1.0
    n_comp, comp_labels = connected_components(W_sym, directed=False)
    comp_sizes = np.bincount(comp_labels)
    degrees = np.array(W_sym.astype(bool).sum(axis=1)).flatten()
    print(f"  Components: {n_comp}, largest: {comp_sizes.max()} ({comp_sizes.max()/N_SUBSAMPLE*100:.1f}%)")
    if n_comp > 1:
        print(f"  Top component sizes: {sorted(comp_sizes, reverse=True)[:10]}")

    # Solve for goals
    all_V = {}
    all_spearmans = []

    for gi, goal_idx in enumerate(goal_indices):
        goal = solver.z_buffer_np[goal_idx]
        goal_raw = solver.scaler_obj.inverse_transform(goal.reshape(1, -1)).flatten()
        solver.v_surrogate = None

        V = solver.solve_for_goal(goal_raw)
        V_np = V.cpu().numpy()

        euclid_dists = np.linalg.norm(solver.z_buffer_np - solver.z_buffer_np[solver.goal_idx], axis=1)
        corr, _ = spearmanr(V_np, euclid_dists)
        all_spearmans.append(corr)
        all_V[gi] = V_np

    mean_sp = np.mean(all_spearmans)
    std_sp = np.std(all_spearmans)
    print(f"\n  Mean Spearman: {mean_sp:.4f} (+/- {std_sp:.4f})")
    print(f"  Per-goal: {[f'{s:.3f}' for s in all_spearmans]}")

    results_by_mode[mode] = {
        'solver': solver, 'all_V': all_V, 'all_spearmans': all_spearmans,
        'degrees': degrees, 'n_comp': n_comp, 'comp_labels': comp_labels,
        'comp_sizes': comp_sizes, 'mean_sp': mean_sp, 'std_sp': std_sp,
    }

# ══════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS
# ══════════════════════════════════════════════════════════════════════

mode_colors = {'knn': '#e8a0bf', 'hierarchical': '#a0c4e8'}
mode_labels = {'knn': 'Spatial K-NN', 'hierarchical': 'Hierarchical'}

# ── Plot 1: Per-goal Spearman comparison ─────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(N_GOALS)
width = 0.35
for i, mode in enumerate(['knn', 'hierarchical']):
    sp = results_by_mode[mode]['all_spearmans']
    bars = ax.bar(x + i*width, sp, width, label=mode_labels[mode],
                  color=mode_colors[mode], edgecolor='white', linewidth=0.8)
ax.set_xlabel('Goal Index')
ax.set_ylabel('Spearman Correlation')
ax.set_title('V Quality per Goal: Spatial K-NN vs Transition Graph')
ax.set_xticks(x + width/2)
ax.set_xticklabels([str(i) for i in range(N_GOALS)])
ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
# Add mean lines
for mode, ls in [('knn', '--'), ('hierarchical', '-')]:
    m = results_by_mode[mode]['mean_sp']
    ax.axhline(m, color=mode_colors[mode], ls=ls, lw=1.5, alpha=0.7,
               label=f'{mode_labels[mode]} mean={m:.3f}')
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/01_spearman_comparison.png")
plt.close(fig)
print(f"Saved 01_spearman_comparison.png")

# ── Plot 2: Degree distribution comparison ───────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax_i, mode in enumerate(['knn', 'hierarchical']):
    deg = results_by_mode[mode]['degrees']
    sns.histplot(deg, bins=60, color=mode_colors[mode], edgecolor='white', linewidth=0.5, ax=axes[ax_i])
    axes[ax_i].axvline(np.median(deg), color='#555555', ls='--', lw=1.5,
                       label=f'Median = {np.median(deg):.0f}')
    axes[ax_i].set_title(f'{mode_labels[mode]} Graph\nmin={deg.min()}, mean={deg.mean():.1f}, max={deg.max()}')
    axes[ax_i].set_xlabel('Node Degree')
    axes[ax_i].set_ylabel('Count')
    axes[ax_i].legend(frameon=True, facecolor='white', edgecolor='#cccccc')
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/02_degree_comparison.png")
plt.close(fig)
print(f"Saved 02_degree_comparison.png")

# ── Plot 3: t-SNE colored by V, best/worst per mode ──────────────────

for mode in ['knn', 'hierarchical']:
    sp_list = results_by_mode[mode]['all_spearmans']
    V_dict = results_by_mode[mode]['all_V']
    solver_m = results_by_mode[mode]['solver']

    sorted_goals = sorted(range(N_GOALS), key=lambda i: sp_list[i])
    worst_2 = sorted_goals[:2]
    best_2 = sorted_goals[-2:]
    show = worst_2 + best_2

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax_i, gi in enumerate(show):
        V_sub = V_dict[gi][tsne_idx]
        sc = axes[ax_i].scatter(z_2d[:, 0], z_2d[:, 1], c=V_sub, cmap='coolwarm',
                                s=4, alpha=0.7, rasterized=True,
                                vmin=np.percentile(V_sub, 2), vmax=np.percentile(V_sub, 98))
        plt.colorbar(sc, ax=axes[ax_i], label='V', shrink=0.8)
        quality = "GOOD" if gi in best_2 else "BAD"
        axes[ax_i].set_title(f'Goal {gi} ({quality})\nSpearman={sp_list[gi]:.3f}', fontsize=11)
        axes[ax_i].set_xticks([]); axes[ax_i].set_yticks([])

    plt.suptitle(f'{mode_labels[mode]} Graph: Best vs Worst Goals', fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{SAVE_DIR}/03_V_tsne_{mode}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved 03_V_tsne_{mode}.png")

# ── Plot 4: V vs Euclidean for best/worst, both modes side by side ───

# Pick the same 2 goals for direct comparison
# Use the goal with best KNN spearman and worst KNN spearman
knn_sp = results_by_mode['knn']['all_spearmans']
best_goal = np.argmax(knn_sp)
worst_goal = np.argmin(knn_sp)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
scatter_idx = rng.choice(N_SUBSAMPLE, 3000, replace=False)

for col, mode in enumerate(['knn', 'hierarchical']):
    solver_m = results_by_mode[mode]['solver']
    for row, gi in enumerate([worst_goal, best_goal]):
        V_np = results_by_mode[mode]['all_V'][gi]
        goal_node = goal_indices[gi]
        euclid = np.linalg.norm(solver_m.z_buffer_np - solver_m.z_buffer_np[goal_node], axis=1)
        sp_val = results_by_mode[mode]['all_spearmans'][gi]
        quality = "Best" if gi == best_goal else "Worst"
        color = accent_colors[0] if mode == 'hierarchical' else accent_colors[3]
        axes[row, col].scatter(euclid[scatter_idx], V_np[scatter_idx], s=3, alpha=0.3,
                               color=color, rasterized=True)
        axes[row, col].set_xlabel('Euclidean Distance')
        axes[row, col].set_ylabel('V')
        axes[row, col].set_title(f'{mode_labels[mode]} | Goal {gi} ({quality})\nr={sp_val:.3f}', fontsize=11)

plt.suptitle('V vs Euclidean Distance: K-NN vs Transition Graph', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/04_V_vs_euclid_comparison.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 04_V_vs_euclid_comparison.png")

# ── Plot 5: V histogram comparison for same goals ────────────────────

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for col, mode in enumerate(['knn', 'hierarchical']):
    for row, gi in enumerate([worst_goal, best_goal]):
        V_np = results_by_mode[mode]['all_V'][gi]
        sp_val = results_by_mode[mode]['all_spearmans'][gi]
        quality = "Best" if gi == best_goal else "Worst"
        color = mode_colors[mode]
        sns.histplot(V_np, bins=80, color=color, edgecolor='white', linewidth=0.3, ax=axes[row, col])
        axes[row, col].set_title(f'{mode_labels[mode]} | Goal {gi} ({quality})\nr={sp_val:.3f}, std={V_np.std():.2f}',
                                  fontsize=11)
        axes[row, col].set_xlabel('V')

plt.suptitle('Value Distribution Comparison', fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(f"{SAVE_DIR}/05_V_hist_comparison.png", bbox_inches='tight')
plt.close(fig)
print(f"Saved 05_V_hist_comparison.png")

# ── Plot 6: Connected components for transition graph ─────────────────

trans_res = results_by_mode['hierarchical']
if trans_res['n_comp'] > 1:
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=trans_res['comp_labels'][tsne_idx],
                    cmap='tab20', s=4, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Component', shrink=0.8)
    ax.set_title(f'Hierarchical Graph: {trans_res["n_comp"]} Connected Components')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(f"{SAVE_DIR}/06_hier_components_tsne.png")
    plt.close(fig)
    print(f"Saved 06_hier_components_tsne.png")

    # Component size bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    sizes_sorted = sorted(trans_res['comp_sizes'], reverse=True)
    ax.bar(range(min(30, len(sizes_sorted))), sizes_sorted[:30],
           color=mode_colors['hierarchical'], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Component Rank')
    ax.set_ylabel('Size (nodes)')
    ax.set_title(f'Transition Graph Components ({trans_res["n_comp"]} total)')
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(f"{SAVE_DIR}/07_hier_component_sizes.png")
    plt.close(fig)
    print(f"Saved 07_hier_component_sizes.png")

# ── Summary ──────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
for mode in ['knn', 'hierarchical']:
    r = results_by_mode[mode]
    print(f"  {mode_labels[mode]:20s}: Spearman = {r['mean_sp']:.4f} (+/- {r['std_sp']:.4f}), "
          f"components = {r['n_comp']}")
print(f"  Previous (20K KNN):     Spearman = 0.28-0.31")

print(f"\nAll plots saved to {SAVE_DIR}/")
print("Done.")
