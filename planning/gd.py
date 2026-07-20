import os

import torch
import torch.nn as nn
import time
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import dijkstra


class MCMollifiedHeat:
    """Value function on a Monte Carlo mollified K-NN graph.

    Supports two solver modes:
      - 'heat': Implicit Euler heat equation on unnormalized Laplacian (I + t*L)u = delta.
      - 'rw_diffusion': Explicit random walk diffusion on normalized transition matrix.
        Iterates v_{t+1} = P^T v_t for rw_steps steps, equivalent to heat diffusion
        on the random walk Laplacian L_rw = I - D^{-1}W.

    Graph construction uses MC sampling to approximate K-NN in O(N * rounds * candidates)
    instead of O(N^2), with Gaussian mollifier weights. The graph is symmetrized for
    undirected reachability. A surrogate MLP provides smooth gradients for planning.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scaler_obj = StandardScaler()

        self.proj_mean = None
        self.proj_std = None

        self.graph_nodes = None
        self.V_graph = None
        self.z_buffer_np = None
        self.v_surrogate = None
        self.solver_mode = 'heat'
        self.rw_steps = 10
        self.P_trans = None  # transition matrix for rw_diffusion

    def compile_environment(self, z_buffer, z_predicted_next=None, n_nodes=None,
                             K_neighbors=32, mc_candidates=4096, chunk_size=8192,
                             use_isomap=False, isomap_landmarks=500,
                             solver_mode='heat', rw_steps=10):
        """
        Builds an undirected reachability graph using Monte Carlo mollified K-NN.

        Transitions (z_curr, z_next) encode reachability: if two latents are
        connected, the agent can move between them. They do NOT encode goal
        direction (the agent often moves away from the goal). The PDE solve
        determines direction; the graph only encodes connectivity.

        The graph is symmetrized: W_sym = (W + W^T) / 2, giving a symmetric
        positive-definite Laplacian L = D - W_sym. This enables Conjugate
        Gradient (faster, guaranteed convergence) instead of GMRES.

        Monte Carlo K-NN replaces the O(N^2) cdist, allowing the full
        dataset (~1M transitions) instead of 55K subsampling.

        If use_isomap=True, after building the spatial K-NN graph, geodesic
        distances (shortest paths through the K-NN graph) are computed via
        landmark Dijkstra. K-NN is then rebuilt in geodesic space, producing
        a graph that respects manifold geometry and avoids cross-fold shortcuts.
        """
        import time as _time

        self.solver_mode = solver_mode
        self.rw_steps = rw_steps

        z_combined = np.vstack([z_buffer, z_predicted_next]) if z_predicted_next is not None else z_buffer
        self.scaler_obj.fit(z_combined)

        z_scaled = self.scaler_obj.transform(z_buffer)
        z_next_scaled = self.scaler_obj.transform(z_predicted_next) if z_predicted_next is not None else z_scaled.copy()

        # Optional subsampling (now optional — default uses all data)
        if n_nodes is not None and z_scaled.shape[0] > n_nodes:
            rng = np.random.RandomState(42)
            idx = rng.choice(z_scaled.shape[0], n_nodes, replace=False)
            z_scaled, z_next_scaled = z_scaled[idx], z_next_scaled[idx]

        self.z_buffer_np = z_scaled
        self.graph_nodes = torch.tensor(z_scaled, dtype=torch.float32, device=self.device)
        N = z_scaled.shape[0]

        t0 = _time.perf_counter()

        z_next_tensor = torch.tensor(z_next_scaled, dtype=torch.float32, device=self.device)

        # For rw_diffusion, use spatial K-NN (z_curr -> z_curr) instead of
        # dynamics K-NN (z_next -> z_curr). Spatial K-NN produces a cleaner
        # graph for random walk diffusion.
        knn_source = self.graph_nodes if solver_mode == 'rw_diffusion' else z_next_tensor

        # MC mollified K-NN: estimate K nearest graph nodes per point
        # by sampling mc_candidates random candidates per round. Avoids the
        # O(N^2) full cdist that OOMs above ~55K nodes.
        # Cap rounds at 50: 50 * 4096 = 200K candidates sampled, sufficient
        # coverage even for N ~ 1M (each node seen ~10x on average).
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
                    dists = torch.cdist(knn_source[start:end], cand_nodes)
                    ck = min(K_neighbors, dists.shape[1])
                    chunk_k_dists, chunk_k_local = torch.topk(-dists, k=ck, dim=1)
                    chunk_k_dists = -chunk_k_dists
                    chunk_k_indices = cand_idx[chunk_k_local]

                    merged_dists = torch.cat([all_k_dists[start:end], chunk_k_dists], dim=1)
                    merged_indices = torch.cat([all_k_indices[start:end], chunk_k_indices], dim=1)
                    best_dists, best_pos = torch.topk(-merged_dists, k=K_neighbors, dim=1)
                    all_k_dists[start:end] = -best_dists
                    all_k_indices[start:end] = merged_indices.gather(1, best_pos)

            # Adaptive local bandwidth from nearest-neighbor distance
            local_sigma = all_k_dists[:, 1].unsqueeze(1) * 2.0
            local_sigma = torch.clamp(local_sigma, min=1e-3)
            self.sigma_perturb = local_sigma.median().item()

            # Gaussian mollifier weights
            W_local = torch.exp(-(all_k_dists ** 2) / (2 * local_sigma ** 2))

        t_knn = _time.perf_counter() - t0

        if use_isomap:
            # ISOMAP: rebuild K-NN using geodesic distances (shortest paths
            # through the spatial K-NN graph). This unfolds the manifold so
            # that points on different folds are correctly separated.
            t_iso0 = _time.perf_counter()

            # Build distance-weighted K-NN graph for Dijkstra
            knn_dists_np = all_k_dists.cpu().numpy()
            knn_indices_np = all_k_indices.cpu().numpy()
            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = knn_indices_np.flatten()
            W_dist = sp.csr_matrix((knn_dists_np.flatten(), (row_idx, col_idx)), shape=(N, N))
            W_dist.setdiag(0)
            W_dist_sym = (W_dist + W_dist.T) * 0.5
            W_dist_sym.eliminate_zeros()

            # Landmark Dijkstra for geodesic distances
            rng_lm = np.random.RandomState(42)
            landmarks = rng_lm.choice(N, min(isomap_landmarks, N), replace=False)
            geo_dists = dijkstra(W_dist_sym, directed=False, indices=landmarks)  # (L, N)
            t_dijk = _time.perf_counter() - t_iso0

            # Rebuild K-NN in geodesic space using GPU
            t_gknn0 = _time.perf_counter()
            geo_t = torch.tensor(geo_dists, dtype=torch.float32, device=self.device)  # (L, N)
            L_lm = len(landmarks)
            geo_knn_dists = np.full((N, K_neighbors), np.inf)
            geo_knn_indices = np.zeros((N, K_neighbors), dtype=np.int64)
            geo_chunk = 500

            with torch.no_grad():
                for start in range(0, N, geo_chunk):
                    end = min(start + geo_chunk, N)
                    chunk_n = end - start
                    d_chunk = geo_t[:, start:end]  # (L, chunk)

                    d_approx = torch.full((chunk_n, N), float('inf'), device=self.device)
                    lm_batch = 100
                    for lb in range(0, L_lm, lm_batch):
                        le = min(lb + lm_batch, L_lm)
                        d_batch = d_chunk[lb:le, :, None] + geo_t[lb:le, None, :]
                        d_approx = torch.minimum(d_approx, d_batch.min(dim=0).values)

                    for ci in range(chunk_n):
                        d_approx[ci, start + ci] = float('inf')

                    topk_dists, topk_indices = torch.topk(-d_approx, k=K_neighbors, dim=1)
                    geo_knn_dists[start:end] = (-topk_dists).cpu().numpy()
                    geo_knn_indices[start:end] = topk_indices.cpu().numpy()

            t_gknn = _time.perf_counter() - t_gknn0

            # Replace K-NN with geodesic K-NN
            geo_sigma = np.median(geo_knn_dists[:, 1]) * 2.0
            geo_sigma = max(geo_sigma, 1e-6)
            self.sigma_perturb = geo_sigma
            W_local = np.exp(-(geo_knn_dists ** 2) / (2 * geo_sigma ** 2))

            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = geo_knn_indices.flatten()
            data = W_local.flatten()
            W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
            W_0.setdiag(0)
            W_sym = (W_0 + W_0.T) * 0.5
            d = np.array(W_sym.sum(axis=1)).flatten()
            self.L_sym = sp.diags(d) - W_sym

            t_iso_total = _time.perf_counter() - t_iso0
            print(f"[ISOMAP Graph] N={N}, K={K_neighbors}, landmarks={len(landmarks)}")
            print(f"[ISOMAP Graph] sigma_perturb={self.sigma_perturb:.6f}")
            print(f"[ISOMAP Graph] Dijkstra: {t_dijk:.1f}s, Geodesic K-NN: {t_gknn:.1f}s, "
                  f"Total: {t_iso_total:.1f}s")

        else:
            # Standard Euclidean K-NN graph
            t1 = _time.perf_counter()
            W_local_np = W_local.cpu().numpy()
            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = all_k_indices.cpu().numpy().flatten()
            data = W_local_np.flatten()

            W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
            W_0.setdiag(0)

            # Symmetrize: reachability is bidirectional
            W_sym = (W_0 + W_0.T) * 0.5

            # Unnormalized graph Laplacian: L = D - W (symmetric positive semi-definite)
            d = np.array(W_sym.sum(axis=1)).flatten()
            self.L_sym = sp.diags(d) - W_sym

            t_graph = _time.perf_counter() - t1

            print(f"[Graph Build] N={N}, K={K_neighbors}, mc_cand={mc_candidates}, rounds={n_rounds}")
            print(f"[Graph Build] sigma_perturb={self.sigma_perturb:.6f} | K-NN: {t_knn:.2f}s, Sparse: {t_graph:.2f}s")

        # Build transition matrix P = D^{-1}W for random walk diffusion.
        # Recover W from L = D - W: diagonal of L is D, off-diag of L is -W.
        if solver_mode == 'rw_diffusion':
            W_recovered = -self.L_sym.copy()
            W_recovered.setdiag(0)
            d_arr = np.array(W_recovered.sum(axis=1)).flatten()
            d_inv = np.where(d_arr > 0, 1.0 / d_arr, 0.0)
            self.P_trans = (sp.diags(d_inv) @ W_recovered).T.tocsr()
            print(f"[RW Diffusion] Built P^T, rw_steps={rw_steps}")

        self.proj_mean = torch.tensor(self.scaler_obj.mean_, dtype=torch.float32, device=self.device)
        self.proj_std = torch.tensor(np.sqrt(self.scaler_obj.var_), dtype=torch.float32, device=self.device)


    def solve_for_goal(self, z_goal, phys_v_mean=10.0):
        """
        Solve for goal-conditioned value function V on the graph.

        Two modes:
          - 'heat': Implicit Euler (I + t*L)u = delta_goal, V = -log(u).
          - 'rw_diffusion': Random walk v_{t+1} = P^T v_t for rw_steps, V = -log(v).

        Then fits a surrogate MLP for smooth gradients during planning.
        """
        import time as _time

        z_goal_proj = self.scaler_obj.transform(z_goal.reshape(1, -1))
        goal_dists = np.linalg.norm(self.z_buffer_np - z_goal_proj, axis=1)
        self.goal_idx = np.argmin(goal_dists)

        N = self.L_sym.shape[0]
        t0 = _time.perf_counter()

        if self.solver_mode == 'rw_diffusion':
            # Random walk diffusion: explicit Euler on L_rw = I - D^{-1}W
            v = np.zeros(N)
            v[self.goal_idx] = 1.0
            for _ in range(self.rw_steps):
                v = self.P_trans @ v
            v = np.maximum(v, 1e-300)
            V_raw = -np.log(v)
        else:
            # Heat equation: implicit Euler on unnormalized Laplacian
            h2 = self.sigma_perturb ** 2
            t = h2 * 500
            b = np.zeros(N)
            b[self.goal_idx] = 1.0
            A_heat = sp.eye(N) + t * self.L_sym
            M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
            u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
            if info != 0:
                print(f"[Warning] CG exited with code {info}")
            u = np.maximum(u, 1e-300)
            V_raw = -np.log(u)

        t_solve = _time.perf_counter() - t0

        # For RW diffusion, -log(v) is very spiky (most mass near goal).
        # Rank-normalize to get a smooth, uniform distribution while preserving
        # the rank order (Spearman is invariant to monotone transforms).
        if self.solver_mode == 'rw_diffusion':
            from scipy.stats import rankdata
            V_graph_np = rankdata(V_raw).astype(np.float64)
            V_graph_np = (V_graph_np / V_graph_np.max()) * phys_v_mean * 2
        else:
            V_raw = V_raw - V_raw.min()
            V_graph_np = V_raw * (phys_v_mean / (V_raw.mean() + 1e-8))

        self.V_graph = torch.tensor(V_graph_np, dtype=torch.float32, device=self.device)

        # Diagnostics
        from scipy.stats import spearmanr
        euclid_dists = np.linalg.norm(self.z_buffer_np - self.z_buffer_np[self.goal_idx], axis=1)
        corr, pval = spearmanr(V_graph_np, euclid_dists)
        solver_tag = f"RW t={self.rw_steps}" if self.solver_mode == 'rw_diffusion' else "Heat CG"
        print(f"[{solver_tag}] Solve: {t_solve:.2f}s | V: [{V_graph_np.min():.2f}, {V_graph_np.max():.2f}] "
              f"mean={V_graph_np.mean():.2f} std={V_graph_np.std():.2f} | "
              f"Spearman={corr:.4f}")

        # Train surrogate MLP for smooth gradients
        t0 = _time.perf_counter()
        D = self.graph_nodes.shape[1]
        if self.v_surrogate is None:
            self.v_surrogate = nn.Sequential(
                nn.Linear(D, 256),
                nn.Mish(),
                nn.Linear(256, 256),
                nn.Mish(),
                nn.Linear(256, 1),
                nn.Softplus(),
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

        t_surr = _time.perf_counter() - t0
        print(f"[Surrogate] Fit: {t_surr:.2f}s | final MSE={loss.item():.4f}")

        return self.V_graph

    def evaluate_rollout(self, z_generated):
        """
        Evaluate V via surrogate MLP. Provides smooth, dense gradients
        everywhere in latent space (0% dead gradients vs 74% with K-NN IDW).
        """
        z_scaled = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)
        with torch.enable_grad():
            V = self.v_surrogate(z_scaled).squeeze(-1)
        return V


def _load_replay_buffer(path, device):
    """Load a latent replay buffer and return (z_curr_np, z_next_np, pca_info).

    Supports two formats:
      - New format: {'z_curr': (N, D), 'z_next': (N, D)}
      - Legacy format: {'z_buffer': (B, T, D)} (derives curr/next from episodes)

    If the buffer contains PCA components, returns them as pca_info dict
    so the planner can project rollout latents into PCA space at planning time.
    """
    loaded = torch.load(path, map_location='cpu', weights_only=True)

    if 'z_curr' in loaded and 'z_next' in loaded:
        z_curr = loaded['z_curr']
        z_next = loaded['z_next']
    elif 'z_buffer' in loaded:
        z_obs = loaded['z_buffer']
        z_curr = z_obs[:, :-1, :].reshape(-1, z_obs.shape[-1])
        z_next = z_obs[:, 1:, :].reshape(-1, z_obs.shape[-1])
    else:
        raise ValueError(
            f"Unknown buffer format. Keys: {list(loaded.keys())}. "
            f"Expected 'z_curr'/'z_next' or 'z_buffer'."
        )

    pca_info = None
    if 'pca_components' in loaded:
        pca_info = {
            'components': loaded['pca_components'].to(device),  # (pca_dim, D)
            'mean': loaded['pca_mean'].to(device),              # (D,)
        }

    return z_curr.numpy(), z_next.numpy(), pca_info


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        var_actions=1.0,
        var_states=0.0,
        num_samples=8,
        action_clamp=0.0,
        action_reg=0.0,
        replay_buffer_path=None,
        latent_key="proprio",
        hjb_tau=0.1,
        use_isomap=False,
        isomap_landmarks=500,
        graph_n_nodes=55000,
        solver_mode='heat',
        rw_steps=10,
        graph_k_neighbors=32,
        use_full_dim=False,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.var_actions = var_actions
        self.var_states = var_states
        self.num_samples = num_samples
        self.action_clamp = action_clamp
        self.action_reg = action_reg
        self.latent_key = latent_key
        self.hjb_tau = hjb_tau
        self.use_isomap = use_isomap
        self.isomap_landmarks = isomap_landmarks
        self.graph_n_nodes = graph_n_nodes
        self.solver_mode = solver_mode
        self.rw_steps = rw_steps
        self.graph_k_neighbors = graph_k_neighbors
        self.use_full_dim = use_full_dim

        # Load replay buffer and build graph
        if replay_buffer_path is None:
            raise ValueError(
                "replay_buffer_path must be set in the planner config. "
                "Generate a buffer with: python build_latent_buffer.py env=<env_name>"
            )

        print(f"Loading replay buffer from {replay_buffer_path}...")
        z_buffer_np, z_next_gt_np, pca_info = _load_replay_buffer(replay_buffer_path, self.device)
        self.pca_info = pca_info
        if pca_info is not None:
            print(f"  PCA projection enabled: {pca_info['mean'].shape[0]} -> {pca_info['components'].shape[0]} dims")
        print(f"  Loaded {z_buffer_np.shape[0]} transitions, latent_dim={z_buffer_np.shape[1]}")

        # For rw_diffusion with PCA buffers, reconstruct full-dim latents for
        # graph construction (better spatial structure in 192D vs 10D PCA).
        # The surrogate MLP still operates in PCA space for planning compatibility.
        graph_z_buffer = z_buffer_np
        graph_z_next = z_next_gt_np
        if self.use_full_dim and pca_info is not None:
            comp_np = pca_info['components'].cpu().numpy()  # (pca_dim, D)
            mean_np = pca_info['mean'].cpu().numpy()         # (D,)
            graph_z_buffer = z_buffer_np @ comp_np + mean_np
            graph_z_next = z_next_gt_np @ comp_np + mean_np
            print(f"  Using full-dim latents for graph: {graph_z_buffer.shape[1]}D")

        self.heat_solver = MCMollifiedHeat()
        print(f"Compiling Latent Environment Graph (solver={self.solver_mode})...")
        self.heat_solver.compile_environment(
            z_buffer=graph_z_buffer,
            z_predicted_next=graph_z_next,
            n_nodes=self.graph_n_nodes,
            K_neighbors=self.graph_k_neighbors,
            use_isomap=self.use_isomap,
            isomap_landmarks=self.isomap_landmarks,
            solver_mode=self.solver_mode,
            rw_steps=self.rw_steps,
        )
        print("Graph built. Ready for planning.")

    def _get_latent(self, z_obses):
        """Extract the latent trajectory from the world model output.

        Uses self.latent_key to select between 'proprio' and 'visual'.
        For visual latents with a patch dimension, pools over patches.
        """
        latent = z_obses[self.latent_key].float()
        # Pool patch dimension if present (e.g. visual: B, T, num_patches, D)
        if latent.dim() == 4:
            latent = latent.mean(dim=2)  # (B, T, D)
        # Apply PCA projection if buffer was built with --pca-dim.
        # Skip PCA when use_full_dim=True: the graph and surrogate MLP operate
        # in the original latent space, which matches the world model output.
        if self.pca_info is not None and not self.use_full_dim:
            shape = latent.shape  # (B, T, D_orig)
            flat = latent.reshape(-1, shape[-1])  # (B*T, D_orig)
            flat = (flat - self.pca_info['mean']) @ self.pca_info['components'].T  # (B*T, pca_dim)
            latent = flat.reshape(*shape[:-1], -1)  # (B, T, pca_dim)
        return latent

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning.
        """
        if isinstance(obs_0, dict):
            n_evals = obs_0["visual"].shape[0]
        else:
            n_evals = obs_0.shape[0]
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = torch.randn(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        return torch.optim.SGD([actions], lr=self.lr)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=False)
        actions = self.init_actions(obs_0, actions).to(self.device)
        actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=self.var_actions).to(actions)
        actions_noise.requires_grad = False
        actions.requires_grad = True
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        n_evals = actions.shape[0]
        max_success_rate = 0.0
        best_dist = np.inf
        best_vis_dist = np.inf

        if isinstance(trans_obs_0, dict):
            trans_obs_0 = {k: v.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1) for k,v in trans_obs_0.items()}
            z_obs_g_detached = {k: v.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach() for k, v in z_obs_g.items()}
        else:
            trans_obs_0 = trans_obs_0[:,0].unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1)
            z_obs_g_detached = z_obs_g.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach()

        total_measured_time = 0.0
        exit_step = self.opt_steps

        # Solve heat equation for the goal using the configured latent key
        goal_latent = self._get_latent(z_obs_g_detached)
        current_goal_np = goal_latent[0, 0].detach().cpu().numpy()
        self.heat_solver.solve_for_goal(current_goal_np)

        current_success = 0.0
        mean_dist = np.inf
        mean_vis_dist = np.inf
        mean_proprio_dist = np.inf

        for i in range(self.opt_steps):
            start = time.perf_counter()
            optimizer.zero_grad()

            # Noise Annealing
            decay_factor = max(0.0, 1.0 - (i / float(self.opt_steps)))
            current_noise = self.var_actions * decay_factor
            actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=current_noise).to(actions)

            eps = torch.randn(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]).to(actions) * actions_noise

            # Expand observations for the batch size (num_samples * B)
            obs_batch = {
                key: arr.repeat_interleave(self.num_samples, dim=0) if arr.shape[0] == actions.shape[0] else arr
                for key, arr in trans_obs_0.items()
            } if isinstance(trans_obs_0, dict) else trans_obs_0

            final_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1)

            # World model rollout
            if self.var_states > 0:
                i_z_obses, i_zs = self.wm.rollout_noisy(obs_0=obs_batch, act=final_noisy_actions, var_states=self.var_states)
            else:
                i_z_obses, i_zs = self.wm.rollout(obs_0=obs_batch, act=final_noisy_actions)

            # Extract latent trajectory using configured key
            latent_rollout = self._get_latent(i_z_obses)
            B, T, D = latent_rollout.shape

            # Objective loss
            loss = self.objective_fn(i_z_obses, z_obs_g_detached, final_noisy_actions)

            # L2 action regularization with annealing: starts strong to prevent
            # adversarial exploitation, decays to let MSE gradients take over
            if self.action_reg > 0:
                reg_weight = self.action_reg * max(0.0, 1.0 - (i / float(self.opt_steps)))
                loss = loss + reg_weight * (actions ** 2).mean()

            # V guidance: backprop directly through surrogate MLP at terminal state
            tau = self.hjb_tau
            terminal_latent = latent_rollout[:, -1, :]
            v_terminal = self.heat_solver.evaluate_rollout(terminal_latent)
            loss = loss + tau * v_terminal.mean()

            # Backpropagate
            total_loss = loss.mean() * self.num_samples
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if self.action_clamp > 0:
                    actions.data.clamp_(-self.action_clamp, self.action_clamp)

            end = time.perf_counter()
            total_measured_time += end - start

            if i % self.eval_every == 0:
                v_val = v_terminal.mean().item() if tau > 0 else 0.0
                print(f"[Plan] step={i:4d} | loss={total_loss.item():.4f} | MSE={loss.mean().item():.4f} | V={v_val:.4f} | act_norm={actions.detach().norm(dim=-1).mean().item():.4f} | noise={current_noise:.4f}")

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                current_success = float(successes.mean())
                if current_success > max_success_rate:
                    max_success_rate = current_success
                mean_dist = logs.get('mean_state_dist')
                mean_vis_dist = logs.get('mean_visual_dist')
                mean_proprio_dist = logs.get('mean_proprio_dist')
                if mean_dist < best_dist and current_success >= max_success_rate:
                    best_dist = mean_dist
                if mean_vis_dist < best_vis_dist and current_success >= max_success_rate:
                    best_vis_dist = mean_vis_dist

                self.wandb_run.log({
                    f"{self.logging_prefix}/loss": total_loss.item(), "step": i + 1,
                    f"{self.logging_prefix}/act_norm": actions.detach().norm(dim=-1).mean().item(),
                    f"{self.logging_prefix}/success": successes.mean(),
                    f"{self.logging_prefix}/noise": actions_noise.mean().item()
                })

                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time})
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    print("######## Success At Step", i + 1, "########")
                    exit_step = i + 1
                    break

            if total_measured_time > 1800:
                exit_step = i + 1
                break

        # Final evaluation
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), filename=f"{self.logging_prefix}_output_final"
        )
        current_success = float(successes.mean())
        if current_success > max_success_rate:
            max_success_rate = current_success
        mean_dist = logs.get('mean_state_dist')
        mean_vis_dist = logs.get('mean_visual_dist')
        mean_proprio_dist = logs.get('mean_proprio_dist')
        if mean_dist < best_dist and current_success >= max_success_rate:
            best_dist = mean_dist
        if mean_vis_dist < best_vis_dist and current_success >= max_success_rate:
            best_vis_dist = mean_vis_dist

        self.wandb_run.log({
            f"{self.logging_prefix}/loss": total_loss.item(), "step": self.opt_steps,
            f"{self.logging_prefix}/act_norm": actions.detach().norm(dim=-1).mean().item(),
            f"{self.logging_prefix}/success": successes.mean(),
            f"{self.logging_prefix}/noise": actions_noise.mean().item()
        })

        logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
        logs.update({"total_measured_time": total_measured_time})
        logs.update({"step": self.opt_steps})
        self.wandb_run.log(logs)
        self.dump_logs(logs)

        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            log_filename = f"seed_success_chunk_{task_id}.txt"
        else:
            log_filename = "seed_success_log.txt"
        with open(f"/scratch/gilbreth/hviswan/grasp/{log_filename}", "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {i}\n")

        return actions, np.full(n_evals, np.inf)
