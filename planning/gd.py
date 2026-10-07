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
        self.x_surrogate = None   # unit direction field (heat method step II)
        # K-NN patches retained for the heat-method gradient (steps II/III).
        # compile_environment discards these once L_sym is assembled, but the
        # least-squares gradient needs the raw neighbour lists and weights.
        self.knn_indices = None
        self.knn_weights = None
        self.solver_mode = 'heat'
        self.rw_steps = 10
        self.P_trans = None  # transition matrix for rw_diffusion

    def compile_environment(self, z_buffer, z_predicted_next=None, n_nodes=None,
                             K_neighbors=32, mc_candidates=4096, chunk_size=8192,
                             use_isomap=False, isomap_landmarks=500,
                             solver_mode='heat', rw_steps=10,
                             use_spatial_knn=False, knn_dedup=True):
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
        else:
            idx = np.arange(z_scaled.shape[0])
        self.node_idx = idx          # buffer row of each vertex (episode lookup)

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
                    # Rounds draw candidates independently, so a vertex already in
                    # the running top-K can be drawn again; without this mask it is
                    # kept twice and csr_matrix below SUMS the duplicate entries,
                    # multiplying that edge's weight (measured before the fix: 100%
                    # of rows on the multi-round roadmaps, 42-46 distinct of 64).
                    if knn_dedup and r > 0:
                        prev = all_k_indices[start:end]
                        prev_ok = torch.isfinite(all_k_dists[start:end])
                        dup = ((chunk_k_indices[:, :, None] == prev[:, None, :])
                               & prev_ok[:, None, :]).any(-1)
                        chunk_k_dists = chunk_k_dists.masked_fill(dup, float('inf'))

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

            self.knn_indices = geo_knn_indices
            self.knn_weights = W_local

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
            self.degree = d

            self.knn_indices = all_k_indices.cpu().numpy()
            self.knn_weights = W_local_np

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
            # Crane et al.: t = m * h^2 with m ~ 1. m=500 diffuses to near
            # equilibrium -- measured u range collapses to 1.7x and V's total
            # variation to 0.081 (vs 6.87 at m=1), leaving no slope to descend.
            t = h2 * getattr(self, 'heat_m', 1.0)
            # 'degree': make tL dimensionless, t = heat_c / median(D_ii). Crane's
            # t = m h^2 is scale-invariant only because their Laplacian carries
            # a 1/h^2 mass normalisation; L = D - W here has dimensionless
            # weights, so t = sigma^2 puts t*D_ii anywhere from ~1e-2 (wall) to
            # ~1e2 (cube) and the solve is in a different regime per env.
            if getattr(self, 'heat_t_rule', 'bandwidth') == 'degree':
                t = getattr(self, 'heat_c', 1.0) / float(np.median(self.degree))
            self.heat_t = t
            b = np.zeros(N)
            b[self.goal_idx] = 1.0
            A_heat = sp.eye(N) + t * self.L_sym
            M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
            u, info = spla.cg(A_heat, b, rtol=1e-8, maxiter=5000, M=M_pre)
            if info != 0:
                print(f"[Warning] CG exited with code {info}")
            u = np.maximum(u, 0.0)
            # -log u, matching the rw_diffusion branch. `u.max() - u` is linear
            # in u and therefore nearly constant once u is smooth.
            V_raw = -np.log(np.maximum(u, 1e-300))

        t_solve = _time.perf_counter() - t0

        # Reshape the potential. V's ORDERING tracks geodesic distance well
        # (spearman 0.93-0.98) but its VALUES do not: fitting V ~ d^p gives
        # p ~ 0.53, i.e. a concave field whose gradient is largest AT the goal
        # and decays outward -- the opposite of what an objective needs, and
        # singular rather than vanishing at the goal. The resolvent (I+tL)^-1
        # decays like exp(-d/lambda), so -log u is at best linear in d; it can
        # never be the d^2/4t bowl Varadhan gives for the true kernel exp(-tL).
        #
        # So rebuild the profile from the part that is reliable. Rank-transform
        # V to a uniform distance surrogate r in [0, 1], then square it:
        # |grad(r^2)| ~ r grows with distance and vanishes at the goal, which is
        # the MSE profile expressed in geodesic distance.
        # Rank maps to quantile, and on an n-dimensional manifold r ~ d^n, so
        # the LINEAR rank transform already yields V ~ d^2.26 here -- measured
        # p=2.263, matching this buffer's TwoNN intrinsic dimension of 2.26.
        # That is the bowl we want; squaring overshoots to p=4.5. On a
        # higher-dimensional manifold use exponent 2/n instead of 1.
        from scipy.stats import rankdata
        r = (rankdata(V_raw).astype(np.float64) - 1.0) / max(len(V_raw) - 1, 1)
        V_graph_np = (r ** getattr(self, 'v_exponent', 1.0)) * phys_v_mean * 2

        # Diagnostic controls: keep V's SCALE but destroy its meaning, to test
        # whether the guidance contributes geodesic information or merely
        # perturbs the optimisation landscape. If a scale-matched nonsense
        # field rescues the same seeds, the term is not doing what it claims.
        ctrl = getattr(self, 'v_control', None)
        if ctrl and ctrl != 'none':
            rng = np.random.RandomState(0)
            lo, hi, mean = V_graph_np.min(), V_graph_np.max(), V_graph_np.mean()
            if ctrl == 'shuffle':          # same values, no spatial structure
                V_graph_np = rng.permutation(V_graph_np)
            elif ctrl == 'constant':       # no gradient at all (should match tau=0)
                V_graph_np = np.full_like(V_graph_np, mean)
            elif ctrl == 'randsmooth':     # smooth, wrong: random linear field
                w = rng.randn(self.z_buffer_np.shape[1])
                p = self.z_buffer_np @ w
                V_graph_np = (p - p.min()) / (np.ptp(p) + 1e-12) * (hi - lo) + lo
            elif ctrl == 'wrong_goal':     # real geodesic structure, wrong target
                gi2 = rng.randint(len(V_graph_np))
                v2 = np.zeros(len(V_graph_np)); v2[gi2] = 1.0
                for _ in range(self.rw_steps):
                    v2 = self.P_trans @ v2 if self.P_trans is not None else v2
                v2 = -np.log(np.maximum(v2, 1e-300))
                V_graph_np = (v2 - v2.min()) / (np.ptp(v2) + 1e-12) * (hi - lo) + lo
            print(f"[V-CONTROL] '{ctrl}' applied | range [{V_graph_np.min():.3f}, "
                  f"{V_graph_np.max():.3f}] mean={V_graph_np.mean():.3f} "
                  f"(original [{lo:.3f}, {hi:.3f}] mean={mean:.3f})")

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

        # The fit below freezes these at the end so the planner can backprop
        # through the surrogate without touching its weights. On a second call
        # (new goal, or n_evals > 1) they are still frozen, which makes
        # loss.backward() raise, so re-enable before refitting.
        for p in self.v_surrogate.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(self.v_surrogate.parameters(), lr=5e-3)
        # surrogate_steps > 250 also anneals the learning rate (cosine): the
        # sharper fields of a small-k roadmap are not fit in 250 constant-lr steps.
        n_steps = int(getattr(self, 'surrogate_steps', 250))
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
                 if n_steps > 250 else None)
        self.v_surrogate.train()
        for step in range(n_steps):
            optimizer.zero_grad()
            pred = self.v_surrogate(self.graph_nodes).squeeze()
            loss = nn.functional.mse_loss(pred, self.V_graph)
            loss.backward()
            optimizer.step()
            if sched is not None:
                sched.step()

        self.v_surrogate.eval()
        for p in self.v_surrogate.parameters():
            p.requires_grad = False

        t_surr = _time.perf_counter() - t0
        print(f"[Surrogate] Fit: {t_surr:.2f}s | final MSE={loss.item():.4f}")

        return self.V_graph

    # ------------------------------------------------------------------
    # Heat method (Crane, Weischedel & Wardetzky, CACM 2017), steps II/III.
    #
    # solve_for_goal above reads distance straight off the diffusion field
    # via -log(u) (Varadhan's formula). That is exact as t -> 0 but poor on
    # an approximate kernel, because it is sensitive to errors in the
    # *magnitude* of u. The heat method keeps only the direction of grad(u),
    # which is the part that survives approximation, and recovers phi by
    # least-squares against that unit field.
    #
    # Step I is unchanged and mode-agnostic: any positive, monotonically
    # decaying u works, so 'rw_diffusion' feeds these as happily as 'heat'.
    # ------------------------------------------------------------------

    def compute_unit_gradient(self, u, chunk_size=2048, rcond=1e-8):
        """Step II: X = -grad(u)/|grad(u)| at every graph node.

        There are no faces on a point cloud, so the per-face gradient of the
        mesh formulation is replaced by a weighted least-squares fit over
        each node's K-NN patch -- find g minimising

            sum_j w_ij ( u_j - u_i - g . (z_j - z_i) )^2

        The system is (K, D) with K < D, so the minimum-norm solution is
        taken. It lands in the span of the neighbour offsets, i.e. the
        estimated tangent space, which is why no explicit projection or
        intrinsic-dimension estimate is needed.

        u decays exponentially with distance, so each patch is rescaled by
        its own maximum before differencing -- cf. computeUnitGradient in
        Geodesics_in_heat, which divides each face by max(u0, u1, u2).
        Without this the differences underflow far from the source. The
        scale cancels under normalisation, so it costs nothing.

        Args:
            u: (N,) non-negative diffusion field from step I.
        Returns:
            X: (N, D) unit vectors in scaled coordinates, pointing away
               from the source. Zero rows where the patch is degenerate.
        """
        if self.knn_indices is None:
            raise RuntimeError("compile_environment must run before compute_unit_gradient")

        dev = self.device
        u_t = torch.as_tensor(np.asarray(u), dtype=torch.float32, device=dev)
        idx = torch.as_tensor(self.knn_indices, dtype=torch.long, device=dev)
        w = torch.as_tensor(self.knn_weights, dtype=torch.float32, device=dev)
        Z = self.graph_nodes
        N, K = idx.shape
        X = torch.zeros_like(Z)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            nb = idx[start:end]                                  # (B, K)
            u_i = u_t[start:end].unsqueeze(1)                    # (B, 1)
            u_j = u_t[nb]                                        # (B, K)

            # Local rescale: bring the patch back to O(1) before differencing.
            scale = torch.maximum(u_j.max(dim=1, keepdim=True).values, u_i)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            du = (u_j - u_i) / scale                             # (B, K)

            delta = Z[nb] - Z[start:end].unsqueeze(1)            # (B, K, D)
            sw = torch.sqrt(torch.clamp(w[start:end], min=0.0))  # (B, K)

            A = (sw.unsqueeze(-1) * delta).double()              # (B, K, D)
            b = (sw * du).double()                               # (B, K)

            # Minimum-norm solution g = A^+ b, via a truncated pseudo-inverse
            # of the (K, K) Gram matrix -- never (D, D), so the ambient
            # dimension costs nothing. A K-NN patch on a d-dimensional
            # manifold spans only d << K directions, so the Gram matrix is
            # badly rank-deficient by construction; a plain solve (or a small
            # ridge) inverts noise in the null space. eigh in float64 with an
            # explicit rank cut is what keeps this stable.
            M = A @ A.transpose(1, 2)                            # (B, K, K)
            evals, evecs = torch.linalg.eigh(M)                  # ascending
            cutoff = rcond * evals[:, -1:].clamp(min=0)
            inv = torch.where(evals > cutoff, 1.0 / evals.clamp(min=1e-300),
                              torch.zeros_like(evals))
            c = evecs @ (inv * torch.einsum('bkj,bk->bj', evecs, b)).unsqueeze(-1)
            g = (A.transpose(1, 2) @ c).squeeze(-1).float()      # (B, D)

            gn = g.norm(dim=-1, keepdim=True)
            X[start:end] = torch.where(gn > 1e-12, -g / gn.clamp(min=1e-12),
                                       torch.zeros_like(g))

        n_dead = int((X.norm(dim=-1) == 0).sum().item())
        print(f"[Heat II] Unit gradient: N={N} K={K} | degenerate patches={n_dead} "
              f"({100.0 * n_dead / N:.2f}%)")
        return X

    def solve_phi(self, X, goal_idx=None):
        """Step III: solve the Poisson problem L phi = div X.

        phi is the least-squares potential whose gradient best matches the
        unit field X, i.e. the minimiser of sum_ij w_ij (phi_j - phi_i -
        X_ij . (z_j - z_i))^2. Setting the derivative to zero gives

            (L phi)_i = - sum_j w_ij X_ij . (z_j - z_i)

        with X_ij = (X_i + X_j)/2 the field on edge (i, j). The reference
        implementation factors the same cotan matrix used in step I and
        solves it against the integrated divergence; L_sym is reused here
        for exactly that reason -- no second operator is built.

        The edge sum expands into four sparse mat-vecs, so this never
        materialises an (nnz, D) intermediate.

        Args:
            X: (N, D) unit vector field from compute_unit_gradient.
            goal_idx: node to shift to zero. Defaults to self.goal_idx.
        Returns:
            phi: (N,) geodesic distance in scaled-coordinate units.
        """
        X_np = X.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(X) else np.asarray(X)
        Z = self.z_buffer_np.astype(np.float64)

        # Recover W from L = D - W so the divergence uses precisely the
        # weights L_sym was assembled from (the K-NN lists are directed and
        # pre-symmetrisation, so they would not match).
        W = -self.L_sym.copy()
        W.setdiag(0.0)
        W.eliminate_zeros()

        d = np.asarray(W.sum(axis=1)).ravel()                # (N,)
        s = np.einsum('ij,ij->i', X_np, Z)                   # X_j . z_j
        flux = 0.5 * (
            np.einsum('ij,ij->i', X_np, W @ Z)               # X_i . sum_j w_ij z_j
            - d * s                                          # - (X_i . z_i) sum_j w_ij
            + (W @ s)                                        # + sum_j w_ij (X_j . z_j)
            - np.einsum('ij,ij->i', W @ X_np, Z)             # - (sum_j w_ij X_j) . z_i
        )
        b = -flux

        # L is singular on the constants; project b onto the compatible
        # subspace, then pin the gauge by shifting the source to zero.
        b -= b.mean()

        M_pre = sp.diags(1.0 / (self.L_sym.diagonal() + 1e-8))
        phi, info = spla.cg(self.L_sym, b, rtol=1e-8, maxiter=5000, M=M_pre)
        if info != 0:
            print(f"[Heat III] Warning: CG exited with code {info}")

        gi = self.goal_idx if goal_idx is None else goal_idx
        phi = phi - phi[gi]
        # phi must increase away from the source; CG returns it up to sign
        # through the singular direction (cf. the mean-sign flip in
        # computeGeodesicDistances).
        if phi.mean() < 0:
            phi = -phi
        phi = phi - phi.min()

        print(f"[Heat III] phi: [{phi.min():.4f}, {phi.max():.4f}] "
              f"mean={phi.mean():.4f} | source residual={abs(phi[gi]):.2e}")
        return phi

    def solve_for_goal_heat_method(self, z_goal, u=None):
        """Steps I-III end to end, returning phi alongside the Varadhan V.

        Does not touch solve_for_goal or the surrogate; run both on the same
        buffer and compare. Step I is reused as-is, including solver_mode.

        Returns:
            dict with 'phi' (heat method), 'V_varadhan' (-log u, unranked)
            and 'X' (the unit field, useful on its own -- a gradient-based
            planner consumes grad(phi) and can read X directly, skipping
            step III entirely).
        """
        if u is None:
            z_goal_proj = self.scaler_obj.transform(np.asarray(z_goal).reshape(1, -1))
            goal_dists = np.linalg.norm(self.z_buffer_np - z_goal_proj, axis=1)
            self.goal_idx = int(np.argmin(goal_dists))

            N = self.L_sym.shape[0]
            if self.solver_mode == 'rw_diffusion':
                u = np.zeros(N)
                u[self.goal_idx] = 1.0
                for _ in range(self.rw_steps):
                    u = self.P_trans @ u
            else:
                t = (self.sigma_perturb ** 2) * 500
                rhs = np.zeros(N)
                rhs[self.goal_idx] = 1.0
                A_heat = sp.eye(N) + t * self.L_sym
                M_pre = sp.diags(1.0 / (A_heat.diagonal() + 1e-8))
                u, info = spla.cg(A_heat, rhs, rtol=1e-8, maxiter=5000, M=M_pre)
                if info != 0:
                    print(f"[Heat I] Warning: CG exited with code {info}")
            u = np.maximum(u, 0.0)

        X = self.compute_unit_gradient(u)
        phi = self.solve_phi(X)

        from scipy.stats import spearmanr
        euclid = np.linalg.norm(self.z_buffer_np - self.z_buffer_np[self.goal_idx], axis=1)
        V_var = -np.log(np.maximum(u, 1e-300))
        print(f"[Compare] Spearman vs Euclid -- phi={spearmanr(phi, euclid).correlation:.4f} "
              f"V_varadhan={spearmanr(V_var, euclid).correlation:.4f}")

        return {'phi': phi, 'V_varadhan': V_var, 'X': X, 'u': u}

    def fit_direction_surrogate(self, X, steps=400, lr=5e-3):
        """Fit a smooth unit vector field to the step-II directions.

        The planner queries at arbitrary rollout latents, not at graph nodes,
        so X has to be interpolated. Fitting the *vector field* directly (as
        opposed to fitting phi and differentiating it) avoids the
        differentiate -> Poisson-integrate -> differentiate round trip, and
        keeps |X| = 1 exactly at query time via an explicit renormalisation.

        X is converted from scaled to unscaled latent coordinates first: a
        gradient is a covector, so it maps as grad_z = grad_z_scaled / sigma.
        The surrogate therefore predicts directions in the same coordinates
        the planner's rollout latents live in, and no conversion is needed
        during planning.

        Args:
            X: (N, D) unit directions at graph nodes, in scaled coordinates.
        """
        with torch.no_grad():
            Xz = X / (self.proj_std + 1e-8)
            Xz = Xz / Xz.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        D = self.graph_nodes.shape[1]
        self.x_surrogate = nn.Sequential(
            nn.Linear(D, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish(),
            nn.Linear(256, D),
        ).to(self.device)

        opt = torch.optim.Adam(self.x_surrogate.parameters(), lr=lr)
        self.x_surrogate.train()
        for _ in range(steps):
            opt.zero_grad()
            pred = self.x_surrogate(self.graph_nodes)
            pred = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            loss = (1.0 - (pred * Xz).sum(-1)).mean()      # mean cosine distance
            loss.backward()
            opt.step()

        self.x_surrogate.eval()
        for p in self.x_surrogate.parameters():
            p.requires_grad = False
        print(f"[Heat II] Direction surrogate: mean cosine error={loss.item():.4f} "
              f"(0 = perfect)")
        return self.x_surrogate

    def set_direction_field(self, X):
        """Store step-II directions for exact nearest-node lookup.

        Preferred over fit_direction_surrogate: the injection
        `loss += tau * <z_T, X.detach()>` uses X as a CONSTANT, so no
        differentiable interpolant is needed and the MLP's fit error (which
        reached cosine error 0.90 -- essentially orthogonal -- when trained on
        a field that was 79% zeros) is removed entirely.

        Converted to unscaled coordinates once, here: a gradient is a covector,
        so grad_z = grad_z_scaled / sigma.
        """
        with torch.no_grad():
            Xz = X / (self.proj_std + 1e-8)
            n = Xz.norm(dim=-1, keepdim=True)
            self.X_nodes = torch.where(n > 1e-12, Xz / n.clamp(min=1e-12),
                                       torch.zeros_like(Xz))
        frac = (self.X_nodes.norm(dim=-1) == 0).float().mean().item()
        print(f"[Heat II] Direction field stored for lookup | "
              f"degenerate nodes={100*frac:.2f}%")
        return self.X_nodes

    @torch.no_grad()
    def lookup_direction(self, z_generated, chunk=4096):
        """Exact X at the nearest graph node (no surrogate, no fit error)."""
        zs = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)
        out = []
        for s in range(0, len(zs), chunk):
            idx = torch.cdist(zs[s:s + chunk], self.graph_nodes).argmin(dim=1)
            out.append(self.X_nodes[idx])
        return torch.cat(out)

    def evaluate_direction(self, z_generated):
        """Unit grad(phi) at arbitrary query latents.

        Returns X pointing AWAY from the source, matching the sign convention
        of the scalar path (V = -log u also increases away from the goal), so
        that `loss += tau * <z, X>` descends toward the goal exactly as
        `loss += tau * V` does.

        Detached by design: it is injected as a known gradient rather than
        differentiated through, so the planner never needs a second-order
        graph.
        """
        with torch.no_grad():
            z_scaled = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)
            X = self.x_surrogate(z_scaled)
            X = X / X.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return X

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
        heat_t_rule="bandwidth",
        heat_c=1.0,
        surrogate_steps=250,
        num_candidates=1,
        cand_chunk=0,
        cand_sync_interval=100,
        guidance_mode='scalar',
        v_control=None,
        mse_weight=1.0,
        metric_weights_path=None,
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
        # Multi-start GD: num_candidates independent action sequences, each
        # optimised by its own gradient; the lowest-objective one is executed.
        # 1 = the original single-sequence planner (plan() path unchanged).
        self.num_candidates = int(num_candidates)
        self.cand_chunk = int(cand_chunk)
        self.cand_sync_interval = int(cand_sync_interval)
        self.action_clamp = action_clamp
        self.action_reg = action_reg
        self.latent_key = latent_key
        self.hjb_tau = hjb_tau
        # 'scalar'    : loss += tau * V(z_T), V = -log u (Varadhan). V is
        #               rank-normalised to ~[0, 2*phys_v_mean] while the MSE is
        #               a mean over D dims, so tau absorbs a ~D*phys_v_mean unit
        #               mismatch (~4000x for reacher) on top of any real weight.
        # 'direction' : loss += tau * (z_T . X_T), |X_T| = 1. The gradient wrt
        #               z_T is exactly X_T, so tau is a step size in latent
        #               units and is dimensionally commensurate with the MSE
        #               gradient. Only the direction of grad(u) is used, which
        #               is the part that survives an approximate kernel
        #               (Crane et al. 2017, step II).
        self.guidance_mode = guidance_mode
        self.v_control = v_control
        # Scales the terminal MSE term. Set to 0 to test whether the heat
        # guidance can drive planning on its own, with no goal-distance
        # objective at all.
        self.mse_weight = mse_weight

        # Optional diagonal latent metric M = diag(w). The plain objective
        # weights all D latent dims equally, but only ~21% of latent variance
        # is task-relevant here, so most of the descent is spent matching
        # nuisance directions. Weighting the terminal term by w concentrates
        # it on the dims that carry task distance. w is learned from TEMPORAL
        # pairs (|step_i - step_j| within an episode), so no state labels and
        # no decoding are involved -- planning stays in latent space.
        self.metric_w = None
        if metric_weights_path is not None:
            w = np.load(metric_weights_path).astype(np.float32)
            self.metric_w = torch.tensor(w, device=self.device)
            print(f"  Latent metric: {len(w)} dims, "
                  f"top-10 carry {100*np.sort(w)[-10:].sum()/w.sum():.1f}% of weight")
        self.use_isomap = use_isomap
        self.isomap_landmarks = isomap_landmarks
        self.graph_n_nodes = graph_n_nodes
        self.solver_mode = solver_mode
        self.rw_steps = rw_steps
        self.graph_k_neighbors = graph_k_neighbors
        self.use_full_dim = use_full_dim

        # Load replay buffer and build graph.
        # At hjb_tau == 0 the guidance term is multiplied by zero, so V is never
        # used: skip the buffer load and the k-NN graph build entirely. This
        # keeps the un-guided baseline runnable on environments that have no
        # latent buffer, and avoids paying graph construction on every run.
        self.heat_solver = None
        self.pca_info = None           # set below when a buffer is loaded
        if hjb_tau == 0 and guidance_mode != 'direction':
            print("[Plan] hjb_tau=0 -> skipping replay buffer / graph (V unused)")
            self._skip_v = True
            return

        self._skip_v = False
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
        self.heat_solver.v_control = getattr(self, 'v_control', None)
        self.heat_solver.heat_t_rule = heat_t_rule   # see solve_for_goal
        self.heat_solver.heat_c = float(heat_c)
        self.heat_solver.surrogate_steps = int(surrogate_steps)
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

    def _plan_multistart(self, obs_0, obs_g):
        """GD with C = num_candidates independent starts (baseline config only:
        no heat guidance, no action/state noise smoothing, num_samples=1).

        Each candidate gets the gradient of its own objective (sum over
        candidates, so gradients are not coupled or rescaled). Backward runs
        in chunks of cand_chunk candidates -- exact, since candidates are
        independent -- to bound memory. At every evaluation, and for the final
        plan, the candidate with the lowest current objective is executed: the
        same selection rule the EM-SMC planner applies to its particles.
        """
        assert self.var_states == 0 and self.heat_solver is None \
            and self.metric_w is None, "multi-start GD supports the baseline configs only"
        C, S = self.num_candidates, self.num_samples
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(obs_0, obs_g, encode_to_vector=False)
        # Mirrors GRASP's own multi-trajectory mode (planning/grasp.py,
        # num_parallel_trajectories): every candidate from the planner's own
        # initialisation; the lowest-loss candidate is executed at each
        # evaluation and returned; and every cand_sync_interval steps (GRASP:
        # gd_interval=100, iter % interval == 0 and iter > 0) the best
        # candidate is copied into all slots (data only, optimiser state kept,
        # as GRASP does).
        cands = [self.init_actions(obs_0, None) for _ in range(C)]
        actions = torch.cat(cands, dim=0).to(self.device).requires_grad_(True)   # (C, H, A)
        optimizer = torch.optim.Adam([actions], lr=self.lr)
        ch = self.cand_chunk if self.cand_chunk > 0 else C * self.num_samples   # rows per rollout

        def expand(x, n):
            if isinstance(x, dict):
                return {k: v[:1].repeat_interleave(n, dim=0) for k, v in x.items()}
            return x[:1].repeat_interleave(n, dim=0)

        max_success_rate, current_success = 0.0, 0.0
        mean_dist = mean_vis_dist = mean_proprio_dist = np.inf
        total_measured_time, exit_step = 0.0, self.opt_steps
        cand_loss = torch.full((C,), float('inf'), device=self.device)
        best = 0
        for i in range(self.opt_steps):
            start = time.perf_counter()
            optimizer.zero_grad()
            decay = max(0.0, 1.0 - (i / float(self.opt_steps)))
            reg_weight = self.action_reg * decay
            noise = self.var_actions * decay        # annealed smoothing noise, as in plan()
            # Rows = (smoothing sample s, candidate j), r = s * C + j. Chunk over
            # rows: each row's gradient depends only on its own rollout, so the
            # accumulated gradient is exact whatever the chunk size; it only
            # bounds memory (C=1, S=12 is smoothed GD on one trajectory).
            eps = torch.randn(S, C, *actions.shape[1:], device=actions.device)
            cand_loss = torch.zeros(C, device=self.device)
            R = S * C
            for r0 in range(0, R, ch):
                r = torch.arange(r0, min(r0 + ch, R), device=actions.device)
                si, ji = r // C, r % C
                n = r.numel()
                ob = expand(trans_obs_0, n)
                gb = ({k: v.detach() for k, v in expand(z_obs_g, n).items()}
                      if isinstance(z_obs_g, dict) else expand(z_obs_g, n).detach())
                an = actions[ji] + noise * eps[si, ji]
                i_z, _ = self.wm.rollout(obs_0=ob, act=an)
                lr_ = self.mse_weight * self.objective_fn(i_z, gb, an) / S      # (n,)
                cand_loss.index_add_(0, ji, lr_.detach())
                lr_.sum().backward()
            if self.action_reg > 0:
                (reg_weight * (actions ** 2).mean(dim=(1, 2))).sum().backward()
            optimizer.step()
            with torch.no_grad():
                if self.action_clamp > 0:
                    actions.data.clamp_(-self.action_clamp, self.action_clamp)
            best = int(torch.argmin(cand_loss))
            if self.cand_sync_interval > 0 and i % self.cand_sync_interval == 0 and i > 0:
                with torch.no_grad():
                    actions.data = actions.data[best:best + 1].repeat(C, 1, 1)
                print(f"[Plan-MS] step={i:4d} | sync: candidate {best} copied to all {C} slots")
            total_measured_time += time.perf_counter() - start
            if i % self.eval_every == 0:
                print(f"[Plan-MS] step={i:4d} | C={C} | best MSE={cand_loss[best].item():.4f} "
                      f"| mean MSE={cand_loss.mean().item():.4f} | best={best}")
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions[best:best + 1].detach(), filename=f"{self.logging_prefix}_output_{i+1}")
                current_success = float(successes.mean())
                max_success_rate = max(max_success_rate, current_success)
                mean_dist = logs.get('mean_state_dist'); mean_vis_dist = logs.get('mean_visual_dist')
                mean_proprio_dist = logs.get('mean_proprio_dist')
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time, "step": i + 1})
                self.wandb_run.log(logs); self.dump_logs(logs)
                if np.all(successes):
                    print("######## Success At Step", i + 1, "########")
                    exit_step = i + 1
                    break
            if total_measured_time > 1800:
                exit_step = i + 1
                break
        final = actions[best:best + 1].detach()
        logs, successes, _, _ = self.evaluator.eval_actions(final, filename=f"{self.logging_prefix}_output_final")
        current_success = max(float(successes.mean()), 0.0)
        max_success_rate = max(max_success_rate, current_success)
        mean_dist = logs.get('mean_state_dist'); mean_vis_dist = logs.get('mean_visual_dist')
        mean_proprio_dist = logs.get('mean_proprio_dist')
        logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
        logs.update({"total_measured_time": total_measured_time, "step": self.opt_steps})
        self.wandb_run.log(logs); self.dump_logs(logs)
        self.exit_step, self.total_measured_time = exit_step, total_measured_time
        results_dir = os.environ.get("GRASP_RESULTS_DIR", os.getcwd())
        with open(os.path.join(results_dir, "seed_success_log.txt"), "a") as f:
            f.write(f"{max_success_rate} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {i}\n")
        return final, np.full(1, np.inf)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        if self.num_candidates > 1 or self.cand_chunk > 0:
            return self._plan_multistart(obs_0, obs_g)
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
        if self.heat_solver is None:
            pass                       # hjb_tau == 0: nothing to solve
        elif self.guidance_mode == 'direction':
            # Steps I-III, then fit the unit field. Step I is shared with the
            # scalar path and honours solver_mode as before.
            out = self.heat_solver.solve_for_goal_heat_method(current_goal_np)
            self.heat_solver.set_direction_field(out['X'])
        else:
            self.heat_solver.solve_for_goal(current_goal_np)

        # Opt-in diagnostic dump (GRASP_DEBUG_DUMP=<path>): records the goal
        # latent and the predicted terminal latent per step, so the objective
        # can be analysed offline without re-running the planner.
        _dump_path = os.environ.get("GRASP_DEBUG_DUMP")
        _dump = {"z_goal": goal_latent[0, 0].detach().cpu().numpy(),
                 "z_term": [], "step": []} if _dump_path else None

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
            if self.metric_w is not None:
                # Metric-weighted terminal term, replacing the uniform mean
                # over latent dims. Weights are normalised to mean 1 upstream,
                # so this stays on the same scale as the unweighted MSE and
                # hjb_tau keeps its meaning.
                diff = latent_rollout[:, -1, :] - goal_latent.reshape(1, -1)
                loss = (self.metric_w * diff.pow(2)).mean(dim=-1)
            else:
                loss = self.objective_fn(i_z_obses, z_obs_g_detached, final_noisy_actions)
            mse_val = loss.mean().item()
            loss = self.mse_weight * loss

            # L2 action regularization with annealing: starts strong to prevent
            # adversarial exploitation, decays to let MSE gradients take over
            if self.action_reg > 0:
                reg_weight = self.action_reg * max(0.0, 1.0 - (i / float(self.opt_steps)))
                loss = loss + reg_weight * (actions ** 2).mean()

            # Heat guidance at the terminal state.
            tau = self.hjb_tau
            terminal_latent = latent_rollout[:, -1, :]
            if self.heat_solver is None:
                v_val = 0.0            # hjb_tau == 0: term contributes nothing
            elif self.guidance_mode == 'direction':
                # Gradient injection: d/dz_T of (z_T . X_T) is exactly X_T, a
                # unit vector, so no scalar potential and no second-order graph
                # are needed. The reported value is a dot product, not a
                # distance, so it is not comparable to the scalar path's V.
                X_terminal = self.heat_solver.lookup_direction(terminal_latent)
                guide = (terminal_latent * X_terminal).sum(dim=-1)
                v_val = guide.mean().item()
                loss = loss + tau * guide.mean()
            else:
                v_terminal = self.heat_solver.evaluate_rollout(terminal_latent)
                v_val = v_terminal.mean().item()
                loss = loss + tau * v_terminal.mean()

            if _dump is not None:
                _dump["z_term"].append(latent_rollout[0, -1, :].detach().cpu().numpy())
                _dump["step"].append(i)

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
                print(f"[Plan] step={i:4d} | total={total_loss.item():.4f} | MSE={mse_val:.4f} | V={v_val:.4f} | act_norm={actions.detach().norm(dim=-1).mean().item():.4f} | noise={current_noise:.4f}")

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

        if _dump is not None:
            _da = self.preprocessor.denormalize_actions(
                rearrange(actions.detach().cpu(), "b t (f d) -> b (t f) d",
                          f=self.evaluator.frameskip)).numpy()
            np.savez(_dump_path, z_goal=_dump["z_goal"], exec_actions=_da,
                     z_term=np.array(_dump["z_term"]), step=np.array(_dump["step"]))
            print(f"[Debug] dumped {len(_dump['z_term'])} terminal latents -> {_dump_path}")

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            log_filename = f"seed_success_chunk_{task_id}.txt"
        else:
            log_filename = "seed_success_log.txt"
        results_dir = os.environ.get("GRASP_RESULTS_DIR", os.getcwd())
        with open(os.path.join(results_dir, log_filename), "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {i}\n")

        return actions, np.full(n_evals, np.inf)
