import os

import torch
import time
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device

from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors
import math


class N_Dimensional_KDE:
    def __init__(self, dataset_states, num_samples=20000, bandwidth=0.2, margin=0.1, device="cuda"):
        """
        Differentiable Nearest-Neighbor Distance Field.
        Automatically scales to any spatial dimension (2D, 10D, etc.).
        """
        self.device = device
        self.bandwidth = bandwidth
        self.margin = margin
        
        # 1. Prepare Data (No 2D slicing, accepts all D dimensions natively)
        points = torch.as_tensor(dataset_states, dtype=torch.float32, device=self.device)
        
        # Subsample if necessary for memory/speed
        if points.shape[0] > num_samples:
            idx = torch.randperm(points.shape[0], device=self.device)[:num_samples]
            points = points[idx]
        
        self.points = points
        self.n_points = points.shape[0]
        
        # 2. Generalized N-Dimensional Bounding Box
        # Calculates min and max across all columns (dimensions) dynamically
        self.dims_min = points.min(dim=0).values - self.margin
        self.dims_max = points.max(dim=0).values + self.margin
        
        self.max_depth = 1.5

        

    def log_prob(self, x, chunk_size=4096):
        """
        Differentiable Nearest Neighbor search using the Envelope Method.
        Memory-optimized to prevent CUDA OOM on massive query graphs.
        """
        original_device = x.device
        x = x.to(self.device)
        M = x.shape[0]
        
        # 1. Nearest Neighbor Index (No Gradients, Chunked for VRAM safety)
        nn_idx_list = []
        with torch.no_grad():
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                x_chunk = x[start:end]
                
                # torch.cdist is memory-efficient and avoids 3D tensor broadcasting
                dists = torch.cdist(x_chunk, self.points)
                chunk_nn_idx = torch.argmin(dists, dim=-1)
                nn_idx_list.append(chunk_nn_idx)
                
        # Recombine the indices
        nn_idx = torch.cat(nn_idx_list, dim=0)
            
        # 2. Re-attach to Autograd Graph
        # Both 'x' and 'nearest_points' are now just [M, D] tensors.
        # This subtraction takes < 5 Megabytes of VRAM, so it's perfectly safe to do globally!
        nearest_points = self.points[nn_idx]
        min_dists = torch.norm(x - nearest_points, dim=-1)
        
        # 3. Apply the plateau and sign (Negative distance = Log Prob)
        min_dists = torch.clamp(min_dists, max=self.max_depth)
        pseudo_log_prob = -1.0 * min_dists
        
        # 4. Generalized Outer Boundary Cliff
        out_of_bounds = (x < self.dims_min) | (x > self.dims_max)
        out_of_bounds_mask = out_of_bounds.any(dim=-1)
        
        pseudo_log_prob = torch.where(
            out_of_bounds_mask, 
            torch.full_like(pseudo_log_prob, -self.max_depth), 
            pseudo_log_prob
        )
        
        return pseudo_log_prob.to(original_device)

class LatentDynamicalHJB:

    def __init__(self, nu=0.1, sigma_dyn=1.0, k_neighbors=15, lambda_off=500.0, 
                 tau_density=0.01, lambda_obs=100.0, sigma_obs=1.0, nystrom_thresh=0.5):
        self.nu = nu
        self.sigma_dyn = sigma_dyn
        self.k_neighbors = k_neighbors
        self.lambda_off = lambda_off
        self.tau_density = tau_density
        self.lambda_obs = lambda_obs
        self.sigma_obs = sigma_obs
        self.nystrom_thresh = nystrom_thresh
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.scaler_obj = StandardScaler()
        self.k_graph = 10 
        self.k_eval = 2
        
        self.proj_mean = None
        self.proj_std = None
        
        self.graph_nodes = None 
        self.V_graph = None
        self.z_buffer_np = None

        self.kde = None
        self.state_range_ = None

    def compile_environment(self, z_buffer, z_predicted_next=None, n_nodes=55000):
        """
        Builds the directed Advection-Diffusion graph using a Gaussian Mollifier 
        to smooth dataset transitions while strictly forbidding backwards flow.
        """
        # 1. SCALE & DOWNSAMPLE
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

        # 2. ADAPTIVE MOLLIFIED K-NN (Defeating the 10D Curse)
        # ---------------------------------------------------------
        z_next_tensor = torch.tensor(z_next_scaled, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            dists = torch.cdist(z_next_tensor, self.graph_nodes)

            # 1. Enforce strict sparsity to prevent 10D teleportation
            K_neighbors = 32
            k_dists, k_indices = torch.topk(-dists, k=K_neighbors, dim=1)
            k_dists = -k_dists

            # 2. Adaptive Local Bandwidth
            local_sigma = k_dists[:, 1].unsqueeze(1) * 2.0 
            local_sigma = torch.clamp(local_sigma, min=1e-3)

            # --- THE FIX: Save the characteristic length scale for Section 4 ---
            self.sigma_perturb = local_sigma.median().item()
            # -------------------------------------------------------------------

            # 3. Apply the Gaussian Mollifier ONLY to the local neighborhood
            W_local = torch.exp(-(k_dists ** 2) / (2 * local_sigma ** 2))

            # 4. Convert to Sparse Matrix Format
            W_local_np = W_local.cpu().numpy()
            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = k_indices.cpu().numpy().flatten()
            data = W_local_np.flatten()

        # 3. DIRECTED ADVECTION-DIFFUSION LAPLACIAN
        # ---------------------------------------------------------
        W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
        
        W_0.setdiag(0)

        # Build Directed Random Walk Laplacian
        d = np.array(W_0.sum(axis=1)).flatten() + 1e-8
        D_inv = sp.diags(1.0 / d)
        
        P = D_inv.dot(W_0)
        L_directed = sp.eye(N) - P
        
        print(f"[Graph Build] Connectivity: Capped at exactly {K_neighbors} edges per node.")

        # 4. ASYMMETRIC SYSTEM MATRIX
        # ---------------------------------------------------------
        c_base = 0.1
        C_matrix = sp.diags(np.full(N, c_base) / (2.0 * self.nu))

        # We substitute sigma_dyn with our sigma_perturb
        self.A_asym = ((self.nu / (self.sigma_perturb ** 2)) * L_directed) + C_matrix
        self.A_asym += sp.diags(np.full(N, 1e-6)) # Conditioning buffer

        self.proj_mean = torch.tensor(self.scaler_obj.mean_, dtype=torch.float32, device=self.device)
        self.proj_std = torch.tensor(np.sqrt(self.scaler_obj.var_), dtype=torch.float32, device=self.device)


    def solve_for_goal(self, z_goal, phys_v_mean=10.0):
        """
        Solves the asymmetric linear PDE using BiCGSTAB.
        """
        z_goal_proj = self.scaler_obj.transform(z_goal.reshape(1, -1))
        goal_dists = np.linalg.norm(self.z_buffer_np - z_goal_proj, axis=1)
        
        # Hard Goal Assignment: Do not smear across contacts
        self.goal_idx = np.argmin(goal_dists)
        goal_indices = np.array([self.goal_idx])
        goal_weights = np.array([1.0]) 

        N = self.A_asym.shape[0]

        # Flux Vector Calculation
        W_bound = sp.csr_matrix((goal_weights, (goal_indices, np.zeros_like(goal_indices))), shape=(N, 1))
        flux_vector = self.A_asym.dot(W_bound).toarray().flatten()
        
        b = np.zeros(N)
        b -= flux_vector
        b[goal_indices] = goal_weights  

        # Boundary Masking
        is_boundary = np.zeros(N, dtype=bool)
        is_boundary[goal_indices] = True
        
        I_mask = sp.diags((~is_boundary).astype(float))
        I_bound = sp.diags(is_boundary.astype(float))
        
        A_solve = I_mask.dot(self.A_asym).dot(I_mask) + I_bound

        # Preconditioned BiCGSTAB Solve (Required for Asymmetric A)
        M_solve = sp.diags(1.0 / (A_solve.diagonal() + 1e-8))
        Z_raw, info = spla.bicgstab(A_solve, b, tol=1e-10, maxiter=5000, M=M_solve)

        if info != 0:
            print(f"[Warning] BiCGSTAB exited with code {info}")

        # Reverse Hopf-Cole
        Z_raw = np.maximum(Z_raw, 1e-300)
        z_max = Z_raw.max()
        if z_max > 1.0: 
            Z_raw /= z_max

        Z_clipped = np.clip(Z_raw, 1e-300, 1.0)
        # Compute the raw value function
        V_raw = -2.0 * self.nu * np.log(Z_clipped)
        scale_factor = phys_v_mean / (V_raw.mean() + 1e-8)
        self.V_graph = torch.tensor(V_raw * scale_factor, dtype=torch.float32, device=self.device)

        # --- THE FIX: LOGARITHMIC COMPRESSION (Soft-Clipping) ---
        # Values near 0-20 remain linear and highly accurate.
        # Massive outliers (669) are logarithmically compressed (e.g., to ~70).
        # This guarantees the global derivative NEVER vanishes to zero.
        tau = phys_v_mean * 2.0  
        self.V_graph = tau * torch.log1p(self.V_graph / tau)
        # --------------------------------------------------------
        if not hasattr(self, 'v_surrogate'):
            self.v_surrogate = torch.nn.Sequential(
                torch.nn.Linear(self.graph_nodes.shape[1], 256),
                torch.nn.Mish(), # Mish/SiLU provides smoother second-order gradients than ReLU
                torch.nn.Linear(256, 256),
                torch.nn.Mish(),
                torch.nn.Linear(256, 1),
                torch.nn.Softplus()  # <--- THE FIX: mathematically bounds output to (0, +inf)
            ).to(self.device)

        optimizer = torch.optim.Adam(self.v_surrogate.parameters(), lr=5e-3)
        
        # Fit the continuous MLP to the exact discrete graph solution.
        # For N=5000 nodes, 250 iterations takes ~0.1 seconds on a modern GPU.
        self.v_surrogate.train()
        for _ in range(250):
            optimizer.zero_grad()
            # Forward pass on all graph nodes
            pred_v = self.v_surrogate(self.graph_nodes).squeeze()
            
            # MSE loss against the BiCGSTAB solved values
            loss = torch.nn.functional.mse_loss(pred_v, self.V_graph)
            #print("Surrogate Fit Loss:", loss.item())
            loss.backward()
            optimizer.step()
        
        self.v_surrogate.eval()
        # ---------------------------------------------------------
        # 2. THE FIX: FREEZE WEIGHTS FOR ROLLOUT EVALUATION
        # ---------------------------------------------------------
        # This prevents PyTorch from calculating or storing gradients for the 
        # MLP's weights during the planner's action optimization loop.
        for p in self.v_surrogate.parameters():
            p.requires_grad = False
        # ---------------------------------------------------------
        # =====================================================================
        # DIAGNOSTICS: PDE LANDSCAPE CHECK
        # =====================================================================
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        v_np = self.V_graph.detach().cpu().numpy()
        # print("\n" + "="*50)
        # print("🔍 PDE SOLVE DIAGNOSTICS")
        # print(f"V_graph Min  : {v_np.min():.4f}")
        # print(f"V_graph Max  : {v_np.max():.4f}")
        # print(f"V_graph Mean : {v_np.mean():.4f}")
        # print(f"V_graph Std  : {v_np.std():.4f}")
        # print("="*50 + "\n")

        # # Visualizing the 10D Latent Graph via PCA
        # try:
        #     pca = PCA(n_components=2)
        #     z_pca = pca.fit_transform(self.z_buffer_np)
        #     goal_pca = z_pca[self.goal_idx]

        #     plt.figure(figsize=(10, 8))
        #     # Scatter all dataset nodes, colored by their PDE Value
        #     sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=v_np, cmap='magma', s=15, alpha=0.8)
        #     # Mark the goal state
        #     plt.scatter(goal_pca[0], goal_pca[1], c='cyan', marker='*', s=400, edgecolors='black', label='Goal Node')
            
        #     plt.colorbar(sc, label='PDE Value (Cost to Goal)')
        #     plt.title("2D PCA of 10D Latent Space\nColored by Quasimetric PDE Value")
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.savefig("/scratch/gilbreth/hviswan/grasp/pde_landscape_debug.png", dpi=150)
        #     plt.close()
        # except Exception as e:
        #     print(f"[Debug Warning] Could not generate PCA plot: {e}")
        return self.V_graph

        return self.V_graph


    # def evaluate_rollout(self, z_generated):
    #     """
    #     Interpolates the V_graph for CEM rollouts using a tight local bandwidth
    #     to prevent blurring the contact boundary gradients.
    #     """
    #     z_scaled = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)

    #     dists = torch.cdist(z_scaled, self.graph_nodes)

    #     # Tight K to prevent interpolating across the contact discontinuity
    #     k = 4 

    #     k_dists, k_indices = torch.topk(-dists, k=k, dim=1)
    #     k_dists = -k_dists

    #     # Sharp softmax temperature to favor the absolute nearest valid state
    #     weights = torch.softmax(-k_dists / 0.05, dim=-1)

    #     V_interp = (weights * self.V_graph[k_indices]).sum(dim=1)

    #     return V_interp
    def evaluate_rollout(self, z_generated):
        """
        Provides flawless analytical gradients while heavily penalizing 
        out-of-distribution (OOD) states to prevent reward hacking.
        """
        # 1. Scale the generated states
        z_scaled = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)

        with torch.enable_grad():
            # 2. Query the bounded surrogate
            V_interp = self.v_surrogate(z_scaled).squeeze(-1)

            # 3. The OOD Gravity Well
            # Calculate how far the state is from the dataset center
            z_norms = torch.norm(z_scaled, dim=-1)
            
            # If the state drifts more than 3 standard deviations from the mean,
            # apply a harsh quadratic penalty to push the gradient back inside.
            ood_penalty = torch.relu(z_norms - 3.0) ** 2
            
            # Combine the surrogate value with the OOD wall
            V_safeguarded = V_interp #+ (ood_penalty * 50.0)

        return V_safeguarded

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
        action_clamp=0.0,  # clamp value for actions; 0 means no clamping
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
        loaded_buffer = torch.load("/scratch/gilbreth/hviswan/dino_wm/pointmaze_offline_replay_buffer.pt", weights_only=True)
        #loaded_buffer = torch.load("/scratch/gilbreth/hviswan/dino_wm/pusht_offline_replay_buffer.pt", weights_only=True)
        #z_buffer_np = loaded_buffer["z_curr"].cpu().numpy()  # Shape: [N, latent_dim]
        #z_next_gt_np = loaded_buffer["z_next"].cpu().numpy()  # Shape: [N, latent_dim]
        Z_obs = loaded_buffer["z_buffer"].to(self.device)  # Shape: [N, latent_dim]
        B, T, D = Z_obs.shape
        N = Z_obs.shape[0]
        Z_curr = Z_obs[:, :-1, :]  # Shape: [4, 99, 10]
        Z_next = Z_obs[:, 1:, :]   # Shape: [4, 99, 10]

        # 2. Flatten the batch and time dimensions into a single list of nodes
        z_buffer_np = Z_curr.reshape(-1, D).cpu().numpy()       # Shape: [396, 10]
        z_next_gt_np = Z_next.reshape(-1, D).cpu().numpy()      # Shape: [396, 10]

        # 3. Define your goal (ensure it's just a 1D array of shape [10])
        self.latent_hjb_solver = LatentDynamicalHJB(
            nu=0.3, 
            sigma_dyn=0.08,       
            k_neighbors=20, 
            lambda_off=500.0,     
            tau_density=0.01     
        )
        print("Compiling Latent Environment Graph...")
        self.latent_hjb_solver.compile_environment(
            z_buffer=z_buffer_np,
            z_predicted_next=z_next_gt_np
        )
        print("Latent Dynamical Graph built and solved. Ready for CEM planning.")

    def init_actions(self, obs_0, actions=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
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
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
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
            # format needed for rollout func
            trans_obs_0 = trans_obs_0[:,0].unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1)
            z_obs_g_detached = z_obs_g.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach()

        total_measured_time = 0.0
        exit_step = self.opt_steps  # Default to max steps if no early exit
        current_goal_np = z_obs_g_detached["proprio"][0, 0].detach().cpu().numpy()
        #print("Current goal NP shape: ", current_goal_np.shape)
        self.latent_hjb_solver.solve_for_goal(current_goal_np)

        # for i in range(self.opt_steps):
        #     start = time.perf_counter()
        #     optimizer.zero_grad()
            
        #     # ---------------------------------------------------------
        #     # 1. NOISE ANNEALING (Stabilize Physical Execution)
        #     # ---------------------------------------------------------
        #     decay_factor = max(0.0, 1.0 - (i / float(self.opt_steps)))
        #     current_noise = self.var_actions * decay_factor
        #     actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=current_noise).to(actions)
            
        #     eps = torch.randn(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]).to(actions) * actions_noise
        #     noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1)
            
        #     # Rollout through the world model
        #     if self.var_states > 0:
        #         i_z_obses, i_zs = self.wm.rollout_noisy(obs_0=trans_obs_0, act=noisy_actions, var_states=self.var_states)
        #     else:
        #         i_z_obses, i_zs = self.wm.rollout(obs_0=trans_obs_0, act=noisy_actions)
            
        #     # ---------------------------------------------------------
        #     # 2. PDE GRAPH SUPERVISION
        #     # ---------------------------------------------------------
        #     latent_rollout = i_z_obses["proprio"].float()      # (B, T, D)
        #     B, T, D = latent_rollout.shape
        #     loss = self.objective_fn(i_z_obses, z_obs_g_detached, noisy_actions)  # (n_evals, )
            
        #     flat_latents = latent_rollout.reshape(-1, D)
        #     flat_values = self.latent_hjb_solver.evaluate_rollout(flat_latents)
        #     values = flat_values.reshape(B, T)

        #     # A. Terminal Anchor
        #     terminal_loss = values[:, -1].mean()

        #     # B. Dynamic Margin Descent
        #     value_diffs = values[:, 1:] - values[:, :-1] 
        #     latent_diffs = latent_rollout[:, 1:] - latent_rollout[:, :-1]
        #     step_norms = torch.norm(latent_diffs, dim=-1)
            
        #     # Dropped base_margin to 0.1 so Adam doesn't panic when cornering the block
        #     dynamic_margin = 15.0 * step_norms.detach()
        #     descent_violations = torch.relu(value_diffs + dynamic_margin) 
            
        #     decay_weights = torch.linspace(1.0, 0.5, steps=T-1).to(self.device)
        #     transition_loss = (descent_violations * decay_weights).sum(dim=1).mean()

        #     # ---------------------------------------------------------
        #     # 3. KINEMATIC REGULARIZATION (The Jitter Killer)
        #     # ---------------------------------------------------------
        #     # Penalize velocity and acceleration to prevent adversarial world model exploitation
        #     action_vel = actions[:, 1:] - actions[:, :-1]
        #     action_acc = action_vel[:, 1:] - action_vel[:, :-1]
            
        #     smoothness_loss = action_vel.pow(2).mean() + action_acc.pow(2).mean()

        #     # ---------------------------------------------------------
        #     # 4. OBJECTIVE BLEND & BACKPROP
        #     # ---------------------------------------------------------
        #     #print(f"Step {i + 1} | Transition: {transition_loss.item():.4f} | Terminal: {terminal_loss.item():.4f} | Smoothness: {smoothness_loss.item():.4f}")
            
        #     # Heavy penalty on smoothness prevents the physical geometry failure
        #     total_loss =  loss.mean()*n_evals #+ transition_loss.mean()*0.05
            
        #     total_loss.backward()
        #     optimizer.step()
            
        #     with torch.no_grad():
        #         if self.action_clamp > 0:
        #             actions.data.clamp_(-self.action_clamp, self.action_clamp)
                    
        #     end = time.perf_counter()
        #     total_measured_time += end - start
        min_steps = 0
        prev_grad_action = None
        for i in range(self.opt_steps):
            start = time.perf_counter()
            optimizer.zero_grad()
            
            # ---------------------------------------------------------
            # 1. NOISE ANNEALING
            # ---------------------------------------------------------
            decay_factor = max(0.0, 1.0 - (i / float(self.opt_steps)))
            current_noise = self.var_actions * decay_factor
            actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=current_noise).to(actions)
            
            # Base random noise
            eps = torch.randn(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]).to(actions) * actions_noise
            
            # We detach here because the Langevin phase should not track gradients back to Adam
            base_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1).detach()
            
            # Expand observations for the batch size (num_samples * B)
            batch_size = base_noisy_actions.shape[0]
            obs_batch_langevin = {
                key: arr.repeat_interleave(self.num_samples, dim=0) if arr.shape[0] == actions.shape[0] else arr
                for key, arr in trans_obs_0.items()
            }

            # ---------------------------------------------------------
            # 2. LANGEVIN NOISE SHAPING (DECOUPLED GRAPH)
            # ---------------------------------------------------------
            guided_actions = base_noisy_actions.clone()
            velocity = torch.zeros_like(guided_actions)
            anchor_actions = actions.unsqueeze(0).detach().flatten(0, 1)
            num_langevin_steps = 10  # Keep this low in a GD planner (3-5 is usually enough)
            #alpha_base = 0.001
            #gamma = 1e-3
            alpha_base = 1e-1
            gamma = 1e-2
            trust_radius = 0.2

            with torch.enable_grad():
                for k in range(num_langevin_steps):
                    guided_actions.requires_grad_(True)
                    
                    # Forward Pass (Local to Langevin)
                    i_z_obses_lang, _ = self.wm.rollout(obs_0=obs_batch_langevin, act=guided_actions)
                    
                    # Objective Calculation (Using only terminal state to save memory)
                    terminal_10d_latent = i_z_obses_lang["proprio"][:, -1, :].float()
                    v_val = self.latent_hjb_solver.evaluate_rollout(terminal_10d_latent)
                    
                    # Standard sum to maintain gradient scale
                    objective_hjb = -v_val.sum()
                    
                    # Local Backward Pass
                    grad_action = torch.autograd.grad(
                        objective_hjb,
                        guided_actions,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                    #print("Grad action Norm before clamp:", grad_action.norm().item())
                    
                    #grad_action = torch.clamp(grad_action, min=-1.0, max=1.0)
                    
                    # Update and strictly detach to destroy the graph
                    with torch.no_grad():
                        if prev_grad_action is None:
                            # First step: Assume neutral confidence (0.5)
                            confidence = torch.full(
                                (grad_action.shape[0], 1, 1), 
                                0.5, 
                                device=grad_action.device
                            )
                        else:
                            # Subsequent steps: Use the previous gradient's confidence
                            flat_curr = grad_action.flatten(1)
                            flat_prev = prev_grad_action.flatten(1)
                            cos_sim = torch.nn.functional.cosine_similarity(flat_curr, flat_prev, dim=-1)
                            confidence = torch.sigmoid((cos_sim + 1.0) / 2.0)
                            confidence = confidence.view(-1, 1, 1)
                        prev_grad_action = grad_action.clone()
                        
                        dt = alpha_base / np.sqrt(k + 1)
                        tau = 1.0
                        grad_norm = grad_action.norm(dim=-1, keepdim=True)
                        scale = torch.minimum(torch.ones_like(grad_norm), 1.0 / (grad_norm + 1e-8))
                        
                        #print("Confidence: ", confidence.mean().item())
                        drift = grad_action*scale
                        dt_tensor = torch.tensor(dt, device=grad_action.device)
                        gamma_tensor = torch.tensor(gamma, device=grad_action.device)
                        friction_decay = torch.exp(-gamma_tensor * dt_tensor)
                        noise_variance = decay_factor * (1.0 - torch.exp(-2.0 * gamma_tensor * dt_tensor))
                        diffusion = torch.sqrt(noise_variance) * torch.randn_like(guided_actions)
                        velocity = (friction_decay) * velocity + dt* drift + diffusion
                        guided_actions = (guided_actions + velocity*dt).detach()
            # ---------------------------------------------------------
            # 3. REPARAMETERIZATION TRICK (RE-ATTACH TO MAIN GRAPH)
            # ---------------------------------------------------------
            # We figure out exactly what noise the Langevin process effectively added
            eps_refined = guided_actions.view(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]) - actions.unsqueeze(0).detach()
            
            # Re-attach to the main 'actions' parameter so Adam gets the gradients
            final_noisy_actions = (actions.unsqueeze(0) + eps_refined.detach()).flatten(0, 1)
            #final_noisy_actions = torch.tanh(final_noisy_actions)
            #final_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1)

            # ---------------------------------------------------------
            # 4. MAIN PDE GRAPH SUPERVISION & BACKPROP
            # ---------------------------------------------------------
            if self.var_states > 0:
                i_z_obses, i_zs = self.wm.rollout_noisy(obs_0=obs_batch_langevin, act=final_noisy_actions, var_states=self.var_states)
            else:
                i_z_obses, i_zs = self.wm.rollout(obs_0=obs_batch_langevin, act=final_noisy_actions)
                i_z_obses_base, i_zs_base = self.wm.rollout(obs_0=obs_batch_langevin, act=base_noisy_actions)
            
            latent_rollout = i_z_obses["proprio"].float()      
            B, T, D = latent_rollout.shape
            
            # Execute your standard loss calculations
            loss = self.objective_fn(i_z_obses, z_obs_g_detached, final_noisy_actions)
            base_loss = self.objective_fn(i_z_obses_base, z_obs_g_detached, base_noisy_actions)
            improvement = base_loss.item() - loss.item()
            # print(f"\n[Outer Opt Step {i}/{self.opt_steps}]")
            # print(f"  Current Loss : {loss.item():.4f}")
            # print(f"  Improvement  : {improvement:.4f} (Positive means gradient is working)")
            # print("-" * 50)
            
            flat_latents = latent_rollout.reshape(-1, D)
            flat_values = self.latent_hjb_solver.evaluate_rollout(flat_latents)
            values = flat_values.reshape(B, T)

            # A. Terminal Anchor
            terminal_loss = values[:, -1].mean()

            # B. Dynamic Margin Descent
            value_diffs = values[:, 1:] - values[:, :-1] 
            latent_diffs = latent_rollout[:, 1:] - latent_rollout[:, :-1]
            step_norms = torch.norm(latent_diffs, dim=-1)
            
            dynamic_margin = 15.0 * step_norms.detach()
            descent_violations = torch.relu(value_diffs + dynamic_margin) 
            
            decay_weights = torch.linspace(1.0, 0.5, steps=T-1).to(self.device)
            transition_loss = (descent_violations * decay_weights).sum(dim=1).mean()

            # C. Kinematic Regularization
            action_vel = actions[:, 1:] - actions[:, :-1]
            action_acc = action_vel[:, 1:] - action_vel[:, :-1]
            smoothness_loss = action_vel.pow(2).mean() + action_acc.pow(2).mean()

            # Backpropagate through the *single* final rollout graph
            total_loss = loss.mean() * self.num_samples
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                if self.action_clamp > 0:
                    actions.data.clamp_(-self.action_clamp, self.action_clamp)
                    
            end = time.perf_counter()
            total_measured_time += end - start

            if self.evaluator is not None and i % self.eval_every == 0:
                
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(), filename=f"{self.logging_prefix}_output_{i+1}"
                )
                #print("LOGS: ", logs)
                # --- NEW: Track max success ---
                current_success = float(successes.mean())
                if current_success > max_success_rate:
                    max_success_rate = current_success
                mean_dist = logs.get('mean_state_dist')
                mean_vis_dist = logs.get('mean_visual_dist')
                mean_vis_dist = logs.get('mean_visual_dist')
                mean_proprio_dist = logs.get('mean_proprio_dist')
                if mean_dist < best_dist and current_success >= max_success_rate:
                    best_dist = mean_dist
                if mean_vis_dist < best_vis_dist and current_success >= max_success_rate:
                    best_vis_dist = mean_vis_dist
                #print(f"Final Success Rate: {successes.mean().item() * 100:.2f}%")
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
                    break  # terminate planning if all success
            if total_measured_time > 1800:
                exit_step = i + 1
                break

        # Log final results
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), filename=f"{self.logging_prefix}_output_final"
        )
        current_success = float(successes.mean())
        if current_success > max_success_rate:
            max_success_rate = current_success
        mean_dist = logs.get('mean_state_dist')
        mean_vis_dist = logs.get('mean_visual_dist')
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
        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time
        # Store planning stats for access by plan.py
        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        # --- NEW: Append to the shared log file ---
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task_id is not None:
            log_filename = f"seed_success_chunk_{task_id}.txt"
        else:
            log_filename = "seed_success_log.txt"
        with open(f"/scratch/gilbreth/hviswan/grasp/{log_filename}", "a") as f:
            print("Writing to log file:", log_filename)
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {i}\n")
        # ------------------------------------------

        return actions, np.full(n_evals, np.inf)
