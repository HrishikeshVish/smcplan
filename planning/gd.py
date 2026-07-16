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

        # Prepare Data (accepts all D dimensions natively)
        points = torch.as_tensor(dataset_states, dtype=torch.float32, device=self.device)

        # Subsample if necessary for memory/speed
        if points.shape[0] > num_samples:
            idx = torch.randperm(points.shape[0], device=self.device)[:num_samples]
            points = points[idx]

        self.points = points
        self.n_points = points.shape[0]

        # Generalized N-Dimensional Bounding Box
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

        # Nearest Neighbor Index (No Gradients, Chunked for VRAM safety)
        nn_idx_list = []
        with torch.no_grad():
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                x_chunk = x[start:end]
                dists = torch.cdist(x_chunk, self.points)
                chunk_nn_idx = torch.argmin(dists, dim=-1)
                nn_idx_list.append(chunk_nn_idx)

        nn_idx = torch.cat(nn_idx_list, dim=0)

        # Re-attach to Autograd Graph
        nearest_points = self.points[nn_idx]
        min_dists = torch.norm(x - nearest_points, dim=-1)

        # Apply the plateau and sign (Negative distance = Log Prob)
        min_dists = torch.clamp(min_dists, max=self.max_depth)
        pseudo_log_prob = -1.0 * min_dists

        # Generalized Outer Boundary Cliff
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

        # Adaptive Mollified K-NN
        z_next_tensor = torch.tensor(z_next_scaled, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dists = torch.cdist(z_next_tensor, self.graph_nodes)

            K_neighbors = 32
            k_dists, k_indices = torch.topk(-dists, k=K_neighbors, dim=1)
            k_dists = -k_dists

            # Adaptive Local Bandwidth
            local_sigma = k_dists[:, 1].unsqueeze(1) * 2.0
            local_sigma = torch.clamp(local_sigma, min=1e-3)

            # Save the characteristic length scale
            self.sigma_perturb = local_sigma.median().item()

            # Apply the Gaussian Mollifier ONLY to the local neighborhood
            W_local = torch.exp(-(k_dists ** 2) / (2 * local_sigma ** 2))

            # Convert to Sparse Matrix Format
            W_local_np = W_local.cpu().numpy()
            row_idx = np.repeat(np.arange(N), K_neighbors)
            col_idx = k_indices.cpu().numpy().flatten()
            data = W_local_np.flatten()

        # Directed Advection-Diffusion Laplacian
        W_0 = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, N))
        W_0.setdiag(0)

        d = np.array(W_0.sum(axis=1)).flatten() + 1e-8
        D_inv = sp.diags(1.0 / d)

        P = D_inv.dot(W_0)
        L_directed = sp.eye(N) - P

        print(f"[Graph Build] Connectivity: Capped at exactly {K_neighbors} edges per node.")

        # Asymmetric System Matrix
        c_base = 0.1
        C_matrix = sp.diags(np.full(N, c_base) / (2.0 * self.nu))

        self.A_asym = ((self.nu / (self.sigma_perturb ** 2)) * L_directed) + C_matrix
        self.A_asym += sp.diags(np.full(N, 1e-6))

        self.proj_mean = torch.tensor(self.scaler_obj.mean_, dtype=torch.float32, device=self.device)
        self.proj_std = torch.tensor(np.sqrt(self.scaler_obj.var_), dtype=torch.float32, device=self.device)


    def solve_for_goal(self, z_goal, phys_v_mean=10.0):
        """
        Solves the asymmetric linear PDE using BiCGSTAB.
        """
        z_goal_proj = self.scaler_obj.transform(z_goal.reshape(1, -1))
        goal_dists = np.linalg.norm(self.z_buffer_np - z_goal_proj, axis=1)

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
        V_raw = -2.0 * self.nu * np.log(Z_clipped)
        scale_factor = phys_v_mean / (V_raw.mean() + 1e-8)
        self.V_graph = torch.tensor(V_raw * scale_factor, dtype=torch.float32, device=self.device)

        # Logarithmic Compression (Soft-Clipping)
        tau = phys_v_mean * 2.0
        self.V_graph = tau * torch.log1p(self.V_graph / tau)

        if not hasattr(self, 'v_surrogate'):
            self.v_surrogate = torch.nn.Sequential(
                torch.nn.Linear(self.graph_nodes.shape[1], 256),
                torch.nn.Mish(),
                torch.nn.Linear(256, 256),
                torch.nn.Mish(),
                torch.nn.Linear(256, 1),
                torch.nn.Softplus()
            ).to(self.device)

        optimizer = torch.optim.Adam(self.v_surrogate.parameters(), lr=5e-3)

        self.v_surrogate.train()
        for _ in range(250):
            optimizer.zero_grad()
            pred_v = self.v_surrogate(self.graph_nodes).squeeze()
            loss = torch.nn.functional.mse_loss(pred_v, self.V_graph)
            loss.backward()
            optimizer.step()

        self.v_surrogate.eval()
        for p in self.v_surrogate.parameters():
            p.requires_grad = False

        return self.V_graph

    def evaluate_rollout(self, z_generated):
        """
        Provides analytical gradients while penalizing out-of-distribution states.
        """
        z_scaled = (z_generated - self.proj_mean) / (self.proj_std + 1e-8)

        with torch.enable_grad():
            V_interp = self.v_surrogate(z_scaled).squeeze(-1)

        return V_interp


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
        replay_buffer_path=None,
        latent_key="proprio",
        hjb_nu=0.3,
        hjb_sigma_dyn=0.08,
        hjb_k_neighbors=20,
        hjb_lambda_off=500.0,
        hjb_tau_density=0.01,
        hjb_tau=0.1,
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
        self.latent_key = latent_key
        self.hjb_tau = hjb_tau

        # Load replay buffer and build HJB graph
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

        self.latent_hjb_solver = LatentDynamicalHJB(
            nu=hjb_nu,
            sigma_dyn=hjb_sigma_dyn,
            k_neighbors=hjb_k_neighbors,
            lambda_off=hjb_lambda_off,
            tau_density=hjb_tau_density,
        )
        print("Compiling Latent Environment Graph...")
        self.latent_hjb_solver.compile_environment(
            z_buffer=z_buffer_np,
            z_predicted_next=z_next_gt_np,
        )
        print("Latent Dynamical Graph built. Ready for planning.")

    def _get_latent(self, z_obses):
        """Extract the latent trajectory from the world model output.

        Uses self.latent_key to select between 'proprio' and 'visual'.
        For visual latents with a patch dimension, pools over patches.
        """
        latent = z_obses[self.latent_key].float()
        # Pool patch dimension if present (e.g. visual: B, T, num_patches, D)
        if latent.dim() == 4:
            latent = latent.mean(dim=2)  # (B, T, D)
        # Apply PCA projection if buffer was built with --pca-dim
        if self.pca_info is not None:
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
            trans_obs_0 = trans_obs_0[:,0].unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1)
            z_obs_g_detached = z_obs_g.unsqueeze(0).repeat_interleave(self.num_samples, dim=0).flatten(0,1).detach()

        total_measured_time = 0.0
        exit_step = self.opt_steps

        # Solve HJB for the goal using the configured latent key
        goal_latent = self._get_latent(z_obs_g_detached)
        current_goal_np = goal_latent[0, 0].detach().cpu().numpy()
        self.latent_hjb_solver.solve_for_goal(current_goal_np)

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

            base_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1).detach()

            # Expand observations for the batch size (num_samples * B)
            obs_batch = {
                key: arr.repeat_interleave(self.num_samples, dim=0) if arr.shape[0] == actions.shape[0] else arr
                for key, arr in trans_obs_0.items()
            } if isinstance(trans_obs_0, dict) else trans_obs_0

            final_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1)

            # World model rollout
            if self.var_states > 0:
                i_z_obses, i_zs = self.wm.rollout_noisy(obs_0=obs_batch, act=final_noisy_actions, var_states=self.var_states)
                i_z_obses_base, i_zs_base = self.wm.rollout_noisy(obs_0=obs_batch, act=base_noisy_actions, var_states=self.var_states)
            else:
                i_z_obses, i_zs = self.wm.rollout(obs_0=obs_batch, act=final_noisy_actions)
                i_z_obses_base, i_zs_base = self.wm.rollout(obs_0=obs_batch, act=base_noisy_actions)

            # Extract latent trajectory using configured key
            latent_rollout = self._get_latent(i_z_obses)
            B, T, D = latent_rollout.shape

            # Objective loss
            loss = self.objective_fn(i_z_obses, z_obs_g_detached, final_noisy_actions)

            # HJB cost-to-go along full trajectory
            tau = self.hjb_tau

            # Evaluate V at all timesteps (gradients flow through world model)
            flat_latents = latent_rollout.reshape(-1, D)
            V_all = self.latent_hjb_solver.evaluate_rollout(flat_latents)
            V_traj = V_all.reshape(B, T)

            # Weighted: later timesteps matter more (terminal emphasis)
            weights = torch.linspace(0.2, 1.0, steps=T, device=self.device)
            v_loss = (V_traj * weights).mean()

            loss = loss + tau * v_loss

            print(f"Step {i + 1} | MSE Loss: {loss.item():.4f} | V_mean: {V_traj.mean().item():.4f} | V_terminal: {V_traj[:,-1].mean().item():.4f}")

            # Backpropagate
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
