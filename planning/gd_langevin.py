"""
GD Planner with Langevin noise shaping over the action space.

Uses underdamped Langevin dynamics guided by the heat-equation value field
to shape the action noise before the main gradient descent update.
The Langevin phase runs a short inner loop that follows the value
function gradient in action space, then the refined noise is
re-attached to the main Adam graph via the reparameterization trick.
"""

import os

import torch
import time
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from .gd import MCMollifiedHeat, _load_replay_buffer


class GDLangevinPlanner(BasePlanner):
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
        hjb_tau=0.1,
        use_isomap=False,
        isomap_landmarks=500,
        graph_n_nodes=55000,
        # Langevin parameters
        num_langevin_steps=10,
        langevin_alpha_base=0.001,
        langevin_gamma=1e-3,
        langevin_trust_radius=0.2,
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
        self.use_isomap = use_isomap
        self.isomap_landmarks = isomap_landmarks
        self.graph_n_nodes = graph_n_nodes

        # Langevin parameters
        self.num_langevin_steps = num_langevin_steps
        self.langevin_alpha_base = langevin_alpha_base
        self.langevin_gamma = langevin_gamma
        self.langevin_trust_radius = langevin_trust_radius

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

        self.heat_solver = MCMollifiedHeat()
        print("Compiling Latent Environment Graph...")
        self.heat_solver.compile_environment(
            z_buffer=z_buffer_np,
            z_predicted_next=z_next_gt_np,
            n_nodes=self.graph_n_nodes,
            use_isomap=self.use_isomap,
            isomap_landmarks=self.isomap_landmarks,
        )
        print("Heat graph built. Ready for planning.")

    def _get_latent(self, z_obses):
        """Extract the latent trajectory using the configured key.

        For visual latents with a patch dimension, pools over patches.
        """
        latent = z_obses[self.latent_key].float()
        if latent.dim() == 4:
            latent = latent.mean(dim=2)
        if self.pca_info is not None:
            shape = latent.shape
            flat = latent.reshape(-1, shape[-1])
            flat = (flat - self.pca_info['mean']) @ self.pca_info['components'].T
            latent = flat.reshape(*shape[:-1], -1)
        return latent

    def init_actions(self, obs_0, actions=None):
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

    def _langevin_noise_shaping(self, base_noisy_actions, obs_batch, decay_factor):
        """Run underdamped Langevin dynamics to shape action noise.

        Follows the HJB cost-to-go gradient in action space to guide
        the noise towards lower-cost regions before the main GD update.

        Args:
            base_noisy_actions: (B*num_samples, T, action_dim) detached noisy actions
            obs_batch: observation dict expanded for num_samples
            decay_factor: noise annealing factor (1.0 -> 0.0 over optimization)

        Returns:
            guided_actions: (B*num_samples, T, action_dim) shaped actions (detached)
        """
        guided_actions = base_noisy_actions.clone()
        velocity = torch.zeros_like(guided_actions)
        prev_grad_action = None

        with torch.enable_grad():
            for k in range(self.num_langevin_steps):
                guided_actions.requires_grad_(True)

                # Forward pass through world model
                i_z_obses_lang, _ = self.wm.rollout(obs_0=obs_batch, act=guided_actions)

                # Compute HJB value at terminal state
                terminal_latent = self._get_latent(i_z_obses_lang)[:, -1, :]
                v_val = self.heat_solver.evaluate_rollout(terminal_latent)

                # We want to minimize v_val, so gradient of -v_val gives descent direction
                objective_hjb = -v_val.sum()

                grad_action = torch.autograd.grad(
                    objective_hjb,
                    guided_actions,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                with torch.no_grad():
                    # Gradient confidence via cosine similarity with previous step
                    if prev_grad_action is None:
                        confidence = torch.full(
                            (grad_action.shape[0], 1, 1),
                            0.5,
                            device=grad_action.device
                        )
                    else:
                        flat_curr = grad_action.flatten(1)
                        flat_prev = prev_grad_action.flatten(1)
                        cos_sim = torch.nn.functional.cosine_similarity(flat_curr, flat_prev, dim=-1)
                        confidence = torch.sigmoid((cos_sim + 1.0) / 2.0)
                        confidence = confidence.view(-1, 1, 1)
                    prev_grad_action = grad_action.clone()

                    # Underdamped Langevin update
                    dt = self.langevin_alpha_base / np.sqrt(k + 1)
                    grad_norm = grad_action.norm(dim=-1, keepdim=True)
                    scale = torch.minimum(torch.ones_like(grad_norm), 1.0 / (grad_norm + 1e-8))

                    drift = grad_action * scale
                    dt_tensor = torch.tensor(dt, device=grad_action.device)
                    gamma_tensor = torch.tensor(self.langevin_gamma, device=grad_action.device)
                    friction_decay = torch.exp(-gamma_tensor * dt_tensor)
                    noise_variance = decay_factor * (1.0 - torch.exp(-2.0 * gamma_tensor * dt_tensor))
                    diffusion = torch.sqrt(noise_variance) * torch.randn_like(guided_actions)

                    velocity = friction_decay * velocity + dt * drift + diffusion
                    guided_actions = (guided_actions + velocity * dt).detach()

        return guided_actions

    def plan(self, obs_0, obs_g, actions=None):
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

        # Solve heat equation for the goal
        goal_latent = self._get_latent(z_obs_g_detached)
        current_goal_np = goal_latent[0, 0].detach().cpu().numpy()
        self.heat_solver.solve_for_goal(current_goal_np)

        current_success = 0.0
        mean_dist = np.inf
        mean_vis_dist = np.inf
        mean_proprio_dist = np.inf
        prev_grad_action = None

        for i in range(self.opt_steps):
            start = time.perf_counter()
            optimizer.zero_grad()

            # Noise Annealing
            decay_factor = max(0.0, 1.0 - (i / float(self.opt_steps)))
            current_noise = self.var_actions * decay_factor
            actions_noise = torch.full((1, 1, 1, self.action_dim), fill_value=current_noise).to(actions)

            eps = torch.randn(self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]).to(actions) * actions_noise

            # Detach base noise for Langevin phase
            base_noisy_actions = (actions.unsqueeze(0) + eps).flatten(0, 1).detach()

            # Expand observations for num_samples * B
            obs_batch = {
                key: arr.repeat_interleave(self.num_samples, dim=0) if arr.shape[0] == actions.shape[0] else arr
                for key, arr in trans_obs_0.items()
            } if isinstance(trans_obs_0, dict) else trans_obs_0

            # Langevin noise shaping
            guided_actions = self._langevin_noise_shaping(
                base_noisy_actions, obs_batch, decay_factor
            )

            # Reparameterization trick: re-attach to main graph
            eps_refined = guided_actions.view(
                self.num_samples, actions.shape[0], actions.shape[1], actions.shape[2]
            ) - actions.unsqueeze(0).detach()
            final_noisy_actions = (actions.unsqueeze(0) + eps_refined.detach()).flatten(0, 1)

            # Main world model rollout
            if self.var_states > 0:
                i_z_obses, i_zs = self.wm.rollout_noisy(obs_0=obs_batch, act=final_noisy_actions, var_states=self.var_states)
            else:
                i_z_obses, i_zs = self.wm.rollout(obs_0=obs_batch, act=final_noisy_actions)

            latent_rollout = self._get_latent(i_z_obses)
            B, T, D = latent_rollout.shape

            # Objective loss + HJB terminal cost
            loss = self.objective_fn(i_z_obses, z_obs_g_detached, final_noisy_actions)
            terminal_latent = latent_rollout[:, -1, :]
            v_val = self.heat_solver.evaluate_rollout(terminal_latent)
            tau = self.hjb_tau
            v_val = (torch.logsumexp(v_val / tau, dim=0)) * tau
            loss = loss + tau * v_val.mean()

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
