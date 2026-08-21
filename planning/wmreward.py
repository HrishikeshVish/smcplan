"""WMReward-style inference-time alignment for latent world-model planning.

Follows "Inference-time Physics Alignment of Video Generative Models with
Latent World Models" (Yuan et al., facebookresearch/WMReward): rather than
descending a single trajectory, SEARCH over sampled candidates and STEER them
using a reward derived from the world model's own priors, scaling test-time
compute instead of changing any weights.

Why this shape fits here. Gradient descent on a learned world model actively
seeks the regions where that model is wrong: on reacher seed 9 the planner
drove its imagined distance-to-goal down 83% while the divergence between the
imagined terminal latent and the ENCODED ACTUAL executed state stayed flat at
~8.9 -- eight times further from reality than from the goal. Sampling
candidates and ranking them cannot exploit the model in that way, and the
surprise term below explicitly penalises the extrapolation that enables it.

The reward has three parts, all evaluated on the imagined rollout:

    R = w_goal * ||z_T - z_g||^2                  terminal agreement
      + w_phi  * phi(z_T)                         geodesic cost-to-go (heat method)
      + w_surprise * mean_t d(z_t, manifold)      WM surprise / extrapolation

phi and the surprise share one nearest-node query against the graph: phi is the
value at the nearest node, surprise is the distance to it. Because search only
needs to SCORE candidates rather than differentiate them, phi is read exactly
off the graph instead of through a surrogate MLP -- the approximation that made
the gradient path unreliable disappears entirely.

Optional steering refines the surviving elites with a few gradient steps on the
differentiable part of the reward (the terminal term only).
"""

import os
import time

import numpy as np
import torch
from einops import repeat

from .base_planner import BasePlanner
from .gd import MCMollifiedHeat, _load_replay_buffer


class WMRewardPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        num_samples,
        topk,
        opt_steps,
        eval_every,
        var_scale,
        replay_buffer_path,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        latent_key="visual",
        w_goal=1.0,
        w_phi=0.0,
        w_surprise=0.0,
        steer_steps=0,
        steer_lr=0.05,
        elite_momentum=0.5,
        graph_n_nodes=55000,
        graph_k_neighbors=32,
        solver_mode="heat",
        rw_steps=20,
        use_isomap=False,
        action_clamp=0.0,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator,
                         wandb_run, log_filename)
        self.horizon = horizon
        self.num_samples = num_samples
        self.topk = topk
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.var_scale = var_scale
        self.latent_key = latent_key
        self.w_goal, self.w_phi, self.w_surprise = w_goal, w_phi, w_surprise
        self.steer_steps, self.steer_lr = steer_steps, steer_lr
        self.elite_momentum = elite_momentum
        self.action_clamp = action_clamp
        self.logging_prefix = logging_prefix
        self.evaluator = evaluator

        self.heat_solver = None
        if self.w_phi != 0.0 or self.w_surprise != 0.0:
            print(f"Loading replay buffer from {replay_buffer_path}...")
            z_curr, z_next, pca_info = _load_replay_buffer(replay_buffer_path, self.device)
            if pca_info is not None:
                print("  NOTE: buffer carries a PCA basis; scoring uses the stored "
                      "coordinates directly.")
            print(f"  {z_curr.shape[0]} transitions, latent_dim={z_curr.shape[1]}")
            self.heat_solver = MCMollifiedHeat()
            self.heat_solver.compile_environment(
                z_buffer=z_curr, z_predicted_next=z_next, n_nodes=graph_n_nodes,
                K_neighbors=graph_k_neighbors, use_isomap=use_isomap,
                solver_mode=solver_mode, rw_steps=rw_steps,
            )
            self._nodes = self.heat_solver.graph_nodes          # scaled coords
            self._phi = None

    # ------------------------------------------------------------------
    def _get_latent(self, z_obses):
        latent = z_obses[self.latent_key] if isinstance(z_obses, dict) else z_obses
        if latent.dim() > 3:
            latent = latent.mean(dim=-2)
        return latent

    @torch.no_grad()
    def _lookup(self, z, chunk=4096):
        """One query gives both reward terms.

        Returns (phi_at_nearest_node, distance_to_that_node). The distance is
        the surprise proxy: a rollout latent far from every buffer point is one
        the world model is extrapolating on, so its prediction is unreliable
        regardless of how good the terminal objective looks.
        """
        zs = (z - self.heat_solver.proj_mean) / (self.heat_solver.proj_std + 1e-8)
        phis, dists = [], []
        for s in range(0, len(zs), chunk):
            d = torch.cdist(zs[s:s + chunk], self._nodes)
            dmin, idx = d.min(dim=1)
            dists.append(dmin)
            phis.append(self._phi[idx])
        return torch.cat(phis), torch.cat(dists)

    @torch.no_grad()
    def _reward(self, i_z_obses, z_obs_g, actions):
        """Lower is better (kept as a cost, to match objective_fn convention)."""
        goal = self.w_goal * self.objective_fn(i_z_obses, z_obs_g, actions)
        if self.heat_solver is None:
            return goal, {"goal": goal.mean().item(), "ctg": 0.0, "surprise": 0.0}

        lat = self._get_latent(i_z_obses)                        # (B, T, D)
        B, T, D = lat.shape
        ctg, dist = self._lookup(lat.reshape(B * T, D))
        ctg = self.w_phi * ctg.view(B, T)[:, -1]
        sur = self.w_surprise * dist.view(B, T).mean(dim=1)
        # report each WEIGHTED contribution, so the three can be balanced
        # against each other rather than guessed.
        return goal + ctg + sur, {"goal": goal.mean().item(),
                                  "ctg": ctg.mean().item(),
                                  "surprise": sur.mean().item()}

    def init_mu_sigma(self, obs_0, actions=None):
        sigma = self.var_scale * torch.ones([1, self.horizon, self.action_dim])
        mu = torch.zeros(1, self.horizon, self.action_dim) if actions is None else actions
        return mu, sigma

    # ------------------------------------------------------------------
    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)

        if self.heat_solver is not None:
            goal_lat = self._get_latent(z_obs_g)[0, 0].detach().cpu().numpy()
            out = self.heat_solver.solve_for_goal_heat_method(goal_lat)
            # Cost-to-go = -log u (Varadhan), NOT the heat method's phi.
            # Measured on this graph: spearman against graph distance is 0.93
            # for -log u versus 0.38 for phi, and phi's minimum is not even at
            # the goal. Crane et al. add steps II/III because -log u is
            # unreliable on an approximate kernel, but the step-I solve here is
            # accurate enough that the correction is a net loss. Kept in
            # natural units -- no rank normalisation, which would destroy the
            # scale the reward weights are calibrated against.
            V = -np.log(np.maximum(out["u"], 1e-300))
            V = V - V.min()
            self._phi = torch.tensor(V, dtype=torch.float32, device=self.device)

        current_success, exit_step = 0.0, self.opt_steps
        mean_dist = mean_vis_dist = mean_proprio_dist = np.inf
        total_measured_time = 0.0

        for i in range(self.opt_steps):
            start = time.perf_counter()

            # ---- SEARCH: sample candidates and rank them by the reward ----
            act = torch.randn(self.num_samples, self.horizon, self.action_dim,
                              device=self.device) * sigma + mu
            act[0] = mu[0]                                   # keep the incumbent
            if self.action_clamp > 0:
                act = act.clamp(-self.action_clamp, self.action_clamp)

            cur_obs = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                       for k, v in trans_obs_0.items()} if isinstance(obs_0, dict) else \
                      repeat(trans_obs_0[0].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
            cur_zg = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=self.num_samples)
                      for k, v in z_obs_g.items()}

            with torch.no_grad():
                i_z_obses, _ = self.wm.rollout(
                    obs_0=cur_obs if isinstance(obs_0, dict) else cur_obs[:, 0], act=act)
            cost, parts = self._reward(i_z_obses, cur_zg, act)

            elite_idx = torch.argsort(cost)[: self.topk]
            elites = act[elite_idx]

            # ---- STEER: refine elites on the differentiable part only ----
            if self.steer_steps > 0:
                e = elites.clone().requires_grad_(True)
                opt = torch.optim.Adam([e], lr=self.steer_lr)
                e_obs = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=len(elites))
                         for k, v in trans_obs_0.items()} if isinstance(obs_0, dict) else None
                e_zg = {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=len(elites))
                        for k, v in z_obs_g.items()}
                for _ in range(self.steer_steps):
                    opt.zero_grad()
                    z_e, _ = self.wm.rollout(obs_0=e_obs, act=e)
                    self.objective_fn(z_e, e_zg, e).mean().backward()
                    opt.step()
                elites = e.detach()

            # ---- refit the sampling distribution ----
            new_mu = elites.mean(dim=0, keepdim=True)
            new_sigma = elites.std(dim=0, keepdim=True) + 1e-6
            m = self.elite_momentum
            mu = m * mu + (1 - m) * new_mu
            sigma = m * sigma + (1 - m) * new_sigma

            total_measured_time += time.perf_counter() - start

            if i % self.eval_every == 0:
                print(f"[WMReward] step={i:4d} | cost={cost[elite_idx[0]].item():.4f} | goal={parts['goal']:.4f} "
                      f"| ctg={parts['ctg']:.4f} | surprise={parts['surprise']:.4f} "
                      f"| sigma={sigma.mean().item():.4f}")

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu.detach(), filename=f"{self.logging_prefix}_output_{i}")
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                current_success = logs.get(f"{self.logging_prefix}/success_rate", 0.0)
                mean_dist = logs.get(f"{self.logging_prefix}/mean_state_dist", np.inf)
                mean_vis_dist = logs.get(f"{self.logging_prefix}/mean_visual_dist", np.inf)
                mean_proprio_dist = logs.get(f"{self.logging_prefix}/mean_proprio_dist", np.inf)
                if current_success >= 1.0:
                    exit_step = i
                    break

        self.exit_step = exit_step
        self.total_measured_time = total_measured_time

        task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        fname = f"seed_success_chunk_{task_id}.txt" if task_id else "seed_success_log.txt"
        with open(os.path.join(os.environ.get("GRASP_RESULTS_DIR", os.getcwd()), fname), "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} {mean_proprio_dist} {exit_step}\n")

        return mu.detach(), np.full(mu.shape[0], np.inf)
