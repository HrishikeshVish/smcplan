"""The PDE roadmap cost of the EM-SMC planner, as a wrapper around any
planner's trajectory objective.

Reviewer WT5H asks whether the proposed cost also improves GD, CEM and GRASP.
Every planner that scores candidates through `objective_fn(i_z, z_g, a)` -- GD
(gd.py:1280), CEM (cem.py:126) and MPPI (mppi.py:76) -- can be given the cost
by wrapping that one call, with no change to the planner itself. GRASP is not
covered: its `goal_loss` is identically zero (grasp.py:962) and its terminal
virtual state is initialised at the goal, so its objective has no slot for a
cost-to-go.

The roadmap, the heat solve and the surrogate fit are `MCMollifiedHeat`, the
same object `GDPlanner.__init__` builds for `EMSMCPlanner`, driven by the same
config values, so the term added here is the C_V of the paper and not a
re-implementation of it.

lambda_v is calibrated as in em_smc.py: std(task) / std(C_V) over the FIRST
batch the planner scores, which is its initial candidate population, then held
fixed for the trial. The weight is therefore set by the same rule as ours, per
seed, rather than by a constant that would not be our cost.
"""

import numpy as np
import torch

from .gd import MCMollifiedHeat, _load_replay_buffer


class RoadmapObjective:
    """objective_fn(i_z, z_g, a) + lambda_v * C_V(rollout)."""

    def __init__(self, objective_fn, replay_buffer_path, latent_key="proprio",
                 graph_n_nodes=3800, graph_k_neighbors=64, use_full_dim=True,
                 use_isomap=False, isomap_landmarks=0, solver_mode="heat",
                 rw_steps=20, heat_t_rule="degree", heat_c=1.0,
                 surrogate_steps=3000, v_traj="all", v_base=2.0,
                 lambda_v="auto", device=None):
        self.objective_fn = objective_fn
        self.latent_key = latent_key
        self.use_full_dim = bool(use_full_dim)
        self.v_traj = v_traj
        self.v_base = float(v_base)
        self.lambda_v = lambda_v          # "auto" until the first batch
        self.device = device
        self._goal_key = None             # re-solve when the goal changes

        print(f"[PDE cost] loading replay buffer {replay_buffer_path}")
        z_buf, z_next, pca_info = _load_replay_buffer(replay_buffer_path, device)
        self.pca_info = pca_info
        graph_z, graph_next = z_buf, z_next
        if self.use_full_dim and pca_info is not None:
            comp = pca_info["components"].cpu().numpy()
            mean = pca_info["mean"].cpu().numpy()
            graph_z = z_buf @ comp + mean
            graph_next = z_next @ comp + mean
        self.solver = MCMollifiedHeat()
        self.solver.v_control = None
        self.solver.heat_t_rule = heat_t_rule
        self.solver.heat_c = float(heat_c)
        self.solver.surrogate_steps = int(surrogate_steps)
        self.solver.compile_environment(
            z_buffer=graph_z, z_predicted_next=graph_next,
            n_nodes=int(graph_n_nodes), K_neighbors=int(graph_k_neighbors),
            use_isomap=bool(use_isomap), isomap_landmarks=int(isomap_landmarks),
            solver_mode=solver_mode, rw_steps=int(rw_steps))
        print(f"[PDE cost] roadmap built (n_nodes={graph_n_nodes}, "
              f"solver={solver_mode}, latent_key={latent_key})")

    # ------------------------------------------------------------------ latents
    def _get_latent(self, z_obses):
        """GDPlanner._get_latent, duplicated so no planner instance is needed."""
        latent = z_obses[self.latent_key].float()
        if latent.dim() == 4:
            latent = latent.mean(dim=2)
        if self.pca_info is not None and not self.use_full_dim:
            shape = latent.shape
            flat = latent.reshape(-1, shape[-1]) - self.pca_info["mean"]
            latent = (flat @ self.pca_info["components"].T).reshape(
                *shape[:-1], -1)
        return latent

    def _solve_if_new_goal(self, z_g):
        g = self._get_latent(z_g)[0, 0].detach()
        key = (float(g.sum()), float(g.pow(2).sum()))
        if key != self._goal_key:
            # CEM scores candidates under no_grad and MPPI's outer solver under
            # inference_mode; the surrogate solver backprops internally, so both
            # have to be lifted here or loss.backward() has nothing to walk.
            with torch.inference_mode(False), torch.enable_grad():
                self.solver.solve_for_goal(g.cpu().numpy().astype(np.float64))
            self._goal_key = key
            self.lambda_v = "auto"        # recalibrate per trial

    # ------------------------------------------------------------------- the cost
    def _value(self, i_z, horizon):
        lat = self._get_latent(i_z)
        if self.v_traj == "all":
            seq = lat[:, -horizon:, :]
            n, T = seq.shape[0], seq.shape[1]
            c = self.v_base ** torch.arange(T, device=seq.device, dtype=seq.dtype)
            c = c / c.sum()
            V = self.solver.evaluate_rollout(seq.reshape(n * T, -1)).view(n, T)
            return (V * c).sum(dim=1)
        return self.solver.evaluate_rollout(lat[:, -1, :])

    def __call__(self, i_z, z_g, a):
        task = self.objective_fn(i_z, z_g, a)
        self._solve_if_new_goal(z_g)
        V = self._value(i_z, int(a.shape[1]))
        if self.lambda_v == "auto":
            with torch.no_grad():
                s_t = task.detach().std().item() if task.numel() > 1 else 0.0
                s_v = V.detach().std().item() if V.numel() > 1 else 0.0
            self.lambda_v = (s_t / s_v) if s_v > 1e-12 else 0.0
            print(f"[PDE cost] lambda_v=auto -> {self.lambda_v:.4f} "
                  f"(std task {s_t:.4f}, std V {s_v:.4f}, v_traj={self.v_traj})")
        return task + self.lambda_v * V
