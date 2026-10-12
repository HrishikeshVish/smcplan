"""MPPI baseline through the official solver of LeWM's planning library.

The update is `stable_worldmodel.solver.mppi.MPPISolver` (stable-worldmodel
0.1.1) unchanged; this file only adapts our frozen world model and planning
objective to its `Costable` interface and drives it one iteration at a time,
so that, as for every other planner in the paper, the current plan is executed
after each iteration. Calling `solve` with n_steps=1 and the previous mean as
`init_action` is the same iteration as one step of `solve` with n_steps=K: the
solver keeps its noise generator across calls and, as standard MPPI, never
updates the variance.
"""
import importlib.util
import os
import sys
import time
import types
from types import SimpleNamespace

import numpy as np
import torch
from einops import repeat
from gymnasium.spaces import Box

from .base_planner import BasePlanner


def _import_official_mppi():
    """Import stable_worldmodel.solver.mppi without running the package
    __init__ when it is not already loaded. On the DINO-WM path this repo's
    local `datasets/` package shadows HuggingFace `datasets`, which the
    package __init__ reaches through lancedb; the solver itself only needs
    solver/{mppi,utils,solver}.py and protocols.py, which are run unchanged."""
    if "stable_worldmodel" not in sys.modules:
        root = os.path.dirname(importlib.util.find_spec("stable_worldmodel").origin)
        for name, path in (("stable_worldmodel", root),
                           ("stable_worldmodel.solver", os.path.join(root, "solver"))):
            pkg = types.ModuleType(name)
            pkg.__path__ = [path]
            sys.modules[name] = pkg
    from stable_worldmodel.solver.mppi import MPPISolver
    return MPPISolver


MPPISolver = _import_official_mppi()


class _WorldModelCost:
    """`Costable` adapter: the planning objective of our rollout, (1, S)."""

    def __init__(self, wm, objective_fn, obs_0, z_obs_g, is_dict):
        self.wm = wm
        self.objective_fn = objective_fn
        self.obs_0 = obs_0
        self.z_obs_g = z_obs_g
        self.is_dict = is_dict

    def parameters(self):
        return self.wm.parameters()

    def _expand(self, x, n):
        if self.is_dict:
            return {k: repeat(v[0].unsqueeze(0), "1 ... -> n ...", n=n)
                    for k, v in x.items()}
        return repeat(x[0].unsqueeze(0), "1 ... -> n ...", n=n)

    @torch.no_grad()
    def get_cost(self, info_dict, action_candidates):
        b, s = action_candidates.shape[:2]
        assert b == 1, "MPPIPlanner plans one env at a time"
        act = action_candidates[0].float()
        obs_0 = self._expand(self.obs_0, s)
        z_g = self._expand(self.z_obs_g, s)
        # Same call signature as planning/cem.py for dict and tensor inputs.
        i_z_obses, _ = self.wm.rollout(
            obs_0=obs_0 if self.is_dict else obs_0[:, 0], act=act)
        loss = self.objective_fn(i_z_obses, z_g, act)
        return loss.view(1, s)

    criterion = get_cost


class MPPIPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        num_samples,
        topk,
        temperature,
        var_scale,
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
        self.num_samples = num_samples
        self.topk = topk
        self.temperature = temperature
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, z_obs_0, trans_obs_g, z_obs_g = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        is_dict = isinstance(obs_0, dict)
        n_evals = (trans_obs_0["visual"].shape[0] if is_dict
                   else trans_obs_0.shape[0])
        assert n_evals == 1, "MPPIPlanner plans one env at a time"
        # planning/cem.py compares tensor observations against the transformed
        # goal rather than its encoding; keep the same objective input.
        goal = z_obs_g if is_dict else trans_obs_g
        cost = _WorldModelCost(self.wm, self.objective_fn, trans_obs_0, goal,
                               is_dict)

        # The solver's generator is seeded from the global torch generator,
        # which plan.py seeds with the trial seed, so trials draw independent
        # noise.
        solver_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        solver = MPPISolver(
            model=cost, batch_size=1, num_samples=self.num_samples,
            var_scale=self.var_scale, n_steps=1, topk=self.topk,
            temperature=self.temperature, device=self.device, seed=solver_seed)
        solver.configure(
            action_space=Box(low=-np.inf, high=np.inf,
                             shape=(1, self.action_dim), dtype=np.float32),
            n_envs=1,
            config=SimpleNamespace(horizon=self.horizon, action_block=1))
        info = {"_": torch.zeros(1, device=self.device)}

        mean = (torch.zeros(1, self.horizon, self.action_dim, device=self.device)
                if actions is None else actions.to(self.device))
        current_success = 0.0
        mean_dist = mean_vis_dist = mean_proprio_dist = np.inf
        total_measured_time = 0.0
        exit_step = self.opt_steps
        i = 0
        for i in range(self.opt_steps):
            start = time.perf_counter()
            out = solver.solve(info, init_action=mean)
            mean = out["actions"].to(self.device)
            total_measured_time += time.perf_counter() - start

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mean, filename=f"{self.logging_prefix}_output_{i+1}")
                current_success = float(successes.mean())
                mean_dist = logs.get("mean_state_dist")
                mean_vis_dist = logs.get("mean_visual_dist")
                mean_proprio_dist = logs.get("mean_proprio_dist")
                self.wandb_run.log({
                    f"{self.logging_prefix}/loss": float(np.mean(out["costs"])),
                    f"{self.logging_prefix}/success": successes.mean(),
                    "step": i + 1})
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"total_measured_time": total_measured_time,
                             "step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    exit_step = i + 1
                    break

        self.exit_step = exit_step
        self.total_measured_time = total_measured_time
        results_dir = os.environ.get("GRASP_RESULTS_DIR", os.getcwd())
        with open(os.path.join(results_dir, "seed_success_log.txt"), "a") as f:
            f.write(f"{current_success} {mean_dist} {mean_vis_dist} "
                    f"{mean_proprio_dist} {i}\n")
        return mean, np.full(n_evals, np.inf)
