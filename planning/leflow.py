"""LeFlow baseline (third_party/LeFlow, official code) under our protocol.

LeFlow samples latent paths from the start to the goal latent with its flow
model, decodes them into action chunks with its inverse dynamics model, rolls
each candidate through the frozen LeWM and keeps the one whose final latent is
closest to the goal (LatentPlannerRuntime.plan, score_mode="rollout_goal").
Here the runtime is given our frozen LeWM module, plans one H-step sequence
from the start observation, and that sequence is executed open-loop once, as a
single-iteration planner. Its actions are in LeWM's normalization (the whole
action column's mean and std, saved by writeup/baselines/train_leflow_cached.py)
and are mapped to the normalization of our harness before execution.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .base_planner import BasePlanner

_LEFLOW = Path(__file__).resolve().parents[1] / "third_party" / "LeFlow"
if str(_LEFLOW) not in sys.path:
    sys.path.insert(0, str(_LEFLOW))
from latent_planner import (InverseDynamics, LatentPathFlow,  # noqa: E402
                            LatentPlannerRuntime)


class LeFlowPlanner(BasePlanner):
    def __init__(self, checkpoint_root, horizon, num_samples=64, flow_steps=16,
                 score_mode="rollout_goal", history_size=3, opt_steps=1,
                 eval_every=1, wm=None, action_dim=None, objective_fn=None,
                 preprocessor=None, evaluator=None, wandb_run=None,
                 logging_prefix="plan_0", log_filename="logs.json", **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator,
                         wandb_run, log_filename)
        self.checkpoint_root = checkpoint_root
        self.horizon = horizon
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.score_mode = score_mode
        self.history_size = history_size
        self.logging_prefix = logging_prefix

    def _runtime(self):
        # one checkpoint per horizon: <root>_h<H>/latent_planner.pt
        ckdir = Path(f"{self.checkpoint_root}_h{self.horizon}")
        payload = torch.load(ckdir / "latent_planner.pt", map_location="cpu",
                             weights_only=False)
        flow = LatentPathFlow(**payload["arch"]["flow"])
        idm = InverseDynamics(**payload["arch"]["inverse_dynamics"])
        flow.load_state_dict(payload["flow_state_dict"])
        idm.load_state_dict(payload["inverse_dynamics_state_dict"])
        rt = LatentPlannerRuntime(lewm=self.wm.jepa, flow=flow, inverse_dynamics=idm,
                                  action_block=payload.get("action_block", 5))
        norm = json.load(open(ckdir / "action_normalizer.json"))
        return rt.to(self.device).eval(), norm

    def _to_harness(self, a, norm):
        """(1, H, fs*d) actions in LeWM's normalization -> ours."""
        d = len(norm["mean"])
        m_lf = torch.tensor(norm["mean"], dtype=a.dtype)
        s_lf = torch.tensor(norm["std"], dtype=a.dtype)
        m = torch.as_tensor(self.preprocessor.action_mean, dtype=a.dtype).reshape(-1)[:d]
        s = torch.as_tensor(self.preprocessor.action_std, dtype=a.dtype).reshape(-1)[:d]
        raw = a.reshape(*a.shape[:-1], -1, d) * s_lf + m_lf
        return ((raw - m) / s).reshape(a.shape)

    @torch.no_grad()
    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, _, trans_obs_g, _ = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        rt, norm = self._runtime()
        gen = torch.Generator(device=self.device).manual_seed(
            int(torch.randint(0, 2**31 - 1, (1,)).item()))
        out = rt.plan({"pixels": trans_obs_0["visual"], "goal": trans_obs_g["visual"]},
                      horizon=self.horizon, num_samples=self.num_samples,
                      flow_steps=self.flow_steps, score_mode=self.score_mode,
                      history_size=self.history_size, generator=gen)
        plan = self._to_harness(out["actions"].float(), norm).to(self.device)
        logs, successes, _, _ = self.evaluator.eval_actions(
            plan, filename=f"{self.logging_prefix}_output_1")
        current_success = float(successes.mean())
        self.exit_step = 1
        self.total_measured_time = 0.0
        results_dir = os.environ.get("GRASP_RESULTS_DIR", os.getcwd())
        with open(os.path.join(results_dir, "seed_success_log.txt"), "a") as f:
            f.write(f"{current_success} {logs.get('mean_state_dist')} "
                    f"{logs.get('mean_visual_dist')} {logs.get('mean_proprio_dist')} 0\n")
        return plan, np.full(plan.shape[0], np.inf)
