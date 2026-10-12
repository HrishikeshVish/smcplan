"""GC-IDM baseline (third_party/GCIDM, official model) under our protocol.

GC-IDM maps (current latent, goal latent, remaining steps) to the next action
and, as released, acts in closed loop at every environment step (eval_idm.py
GCIDMPolicy.get_action). Two modes:

- "open_loop" (our adaptation, disclosed as such): an IDM trained with
  frameskip 5 outputs one action block per world-model step; the block is
  appended to the plan and the plan is rolled through our frozen LeWM to give
  the next latent, for H steps, with the remaining horizon counted in raw
  environment steps as in its training data. The H-step plan is then executed
  open-loop once, as for every other planner.
- "closed_loop" (the released usage, a reference): the frameskip-1 IDM acts at
  every one of the 5H environment steps from the encoded observation, with the
  remaining budget min(5H - t, max_horizon); success is checked on the final
  state, as for the open-loop planners.

GC-IDM outputs raw environment actions; they are mapped to the normalization of
our harness where our rollout or evaluator needs them.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .base_planner import BasePlanner

_GCIDM = Path(__file__).resolve().parents[1] / "third_party" / "GCIDM"
if str(_GCIDM) not in sys.path:
    sys.path.insert(0, str(_GCIDM))
from idm.model import GoalConditionedIDM, IDMConfig  # noqa: E402


class GCIDMPlanner(BasePlanner):
    def __init__(self, checkpoint_root, horizon, mode="open_loop", opt_steps=1,
                 eval_every=1, wm=None, action_dim=None, objective_fn=None,
                 preprocessor=None, evaluator=None, wandb_run=None,
                 logging_prefix="plan_0", log_filename="logs.json", **kwargs):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator,
                         wandb_run, log_filename)
        assert mode in ("open_loop", "closed_loop"), mode
        self.mode = mode
        self.checkpoint_root = checkpoint_root
        self.horizon = horizon
        self.logging_prefix = logging_prefix

    def _load(self, fs):
        ck = torch.load(f"{self.checkpoint_root}_fs{fs}.pt", map_location="cpu",
                        weights_only=False)
        idm = GoalConditionedIDM(IDMConfig(**ck["config"]))
        idm.load_state_dict(ck["model_state_dict"])
        return idm.to(self.device).eval()

    def _stats(self, d):
        m = torch.as_tensor(self.preprocessor.action_mean).reshape(-1)[:d].to(self.device)
        s = torch.as_tensor(self.preprocessor.action_std).reshape(-1)[:d].to(self.device)
        return m.float(), s.float()

    def _encode(self, trans_obs):
        z = self.wm.encode_obs(trans_obs)["visual"]   # (B, T, 1, D)
        return z[:, -1, 0]

    def _log(self, success, logs):
        self.exit_step = 1
        self.total_measured_time = 0.0
        results_dir = os.environ.get("GRASP_RESULTS_DIR", os.getcwd())
        with open(os.path.join(results_dir, "seed_success_log.txt"), "a") as f:
            f.write(f"{success} {logs.get('mean_state_dist')} "
                    f"{logs.get('mean_visual_dist')} {logs.get('mean_proprio_dist')} 0\n")

    @torch.no_grad()
    def plan(self, obs_0, obs_g, actions=None):
        trans_obs_0, _, trans_obs_g, _ = self.preprocess_start_goal(
            obs_0, obs_g, encode_to_vector=False)
        z_goal = self._encode(trans_obs_g)
        if self.mode == "open_loop":
            return self._plan_open_loop(trans_obs_0, z_goal)
        return self._run_closed_loop(z_goal)

    def _plan_open_loop(self, trans_obs_0, z_goal):
        idm = self._load(5)
        fs = idm.cfg.frameskip
        d = self.action_dim // fs
        m, s = self._stats(d)
        z = self._encode(trans_obs_0)
        plan = torch.zeros(1, 0, self.action_dim, device=self.device)
        for k in range(self.horizon):
            remaining = torch.full((1,), fs * (self.horizon - k), device=self.device,
                                   dtype=torch.long)
            raw = idm(z, z_goal, remaining).reshape(1, 1, fs, d)
            block = ((raw - m) / s).reshape(1, 1, self.action_dim)
            plan = torch.cat([plan, block], dim=1)
            z_obses, _ = self.wm.rollout(obs_0=trans_obs_0, act=plan)
            z = z_obses["visual"][:, -1, 0]
        logs, successes, _, _ = self.evaluator.eval_actions(
            plan, filename=f"{self.logging_prefix}_output_1")
        self._log(float(successes.mean()), logs)
        return plan, np.full(1, np.inf)

    def _run_closed_loop(self, z_goal):
        idm = self._load(1)
        ev = self.evaluator
        env = ev.env.envs[0]
        seed = int(np.asarray(ev.seed).reshape(-1)[0])
        env.reset(seed=seed)
        ev.env._set_env_state(env, np.asarray(ev.state_0)[0])
        budget = ev.frameskip * self.horizon
        actions = []
        for t in range(budget):
            img = np.ascontiguousarray(env.render())
            # proprio is not encoded by LeWM; keep the evaluator's shape
            obs = {"visual": img[None, None],
                   "proprio": np.zeros((1, 1, np.asarray(ev.obs_0["proprio"]).shape[-1]),
                                       np.float32)}
            trans = self.preprocessor.transform_obs(obs)
            trans = {k: torch.as_tensor(v).to(self.device) for k, v in trans.items()}
            z = self._encode(trans)
            remaining = torch.full((1,), min(budget - t, idm.max_horizon),
                                   device=self.device, dtype=torch.long)
            a = idm(z, z_goal, remaining)[0].cpu().numpy()
            env.step(a)
            actions.append(a)
        final_state = ev.env._get_env_state(env)[None]
        res = ev.env.eval_state(ev.state_g, final_state)
        success = float(np.mean(res["success"]))
        print("Success rate: ", success)
        print(res)
        self._log(success, {"mean_state_dist": float(np.mean(res["state_dist"]))})
        # The executed raw actions, in our normalization and grouped into
        # world-model blocks, so that plan.py's final replay of the returned
        # plan reproduces this (deterministic) trajectory.
        raw = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=self.device)
        m, s = self._stats(raw.shape[-1])
        plan = ((raw - m) / s).reshape(1, self.horizon, self.action_dim)
        return plan, np.full(1, np.inf)
