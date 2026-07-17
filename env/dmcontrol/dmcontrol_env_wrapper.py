"""
Environment wrapper for DMControl tasks (reacher, etc.) used with LeWM.

Uses stable_worldmodel's World API for env management, but exposes the
interface expected by grasp_langevin's PlanWorkspace and PlanEvaluator.
"""

import numpy as np
import torch
import gymnasium as gym

# Import stable_worldmodel.envs to register swm/* gymnasium environments
try:
    import stable_worldmodel.envs  # noqa: F401
except ImportError:
    pass


class DMControlEnvWrapper:
    """Wraps a DMControl gym environment for grasp_langevin planning.

    Provides the interface expected by PlanEvaluator:
      - reset(seed) -> obs, info
      - step(action) -> obs, reward, done, info
      - rollout(seeds, init_states, actions) -> (obses, states)
      - prepare(seeds, states) -> (obs, state)
    """

    def __init__(self, env_name, n_envs, max_episode_steps=100, image_size=224, render_kwargs=None):
        self.env_name = env_name
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size

        # Create environments
        self.envs = []
        for _ in range(n_envs):
            env = gym.make(
                env_name,
                max_episode_steps=max_episode_steps,
                render_mode="rgb_array",
            )
            self.envs.append(env)

        # Get action/obs dims from first env
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

    def reset(self, seeds=None):
        """Reset all environments.

        Returns:
            obs: dict with 'visual' (n_envs, H, W, C), 'proprio' (n_envs, D)
        """
        if seeds is None:
            seeds = [None] * self.n_envs

        visuals = []
        proprios = []
        infos = []
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=seeds[i] if seeds[i] is not None else None)
            img = env.render()
            visuals.append(img)
            proprios.append(obs if isinstance(obs, np.ndarray) else np.zeros(0))
            infos.append(info)

        obs_dict = {
            'visual': np.stack(visuals),  # (n_envs, H, W, C)
            'proprio': np.stack(proprios),  # (n_envs, D)
        }
        return obs_dict, infos

    def step(self, actions):
        """Step all environments.

        Args:
            actions: (n_envs, action_dim) numpy array

        Returns:
            obs, rewards, dones, infos
        """
        visuals = []
        proprios = []
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[i])
            img = env.render()
            visuals.append(img)
            proprios.append(obs if isinstance(obs, np.ndarray) else np.zeros(0))
            rewards.append(reward)
            dones.append(terminated or truncated)
            infos.append(info)

        obs_dict = {
            'visual': np.stack(visuals),
            'proprio': np.stack(proprios),
        }
        return obs_dict, np.array(rewards), np.array(dones), infos

    def rollout(self, seeds, init_states, actions):
        """Rollout actions from initial states.

        Args:
            seeds: list of seeds
            init_states: (n_envs, state_dim) initial states
            actions: (n_envs, T, action_dim) action sequences

        Returns:
            obses: dict with 'visual' (n_envs, T+1, H, W, C)
            states: (n_envs, T+1, state_dim) state trajectories
        """
        n_steps = actions.shape[1]
        all_visuals = []
        all_proprios = []
        all_states = []

        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=int(seeds[i]) if seeds[i] is not None else None)

            # Set state if available
            if init_states is not None:
                self._set_env_state(env, init_states[i])

            # Collect initial observation
            img = env.render()
            ep_visuals = [img]
            init_obs = obs if isinstance(obs, np.ndarray) else np.zeros(0)
            ep_proprios = [init_obs]
            ep_states = [self._get_env_state(env)]

            for t in range(n_steps):
                obs, reward, terminated, truncated, info = env.step(actions[i, t])
                img = env.render()
                ep_visuals.append(img)
                ep_proprios.append(obs if isinstance(obs, np.ndarray) else init_obs)
                ep_states.append(self._get_env_state(env))

            all_visuals.append(np.stack(ep_visuals))
            all_proprios.append(np.stack(ep_proprios))
            all_states.append(np.stack(ep_states))

        obses = {
            'visual': np.stack(all_visuals),    # (n_envs, T+1, H, W, C)
            'proprio': np.stack(all_proprios),   # (n_envs, T+1, D)
        }
        states = np.stack(all_states)  # (n_envs, T+1, state_dim)
        return obses, states

    def prepare(self, seeds, states):
        """Reset envs and set to specific states.

        Returns:
            obs: dict, state: np.array
        """
        visuals = []
        proprios = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=int(seeds[i]) if seeds[i] is not None else None)
            if states is not None:
                self._set_env_state(env, states[i])
            img = env.render()
            visuals.append(img)
            proprios.append(obs if isinstance(obs, np.ndarray) else np.zeros(0))

        obs_dict = {
            'visual': np.stack(visuals),
            'proprio': np.stack(proprios),
        }
        return obs_dict, states

    def eval_state(self, goal_states, cur_states):
        """Evaluate whether current states have reached goals (batched).

        Args:
            goal_states: (n_envs, state_dim) goal states
            cur_states: (n_envs, state_dim) current states

        Thresholds match le-wm / stable-worldmodel:
          - Reacher (swm/ReacherDMControl): per-joint qpos tolerance of 0.05 rad
          - Cube (swm/OGBCube): Euclidean position tolerance of 0.04 m
        """
        goal_states = np.asarray(goal_states)
        cur_states = np.asarray(cur_states)
        state_dists = np.linalg.norm(goal_states - cur_states, axis=-1)

        if 'Cube' in self.env_name or 'OGB' in self.env_name:
            successes = state_dists <= 0.04
            return {
                'success': successes,
                'state_dist': state_dists,
            }
        else:
            n_qpos = goal_states.shape[-1] // 2
            qpos_diff = np.abs(goal_states[:, :n_qpos] - cur_states[:, :n_qpos])
            qvel_diff = np.abs(goal_states[:, n_qpos:] - cur_states[:, n_qpos:])
            successes = np.all(qpos_diff < 0.05, axis=-1)

            return {
                'success': successes,
                'state_dist': state_dists,
                'qpos_diff': qpos_diff,
                'qvel_diff': qvel_diff,
            }

    def update_env(self, env_info):
        """No-op for DMControl."""
        pass

    def sample_random_init_goal_states(self, seeds):
        """Sample random init and goal states."""
        init_states = []
        goal_states = []
        for i, env in enumerate(self.envs):
            env.reset(seed=int(seeds[i]))
            init_states.append(self._get_env_state(env))
            # Take random steps to get a different state as goal
            for _ in range(10):
                env.step(env.action_space.sample())
            goal_states.append(self._get_env_state(env))
        return np.stack(init_states), np.stack(goal_states)

    def _get_physics(self, env):
        """Get the MuJoCo physics object from the environment.

        The swm DMControlWrapper stores the dm_control env at self.env,
        so physics is at env.unwrapped.env.physics (not env.unwrapped.physics).
        """
        unwrapped = env.unwrapped
        # swm DMControlWrapper: unwrapped.env is the dm_control Environment
        if hasattr(unwrapped, 'env') and hasattr(unwrapped.env, 'physics'):
            return unwrapped.env.physics
        # Direct dm_control env
        if hasattr(unwrapped, 'physics'):
            return unwrapped.physics
        return None

    def _get_env_state(self, env):
        """Extract state from environment."""
        physics = self._get_physics(env)
        if physics is not None:
            qpos = physics.data.qpos.copy()
            qvel = physics.data.qvel.copy()
            return np.concatenate([qpos, qvel])
        return np.zeros(0)

    def _set_env_state(self, env, state):
        """Set environment state."""
        unwrapped = env.unwrapped
        # Prefer the wrapper's set_state() method (handles act array, forward())
        if hasattr(unwrapped, 'set_state'):
            physics = self._get_physics(env)
            n = len(physics.data.qpos)
            unwrapped.set_state(state[:n], state[n:])
            return
        physics = self._get_physics(env)
        if physics is not None:
            n = len(physics.data.qpos)
            physics.data.qpos[:] = state[:n]
            physics.data.qvel[:] = state[n:]
            physics.forward()
