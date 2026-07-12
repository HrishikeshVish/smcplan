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

    def eval_state(self, goal_state, cur_state):
        """Evaluate whether the current state has reached the goal.

        Thresholds match le-wm / stable-worldmodel:
          - Reacher (swm/ReacherDMControl): per-joint qpos tolerance of 0.05 rad
          - Cube (swm/OGBCube): Euclidean position tolerance of 0.04 m
        """
        state_dist = np.linalg.norm(goal_state - cur_state)
        if 'Cube' in self.env_name or 'OGB' in self.env_name:
            # Cube: compare object position (first 3 dims of qpos)
            success = state_dist <= 0.04
        else:
            # Reacher: per-joint qpos threshold of 0.05 rad
            n_qpos = len(goal_state) // 2
            qpos_diff = np.abs(goal_state[:n_qpos] - cur_state[:n_qpos])
            success = bool(np.all(qpos_diff < 0.05))
        return {
            'success': success,
            'state_dist': state_dist,
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

    def _get_env_state(self, env):
        """Extract state from environment."""
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'physics'):
            # DMControl via dm_control
            qpos = unwrapped.physics.data.qpos.copy()
            qvel = unwrapped.physics.data.qvel.copy()
            return np.concatenate([qpos, qvel])
        return np.zeros(0)

    def _set_env_state(self, env, state):
        """Set environment state."""
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'physics'):
            n = len(unwrapped.physics.data.qpos)
            unwrapped.physics.data.qpos[:] = state[:n]
            unwrapped.physics.data.qvel[:] = state[n:]
            unwrapped.physics.forward()
