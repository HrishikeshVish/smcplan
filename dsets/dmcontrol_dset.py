"""
Dataset loader for DMControl HDF5 datasets from le-wm / stable-worldmodel.

Uses stable_worldmodel.data for HDF5 reading (same as le-wm), then wraps
the result to match grasp_langevin's PlanWorkspace interface:
  - dset[i] returns (obs, act, state, env_info)
  - dset.action_dim, action_mean, action_std, etc.
"""

import logging

import numpy as np
import torch
from torchvision import transforms

log = logging.getLogger(__name__)


class DMControlDataset:
    """Wraps a stable_worldmodel HDF5Dataset for grasp_langevin planners.

    Reads the HDF5 file via stable_worldmodel.data (which handles SWMR mode,
    hdf5plugin codecs, and lazy per-worker file handles), then exposes
    episode-level access with the (obs, act, state, env_info) convention.
    """

    def __init__(
        self,
        data_path,
        num_hist=3,
        num_pred=1,
        frameskip=5,
        img_size=224,
        split="valid",
        split_ratio=0.9,
    ):
        self.data_path = data_path
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.img_size = img_size

        log.info(f"Loading DMControl dataset from {data_path}")

        from stable_worldmodel.data import load_dataset

        swm_dset = load_dataset(
            data_path,
            frameskip=frameskip,
            num_steps=num_hist + num_pred,
        )

        # swm_dset has .lengths (episode lengths) and .offsets (start offsets)
        self._swm = swm_dset
        n_episodes = len(swm_dset.lengths)
        n_train = int(n_episodes * split_ratio)

        if split == "train":
            self._ep_range = range(0, n_train)
        else:
            self._ep_range = range(n_train, n_episodes)

        # Compute action statistics from training episodes
        train_actions = []
        for ep_idx in range(n_train):
            ep = swm_dset.load_episode(ep_idx)
            if 'action' in ep:
                a = ep['action'].numpy() if isinstance(ep['action'], torch.Tensor) else np.array(ep['action'])
                train_actions.append(a)

        if train_actions:
            all_actions = np.concatenate(train_actions, axis=0)
            valid_mask = ~np.isnan(all_actions).any(axis=1)
            all_actions = all_actions[valid_mask]
            self.action_dim = all_actions.shape[1]
            self.action_mean = torch.tensor(np.nanmean(all_actions, axis=0), dtype=torch.float32)
            self.action_std = torch.tensor(np.nanstd(all_actions, axis=0) + 1e-6, dtype=torch.float32)
        else:
            self.action_dim = 0
            self.action_mean = torch.zeros(0)
            self.action_std = torch.ones(0)

        # State/proprio stats
        self.state_dim = 0
        self.state_mean = torch.zeros(0)
        self.state_std = torch.ones(0)
        self.proprio_dim = 0
        self.proprio_mean = torch.zeros(0)
        self.proprio_std = torch.ones(0)

        cols = swm_dset.column_names
        if 'observation' in cols:
            obs_data = []
            for ep_idx in range(n_train):
                ep = swm_dset.load_episode(ep_idx)
                o = ep['observation'].numpy() if isinstance(ep['observation'], torch.Tensor) else np.array(ep['observation'])
                obs_data.append(o)
            if obs_data:
                all_obs = np.concatenate(obs_data, axis=0)
                valid_obs = all_obs[~np.isnan(all_obs).any(axis=1)]
                self.state_dim = all_obs.shape[1]
                self.state_mean = torch.tensor(np.nanmean(valid_obs, axis=0), dtype=torch.float32)
                self.state_std = torch.tensor(np.nanstd(valid_obs, axis=0) + 1e-6, dtype=torch.float32)

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        log.info(
            f"DMControl dataset: {len(self._ep_range)} episodes, "
            f"action_dim={self.action_dim}, state_dim={self.state_dim}"
        )

    def __len__(self):
        return len(self._ep_range)

    def __getitem__(self, idx):
        """Get a full episode trajectory.

        Returns:
            obs: dict with 'visual' (T, C, H, W) float tensor, 'proprio' (T, D)
            act: (T, action_dim) float tensor
            state: (T, state_dim) float tensor
            env_info: dict with metadata
        """
        ep_idx = self._ep_range[idx]
        ep = self._swm.load_episode(ep_idx)

        # Pixels
        pixels = ep['pixels']  # torch tensor from HDF5Dataset
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.tensor(np.array(pixels))
        # HDF5Dataset already handles HWC->CHW permutation
        pixels = pixels.float() / 255.0

        # Actions
        actions = ep.get('action', torch.zeros(pixels.shape[0], self.action_dim))
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(np.array(actions), dtype=torch.float32)
        actions = actions.float()
        actions = torch.nan_to_num(actions, nan=0.0)
        if self.action_std.numel() > 0:
            actions = (actions - self.action_mean) / self.action_std

        # State
        T = pixels.shape[0]
        cols = self._swm.column_names
        if 'observation' in cols and 'observation' in ep:
            state = ep['observation']
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(np.array(state), dtype=torch.float32)
            state = state.float()
        elif 'qpos' in cols and 'qpos' in ep:
            qpos = ep['qpos']
            qvel = ep['qvel']
            if not isinstance(qpos, torch.Tensor):
                qpos = torch.tensor(np.array(qpos), dtype=torch.float32)
                qvel = torch.tensor(np.array(qvel), dtype=torch.float32)
            state = torch.cat([qpos.float(), qvel.float()], dim=-1)
        else:
            state = torch.zeros(T, 0)

        obs = {
            'visual': pixels,
            'proprio': torch.zeros(T, self.proprio_dim),
        }

        env_info = {'episode': ep_idx}
        return obs, actions, state, env_info


def load_dmcontrol_dataset(data_path, num_hist=3, num_pred=1, frameskip=5, img_size=224):
    """Load DMControl dataset and return train/valid splits.

    Returns:
        Tuple of (None, dict) to match grasp_langevin convention:
          (_, {"train": dataset, "valid": dataset})
    """
    train_dset = DMControlDataset(
        data_path, num_hist=num_hist, num_pred=num_pred,
        frameskip=frameskip, img_size=img_size, split="train",
    )
    valid_dset = DMControlDataset(
        data_path, num_hist=num_hist, num_pred=num_pred,
        frameskip=frameskip, img_size=img_size, split="valid",
    )
    return None, {"train": train_dset, "valid": valid_dset}
