"""
Dataset loader for DMControl HDF5 datasets from le-wm / stable-worldmodel.

Uses stable_worldmodel.data for HDF5 reading (same as le-wm):
  - HDF5Dataset with SWMR mode, hdf5plugin codecs, lazy per-worker handles
  - get_col_data() for fast column-level stats
  - load_episode() for episode access at __getitem__ time
"""

import logging

import numpy as np
import torch
from torchvision import transforms

log = logging.getLogger(__name__)


class DMControlDataset:
    """Wraps a stable_worldmodel HDF5Dataset for grasp_langevin planners."""

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
        self._swm = swm_dset

        # Split episodes
        n_episodes = len(swm_dset.lengths)
        n_train = int(n_episodes * split_ratio)
        if split == "train":
            self._ep_range = range(0, n_train)
        else:
            self._ep_range = range(n_train, n_episodes)

        # Compute stats using get_col_data (single bulk read, like le-wm)
        cols = swm_dset.column_names

        # Train mask: rows belonging to training episodes
        train_end = int(swm_dset.offsets[n_train]) if n_train < n_episodes else int(swm_dset.offsets[-1] + swm_dset.lengths[-1])

        all_actions = swm_dset.get_col_data('action')[:train_end]
        valid = ~np.isnan(all_actions).any(axis=1)
        all_actions = all_actions[valid]
        self.action_dim = all_actions.shape[1]
        self.action_mean = torch.from_numpy(np.nanmean(all_actions, axis=0).astype(np.float32))
        self.action_std = torch.from_numpy((np.nanstd(all_actions, axis=0) + 1e-6).astype(np.float32))

        # State stats (prefer qpos/qvel to match env state format)
        self.state_dim = 0
        self.state_mean = torch.zeros(0)
        self.state_std = torch.ones(0)
        if 'qpos' in cols:
            qpos_data = swm_dset.get_col_data('qpos')[:train_end]
            qvel_data = swm_dset.get_col_data('qvel')[:train_end]
            state_data = np.concatenate([qpos_data, qvel_data], axis=-1)
            valid_state = state_data[~np.isnan(state_data).any(axis=1)]
            self.state_dim = state_data.shape[1]
            self.state_mean = torch.from_numpy(np.nanmean(valid_state, axis=0).astype(np.float32))
            self.state_std = torch.from_numpy((np.nanstd(valid_state, axis=0) + 1e-6).astype(np.float32))
        elif 'observation' in cols:
            obs_data = swm_dset.get_col_data('observation')[:train_end]
            valid_obs = obs_data[~np.isnan(obs_data).any(axis=1)]
            self.state_dim = obs_data.shape[1]
            self.state_mean = torch.from_numpy(np.nanmean(valid_obs, axis=0).astype(np.float32))
            self.state_std = torch.from_numpy((np.nanstd(valid_obs, axis=0) + 1e-6).astype(np.float32))

        self.proprio_dim = 0
        self.proprio_mean = torch.zeros(0)
        self.proprio_std = torch.ones(0)

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

        # Pixels (HDF5Dataset handles HWC->CHW)
        pixels = ep['pixels']
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.from_numpy(np.array(pixels))
        pixels = pixels.float() / 255.0

        # Actions
        actions = ep.get('action', torch.zeros(pixels.shape[0], self.action_dim))
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.array(actions))
        actions = torch.nan_to_num(actions.float(), nan=0.0)
        actions = (actions - self.action_mean) / self.action_std

        # State: prefer qpos/qvel (matches le-wm and env's set_state interface)
        T = pixels.shape[0]
        cols = self._swm.column_names
        if 'qpos' in cols and 'qpos' in ep:
            qpos = ep['qpos'] if isinstance(ep['qpos'], torch.Tensor) else torch.from_numpy(np.array(ep['qpos']))
            qvel = ep['qvel'] if isinstance(ep['qvel'], torch.Tensor) else torch.from_numpy(np.array(ep['qvel']))
            state = torch.cat([qpos.float(), qvel.float()], dim=-1)
        elif 'observation' in cols and 'observation' in ep:
            state = ep['observation']
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(np.array(state))
            state = state.float()
        else:
            state = torch.zeros(T, 0)

        obs = {
            'visual': pixels,
            'proprio': torch.zeros(T, self.proprio_dim),
        }
        return obs, actions, state, {'episode': ep_idx}


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
