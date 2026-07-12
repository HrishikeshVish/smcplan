"""
Dataset loader for DMControl HDF5 datasets from le-wm / stable-worldmodel.

Loads data in the format expected by grasp_langevin's PlanWorkspace:
  - dset[i] returns (obs, act, state, env_info)
  - dset.action_dim, action_mean, action_std, etc.
"""

import os
import logging

import h5py
import numpy as np
import torch
from torchvision import transforms

log = logging.getLogger(__name__)


class DMControlDataset:
    """HDF5 dataset for DMControl tasks (reacher, cube, etc.).

    Expected HDF5 structure (from le-wm / stable-worldmodel):
      - pixels: (N, C, H, W) uint8 images
      - action: (N, action_dim) float actions
      - episode_idx or ep_idx: (N,) int episode indices
      - step_idx: (N,) int step indices within episodes
      - Optional: qpos, qvel, observation
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

        self.h5 = h5py.File(data_path, 'r')

        # Detect column names
        self.ep_col = 'episode_idx' if 'episode_idx' in self.h5 else 'ep_idx'

        # Load metadata
        self.episode_indices = np.array(self.h5[self.ep_col])
        self.step_indices = np.array(self.h5['step_idx'])
        self.actions_all = np.array(self.h5['action'])

        # Get unique episodes and split
        unique_eps = np.unique(self.episode_indices)
        n_train = int(len(unique_eps) * split_ratio)
        if split == "train":
            self.episodes = unique_eps[:n_train]
        else:
            self.episodes = unique_eps[n_train:]

        # Build episode index mapping
        self._build_episode_map()

        # Compute statistics from training data
        train_eps = unique_eps[:n_train]
        train_mask = np.isin(self.episode_indices, train_eps)
        train_actions = self.actions_all[train_mask]
        valid_mask = ~np.isnan(train_actions).any(axis=1)
        train_actions = train_actions[valid_mask]

        self.action_dim = self.actions_all.shape[1]
        self.action_mean = torch.tensor(np.nanmean(train_actions, axis=0), dtype=torch.float32)
        self.action_std = torch.tensor(np.nanstd(train_actions, axis=0) + 1e-6, dtype=torch.float32)

        # State/proprio stats (use zeros if not available)
        if 'observation' in self.h5:
            obs_data = np.array(self.h5['observation'])[train_mask]
            valid_obs = obs_data[~np.isnan(obs_data).any(axis=1)]
            self.state_dim = obs_data.shape[1]
            self.state_mean = torch.tensor(np.nanmean(valid_obs, axis=0), dtype=torch.float32)
            self.state_std = torch.tensor(np.nanstd(valid_obs, axis=0) + 1e-6, dtype=torch.float32)
        else:
            self.state_dim = 0
            self.state_mean = torch.zeros(0)
            self.state_std = torch.ones(0)

        self.proprio_dim = 0
        self.proprio_mean = torch.zeros(0)
        self.proprio_std = torch.ones(0)

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        log.info(
            f"DMControl dataset: {len(self.episodes)} episodes, "
            f"action_dim={self.action_dim}, state_dim={self.state_dim}"
        )

    def _build_episode_map(self):
        """Build mapping from episode index to data rows."""
        self.episode_map = {}
        for ep in self.episodes:
            mask = self.episode_indices == ep
            indices = np.where(mask)[0]
            self.episode_map[ep] = indices

    @staticmethod
    def _read_slice(dataset, indices):
        """Read from an HDF5 dataset using contiguous slices to avoid plugin errors."""
        if len(indices) > 0 and np.all(np.diff(indices) == 1):
            return np.array(dataset[indices[0]:indices[-1] + 1])
        lo, hi = int(indices.min()), int(indices.max()) + 1
        block = np.array(dataset[lo:hi])
        return block[indices - lo]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        """Get a trajectory.

        Returns:
            obs: dict with 'visual' (T, C, H, W) float tensor, 'proprio' (T, D)
            act: (T, action_dim) float tensor
            state: (T, state_dim) float tensor
            env_info: dict with metadata
        """
        ep = self.episodes[idx]
        row_indices = self.episode_map[ep]
        # Sort by step_idx
        steps = self.step_indices[row_indices]
        sort_order = np.argsort(steps)
        row_indices = row_indices[sort_order]

        # Load pixels – use contiguous slices to avoid HDF5 plugin errors
        pixels = self._read_slice(self.h5['pixels'], row_indices)
        if pixels.ndim == 4 and pixels.shape[-1] == 3:
            # HWC -> CHW
            pixels = np.transpose(pixels, (0, 3, 1, 2))
        pixels = torch.tensor(pixels, dtype=torch.float32) / 255.0

        # Load actions
        actions = torch.tensor(
            np.nan_to_num(self.actions_all[row_indices], nan=0.0),
            dtype=torch.float32,
        )
        # Normalize
        actions = (actions - self.action_mean) / self.action_std

        # Load state
        if 'observation' in self.h5:
            state = torch.tensor(
                self._read_slice(self.h5['observation'], row_indices),
                dtype=torch.float32,
            )
        elif 'qpos' in self.h5:
            qpos = self._read_slice(self.h5['qpos'], row_indices)
            qvel = self._read_slice(self.h5['qvel'], row_indices)
            state = torch.tensor(np.concatenate([qpos, qvel], axis=-1), dtype=torch.float32)
        else:
            state = torch.zeros(len(row_indices), 0)

        obs = {
            'visual': pixels,
            'proprio': torch.zeros(len(row_indices), self.proprio_dim),
        }

        env_info = {'episode': int(ep)}
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
