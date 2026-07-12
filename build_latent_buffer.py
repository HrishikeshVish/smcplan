"""
Build a latent replay buffer for the HJB cost-to-go solver in gd.py.

Encodes dataset episodes through a pretrained world model and saves
(z_curr, z_next) transition pairs in the model's latent space.

Supports:
  - DINO-WM (pointmaze, pusht): uses proprio or visual latents
  - LeWM (reacher, cube): uses visual (CLS token) latents

Usage:
  # PointMaze (DINO-WM, proprio latents)
  python build_latent_buffer.py env=pointmaze

  # PushT (DINO-WM, visual latents)
  python build_latent_buffer.py env=pusht

  # Reacher (LeWM, visual latents)
  python build_latent_buffer.py env=lewm_reacher
"""

import os
import torch
import hydra
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

os.environ.setdefault("DATASET_DIR", "./data")


def encode_dino_wm_dataset(wm, dset, latent_key, device, max_episodes=8000):
    """Encode a DINO-WM dataset into latent transition pairs.

    Args:
        wm: Pretrained DINO world model
        dset: Dataset returning (obs_dict, actions, state, info) per episode
        latent_key: 'proprio' or 'visual' - which latent to extract
        device: torch device
        max_episodes: max episodes to process

    Returns:
        z_curr, z_next: (N_transitions, D) tensors
    """
    loader = DataLoader(dset, batch_size=1, shuffle=True)
    z_curr_list = []
    z_next_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding dataset")):
            if i >= max_episodes:
                break

            obs = batch[0]
            trans_obs = {
                "visual": obs["visual"].float().to(device),
                "proprio": obs["proprio"].float().to(device),
            }

            z_obs = wm.encode_obs(trans_obs)
            latent_ep = z_obs[latent_key].cpu().squeeze(0)  # (T, ..., D)

            # Flatten spatial dims if present (e.g. visual patches)
            if latent_ep.dim() > 2:
                latent_ep = latent_ep.mean(dim=-2)  # pool patches -> (T, D)

            if latent_ep.shape[0] > 1:
                z_curr_list.append(latent_ep[:-1])
                z_next_list.append(latent_ep[1:])

    z_curr = torch.cat(z_curr_list, dim=0)
    z_next = torch.cat(z_next_list, dim=0)
    return z_curr, z_next


def encode_lewm_dataset(wm, dset, device, max_episodes=8000):
    """Encode a LeWM dataset into latent transition pairs.

    Args:
        wm: LeWMWorldModel wrapper
        dset: DMControlDataset returning (obs_dict, actions, state, info)
        device: torch device
        max_episodes: max episodes to process

    Returns:
        z_curr, z_next: (N_transitions, D) tensors
    """
    z_curr_list = []
    z_next_list = []

    with torch.no_grad():
        for i in tqdm(range(min(len(dset), max_episodes)), desc="Encoding dataset"):
            obs, actions, state, info = dset[i]

            pixels = obs["visual"].unsqueeze(0).float().to(device)  # (1, T, C, H, W)
            trans_obs = {
                "visual": pixels,
                "proprio": torch.zeros(1, pixels.shape[1], 0, device=device),
            }

            z_obs = wm.encode_obs(trans_obs)
            # visual is (1, T, 1, D) for LeWM - squeeze batch and patch dims
            latent_ep = z_obs["visual"].squeeze(0).squeeze(1).cpu()  # (T, D)

            if latent_ep.shape[0] > 1:
                z_curr_list.append(latent_ep[:-1])
                z_next_list.append(latent_ep[1:])

    z_curr = torch.cat(z_curr_list, dim=0)
    z_next = torch.cat(z_next_list, dim=0)
    return z_curr, z_next


@hydra.main(version_base="1.2", config_path="conf", config_name="build_latent_buffer")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = cfg.env
    max_episodes = cfg.get("max_episodes", 8000)

    is_lewm = env_name.startswith("lewm_")

    if is_lewm:
        # Load LeWM model and DMControl dataset
        from plan import load_lewm_model, ALL_LEWM_ENVS
        base_path = os.path.dirname(os.path.abspath(__file__))
        short_name = env_name.replace("lewm_", "")
        env_cfg = OmegaConf.load(os.path.join(base_path, f"conf/env/{short_name}.yaml"))

        print(f"Loading LeWM model for {env_name}...")
        wm = load_lewm_model(env_cfg, ckpt_base_path=cfg.ckpt_base_path, device=device)
        wm.eval()

        print(f"Loading dataset...")
        _, dsets = hydra.utils.call(
            env_cfg.dataset,
            num_hist=env_cfg.get("history_size", 3),
            num_pred=1,
            frameskip=5,
        )
        dset = dsets["train"]

        z_curr, z_next = encode_lewm_dataset(wm, dset, device, max_episodes)
    else:
        # Load DINO-WM model and dataset
        from hydra.utils import to_absolute_path
        from utils import load_model

        dino_run_dir = Path(to_absolute_path(cfg.checkpoint_dir))
        model_cfg = OmegaConf.load(dino_run_dir / "hydra.yaml")
        model_ckpt = dino_run_dir / "checkpoints" / "model_latest.pth"

        print(f"Loading DINO-WM from {dino_run_dir}...")
        wm = load_model(model_ckpt, model_cfg, num_action_repeat=1, device=device)
        wm.to(device)
        wm.eval()
        wm.requires_grad_(False)

        dataset_path = to_absolute_path(model_cfg.env.dataset.data_path)

        if env_name == "pointmaze":
            from datasets.point_maze_dset import PointMazeDataset
            from datasets.img_transforms import default_transform
            dset = PointMazeDataset(
                data_path=dataset_path,
                n_rollout=None,
                transform=default_transform(img_size=224),
                normalize_action=True,
            )
        elif env_name == "pusht":
            from datasets.pusht_dset import PushTDataset
            from datasets.img_transforms import default_transform
            dset = PushTDataset(
                data_path=dataset_path + "/train",
                n_rollout=None,
                transform=default_transform(img_size=224),
                normalize_action=True,
            )
        else:
            raise ValueError(f"Unknown DINO-WM env: {env_name}")

        latent_key = cfg.get("latent_key", "proprio")
        z_curr, z_next = encode_dino_wm_dataset(wm, dset, latent_key, device, max_episodes)

    print(f"Transitions: {z_curr.shape[0]}, latent_dim: {z_curr.shape[1]}")

    buffer = {"z_curr": z_curr, "z_next": z_next}
    save_path = cfg.output_path
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(buffer, save_path)
    print(f"Saved replay buffer to {save_path}")


if __name__ == "__main__":
    main()
