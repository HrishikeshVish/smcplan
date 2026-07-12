"""
Build a latent replay buffer for the HJB cost-to-go solver in gd.py.

Encodes dataset episodes through a pretrained world model and saves
(z_curr, z_next) transition pairs in the model's latent space.

Usage:
  # Reacher (LeWM, visual latents)
  python build_latent_buffer.py --env lewm_reacher

  # PointMaze (DINO-WM, proprio latents)
  python build_latent_buffer.py --env pointmaze --checkpoint-dir ./checkpoints/pointmaze

  # PushT (DINO-WM, proprio latents)
  python build_latent_buffer.py --env pusht --checkpoint-dir ./checkpoints/pusht --max-episodes 500
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

os.environ.setdefault("DATASET_DIR", "/scratch/gilbreth/hviswan/grasp/data")


def encode_dino_wm_dataset(wm, dset, latent_key, device, max_episodes=8000):
    """Encode a DINO-WM dataset into latent transition pairs."""
    from torch.utils.data import DataLoader
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
    """Encode a LeWM dataset into latent transition pairs."""
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


def main():
    parser = argparse.ArgumentParser(description="Build latent replay buffer for HJB solver")
    parser.add_argument("--env", required=True, help="Environment: pointmaze, pusht, lewm_reacher, lewm_cube")
    parser.add_argument("--checkpoint-dir", default=None, help="DINO-WM checkpoint dir (ignored for LeWM)")
    parser.add_argument("--ckpt-base-path", default="/scratch/gilbreth/hviswan/grasp/checkpoints", help="LeWM checkpoint base path")
    parser.add_argument("--output-path", default=None, help="Output .pt path (auto-generated if not set)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to encode (default: all)")
    parser.add_argument("--latent-key", default="proprio", help="Latent key for DINO-WM: proprio or visual")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = args.env
    is_lewm = env_name.startswith("lewm_")

    if args.output_path is None:
        args.output_path = f"/scratch/gilbreth/hviswan/grasp/latent_buffers/{env_name}_buffer.pt"

    if is_lewm:
        from plan import load_lewm_model
        import hydra

        base_path = os.path.dirname(os.path.abspath(__file__))
        short_name = env_name.replace("lewm_", "")
        env_cfg = OmegaConf.load(os.path.join(base_path, f"conf/env/{short_name}.yaml"))

        print(f"Loading LeWM model for {env_name}...")
        wm = load_lewm_model(env_cfg, ckpt_base_path=args.ckpt_base_path, device=device)
        wm.eval()

        print(f"Loading dataset...")
        _, dsets = hydra.utils.call(
            env_cfg.dataset,
            num_hist=env_cfg.get("history_size", 3),
            num_pred=1,
            frameskip=5,
        )
        dset = dsets["train"]

        n_episodes = len(dset)
        print(f"Dataset has {n_episodes} episodes")

        # Encode one episode to get latent dim
        obs, _, _, _ = dset[0]
        with torch.no_grad():
            pixels = obs["visual"].unsqueeze(0).float().to(device)
            z = wm.encode_obs({"visual": pixels, "proprio": torch.zeros(1, pixels.shape[1], 0, device=device)})
            latent_dim = z["visual"].shape[-1]
        print(f"Latent dimension: {latent_dim}")
        print(f"Episode length: {obs['visual'].shape[0]} frames")

        max_episodes = args.max_episodes if args.max_episodes is not None else n_episodes
        print(f"Encoding {max_episodes} / {n_episodes} episodes...")

        z_curr, z_next = encode_lewm_dataset(wm, dset, device, max_episodes)
    else:
        from hydra.utils import to_absolute_path
        from utils import load_model

        checkpoint_dir = args.checkpoint_dir
        if checkpoint_dir is None:
            checkpoint_dir = f"./checkpoints/{env_name}"

        dino_run_dir = Path(os.path.abspath(checkpoint_dir))
        model_cfg = OmegaConf.load(dino_run_dir / "hydra.yaml")
        model_ckpt = dino_run_dir / "checkpoints" / "model_latest.pth"

        print(f"Loading DINO-WM from {dino_run_dir}...")
        wm = load_model(model_ckpt, model_cfg, num_action_repeat=1, device=device)
        wm.to(device)
        wm.eval()
        wm.requires_grad_(False)

        dataset_path = str(dino_run_dir.parent / model_cfg.env.dataset.data_path) if not os.path.isabs(model_cfg.env.dataset.data_path) else model_cfg.env.dataset.data_path

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

        n_episodes = len(dset)
        print(f"Dataset has {n_episodes} episodes")

        max_episodes = args.max_episodes if args.max_episodes is not None else n_episodes
        print(f"Encoding {max_episodes} / {n_episodes} episodes...")

        latent_key = args.latent_key
        z_curr, z_next = encode_dino_wm_dataset(wm, dset, latent_key, device, max_episodes)

    print(f"Total transitions: {z_curr.shape[0]}, latent_dim: {z_curr.shape[1]}")

    buffer = {"z_curr": z_curr, "z_next": z_next}
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(buffer, args.output_path)
    print(f"Saved replay buffer to {args.output_path}")


if __name__ == "__main__":
    main()
