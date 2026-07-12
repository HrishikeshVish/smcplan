import os
import torch
import hydra
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from pathlib import Path
from omegaconf import OmegaConf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models.proprio_decoder import ProprioDecoder, SpatialStateDecoder
from datasets.img_transforms import default_transform

# --- FIX: Provide a default DATASET_DIR before Hydra parses the config ---
os.environ.setdefault("DATASET_DIR", "./data")
# --- DINO-WM Imports ---
# Adjust these import paths if your DINO codebase has them located elsewhere
#from datasets.point_maze_dset import PointMazeDataset
from datasets.pusht_dset import PushTDataset
from utils import load_model 

from hydra.core.hydra_config import HydraConfig
import wandb

from hydra.utils import to_absolute_path 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from preprocessor import Preprocessor

@hydra.main(version_base="1.2", config_path="conf", config_name="train_proprio_decoder_pusht")
def main(cfg):
    # --- 2. Setup your Environment (Assuming 'dset', 'wm', 'preprocessor' are loaded) ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Loading Pre-Trained DINO-WM...")
    
    # 1. Point to your specific DINO run directory
    dino_run_dir = Path(to_absolute_path("./checkpoints/pusht"))
    
    # 2. Load the EXACT config that was used to train the DINO model
    model_cfg_path = dino_run_dir / "hydra.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Could not find DINO config at {model_cfg_path}. Ensure the hydra.yaml is in the run directory.")
    
    model_cfg = OmegaConf.load(model_cfg_path)
    
    # 3. Load the model checkpoint
    model_ckpt = dino_run_dir / "checkpoints" / "model_latest.pth"
    
    # Initialize using the true config
    wm = load_model(model_ckpt, model_cfg, num_action_repeat=1, device=device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm.to(device)
    wm.eval()
    wm.requires_grad_(False)

    print("Extracting dataset into latent space...")
    # Use a large batch size since we are just doing forward passes
    print("Loading Offline Dataset for Implicit Geometry...")
    # Use the dataset path defined in the loaded DINO config, 
    # ensuring it's an absolute path so Hydra doesn't get lost
    dataset_path = to_absolute_path(model_cfg.env.dataset.data_path)
    dset = PushTDataset(
        data_path=dataset_path+'/train',
        n_rollout=None, 
        transform=default_transform(img_size=224),
        normalize_action=True
    )
    all_states = dset.states.view(-1, dset.state_dim) 
    # Flatten to get all points
    all_states = dset.states.view(-1, dset.state_dim).cpu().numpy()
    preprocessor = Preprocessor(
            action_mean=dset.action_mean,
            action_std=dset.action_std,
            state_mean=dset.state_mean,
            state_std=dset.state_std,
            proprio_mean=dset.proprio_mean,
            proprio_std=dset.proprio_std,
            transform=dset.transform,
        )

    loader = DataLoader(dset, batch_size=1, shuffle=False)

    latents_list = []
    targets_list = []
    

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding Dataset")):
            # Adjust unpacking based on your exact dataset __getitem__ return structure
            # Assuming batch[0] is obs_dict
            #print("Batch keys: ", batch[0].keys())

            if i >=500:
                break
            obs = batch[0] 
            trans_obs = {
                "visual": obs["visual"].float().to(device),
                "proprio": obs["proprio"].float().to(device)
            }
            # print("Original obs keys: ", obs.keys())
            # print("Original obs 'proprio' shape: ", obs['proprio'].shape)
            # print("Original obs 'visual' shape: ", obs['visual'].shape)
            
            # 1. Transform to normalized physical space (This is our Ground Truth Target)
            #trans_obs = preprocessor.transform_obs(obs)
            #trans_obs = trans_obs.to(device)
            
            # 2. Encode to latent space (This is our Input)
            z_obs = wm.encode_obs(trans_obs)
            #print("Encoded latent shape: ", z_obs["visual"].shape)
            #print("Target state shape: ", z_obs["proprio"].shape)
            
            # Store them
            B, T, H, W = z_obs["visual"].shape
            pooled_visual = z_obs['visual']
            latents_list.append(pooled_visual.cpu())
            raw_target = obs["state"][...,:5]  # Assuming the first 5 dims are the ones we want to predict
            state_mean = dset.state_mean[:5].cpu()
            state_std = dset.state_std[:5].cpu()
            normalized_target = (raw_target - state_mean) / state_std
            targets_list.append(normalized_target.cpu())
    # Reshape [1, T, D] -> [T, D] for every trajectory in the list
    latents_flat = [x.view(-1, x.shape[-2], x.shape[-1]) for x in latents_list]
    targets_flat = [y.view(-1, y.shape[-1]) for y in targets_list]

    # Now perfectly concatenate all frames into one massive 2D tensor
    X = torch.cat(latents_flat, dim=0)
    Y = torch.cat(targets_flat, dim=0)

    print(f"Extraction complete! Dataset shape: X={X.shape}, Y={Y.shape}")

    # --- 3. Train the Decoder ---
    print("Training Proprio Decoder...")
    #decoder = ProprioDecoder(latent_dim=X.shape[-1], proprio_dim=Y.shape[-1]).to(device)
    decoder = SpatialStateDecoder(proprio_dim=Y.shape[-1]).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Create a fast TensorDataset for the MLP
    train_dset = TensorDataset(X.to(device), Y.to(device))
    
    # CRITICAL SPEED FIX: Do NOT use batch_size=1 here! 
    # The frames are flat now. Use 256 or 1024 so the MLP trains in seconds.
    train_loader = DataLoader(train_dset, batch_size=256, shuffle=True)

    epochs = 400
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            preds = decoder(batch_x)
            loss = criterion(preds, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | MSE Loss: {avg_loss:.6f}")

    # --- 4. Save the Weights ---
    save_path = "/scratch/gilbreth/hviswan/dino_wm/proprio_decoder_weights_pusht.pth"
    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder training finished. Weights saved to {save_path}")

if __name__ == "__main__":
    main()