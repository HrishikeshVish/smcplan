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
from models.proprio_decoder import ProprioDecoder
from datasets.img_transforms import default_transform

# --- FIX: Provide a default DATASET_DIR before Hydra parses the config ---
os.environ.setdefault("DATASET_DIR", "./data")
# --- DINO-WM Imports ---
# Adjust these import paths if your DINO codebase has them located elsewhere
from datasets.point_maze_dset import PointMazeDataset
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
    # dset = PointMazeDataset(
    #     data_path=dataset_path,
    #     n_rollout=None, 
    #     transform=default_transform(img_size=224),
    #     normalize_action=True
    # )
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

    loader = DataLoader(dset, batch_size=1, shuffle=True)

    latents_list = []
    targets_list = []

    Z_curr_list = []
    Z_next_list = []
    Y_curr_list = []
    Y_next_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding Dataset")):
            if i > 8000: break
                
            obs = batch[0] 
            trans_obs = {
                "visual": obs["visual"].float().to(device),
                "proprio": obs["proprio"].float().to(device)
            }
            
            z_obs = wm.encode_obs(trans_obs)

            latent_ep = z_obs["proprio"].cpu().squeeze(0)    # Shape: [T_i, Dim]
            target_ep = trans_obs["proprio"].cpu().squeeze(0) # Shape: [T_i, Dim]
            
            # Slice into current and next states while we still know the episode boundaries!
            if latent_ep.shape[0] > 1:
                Z_curr_list.append(latent_ep[:-1])
                Z_next_list.append(latent_ep[1:])
                
                Y_curr_list.append(target_ep[:-1])
                Y_next_list.append(target_ep[1:])

    # Concatenate into massive flat 2D tensors -> Shape: [Total_Transitions, Dim]
    Z_curr_all = torch.cat(Z_curr_list, dim=0)
    Z_next_all = torch.cat(Z_next_list, dim=0)
    Y_curr_all = torch.cat(Y_curr_list, dim=0)
    Y_next_all = torch.cat(Y_next_list, dim=0)

    buffer_dict = {
        "z_curr": Z_curr_all,
        "z_next": Z_next_all,
        "y_curr": Y_curr_all,
        "y_next": Y_next_all
    }

    torch.save(buffer_dict, "pusht_offline_replay_buffer.pt")
    print(f"Saved {Z_curr_all.shape[0]} valid transitions.")
    exit()


    # If the World Model returns sequences [Batch, Time, Dim], flatten them to [Batch * Time, Dim]
    if X_raw.dim() > 2:
        X = X_raw.view(-1, X_raw.shape[-1])
        Y = Y_raw.view(-1, Y_raw.shape[-1])
    else:
        X, Y = X_raw, Y_raw

    print(f"Extraction complete! Dataset shape: X={X.shape}, Y={Y.shape}")

    # --- 3. Train the Decoder ---
    print("Training Proprio Decoder...")
    decoder = ProprioDecoder(latent_dim=X.shape[-1], proprio_dim=Y.shape[-1]).to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Create a fast TensorDataset for the MLP
    train_dset = TensorDataset(X.to(device), Y.to(device))
    train_loader = DataLoader(train_dset, batch_size=1024, shuffle=True)

    epochs = 100
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
    save_path = "/scratch/gilbreth/hviswan/dino_wm/proprio_decoder_weights.pth"
    torch.save(decoder.state_dict(), save_path)
    print(f"Decoder training finished. Weights saved to {save_path}")

if __name__ == "__main__":
    main()