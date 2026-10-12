"""Train LeFlow (third_party/LeFlow, official code) on precomputed LeWM embeddings.

The official trainer (train_latent_planner.py) encodes every clip's frames with
the frozen LeWM at every step. The encoder is frozen and in eval mode and the
image preprocessing (ImageNet normalization, resize to 224 on 224-px frames) is
deterministic, so the embeddings are the same at every epoch; we compute them
once with GC-IDM's extractor, which gives bit-identical embeddings to LeWM's
encode (checked), and train on them. Everything else is the official code and
configuration: LatentPathFlow, InverseDynamics, flow_matching_loss,
inverse_dynamics_loss, lewm_consistency_loss and checkpoint_payload are
imported unchanged, and the hyperparameters are those of
config/train/latent_planner.yaml with the README's overrides for the main
models (max_horizon=20, action_block=5, epochs=10, batch_size=128).

A clip is built as stable_worldmodel's HDF5Dataset builds it for the official
trainer (frameskip=5, num_steps=H+1): frames start, start+5, ..., start+5H, and
the 5(H+1) raw actions from `start`, normalized per dimension with the mean and
(unbiased) std of the whole action column (LeFlow utils.get_column_normalizer),
reshaped to (H+1, 5*action_dim), of which the first H chunks are used.

Usage (leflow venv):
  python train_leflow_cached.py --embeddings E.npz --h5 D.h5 --lewm W.pt \
      --horizon H --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

LEFLOW = Path(__file__).resolve().parents[2] / "third_party" / "LeFlow"
sys.path.insert(0, str(LEFLOW))
from latent_planner import (  # noqa: E402
    InverseDynamics, LatentPathFlow, checkpoint_payload, flow_matching_loss,
    inverse_dynamics_loss, lewm_consistency_loss, load_lewm, smoothness_loss)

# config/train/latent_planner.yaml + README overrides for the main models
CFG = dict(seed=3072, train_split=0.9, epochs=10, log_interval=50, val_batches=20,
           grad_clip_norm=1.0, lewm_history_size=3, batch_size=128,
           max_horizon=20, action_block=5,
           flow=dict(hidden_dim=512, depth=4, time_dim=64, dropout=0.0),
           inverse_dynamics=dict(hidden_dim=512, depth=3, dropout=0.0),
           loss=dict(flow=1.0, inverse=1.0, consistency=0.1, smoothness=0.0),
           optimizer=dict(lr=1e-4, weight_decay=1e-4))


class CachedClips(torch.utils.data.Dataset):
    def __init__(self, emb, act_norm, ep_ids, horizon, fs):
        self.emb, self.act, self.H, self.fs = emb, act_norm, horizon, fs
        span = (horizon + 1) * fs
        # episode boundaries (rows are contiguous per episode)
        change = np.flatnonzero(np.diff(ep_ids)) + 1
        starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(ep_ids)]])
        self.idx = np.concatenate([np.arange(s, e - span + 1) for s, e in zip(starts, ends)
                                   if e - s >= span])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        g = int(self.idx[i]); H, fs = self.H, self.fs
        z = self.emb[g: g + fs * H + 1: fs]                       # (H+1, D)
        a = self.act[g: g + fs * (H + 1)].reshape(H + 1, -1)[:H]  # (H, fs*d)
        return torch.from_numpy(z), torch.from_numpy(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True)
    p.add_argument("--h5", required=True, help="dataset, for the action normalizer")
    p.add_argument("--lewm", required=True, help="frozen LeWM weights.pt")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-steps", type=int, default=0, help="smoke test only")
    a = p.parse_args()
    torch.manual_seed(CFG["seed"])
    dev = torch.device(a.device)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # utils.get_column_normalizer: whole action column, NaN rows dropped, torch std
    with h5py.File(a.h5, "r") as f:
        col = torch.from_numpy(np.array(f["action"][:]))
    col = col[~torch.isnan(col).any(dim=1)]
    mean, std = col.mean(0, keepdim=True), col.std(0, keepdim=True)
    json.dump({"mean": mean[0].tolist(), "std": std[0].tolist()},
              open(out / "action_normalizer.json", "w"))

    d = np.load(a.embeddings)
    emb = d["embeddings"].astype(np.float32)
    act = torch.from_numpy(d["actions"].astype(np.float64))
    act = torch.nan_to_num(((act - mean) / std).float(), nan=0.0).numpy()
    ds = CachedClips(emb, act, d["episode_ids"], a.horizon, CFG["action_block"])
    gen = torch.Generator().manual_seed(CFG["seed"])
    n_tr = int(round(len(ds) * CFG["train_split"]))
    tr, va = torch.utils.data.random_split(ds, [n_tr, len(ds) - n_tr], generator=gen)
    kw = dict(batch_size=CFG["batch_size"], num_workers=6, pin_memory=True)
    tl = torch.utils.data.DataLoader(tr, shuffle=True, drop_last=True, generator=gen, **kw)
    vl = torch.utils.data.DataLoader(va, shuffle=False, **kw)
    print(f"clips: {len(ds)} ({n_tr} train), steps/epoch {len(tl)}", flush=True)

    lewm = load_lewm(a.lewm).to(dev).eval(); lewm.requires_grad_(False)
    flow = LatentPathFlow(latent_dim=emb.shape[1], max_horizon=CFG["max_horizon"],
                          **CFG["flow"]).to(dev)
    idm = InverseDynamics(latent_dim=emb.shape[1], action_dim=act.shape[1] * CFG["action_block"],
                          **CFG["inverse_dynamics"]).to(dev)
    params = list(flow.parameters()) + list(idm.parameters())
    opt = torch.optim.AdamW(params, **CFG["optimizer"])
    L = CFG["loss"]

    def step(z, acts):
        z, acts = z.to(dev, non_blocking=True), acts.to(dev, non_blocking=True)
        lf = flow_matching_loss(flow, z)
        li, pred = inverse_dynamics_loss(idm, z, acts)
        lc = lewm_consistency_loss(lewm, z, pred, history_size=CFG["lewm_history_size"])
        ls = smoothness_loss(z)
        tot = L["flow"] * lf + L["inverse"] * li + L["consistency"] * lc + L["smoothness"] * ls
        return tot, dict(flow=lf.item(), inverse=li.item(), consistency=lc.item())

    gstep, t0 = 0, time.time()
    for ep in range(CFG["epochs"]):
        flow.train(); idm.train()
        for z, acts in tl:
            tot, m = step(z, acts)
            opt.zero_grad(set_to_none=True); tot.backward()
            torch.nn.utils.clip_grad_norm_(params, CFG["grad_clip_norm"]); opt.step()
            gstep += 1
            if gstep % CFG["log_interval"] == 0:
                print(f"epoch={ep+1} step={gstep} loss={tot.item():.4f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in m.items())
                      + f" {time.time()-t0:.0f}s", flush=True)
            if a.max_steps and gstep >= a.max_steps:
                break
        flow.eval(); idm.eval(); vs, n = {}, 0
        with torch.no_grad():
            for i, (z, acts) in enumerate(vl):
                _, m = step(z, acts)
                for k, v in m.items():
                    vs[k] = vs.get(k, 0.0) + v * len(z)
                n += len(z)
                if i + 1 >= CFG["val_batches"]:
                    break
        print(f"epoch={ep+1} validation " + " ".join(f"{k}={v/n:.4f}" for k, v in vs.items()), flush=True)
        payload = checkpoint_payload(lewm_checkpoint=a.lewm, action_block=CFG["action_block"],
                                     flow=flow, inverse_dynamics=idm,
                                     cfg=dict(CFG, horizon=a.horizon, embeddings=a.embeddings))
        torch.save(payload, out / "latent_planner.pt")
        if a.max_steps and gstep >= a.max_steps:
            break
    print(f"done {time.time()-t0:.0f}s -> {out/'latent_planner.pt'}", flush=True)


if __name__ == "__main__":
    main()
