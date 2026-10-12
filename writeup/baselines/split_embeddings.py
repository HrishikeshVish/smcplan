"""Keep the train-split episodes of an embeddings file from GC-IDM's extractor.

Our planners and roadmaps use the first 90% of episodes (dsets/dmcontrol_dset.py,
split_ratio=0.9) and Cube draws its evaluation goals from the remaining 10%, so
the learned baselines are trained on the same first 90% only.

Usage: python split_embeddings.py <all.npz> <train.npz> [split_ratio]
"""
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
d = dict(np.load(src))
ep = d["episode_ids"]
n_episodes = int(ep.max()) + 1
n_train = int(n_episodes * ratio)
keep = ep < n_train
out = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(ep) else v)
       for k, v in d.items()}
np.savez(dst, **out)
print(f"{src}: {n_episodes} episodes, kept {n_train} ({keep.sum()} of {len(ep)} frames) -> {dst}")
