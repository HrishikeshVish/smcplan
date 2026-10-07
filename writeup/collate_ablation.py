"""Collate the kernel ablation into the paper table.

Two roots, deliberately kept separate: sweep_aft/ablation (seeds where FK beat the
baselines) and sweep_aft/ablation_fkfail (seeds FK failed in the main table sweep).
Reporting only the first would condition the table on FK success, so the summary
prints both regimes and their union.

result.txt holds "<ok> <exit_step> <seconds>"; exit_step is the env step at which
the episode ended (== solve step when ok=1, == the step cap when ok=0).
"""
import os, glob, sys
from collections import defaultdict

ROOTS = {"FK-success seeds": "sweep_aft/ablation",
         "FK-failure seeds": "sweep_aft/ablation_fkfail"}
KERNELS = ["fk", "mala", "hmc", "aft"]
LABEL = {"fk": "FK (ours)", "mala": "MALA", "hmc": "HMC", "aft": "AFT"}
ENVS = ["pointmaze", "cube", "reacher", "pusht"]


def load(root):
    out = defaultdict(dict)                      # (env,H,seed) -> kernel -> (ok, step)
    for k in KERNELS:
        for f in glob.glob(f"{root}/{k}/*/H*/seed_*/result.txt"):
            p = f.split("/")
            env, H, seed = p[-4], int(p[-3][1:]), int(p[-2].split("_")[1])
            v = open(f).read().split()
            if not v:
                continue
            step = v[1] if len(v) > 1 else "?"
            out[(env, H, seed)][k] = (v[0] == "1", step)
    return out


def block(title, data):
    print(f"\n### {title}")
    if not data:
        print("  (no results yet)")
        return {}
    hdr = f"{'env':<10} {'H':>3} {'seed':>5} " + " ".join(f"{LABEL[k]:>10}" for k in KERNELS)
    print(hdr); print("-" * len(hdr))
    tot = {k: [0, 0] for k in KERNELS}           # [solved, run]
    for env in ENVS:
        for (e, H, s) in sorted(data, key=lambda t: (t[1], t[2])):
            if e != env:
                continue
            cells = []
            for k in KERNELS:
                r = data[(e, H, s)].get(k)
                if r is None:
                    cells.append(f"{'-':>10}")
                else:
                    tot[k][1] += 1; tot[k][0] += r[0]
                    cells.append(f"{('OK ' + r[1]) if r[0] else 'x':>10}")
            print(f"{e:<10} {H:>3} {s:>5} " + " ".join(cells))
    print("-" * len(hdr))
    print(f"{'TOTAL':<10} {'':>3} {'':>5} " +
          " ".join(f"{str(tot[k][0]) + '/' + str(tot[k][1]):>10}" for k in KERNELS))
    return tot


grand = {k: [0, 0] for k in KERNELS}
for title, root in ROOTS.items():
    t = block(f"{title}  ({root})", load(root))
    for k, v in t.items():
        grand[k][0] += v[0]; grand[k][1] += v[1]
print("\n### Both regimes pooled")
for k in KERNELS:
    n, d = grand[k]
    print(f"  {LABEL[k]:<10} {n}/{d}" + (f"  ({100*n/d:.0f}%)" if d else ""))
