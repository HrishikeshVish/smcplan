"""Compare the EM planner (planning/em_smc.py) against the recorded FKSMCPlanner
baseline on the same pointmaze H5 seeds, same budget, same population size.

result.txt in both trees is "<success 0|1> <exit_step>".
"""
import os
import sys

BASE = "/home/hviswan/Documents/grasp"
FK = f"{BASE}/sweep_fk/table/pointmaze/fk_H5"
EM = f"{BASE}/sweep_emsmc/pointmaze_H5"


def load(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for name in os.listdir(d):
        if not name.startswith("seed_"):
            continue
        f = os.path.join(d, name, "result.txt")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        parts = open(f).read().split()
        try:
            out[int(name[5:])] = (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            continue
    return out


fk = load(FK)
arms = {a: load(f"{EM}/{a}") for a in sorted(os.listdir(EM))} if os.path.isdir(EM) else {}

for arm, res in arms.items():
    common = sorted(set(res) & set(fk))
    if not common:
        print(f"{arm}: no overlapping seeds yet")
        continue
    fs = sum(fk[s][0] for s in common)
    es = sum(res[s][0] for s in common)
    # mean iterations to success, over seeds BOTH solve (the only fair set)
    both_ok = [s for s in common if fk[s][0] and res[s][0]]
    fi = sum(fk[s][1] for s in both_ok) / len(both_ok) if both_ok else float("nan")
    ei = sum(res[s][1] for s in both_ok) / len(both_ok) if both_ok else float("nan")
    win = sum(1 for s in both_ok if res[s][1] < fk[s][1])
    print(f"\n=== EM-SMC arm '{arm}' vs FK, n={len(common)} shared seeds ===")
    print(f"  success      FK {fs}/{len(common)} ({100*fs/len(common):.0f}%)   "
          f"EM {es}/{len(common)} ({100*es/len(common):.0f}%)")
    print(f"  mean iters   FK {fi:.1f}   EM {ei:.1f}   "
          f"(over {len(both_ok)} seeds both solve)")
    print(f"  EM faster on {win}/{len(both_ok)}")
    only_em = [s for s in common if res[s][0] and not fk[s][0]]
    only_fk = [s for s in common if fk[s][0] and not res[s][0]]
    if only_em:
        print(f"  EM solves, FK fails: {only_em}")
    if only_fk:
        print(f"  FK solves, EM fails: {only_fk}")
    print("  seed   FK      EM")
    for s in common:
        mark = "" if fk[s][0] == res[s][0] else "   <-- differs"
        print(f"  {s:5d}  {fk[s][0]} {fk[s][1]:3d}   {res[s][0]} {res[s][1]:3d}{mark}")
