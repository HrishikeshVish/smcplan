"""Aggregate trajectory GEOMETRY for the heat-V vs L2-objective ablation.

Answers "are the ablation's trajectories longer, or do they stop farther back?"
with numbers rather than the hand-picked videos. Reads the (x,y) paths dumped by
plan.py's GRASP_TRAJ_DUMP hook (pointmaze state is [qpos(2), qvel(2)]).

Metrics, per seed:
  path_len     sum of |dx| along the executed path -- total distance travelled.
  net_disp     |x_end - x_start| -- straight-line progress actually made.
  wander       path_len / net_disp. 1.0 = a straight shot; large = circling.
  final_gap    |x_end - x_goal|. success iff < 0.5 (point_maze_wrapper.py:48).
  closest_gap  min over the path of |x_t - x_goal| -- the best the run ever got.
               Separates "never got near" from "got there and drifted off".

Note on final_gap: the episode early-exits on success, so every solved run ends
just inside 0.5 by construction and the metric is censored for those. Compare it
on failures, and use path_len/wander for the solved ones.
"""
import os, glob, sys, statistics as st
import numpy as np

ARMS = ["none", "objective_raw"]


def metrics(p):
    z = np.load(p, allow_pickle=True)
    xy = np.asarray(z["e_states"])
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    g = np.asarray(z["state_g"]).ravel()[:2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(step.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    gaps = np.linalg.norm(xy - g[None, :], axis=1)
    return {
        "path_len": path,
        "net_disp": net,
        "wander": path / net if net > 1e-6 else float("nan"),
        "final_gap": float(gaps[-1]),
        "closest_gap": float(gaps.min()),
        "success": float(z["success"]),
        "exit": int(z["exit_step"]),
        "T": len(xy),
    }


def col(d, sel, k):
    return [d[s][k] for s in sel if s in d and not np.isnan(d[s][k])]


def line(name, a, b, sel, k, prec=2):
    va, vb = col(a, sel, k), col(b, sel, k)
    if not va or not vb:
        return
    w = sum(a[s][k] < b[s][k] for s in sel)
    print(f"  {name:<14} ours {st.mean(va):7.{prec}f} (med {st.median(va):6.{prec}f})   "
          f"L2obj {st.mean(vb):7.{prec}f} (med {st.median(vb):6.{prec}f})   "
          f"delta {st.mean(va)-st.mean(vb):+7.{prec}f}   ours lower {w}/{len(sel)}")


def main(root):
    data = {}
    for arm in ARMS:
        d = {}
        for p in glob.glob(f"{root}/{arm}/seed_*/traj.npz"):
            s = int(os.path.basename(os.path.dirname(p)).replace("seed_", ""))
            try:
                d[s] = metrics(p)
            except Exception as e:
                print(f"  skip {arm} seed {s}: {e}")
        data[arm] = d
    a, b = data["none"], data["objective_raw"]
    S = sorted(set(a) & set(b))
    print(f"paired seeds with trajectories: {len(S)}  "
          f"(ours {len(a)}, L2obj {len(b)})\n")
    if not S:
        return
    groups = [
        ("ALL PAIRED SEEDS", S),
        ("SEEDS L2OBJ FAILED", [s for s in S if b[s]["success"] < 0.5]),
        ("SEEDS BOTH SOLVED", [s for s in S if a[s]["success"] > 0.5 and b[s]["success"] > 0.5]),
    ]
    for title, sel in groups:
        if not sel:
            continue
        print(f"{title}  n={len(sel)}")
        line("path_len", a, b, sel, "path_len")
        line("net_disp", a, b, sel, "net_disp")
        line("wander", a, b, sel, "wander")
        line("final_gap", a, b, sel, "final_gap", 3)
        line("closest_gap", a, b, sel, "closest_gap", 3)
        line("exit_step", a, b, sel, "exit", 1)
        print()
    sr = {k: sum(v[s]["success"] for s in S) for k, v in (("ours", a), ("L2obj", b))}
    print(f"success: ours {sr['ours']:.0f}/{len(S)}   L2obj {sr['L2obj']:.0f}/{len(S)}")


if __name__ == "__main__":
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo, "sweep_fk", "ablate_trajs"))
