"""Figures 6-8: rolled-out plans as a sequence of frames.

One row per planner. Column 0 is t=0, identical across rows because the initial
condition is shared; the remaining executed columns are sampled at equal
FRACTIONS of each row's OWN episode, because the rows run for different numbers
of steps (a run stops when it succeeds). Sampling absolute timesteps truncated to
the shorter episode, as an earlier version did, showed the two rows at different
fractions of their own progress and so compared unlike points. A GOAL frame
(ground truth) closes every row.

The goal frame is not decoration. Without it these strips show a scene changing
with no indication of what was being aimed at, so a reader cannot tell a
successful rollout from a failed one -- on Cube in particular the block simply
appears to be put down somewhere. The goal image is the evaluator's own
obs_g, captured by plan.py alongside the executed frames.

Frames come from the executed rollout captured by plan.py's GRASP_TRAJ_DUMP_FRAMES
hook. Cube frames are re-rendered from the stored qpos/qvel at a brighter camera
(see render_cube_frames) because the env's own view is too dark to read in print
(mean brightness ~58/255).

Usage:  python figs/frame_strip.py ENV
        ENV in {pointmaze, pusht, cube}
"""
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, rounded_frame, round_image, save, outlined, PINK, PINK_DARK,  # noqa: E402
                        GREY_DARK, GOLD, INK)
OURS, GD = PINK, GREY_DARK

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCOLS = 6


def _goal_img(z):
    if "goal_visual" not in z.files:
        return None
    g = np.asarray(z["goal_visual"])
    while g.ndim > 3:                      # (1,1,H,W,C) -> (H,W,C)
        g = g[0]
    if g.shape[0] in (1, 3) and g.ndim == 3 and g.shape[0] < g.shape[-1]:
        g = np.transpose(g, (1, 2, 0))     # CHW -> HWC
    return g.astype(np.uint8)


def load_frames(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    f = z["frames"]
    f = f[0] if f.ndim == 5 else f
    s = np.asarray(z["e_states"])
    s = s[0] if s.ndim == 3 else s
    return (f, s, np.asarray(z["state_g"]).ravel(), float(z["success"]),
            int(z["exit_step"]), _goal_img(z))


def render_cube_frames(npz_path, size=320, cam=2, alpha=2.2, beta=18):
    """Re-render cube from stored qpos/qvel; the default camera is too dark."""
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    z = np.load(npz_path, allow_pickle=True)
    s = np.asarray(z["e_states"])
    s = s[0] if s.ndim == 3 else s
    env = gym.make("swm/OGBCube-v0")
    env.reset(seed=0)
    u = env.unwrapped
    m, d = u.model, u.data
    r = mujoco.Renderer(m, height=224, width=224)
    out = []
    for t in range(len(s)):
        d.qpos[:] = s[t, :m.nq]
        d.qvel[:] = s[t, m.nq:]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        img = cv2.resize(r.render(), (size, size), interpolation=cv2.INTER_LANCZOS4)
        out.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))
    # Goal frame: render the GOAL state through the same camera, so the target
    # block pose is unambiguous rather than inferred from a marker in the scene.
    g = np.asarray(z["state_g"]).ravel()
    d.qpos[:] = g[:m.nq]
    d.qvel[:] = g[m.nq:]
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cam)
    gimg = cv2.resize(r.render(), (size, size), interpolation=cv2.INTER_LANCZOS4)
    gimg = cv2.convertScaleAbs(gimg, alpha=alpha, beta=beta)
    r.close()
    env.close()
    return (np.stack(out), s, g, float(z["success"]), int(z["exit_step"]), gimg)


# 2026-10-07: equal-particle protocol of Table 1 (writeup/render_frames_eq12.sh).
# Main text: ours (JALA-EM) vs GD with 12 particles, on a seed where ours solves
# and GD-12 does not in the 100-seed sweep. Appendix (*_app): the seeds of the
# earlier strips, with GD at 1 particle (the earlier renders) and at 12.
R = f"{ROOT}/figs/renders"
SPECS = {
    "pointmaze": dict(
        title="point-maze  H=10  (seed 88)",
        rows=[("ours", f"{R}/pm_ours12_s88/traj.npz"),
              ("gd",   f"{R}/pm_gd12_s88/traj.npz")]),
    "pusht": dict(
        title="push-t  H=5  (seed 10)",
        rows=[("ours", f"{R}/pusht_ours12_s10/traj.npz"),
              ("gd",   f"{R}/pusht_gd12_s10/traj.npz")]),
    "pointmaze_app": dict(
        title="point-maze  H=10  (seed 24)",
        rows=[("ours", f"{R}/pm_ours12_s24/traj.npz"),
              ("gd1",  f"{R}/pm_gd/traj.npz"),
              ("gd12", f"{R}/pm_gd12_s24/traj.npz")]),
    "pusht_app": dict(
        title="push-t  H=5  (seed 20)",
        rows=[("ours", f"{R}/pusht_ours12_s20/traj.npz"),
              ("gd1",  f"{R}/pusht_gd/traj.npz"),
              ("gd12", f"{R}/pusht_gd12_s20/traj.npz")]),
    "cube": dict(
        title="cube  (seed 17)", cube=True,
        rows=[("ours", f"{ROOT}/figs/renders/cube_fk/traj.npz"),
              ("gd",   f"{ROOT}/figs/renders/cube_gd/traj.npz")]),
}
LABEL = {"ours": "ours", "gd": "GD", "gd1": "GD (1)", "gd12": "GD (12)"}
COLOR = {"ours": OURS, "gd": GD, "gd1": GD, "gd12": GD}


def iters_to_success(npz_path):
    """1 + index of the first successful evaluation in the run's run.log (the
    paper's iteration count), or None. exit_step in the dump is not used: its
    convention differs between planners (0-based for JALA-EM, 1-based for GD)."""
    try:
        lines = [l for l in open(os.path.join(os.path.dirname(npz_path), "run.log"),
                                 errors="ignore") if "Success rate:" in l]
    except OSError:
        return None
    for i, l in enumerate(lines):
        if "1.0" in l.split("Success rate:")[1]:
            return i + 1
    return None


def ball_track(frames):
    """Pixel centroid of the green point-maze ball in every frame (NaN if unseen)."""
    f = frames.astype(np.int16)
    g = (f[..., 1] > 150) & (f[..., 1] - f[..., 0] > 60) & (f[..., 1] - f[..., 2] > 60)
    out = np.full((len(frames), 2), np.nan)
    for i, m in enumerate(g):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[i] = xs.mean(), ys.mean()
    return out


def main(env):
    use_style(7)
    spec = SPECS[env]
    loader = render_cube_frames if spec.get("cube") else load_frames
    rows = [(k, loader(p)) for k, p in spec["rows"]]
    its = {k: iters_to_success(p) for k, p in spec["rows"]}
    # Equal fractions of each row's own episode; column 0 is always t=0.
    FRACS = np.linspace(0.0, 1.0, NCOLS)

    def row_idx(frames, exit_step):
        # Span the WHOLE stored trajectory. e_states[0] == state_0 and the final
        # entry is the state the evaluator scores, so the last frame is the
        # outcome; truncating at exit_step cut a solved run off mid-flight,
        # after it had passed the goal but before it settled, which read as the
        # agent overshooting. exit_step counts optimisation iterations and does
        # not index this array.
        return (FRACS * (len(frames) - 1)).round().astype(int)

    FW = 7.2
    x0, tw, gap, goalgap = 0.105, 0.108, 0.010, 0.030
    nrow = len(rows)
    TOP, ROWH, RGAP, BOT = 0.42, tw * FW, 0.166, 0.40     # inches
    FH = TOP + nrow * ROWH + (nrow - 1) * RGAP + BOT
    fig = plt.figure(figsize=(FW, FH))
    trail = env.startswith("pointmaze")
    th = tw * FW / FH
    for r, (key, (frames, states, goal, ok, exit_step, gimg)) in enumerate(rows):
        y = (FH - TOP - (r + 1) * ROWH - r * RGAP) / FH
        cxy = ball_track(frames) if trail else None
        gxy = ball_track(gimg[None])[0] if (trail and gimg is not None) else None
        col = PINK_DARK if key == "ours" else GREY_DARK
        fig.text(x0 - 0.015, y + th * 0.62, LABEL[key], ha="right", va="center", fontsize=8.5,
                 color=col)
        fig.text(x0 - 0.015, y + th * 0.36,
                 f"solved\n@ iter {its[key]}" if its[key] is not None else "not\nsolved",
                 ha="right", va="center", fontsize=6, color=col, linespacing=1.1)
        idx = row_idx(frames, exit_step)
        for c, t in enumerate(idx):
            ax = fig.add_axes([x0 + c * (tw + gap), y, tw, th])
            round_image(ax, frames[t], r=0.07, ec=PINK if key == "ours" else "#9A9A9A",
                        lw=1.4 if key == "ours" else 0.8)
            if gxy is not None:
                ax.scatter(*gxy, marker="*", s=34, fc=GOLD, ec=INK, lw=0.5, zorder=5, alpha=0.9)
            if cxy is not None and t > 0:
                ax.plot(cxy[:t + 1, 0], cxy[:t + 1, 1], color="white", lw=1.1,
                        solid_capstyle="round", path_effects=outlined(2.2), zorder=6)
                ax.scatter(*cxy[t], s=9, fc=PINK if key == "ours" else "white", ec=INK,
                           lw=0.6, zorder=7)
            # No absolute index is printed: the dump is indexed by optimisation
            # iteration, not by rollout step, so a number here would be read as
            # an episode timestep and mislead. The fraction labels carry the
            # ordering, and the per-row "solved @ iter" annotation carries the
            # iteration count where it is correctly named.
            if r == 0:
                lab = "start" if c == 0 else ("end" if c == NCOLS - 1
                                               else f"{FRACS[c]:.0%}")
                fig.text(x0 + c * (tw + gap) + tw / 2, y + th + 0.089 / FH, lab,
                         ha="center", va="center", fontsize=6.5)
        # arrow of time
        if r == len(rows) - 1:
            fig.patches.append(plt.matplotlib.patches.FancyArrowPatch(
                (x0, y - 0.1275 / FH), (x0 + NCOLS * (tw + gap) - gap, y - 0.1275 / FH),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.8,
                color=INK))
            fig.text(x0 + (NCOLS * (tw + gap)) / 2, y - 0.23 / FH,
                     "executed rollout, as a fraction of each row's own episode",
                     ha="center", va="center", fontsize=6.5)
        # goal
        gx = x0 + NCOLS * (tw + gap) + goalgap
        axg = fig.add_axes([gx, y, tw, th])
        if gimg is not None:
            round_image(axg, gimg, r=0.07, ec=INK, lw=1.2)
        else:
            axg.set_axis_off()
            axg.text(0.5, 0.5, "goal\nunavailable", ha="center", va="center", fontsize=7)
        if gxy is not None:
            axg.scatter(*gxy, marker="*", s=60, fc=GOLD, ec=INK, lw=0.6, zorder=8)
        else:
            axg.scatter([0.1], [0.88], marker="*", s=70, fc=GOLD, ec=INK, lw=0.6,
                        transform=axg.transAxes, zorder=8, clip_on=False)
        if r == 0:
            fig.text(gx + tw / 2, y + th + 0.089 / FH, "ground truth goal", ha="center", va="center", fontsize=6.5)
    fig.text(0.5, 1 - 0.153 / FH, spec["title"], ha="center", va="center", fontsize=8.5)
    rounded_frame(fig, 0.004, 0.01, 0.992, 0.98, r=0.035)
    save(fig, f"fig_frames_{env}")
    print("wrote", f"fig_frames_{env}")
    for key, (frames, states, goal, ok, ex, gimg) in rows:
        print(f"  {key:<5} frames={len(frames):<4} success={ok:.0f} iters={its[key]} "
              f"goal={'yes' if gimg is not None else 'MISSING'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cube")
