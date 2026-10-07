"""Side-by-side cube video with the BLOCK's path traced, in the style of
submission_video_draft3 / the pointmaze ablation videos.

Why this re-renders instead of reusing the dumped frames:
  * The LeWM cube checkpoint has NO decoder (config.json has no 'decoder' key),
    so planning/evaluator.py's video writer -- which decodes latents through a
    VQVAE -- cannot run for cube at all. That is what produced the pointmaze
    videos.
  * The env's own render() is a dim 224x224 view (mean brightness ~58). We hold
    the executed qpos/qvel from the trajectory dump, so we can re-render each
    step from a chosen camera and brighten it, which is far more legible.

Why the trail is PROJECTED rather than colour-detected: the pointmaze agent was
an unambiguous green dot on a flat 2-D maze, but this scene is dark, 3-D and
contains several reddish objects (the block AND its target marker). We already
hold the exact block position in state[14:17] (verified bit-identical to the
dataset's privileged_block_0_pos), so projecting it through MuJoCo's camera is
exact and cannot drift onto the wrong object.

Usage: python writeup/compose_cube_video.py SEED [out.mp4]
"""
import os
import sys
import cv2
import numpy as np
import mujoco
import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers swm/ envs)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "sweep_cube_leflow", "renders")
ARMS = [("gd", "Gradient descent (baseline)"), ("fk", "FK roadmap tilt (ours)")]
CAM = 2                      # side_pixels: the clearest of the four cameras
PANEL, PAD, TOPBAR = 600, 55, 150
BRIGHT_A, BRIGHT_B = 2.2, 18  # the raw render is too dark to read
GREEN, GOAL = (90, 220, 90), (70, 70, 240)


def projector(model, data, cam_id, W, H):
    """World point -> pixel. cam_xmat columns are the camera's right/up/back
    axes in world coords; MuJoCo cameras look down their -z."""
    f = 0.5 * H / np.tan(0.5 * np.deg2rad(model.cam_fovy[cam_id]))

    def proj(point):
        p = np.asarray(point) - data.cam_xpos[cam_id]
        c = data.cam_xmat[cam_id].reshape(3, 3).T @ p
        z = -c[2]
        if z <= 1e-6:
            return None
        return (int(round(W / 2 + f * c[0] / z)), int(round(H / 2 - f * c[1] / z)))
    return proj


def render_arm(seed, arm):
    z = np.load(os.path.join(ROOT, arm, f"seed_{seed}", "traj.npz"), allow_pickle=True)
    s = np.asarray(z["e_states"])
    s = s[0] if s.ndim == 3 else s
    goal = np.asarray(z["state_g"]).ravel()
    meta = {"success": float(z["success"]), "exit": int(z["exit_step"])}

    env = gym.make("swm/OGBCube-v0")
    env.reset(seed=int(seed))
    u = env.unwrapped
    m, d = u.model, u.data
    nq = m.nq
    r = mujoco.Renderer(m, height=224, width=224)

    frames, pts = [], []
    for t in range(len(s)):
        d.qpos[:] = s[t, :nq]
        d.qvel[:] = s[t, nq:]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=CAM)
        img = cv2.resize(r.render(), (PANEL, PANEL), interpolation=cv2.INTER_LANCZOS4)
        frames.append(cv2.convertScaleAbs(img, alpha=BRIGHT_A, beta=BRIGHT_B))
        pr = projector(m, d, CAM, PANEL, PANEL)
        pts.append(pr(s[t, 14:17]))
        if t == 0:
            goal_uv = pr(goal[14:17])
    r.close()
    env.close()
    return frames, pts, goal_uv, meta, float(np.linalg.norm(goal[14:17] - s[0, 14:17]))


def panel(frame, trail, goal_uv, label, meta):
    img = frame.copy()
    if goal_uv:
        cv2.drawMarker(img, goal_uv, GOAL, cv2.MARKER_TILTED_CROSS, 30, 3)
        cv2.circle(img, goal_uv, 17, GOAL, 2)
    tr = [p for p in trail if p is not None]
    if len(tr) > 1:
        cv2.polylines(img, [np.array(tr, np.int32)], False, GREEN, 4, cv2.LINE_AA)
    if tr:
        cv2.circle(img, tr[-1], 7, (255, 255, 255), -1)
        cv2.circle(img, tr[-1], 7, (40, 120, 40), 2)

    out = np.zeros((PANEL + 46, PANEL, 3), np.uint8)
    out[46:] = img
    cv2.putText(out, label, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    ok = meta["success"] > 0.5
    badge = f"solved, {meta['exit']} iters" if ok else "not solved"
    col = GREEN if ok else (90, 90, 240)
    cv2.rectangle(out, (8, 54), (20 + 13 * len(badge), 88), (0, 0, 0), -1)
    cv2.putText(out, badge, (14, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2, cv2.LINE_AA)
    if ok:
        cx, cy = PANEL - 42, 88
        cv2.circle(out, (cx, cy), 22, (80, 200, 80), -1)
        cv2.polylines(out, [np.array([[cx - 10, cy], [cx - 3, cy + 8], [cx + 11, cy - 9]], np.int32)],
                      False, (255, 255, 255), 4, cv2.LINE_AA)
    return out


def main(seed, out_path):
    arms = [(a, lab) + render_arm(seed, a) for a, lab in ARMS]
    n = max(len(a[2]) for a in arms)
    disp = arms[0][6]
    W = PAD * 3 + PANEL * 2
    H = TOPBAR + PANEL + 46 + PAD
    W += W % 2
    H += H % 2                                  # libx264 needs even dimensions
    import imageio
    vw = imageio.get_writer(out_path, fps=6, codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1, ffmpeg_params=["-crf", "18", "-movflags", "+faststart"])
    for t in range(n + 10):                      # +10 holds the final frame
        ti = min(t, n - 1)
        canvas = np.zeros((H, W, 3), np.uint8)
        cv2.putText(canvas, f"OGBench cube, seed {seed}", (PAD, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"green = block path   red cross = target   block must move {disp:.3f} m",
                    (PAD, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 1, cv2.LINE_AA)
        for k, (_, lab, frames, pts, guv, meta, _d) in enumerate(arms):
            j = min(ti, len(frames) - 1)
            p = panel(frames[j], pts[:j + 1], guv, lab, meta)
            x0 = PAD + k * (PANEL + PAD)
            canvas[TOPBAR:TOPBAR + p.shape[0], x0:x0 + PANEL] = p
        vw.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    vw.close()
    print(f"wrote {out_path}  ({n} frames, {W}x{H}, block displacement {disp:.3f} m)")
    for a, _l, _f, pts, _g, meta, _d in arms:
        print(f"  {a:<4} success={meta['success']:.0f} exit={meta['exit']:<4} block located {sum(p is not None for p in pts)}/{len(pts)}")


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else "43"
    main(sd, sys.argv[2] if len(sys.argv) > 2 else f"sweep_cube_leflow/renders/cube_seed{sd}.mp4")
