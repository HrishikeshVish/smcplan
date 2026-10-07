"""Compose the side-by-side ablation video, in the style of submission_video_draft3.

The official DINO-WM writer (planning/evaluator.py:239) emits a 2x2 grid per
frame: [executed | goal] on top, [imagined | goal] below. It draws the agent as
a dot but leaves NO trail, so a still frame says nothing about the path taken.
Here we take the executed quadrant, locate the agent in each frame by its green
colour, and accumulate the centroids into a polyline -- the same device as the
green end-effector trace in the reference video.

Locating the agent by colour rather than by projecting e_states avoids needing
the mujoco camera intrinsics: the dot the renderer drew IS ground truth for
where the agent appeared, so the trace cannot drift from the picture.

Usage:
  python writeup/compose_ablation_video.py SEED [out.mp4]
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep_fk", "renders")
ARMS = [("objective_raw", "L2 objective tilt (ablation)"),
        ("none", "Heat cost-to-go tilt (ours)")]
PANEL = 620          # rendered size of one maze panel
PAD, TOPBAR = 55, 150
GREEN = (90, 220, 90)        # BGR trail
TRAIL_W = 4


def agent_xy(quad):
    """Centroid of the green agent dot in one panel, or None."""
    hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
    # The maze is orange/tan and the floor dark blue, so a broad green window
    # is unambiguous here.
    m = cv2.inRange(hsv, np.array([35, 80, 60]), np.array([90, 255, 255]))
    if m.sum() == 0:
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()), float(ys.mean())


def load_arm(seed, arm):
    d = os.path.join(ROOT, arm, f"seed_{seed}")
    cap = cv2.VideoCapture(os.path.join(d, "rollout.mp4"))
    frames, pts, goal = [], [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        ex = fr[: h // 2, : w // 2]              # executed rollout
        gl = fr[: h // 2, w // 2:]               # goal panel
        frames.append(ex)
        pts.append(agent_xy(ex))
        if goal is None:
            goal = agent_xy(gl)
    cap.release()
    meta = {}
    p = os.path.join(d, "traj.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        meta = {"success": float(z["success"]), "exit": int(z["exit_step"])}
    return frames, pts, goal, meta


def draw_panel(frame, pts_so_far, goal, label, meta, step):
    """One maze panel, upscaled, with the accumulated trail drawn on top."""
    h, w = frame.shape[:2]
    sc = PANEL / float(w)
    img = cv2.resize(frame, (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    if goal is not None:
        gx, gy = int(goal[0] * sc), int(goal[1] * sc)
        cv2.drawMarker(img, (gx, gy), (70, 70, 240), cv2.MARKER_TILTED_CROSS, 26, 3)
        cv2.circle(img, (gx, gy), 15, (70, 70, 240), 2)
    track = [(int(p[0] * sc), int(p[1] * sc)) for p in pts_so_far if p is not None]
    if len(track) > 1:
        cv2.polylines(img, [np.array(track, np.int32)], False, GREEN, TRAIL_W, cv2.LINE_AA)
    if track:
        cv2.circle(img, track[-1], 7, (255, 255, 255), -1)
        cv2.circle(img, track[-1], 7, (40, 120, 40), 2)

    out = np.zeros((PANEL + 46, PANEL, 3), np.uint8)
    out[46:] = img
    cv2.putText(out, label, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    if meta:
        ok = meta["success"] > 0.5
        badge = f"{meta['exit']} iters" if ok else "no solution (150 cap)"
        col = (90, 220, 90) if ok else (90, 90, 240)
        cv2.rectangle(out, (8, 54), (18 + 13 * len(badge), 88), (0, 0, 0), -1)
        cv2.putText(out, badge, (14, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        if ok:
            cx, cy = PANEL - 42, 88
            cv2.circle(out, (cx, cy), 22, (80, 200, 80), -1)
            cv2.polylines(out, [np.array([[cx - 10, cy], [cx - 3, cy + 8], [cx + 11, cy - 9]], np.int32)],
                          False, (255, 255, 255), 4, cv2.LINE_AA)
    return out


def main(seed, out_path):
    arms = [(name, label) + load_arm(seed, name) for name, label in ARMS]
    n = max(len(a[2]) for a in arms)
    W = PAD * 3 + PANEL * 2
    H = TOPBAR + PANEL + 46 + PAD
    # libx264 + yuv420p requires EVEN dimensions, and macro_block_size=1 below
    # disables imageio's auto-padding, so an odd canvas makes ffmpeg die with a
    # broken pipe and leave a 0-byte file. (cv2's mp4v tolerated odd sizes,
    # which is why this only appeared after switching to H.264.)
    W += W % 2
    H += H % 2
    # imageio+ffmpeg (libx264/yuv420p), NOT cv2's "mp4v": cv2 writes MPEG-4
    # Part 2, which browsers and QuickTime largely refuse to decode -- the file
    # looks corrupt to the viewer even though it is intact.
    import imageio
    vw = imageio.get_writer(out_path, fps=10, codec="libx264",
                            pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-crf", "18", "-movflags", "+faststart"])
    for t in range(n + 12):                      # +12 = hold the final frame
        ti = min(t, n - 1)
        canvas = np.zeros((H, W, 3), np.uint8)
        cv2.putText(canvas, f"Point maze, goal_H=10, seed {seed}", (PAD, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "FK tilt quantity ablation, 3 particles - green trail = executed path",
                    (PAD, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (170, 170, 170), 1, cv2.LINE_AA)
        for k, (name, label, frames, pts, goal, meta) in enumerate(arms):
            j = min(ti, len(frames) - 1)
            panel = draw_panel(frames[j], pts[: j + 1], goal, label, meta, j)
            x0 = PAD + k * (PANEL + PAD)
            canvas[TOPBAR:TOPBAR + panel.shape[0], x0:x0 + PANEL] = panel
        vw.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    vw.close()
    print(f"wrote {out_path}  ({n} frames, {W}x{H})")
    for name, label, frames, pts, goal, meta in arms:
        got = sum(p is not None for p in pts)
        print(f"  {name:<14} frames={len(frames):<4} agent located in {got}  meta={meta}")


if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "23"
    out = sys.argv[2] if len(sys.argv) > 2 else f"sweep_fk/renders/ablation_seed{seed}.mp4"
    main(seed, out)
