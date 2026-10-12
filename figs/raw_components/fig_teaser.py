"""Teaser (layout from fig_styles/IMG_1358.PNG).

Top-left   latent world models: proprio (Point maze, Push-T -> state vector) and
           visual (Reacher, Cube -> patch tokens).
Top-right  sparse roadmap: the planner's real Point-maze graph, coloured by the
           real heat-equation V for the goal (goal starred).
Bottom     sequential Monte-Carlo -- ILLUSTRATIVE, in
           the style of Fig. 1 of Park et al. 2025 (Flow Q-learning): tempered
           densities of a synthetic 3-mode cost-to-go, beta 0 -> 1, with particles
           drawn from each; the roadmap supplies w_i ∝ exp(-beta V).
Previous version kept as fig_teaser_v1.py.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_closing, binary_opening
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, round_image, outlined, save,  # noqa: E402
                        VALUE_CMAP, PINK, PINK_DARK, PINK_LIGHT, GREY, GREY_DARK,
                        GOLD, INK, NAVY)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "renders")
TRACE = os.path.join(R, "pm25_fk_s3", "traj.npz")
VALUE = os.path.join(R, "pm25_value_s3.npz")
OFFSET = np.array([1.83, 1.90])      # offline-buffer frame -> env frame

use_style(7 * 1.2)
FW, FH = 7.2, 3.6
fig = plt.figure(figsize=(FW, FH))
AR = FW / FH


def box(x, y, w, h, lw=1.5, r=0.025, fc="none"):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                      transform=fig.transFigure, fc=fc, ec=INK, lw=lw, zorder=0))


def arrow(p0, p1, rad=0.0, lw=1.0, ms=8, color=INK):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=ms, lw=lw, color=color, clip_on=False,
                                       connectionstyle=f"arc3,rad={rad}", zorder=20))


# One content height for everything in the top row (tiles, latent glyphs, maze,
# colour bar) so it reads as a single composite strip.
CH = 0.25                       # content height, figure fraction
CW = CH / AR                    # width of a square of that height
CY = 0.615                      # bottom of the top-row content
TITLE_Y = CY + CH + 0.075

# ============================ top-left: world models =========================
groups = [("proprio (DINO-WM)", ["pointmaze.png", "pusht.png"], "proprio", 0.012),
          ("visual (LeWM)", ["reacher.png", "cube.png"], "visual", 0.352)]
for label, tiles, kind, gx in groups:
    for k, f in enumerate(tiles):
        ax = fig.add_axes([gx + k * (CW + 0.008), CY, CW, CH])
        round_image(ax, imageio.imread(os.path.join(R, "tiles", f)), r=0.1)
    ax_end = gx + 2 * CW + 0.008
    arrow((ax_end + 0.004, CY + CH / 2), (ax_end + 0.027, CY + CH / 2), ms=6, lw=0.9)
    gw = 0.034 if kind == "proprio" else CW          # vector is narrow, patch grid square
    g = fig.add_axes([ax_end + 0.031, CY, gw, CH])
    g.set_xlim(0, 1); g.set_ylim(0, 1); g.set_axis_off()
    if kind == "proprio":
        for i in range(4):
            g.add_patch(Rectangle((0.08, 0.02 + i * 0.245), 0.84, 0.225,
                                  fc=PINK if i % 2 == 0 else PINK_LIGHT, ec=INK, lw=0.6))
    else:
        rng = np.random.default_rng(1)
        for i in range(4):
            for j in range(4):
                g.add_patch(Rectangle((0.02 + j * 0.245, 0.02 + i * 0.245), 0.225, 0.225,
                                      fc=VALUE_CMAP(0.3 + 0.65 * rng.random()), ec=INK, lw=0.4))
    fig.text(gx + (ax_end + 0.031 + gw - gx) / 2, CY - 0.04, label, ha="center", va="center",
             fontsize=9.6, color=PINK_DARK if kind == "proprio" else NAVY)
WM_RIGHT = 0.352 + 2 * CW + 0.008 + 0.031 + CW
fig.text((0.012 + WM_RIGHT) / 2, TITLE_Y, "latent world models", ha="center", va="center",
         fontsize=10.2)

# thin grey divider between the two top panels
DIV_X = WM_RIGHT + 0.017
fig.add_artist(plt.Line2D([DIV_X, DIV_X], [CY - 0.05, TITLE_Y + 0.02], transform=fig.transFigure,
                          color=GREY, lw=0.8))
# ============================ top-right: roadmap =============================
RX = DIV_X + 0.017
fig.text(RX + (CW + 0.06) / 2, TITLE_Y, "roadmap", ha="center", va="center", fontsize=10.2)
val = np.load(VALUE)
rm = np.load(VALUE.replace(".npz", "_roadmap.npz"))
tr = np.load(TRACE, allow_pickle=True)
xy = val["xy"] + OFFSET
nxy, nV = rm["node_xy"] + OFFSET, rm["node_V"]
path = np.asarray(tr["e_states"])[0][:, :2]
goal = np.asarray(tr["state_g"]).ravel()[:2]
lo, hi = xy.min(0) - 0.12, xy.max(0) + 0.12
H2, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=140, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
free = binary_opening(binary_closing(H2 > 0, iterations=3), iterations=2)
am = fig.add_axes([RX, CY, CW, CH])
am.set_aspect("equal")
am.imshow(np.where(free.T, 1.0, np.nan), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
          cmap=plt.matplotlib.colors.ListedColormap(["#EEF0F7"]), vmin=0, vmax=1)
am.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, gaussian_filter(free.T.astype(float), 1.0),
           levels=[0.5], colors=[GREY], linewidths=0.7)
knn = rm["knn"][:, 1:3]
segs = np.stack([np.repeat(nxy, knn.shape[1], 0), nxy[knn.ravel()]], 1)
short = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1) < 0.45
am.add_collection(LineCollection(segs[short], colors="#B9BCCB", linewidths=0.2, alpha=0.7))
vmin, vmax = np.percentile(nV, [2, 98])
am.scatter(nxy[:, 0], nxy[:, 1], c=nV, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax, s=1.2, lw=0, zorder=3)
am.scatter(*goal, marker="*", s=80, fc=GOLD, ec=INK, lw=0.6, zorder=6)
am.set_xlim(xe[0], xe[-1]); am.set_ylim(ye[-1], ye[0]); am.set_axis_off()
# legend: cost-to-go colour bar, vertical, right of the maze
cb = fig.add_axes([RX + CW + 0.012, CY + 0.02, 0.011, CH - 0.04])
cb.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", cmap=VALUE_CMAP)
cb.set_axis_off()
fig.text(RX + CW + 0.027, CY + CH - 0.03, "near\ngoal", ha="left", va="center", fontsize=6.96, linespacing=1.0)
fig.text(RX + CW + 0.027, CY + 0.025, "far", ha="left", va="center", fontsize=6.96)
fig.text(RX + CW + 0.027, CY + CH / 2, r"$C$", ha="left", va="center", fontsize=9.6)

# ============================ bottom: SMC (illustrative) ====================
# Synthetic, FQL-Fig.1-style: tempered densities p_b ∝ p0^(1-b) * exp(-V)^b of a
# 3-mode toy cost-to-go, with particles drawn from each. Not data.
BX, BY, BW, BH = 0.008, 0.012, 0.984, 0.505
box(BX, BY, BW, BH)
fig.text(BX + BW / 2, BY + BH - 0.04, "sequential Monte-Carlo",
         ha="center", va="center", fontsize=10.2)
# (world model -> SMC arrow dropped with the top frames)

g = np.linspace(-4.3, 4.3, 320)                     # wide enough that every density sits inside
GX, GY = np.meshgrid(g, g)
log_p0 = -(GX ** 2 + GY ** 2) / (2 * 1.1 ** 2)
modes = []
for ang in (90, 210, 330):
    t = np.deg2rad(ang); mu = 1.8 * np.array([np.cos(t), np.sin(t)])
    R2 = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    modes.append((mu, R2 @ np.diag([0.28, 0.11]) @ R2.T))       # petals, long radially
def log_mix(X, Y):
    out = []
    for mu, S in modes:
        d = np.stack([X - mu[0], Y - mu[1]], -1); Si = np.linalg.inv(S)
        out.append(-0.5 * np.einsum("...i,ij,...j->...", d, Si, d) - 0.5 * np.log(np.linalg.det(S)))
    return np.logaddexp.reduce(np.stack(out), axis=0)
log_tgt = log_mix(GX, GY)

betas = [0.0, 0.25, 0.6, 1.0]
names = ["initial", "", "", "after"]
pw, gap = CW, 0.06
ph = pw * AR
wrow = len(betas) * pw + (len(betas) - 1) * gap
px0 = BX + (BW - wrow) / 2                          # centred in the SMC box
py = BY + 0.085
rng = np.random.default_rng(2)
for k, b in enumerate(betas):
    lp = (1 - b) * log_p0 + b * log_tgt
    pz = np.exp(lp - lp.max())
    col = VALUE_CMAP(0.78 * (1 - b))
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list("c", ["white", col])
    x = px0 + k * (pw + gap)
    ax = fig.add_axes([x, py, pw, ph])
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.1",
                          transform=ax.transAxes, fc="none", ec="none")
    ax.add_patch(clip)
    levels = [0.02, 0.1, 0.3, 0.55, 0.8, 1.0]
    cf = ax.contourf(GX, GY, pz, levels=levels, cmap=cm, vmin=-0.15, vmax=1.0)
    cl = ax.contour(GX, GY, pz, levels=levels[:-1], colors=[col], linewidths=0.7)
    flat = (pz / pz.sum()).ravel()
    idx = rng.choice(flat.size, 45, p=flat)
    ptx = GX.ravel()[idx] + rng.normal(0, 0.03, 45); pty = GY.ravel()[idx] + rng.normal(0, 0.03, 45)
    sc = ax.scatter(ptx, pty, s=6, fc="white" if b == 0 else PINK, ec=INK, lw=0.35, zorder=3)
    for art in (cf, cl, sc):
        art.set_clip_path(clip)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.1",
                                transform=ax.transAxes, fc="none", ec=INK, lw=0.9, zorder=5))
    ax.set_xlim(g[0], g[-1]); ax.set_ylim(g[0], g[-1])
    ax.set_axis_off()
    if names[k]:                                    # illustrative: no tempering labels
        fig.text(x + pw / 2, py - 0.028, names[k], ha="center", va="center", fontsize=9.0)
    if k < len(betas) - 1:
        arrow((x + pw + 0.008, py + ph / 2), (x + pw + gap - 0.008, py + ph / 2), ms=6, lw=0.8)
# anneal arrow across the top of the panels
ya = py + ph + 0.024
arrow((px0, ya), (px0 + wrow, ya), ms=8, lw=1.0)
fig.text(px0 + wrow / 2, ya + 0.022, "propose  $\cdot$  resample  $\cdot$  update",
         ha="center", va="center", fontsize=8.4)
# the roadmap supplies the weights
fig.text(px0 + wrow, ya + 0.024, r"$w_i \propto e^{-C(\mathbf{a}_i)}$", ha="right",
         va="center", fontsize=10.2)
arrow((RX + CW / 2, CY - 0.012), (px0 + wrow - 0.05, ya + 0.045), ms=8)      # roadmap -> weights

save(fig, "fig_teaser")
print("wrote fig_teaser")
