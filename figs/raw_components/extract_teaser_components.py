"""Extract every component of the teaser as a standalone file, at source resolution.

The drawing code here is COPIED VERBATIM from fig_teaser.py -- same arrays, same
transforms, same styling constants -- so each component is what the teaser shows,
only rendered into its own figure at full size instead of into a sub-axes. An
earlier version of this script re-implemented the panels approximately and
produced things that did not match (wrong offset sign, missing free-space mask
and kNN edges, isotropic instead of anisotropic SMC modes); if fig_teaser.py
changes, mirror the change here rather than re-deriving it.

Output: figs/raw_components/
"""
import os
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, round_image, VALUE_CMAP, PINK,  # noqa: E402
                        PINK_LIGHT, PINK_DARK, GREY, NAVY, GOLD, INK)

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "renders")
OUT = os.path.join(HERE, "raw_components")
os.makedirs(OUT, exist_ok=True)
TRACE = os.path.join(R, "pm25_fk_s3", "traj.npz")
VALUE = os.path.join(R, "pm25_value_s3.npz")
OFFSET = np.array([1.83, 1.90])          # same sign convention as the teaser
use_style(7 * 1.2)

# ---- environment tiles ---------------------------------------------------
for f in ("pointmaze.png", "pusht.png", "reacher.png", "cube.png", "wall.png"):
    src = os.path.join(R, "tiles", f)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, "tile_" + f))
# also as the teaser draws them: rounded corners, r=0.1
for f in ("pointmaze.png", "pusht.png", "reacher.png", "cube.png"):
    src = os.path.join(R, "tiles", f)
    if not os.path.exists(src):
        continue
    fig = plt.figure(figsize=(2.4, 2.4))
    ax = fig.add_axes([0, 0, 1, 1])
    round_image(ax, imageio.imread(src), r=0.1)
    ax.set_axis_off()
    fig.savefig(os.path.join(OUT, "tile_rounded_" + f.replace(".png", ".pdf")))
    plt.close(fig)
print("tiles: verbatim copies + rounded versions")

# ---- latent glyphs (verbatim from the teaser) ----------------------------
fig = plt.figure(figsize=(0.55, 2.4)); g = fig.add_axes([0, 0, 1, 1])
g.set_xlim(0, 1); g.set_ylim(0, 1); g.set_axis_off()
for i in range(4):
    g.add_patch(Rectangle((0.08, 0.02 + i * 0.245), 0.84, 0.225,
                          fc=PINK if i % 2 == 0 else PINK_LIGHT, ec=INK, lw=0.6))
fig.savefig(os.path.join(OUT, "glyph_proprio.pdf")); plt.close(fig)

fig = plt.figure(figsize=(2.4, 2.4)); g = fig.add_axes([0, 0, 1, 1])
g.set_xlim(0, 1); g.set_ylim(0, 1); g.set_axis_off()
rng = np.random.default_rng(1)                       # same seed as the teaser
for i in range(4):
    for j in range(4):
        g.add_patch(Rectangle((0.02 + j * 0.245, 0.02 + i * 0.245), 0.225, 0.225,
                              fc=VALUE_CMAP(0.3 + 0.65 * rng.random()), ec=INK, lw=0.4))
fig.savefig(os.path.join(OUT, "glyph_visual.pdf")); plt.close(fig)
print("glyphs: proprio vector, visual patch grid")

# ---- roadmap panel (verbatim) -------------------------------------------
val = np.load(VALUE); rm = np.load(VALUE.replace(".npz", "_roadmap.npz"))
tr = np.load(TRACE, allow_pickle=True)
xy = val["xy"] + OFFSET
nxy, nV = rm["node_xy"] + OFFSET, rm["node_V"]
goal = np.asarray(tr["state_g"]).ravel()[:2]
lo, hi = xy.min(0) - 0.12, xy.max(0) + 0.12
H2, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=140, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
free = binary_opening(binary_closing(H2 > 0, iterations=3), iterations=2)
# In the teaser this panel is CW x CH of a 7.2 x 3.6in figure = 0.9 x 0.9in.
# Rendering the same ABSOLUTE marker and line sizes into a larger figure makes
# the nodes specks and lets the kNN edges dominate, which is what went wrong
# first time. Scale point AREAS by SC^2 and line widths by SC so the panel looks
# identical, only bigger.
PANEL_IN = 0.9
SIZE_IN = 4.0
SC = SIZE_IN / PANEL_IN
fig = plt.figure(figsize=(SIZE_IN, SIZE_IN)); am = fig.add_axes([0, 0, 1, 1])
am.set_aspect("equal")
am.imshow(np.where(free.T, 1.0, np.nan), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
          cmap=plt.matplotlib.colors.ListedColormap(["#EEF0F7"]), vmin=0, vmax=1)
am.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2,
           gaussian_filter(free.T.astype(float), 1.0), levels=[0.5], colors=[GREY],
           linewidths=0.7 * SC)
knn = rm["knn"][:, 1:3]
segs = np.stack([np.repeat(nxy, knn.shape[1], 0), nxy[knn.ravel()]], 1)
short = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1) < 0.45
am.add_collection(LineCollection(segs[short], colors="#B9BCCB",
                                 linewidths=0.2 * SC, alpha=0.7))
vmin, vmax = np.percentile(nV, [2, 98])
am.scatter(nxy[:, 0], nxy[:, 1], c=nV, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax,
           s=1.2 * SC ** 2, lw=0, zorder=3)
am.scatter(*goal, marker="*", s=80 * SC ** 2, fc=GOLD, ec=INK, lw=0.6 * SC, zorder=6)
am.set_xlim(xe[0], xe[-1]); am.set_ylim(ye[-1], ye[0]); am.set_axis_off()
fig.savefig(os.path.join(OUT, "roadmap.pdf"))
fig.savefig(os.path.join(OUT, "roadmap.png"), dpi=600)
plt.close(fig)
print(f"roadmap: {len(nV)} nodes, free-space mask, kNN edges, y-axis inverted as in the teaser")

# ---- colour bar (verbatim) ----------------------------------------------
fig = plt.figure(figsize=(0.9, 2.6)); cb = fig.add_axes([0.05, 0.03, 0.22, 0.94])
cb.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", cmap=VALUE_CMAP)
cb.set_axis_off()
fig.text(0.34, 0.95, "near\ngoal", ha="left", va="center", fontsize=6.96, linespacing=1.0)
fig.text(0.34, 0.05, "far", ha="left", va="center", fontsize=6.96)
fig.text(0.34, 0.5, r"$C$", ha="left", va="center", fontsize=9.6)
fig.savefig(os.path.join(OUT, "colorbar_C.pdf")); plt.close(fig)
print("colorbar")

# ---- SMC stages (verbatim) ----------------------------------------------
g = np.linspace(-4.3, 4.3, 320)
GX, GY = np.meshgrid(g, g)
log_p0 = -(GX ** 2 + GY ** 2) / (2 * 1.1 ** 2)
modes = []
for ang in (90, 210, 330):
    t = np.deg2rad(ang); mu = 1.8 * np.array([np.cos(t), np.sin(t)])
    R2 = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    modes.append((mu, R2 @ np.diag([0.28, 0.11]) @ R2.T))
def log_mix(X, Y):
    out = []
    for mu, S in modes:
        d = np.stack([X - mu[0], Y - mu[1]], -1); Si = np.linalg.inv(S)
        out.append(-0.5 * np.einsum("...i,ij,...j->...", d, Si, d) - 0.5 * np.log(np.linalg.det(S)))
    return np.logaddexp.reduce(np.stack(out), axis=0)
log_tgt = log_mix(GX, GY)
betas = [0.0, 0.25, 0.6, 1.0]
rng = np.random.default_rng(2)                       # same seed as the teaser
for k, b in enumerate(betas):
    lp = (1 - b) * log_p0 + b * log_tgt
    pz = np.exp(lp - lp.max())
    col = VALUE_CMAP(0.78 * (1 - b))
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list("c", ["white", col])
    fig = plt.figure(figsize=(2.4, 2.4)); ax = fig.add_axes([0, 0, 1, 1])
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
    ax.set_xlim(g[0], g[-1]); ax.set_ylim(g[0], g[-1]); ax.set_axis_off()
    fig.savefig(os.path.join(OUT, f"smc_stage{k+1}.pdf")); plt.close(fig)
print("smc stages 1-4 (anisotropic petals, contourf, 45 particles, rounded clip)")
