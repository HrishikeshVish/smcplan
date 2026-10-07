"""Teaser: latent planning across domains, sparse-roadmap value, SMC.

Three framed panels, house style (iclr_style.py):
  (1) world models -- proprio latents (Point maze, Push-T) vs visual latents
      (Reacher, Cube), rendered env tiles + a latent glyph per row.
  (2) sparse roadmap -- the planner's real Point-maze graph (3800 nodes placed
      by exact latent match to the offline states), coloured by the real
      heat-equation V for the goal, with our real plan trace on top.
  (3) sequential Monte-Carlo -- after Fig. 1 of Yang (2026, TSMC): stacked
      tempered densities p_beta ∝ p0 exp(-beta V) over a REAL 2-D slice of V
      through action space (viz_dumps/seed33_plane), and a particle population
      reweighted/resampled on that same slice. The particle panel and the prior
      width are illustrative (the real run has 3 particles, prior std 1).
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, zoom, binary_closing, binary_opening
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, rounded_frame, round_image, outlined, save,  # noqa: E402
                        VALUE_CMAP, PINK, PINK_DARK, PINK_LIGHT, GREY, GREY_LIGHT,
                        GREY_DARK, GOLD, INK, NAVY)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "renders")
TRACE = os.environ.get("TEASER_TRACE", os.path.join(R, "pm25_fk_s3", "traj.npz"))
VALUE = os.environ.get("TEASER_VALUE", os.path.join(R, "pm25_value_s3.npz"))
OFFSET = np.array([1.83, 1.90])      # offline-buffer frame -> env frame (fit to data support)

use_style(7)
FW, FH = 7.2, 2.45
fig = plt.figure(figsize=(FW, FH))


def ax_at(x, y, w, h=None, square=False, **kw):
    if square:
        h = w * FW / FH
    return fig.add_axes([x, y, w, h], **kw)


def arrow(x0, y0, x1, y1, lw=1.1, ms=8, color=INK):
    fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                       color=color, clip_on=False))


# ============================ (1) world models ==============================
X1, W1 = 0.004, 0.300
rounded_frame(fig, X1, 0.035, W1, 0.925, r=0.03)
fig.text(X1 + W1 / 2, 0.905, "latent world models", ha="center", va="center", fontsize=8.5)

tile_w = 0.074
rows = [("proprio latents", [("point-maze", "pointmaze.png"), ("push-t", "pusht.png")], "proprio"),
        ("visual latents", [("reacher", "reacher.png"), ("cube", "cube.png")], "visual")]
for r, (rlabel, tiles, kind) in enumerate(rows):
    y0 = 0.515 - r * 0.385
    fig.text(X1 + 0.016, y0 + tile_w * FW / FH + 0.04, rlabel, ha="left", va="center",
             fontsize=7, color=PINK_DARK if kind == "proprio" else NAVY)
    for c, (name, f) in enumerate(tiles):
        x = X1 + 0.016 + c * (tile_w + 0.012)
        ax = ax_at(x, y0, tile_w, square=True)
        img = imageio.imread(os.path.join(R, "tiles", f))
        round_image(ax, img, r=0.10)
        fig.text(x + tile_w / 2, y0 - 0.032, name, ha="center", va="center", fontsize=6)
    # encoder arrow + latent glyph
    gx = X1 + 0.016 + 2 * (tile_w + 0.012) + 0.004
    ymid = y0 + tile_w * FW / FH / 2
    arrow(gx, ymid, gx + 0.024, ymid, ms=6, lw=0.9)
    g = ax_at(gx + 0.030, y0 + 0.02, 0.05, tile_w * FW / FH - 0.04)
    g.set_xlim(0, 1); g.set_ylim(0, 1); g.set_axis_off()
    rng = np.random.default_rng(3 + r)
    if kind == "proprio":
        # a short state-like vector: few, meaningful coordinates
        for k in range(4):
            g.add_patch(Rectangle((0.33, 0.08 + k * 0.215), 0.34, 0.19,
                                  fc=PINK_LIGHT if k % 2 else PINK, ec=INK, lw=0.6))
        g.text(0.5, -0.12, r"$z\in\mathbb{R}^{d}$", ha="center", va="center", fontsize=6.5)
    else:
        # a grid of patch tokens
        n = 4
        for i in range(n):
            for j in range(n):
                g.add_patch(Rectangle((0.05 + j * 0.225, 0.06 + i * 0.225), 0.2, 0.2,
                                      fc=VALUE_CMAP(0.35 + 0.6 * rng.random()), ec=INK, lw=0.4))
        g.text(0.5, -0.14, "patch tokens", ha="center", va="center", fontsize=5.5)

# ============================ (2) sparse roadmap =============================
X2, W2 = 0.318, 0.285
rounded_frame(fig, X2, 0.035, W2, 0.925, r=0.03)
fig.text(X2 + W2 / 2, 0.905, "sparse roadmap value", ha="center", va="center", fontsize=8.5)
arrow(X1 + W1 + 0.001, 0.5, X2 - 0.001, 0.5, ms=7)

val = np.load(VALUE)
rm = np.load(VALUE.replace(".npz", "_roadmap.npz"))
tr = np.load(TRACE, allow_pickle=True)
xy = val["xy"] + OFFSET
nxy = rm["node_xy"] + OFFSET
nV = rm["node_V"]
path = np.asarray(tr["e_states"])[0][:, :2]
goal = np.asarray(tr["state_g"]).ravel()[:2]
start = path[0]

# free space = support of the offline data (closed/opened to clean the edge)
lo, hi = xy.min(0) - 0.12, xy.max(0) + 0.12
nb = 140
H, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=nb, range=[[lo[0], hi[0]], [lo[1], hi[1]]])
free = binary_opening(binary_closing(H > 0, iterations=3), iterations=2)

am = ax_at(X2 + 0.024, 0.14, 0.237, square=False, h=0.70)
am.set_aspect("equal")
am.imshow(np.where(free.T, 1.0, np.nan), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
          cmap=plt.matplotlib.colors.ListedColormap(["#EEF0F7"]), vmin=0, vmax=1, zorder=0,
          interpolation="bilinear")
am.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, gaussian_filter(free.T.astype(float), 1.0),
           levels=[0.5], colors=[GREY], linewidths=0.8, zorder=1)
# kNN edges (3 nearest per node, in latent space), faint
knn = rm["knn"][:, 1:3]
segs = np.stack([np.repeat(nxy, knn.shape[1], 0), nxy[knn.ravel()]], 1)
short = np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1) < 0.45
am.add_collection(LineCollection(segs[short], colors="#B9BCCB", linewidths=0.25, alpha=0.7, zorder=2))
vmin, vmax = np.percentile(nV, [2, 98])
am.scatter(nxy[:, 0], nxy[:, 1], c=nV, cmap=VALUE_CMAP, vmin=vmin, vmax=vmax, s=2.2, lw=0, zorder=3)
am.plot(path[:, 0], path[:, 1], color="white", lw=1.5, zorder=5, solid_capstyle="round",
        path_effects=outlined(3.0))
am.scatter(*start, s=22, fc="white", ec=INK, lw=1.0, zorder=6)
am.scatter(*goal, marker="*", s=120, fc=GOLD, ec=INK, lw=0.8, zorder=6)
am.set_xlim(xe[0], xe[-1]); am.set_ylim(ye[-1], ye[0])   # y flipped: matches the render
am.set_axis_off()

cb = ax_at(X2 + 0.085, 0.075, 0.13, 0.022)
cb.imshow(np.linspace(0, 1, 256)[None], aspect="auto", cmap=VALUE_CMAP)
cb.set_axis_off()
fig.text(X2 + 0.080, 0.086, "near goal", ha="right", va="center", fontsize=6)
fig.text(X2 + 0.220, 0.086, "far", ha="left", va="center", fontsize=6)

# ============================ (3) SMC =======================================
X3, W3 = 0.616, 0.380
rounded_frame(fig, X3, 0.035, W3, 0.925, r=0.03)
fig.text(X3 + W3 / 2, 0.905, "sequential Monte-Carlo", ha="center", va="center", fontsize=8.5)
arrow(X2 + W2 + 0.001, 0.5, X3 - 0.001, 0.5, ms=7)

P = np.load(os.path.join(HERE, "..", "viz_dumps", "seed33_plane", "plane.npz"))
PI = np.load(os.path.join(HERE, "..", "viz_dumps", "seed33_plane_in.npz"))
xs, ys, Vp = P["xs"], P["ys"], P["V"]
Vs = gaussian_filter(zoom(Vp, 4, order=1), 2.0)
gx = np.linspace(xs[0], xs[-1], Vs.shape[1]); gy = np.linspace(ys[0], ys[-1], Vs.shape[0])
Vn = (Vs - Vs.min()) / (Vs.max() - Vs.min())
# The real prior mean sits at the slice edge; the illustration centres it.
c = np.array([(gx[0] + gx[-1]) / 2, (gy[0] + gy[-1]) / 2])
SIG = 6.0                               # widened for legibility (real: 1)
GX, GY = np.meshgrid(gx, gy)
prior = ((GX - c[0]) ** 2 + (GY - c[1]) ** 2) / (2 * SIG ** 2)
betas = [0.0, 0.3, 1.0]
BSCALE = 6.0                            # V normalised to [0, 1] then scaled

# (a) stacked tempered negative log-densities, TSMC Fig. 1(a) style
a3 = fig.add_axes([X3 - 0.012, 0.07, 0.175, 0.80], projection="3d")
lvl_col = [NAVY, "#8E8FE3", PINK]
for k, b in enumerate(betas):
    E = prior + BSCALE * b * Vn
    E = (E - E.min()) / (E.max() - E.min())
    a3.plot_surface(GX, GY, 0.85 * E + 1.25 * k, color=lvl_col[k], alpha=0.55, lw=0,
                    rstride=3, cstride=3, antialiased=True, shade=True)
    a3.text(gx[-1] + 4, gy[0], 1.25 * k + 0.4, fr"$\beta={b:g}$", fontsize=6.5)
a3.view_init(elev=18, azim=-58)
a3.set_axis_off()
a3.set_box_aspect((1, 1, 1.35))

# (b) particles on the slice: prior -> reweight + resample -> move
rng = np.random.default_rng(0)
N = 160
interp = RegularGridInterpolator((gy, gx), Vn, bounds_error=False, fill_value=1.0)
pts = c + SIG * 0.6 * rng.standard_normal((N, 2))
pts = np.clip(pts, [gx[0], gy[0]], [gx[-1], gy[-1]])
snaps = []
w = np.full(N, 1 / N)
bprev = 0.0
for b in betas:
    logw = np.log(w) - (b - bprev) * BSCALE * interp(pts[:, ::-1])
    w = np.exp(logw - logw.max()); w /= w.sum()
    snaps.append((pts.copy(), w.copy()))
    if b > 0:                           # systematic resample, then a small move
        u0 = (rng.random() + np.arange(N)) / N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u0), N - 1)
        pts = pts[idx] + 0.9 * rng.standard_normal((N, 2))
        w = np.full(N, 1 / N)
    bprev = b

pw = 0.056
labels = ["prior", "reweight", "resample\n+ move"]
for k, (b, (p, wk)) in enumerate(zip(betas, snaps)):
    x = X3 + 0.172 + k * (pw + 0.010)
    ax = ax_at(x, 0.30, pw, square=True)
    E = prior + BSCALE * b * Vn
    ax.contourf(gx, gy, E, levels=12, cmap=VALUE_CMAP, alpha=0.9)
    if k == 2:
        p, wk = pts, np.full(N, 1 / N)
    sz = 1.0 + 16 * wk / wk.max() * (0.4 if k == 0 else 1.0)
    ax.scatter(p[:, 0], p[:, 1], s=sz, fc="white" if k == 0 else PINK, ec=INK, lw=0.25, zorder=3)
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(gy[0], gy[-1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GREY); s.set_linewidth(0.6)
    ax.set_title(fr"$\beta={b:g}$", fontsize=6.5, pad=2)
    fig.text(x + pw / 2, 0.235, labels[k], ha="center", va="top", fontsize=5.8, linespacing=1.1)
    if k < 2:
        arrow(x + pw + 0.001, 0.30 + pw * FW / FH / 2, x + pw + 0.009, 0.30 + pw * FW / FH / 2,
              ms=4, lw=0.7)
fig.text(X3 + 0.172 + 1.5 * pw + 0.010, 0.13,
         r"weights $w_i \propto e^{-\beta V(\mathbf{a}_i)}$", ha="center", va="center", fontsize=6.3)
fig.text(X3 + 0.172 + 1.5 * pw + 0.010, 0.075,
         "move: gradient step on the loss", ha="center", va="center", fontsize=5.6,
         color=GREY_DARK)

save(fig, "fig_teaser")
print("wrote fig_teaser.pdf/png")
