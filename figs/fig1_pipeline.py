"""Figure 1: the planner pipeline.

(a) The four stages of one iteration.
(b) The same iteration drawn as the particle population itself: circles are
    particles, colour is the cost structure C, area is importance weight. SELECT shows
    killed particles hollow and crossed, with arrows marking which survivor is
    cloned into each freed slot. Only MOVE displaces anything -- selection
    changes which particles exist, never where they are, which is the property
    the method rests on.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iclr_style import (use_style, rounded_frame, save, VALUE_CMAP, PINK, PINK_DARK,  # noqa: E402
                        PINK_LIGHT, GREY, GREY_DARK, INK, NAVY, GREY_LIGHT)
OURS = INK

MOVE_C = PINK_DARK
# Panel fill: the one neutral the palette lacks, kept as a named constant so it
# is not mistaken for an ad-hoc colour. Lighter than GREY_LIGHT so text reads.
PANEL_BG = "#EFF0F3"
STAGES = [
    ("evaluate", "roll out every particle\nread $C_i$ at the final latent", OURS, PANEL_BG),
    ("weigh", r"$\log W_i \leftarrow \log W_i - \Delta\beta_k C_i/\alpha$" "\nadvance $\\beta$ to hold ESS", OURS, PANEL_BG),
    ("select", "if ESS $< \\tau P$: resample\nclone survivors, drop rest", OURS, PANEL_BG),
    ("move", "gradient step on $L$\nfor every particle", MOVE_C, PINK_LIGHT),
]
W, H, GAP, Y = 2.05, 1.05, 0.42, 1.05


def boxes(ax):
    ax.set_xlim(-0.15, 4 * W + 3 * GAP + 0.15)
    ax.set_ylim(0.0, 2.55)
    ax.axis("off")
    xs = []
    for i, (name, body, edge, face) in enumerate(STAGES):
        x = i * (W + GAP)
        xs.append(x)
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.035,rounding_size=0.14",
                                    linewidth=1.4, edgecolor=edge, facecolor=face))
        ax.text(x + W / 2, Y + H - 0.26, name, ha="center", va="center",
                fontsize=9.5, color=edge)
        ax.text(x + W / 2, Y + 0.34, body, ha="center", va="center",
                fontsize=7.2, linespacing=1.45)
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.03, Y + H / 2),
                                         (x + W + GAP - 0.03, Y + H / 2),
                                         arrowstyle="-|>", mutation_scale=10,
                                         color=INK, lw=1.1))
    x_end, x_start, yb = xs[-1] + W, xs[0], 0.52
    ax.plot([x_end - W / 2, x_end - W / 2], [Y - 0.03, yb], color=GREY, lw=1.0, ls=(0, (5, 3)))
    ax.plot([x_end - W / 2, x_start + W / 2], [yb, yb], color=GREY, lw=1.0, ls=(0, (5, 3)))
    ax.add_patch(FancyArrowPatch((x_start + W / 2, yb), (x_start + W / 2, Y - 0.03),
                                 arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.0))
    ax.text((x_start + x_end) / 2, yb - 0.21,
            r"repeat for $K$ iterations, then return $\mathrm{argmin}_i\,L_i$",
            ha="center", va="center", fontsize=7.4, color=GREY)
    ax.text(xs[1] + W / 2, 2.34,
            r"$C$ acts only through the weights $-$ never through the gradient",
            ha="center", va="center", fontsize=7.6, color=GREY_DARK)
    return [x + W / 2 for x in xs]


def particles(ax, centres, P=8, seed=0):
    rng = np.random.default_rng(seed)
    V = np.sort(rng.gamma(2.0, 1.1, size=P))
    w = np.exp(-0.9 * V); w /= w.sum()
    u = (rng.random() + np.arange(P)) / P
    anc = np.searchsorted(np.cumsum(w), u).clip(max=P - 1)

    ax.set_xlim(-0.15, 4 * W + 3 * GAP + 0.15)
    ax.set_ylim(-0.52, 1.18)
    ax.axis("off")
    ys = np.linspace(0.04, 0.96, P)
    cmap, norm = VALUE_CMAP, plt.Normalize(V.min(), V.max())
    X = centres

    for i, y in enumerate(ys):
        ax.scatter(X[0], y, s=55, color=cmap(norm(V[i])), edgecolor=INK, lw=0.5, zorder=3)
        ax.plot([X[0] + .10, X[1] - .10], [y, y], color=GREY_LIGHT, lw=0.6, zorder=1)
        ax.scatter(X[1], y, s=26 + 560 * w[i], color=cmap(norm(V[i])),
                   edgecolor=INK, lw=0.5, zorder=3)
        ax.plot([X[1] + .10, X[2] - .10], [y, y], color=GREY_LIGHT, lw=0.6, zorder=1)
    kept = set(anc.tolist())
    for i, y in enumerate(ys):
        if i in kept:
            ax.scatter(X[2], y, s=26 + 560 * w[i], color=cmap(norm(V[i])),
                       edgecolor=INK, lw=0.5, zorder=3)
        else:
            ax.scatter(X[2], y, s=50, facecolor="white", edgecolor=GREY, lw=0.7, zorder=3)
            for d in (-1, 1):
                ax.plot([X[2] - .055, X[2] + .055], [y - d * .022, y + d * .022],
                        color=GREY, lw=0.8, zorder=4)
    shift = rng.normal(0, 0.028, size=P)
    for dst, src in enumerate(anc):
        y1 = ys[dst] + shift[dst]
        ax.annotate("", xy=(X[3] - .12, y1), xytext=(X[2] + .10, ys[src]),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7,
                                    connectionstyle="arc3,rad=0.10"), zorder=2)
        ax.scatter(X[3], y1, s=55, color=PINK, edgecolor=INK, lw=0.6, zorder=3)

    ax.text(X[1], -0.16, "area $\\propto$ weight $w_i$", ha="center", va="top",
            fontsize=6.8, color=GREY_DARK)
    ax.text(X[2], -0.16, "clones replace killed particles\nnothing moves here",
            ha="center", va="top", fontsize=6.8, color=GREY_DARK, linespacing=1.4)
    ax.text(X[3], -0.16, "only here do particles move", ha="center", va="top",
            fontsize=6.8, color=MOVE_C)
    ax.text(X[0], -0.16, "colour $=$ cost structure $C_i$", ha="center", va="top",
            fontsize=6.8, color=GREY_DARK)


def main():
    use_style(7.5)
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.05,
                          left=0.04, right=0.985, top=0.97, bottom=0.03)
    ax_top = fig.add_subplot(gs[0])
    centres = boxes(ax_top)
    particles(fig.add_subplot(gs[1]), centres)
    fig.text(0.018, 0.93, "(a)", fontsize=9)
    fig.text(0.018, 0.47, "(b)", fontsize=9)
    rounded_frame(fig, 0.004, 0.005, 0.992, 0.99, r=0.03)
    save(fig, "fig1_pipeline")
    print("wrote fig1_pipeline")


if __name__ == "__main__":
    main()
