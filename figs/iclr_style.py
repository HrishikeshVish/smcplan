"""House style for the ICLR figures, matched to fig_styles/ (the other paper).

Typewriter type (Computer Modern Typewriter, shipped with matplotlib), thick
rounded black frames, pink = ours, grey = baselines, and a pink -> violet ->
blue -> navy value colormap (pink = near the goal). Every figure imports this
so a reader learns the encoding once.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

_TTF = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "cmtt10.ttf")
font_manager.fontManager.addfont(_TTF)
MONO = font_manager.FontProperties(fname=_TTF).get_name()   # "cmtt10"

# --- palette ---------------------------------------------------------------
PINK = "#F27DB4"        # ours
PINK_DARK = "#D9468C"
PINK_LIGHT = "#FAD2E4"
GREY = "#8C909B"        # baselines
GREY_DARK = "#5E626D"
GREY_LIGHT = "#D6D8DD"
BLUE = "#7F93E0"        # a third series when needed
NAVY = "#1C2464"
GOLD = "#F6BE2C"
INK = "#1A1A1A"

METHOD_COLOR = {"ours": PINK, "gd": GREY_DARK, "grasp": GREY, "cem": BLUE}
METHOD_LABEL = {"ours": "ours", "gd": "GD", "grasp": "GRASP", "cem": "CEM"}

VALUE_CMAP = LinearSegmentedColormap.from_list(
    "near_far", ["#F57DB5", "#C58BDD", "#8E8FE3", "#5A80DA", "#2F4BA6", NAVY])


def use_style(base=8):
    plt.rcParams.update({
        "font.family": MONO,
        "font.size": base,
        "axes.titlesize": base + 1,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,        # cmtt10 has no U+2212
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def rounded_frame(fig, x0, y0, w, h, lw=1.6, pad=0.0, r=0.025, **kw):
    """Thick rounded black frame in figure coordinates (the reference's box)."""
    box = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad={pad},rounding_size={r}",
                         transform=fig.transFigure, fill=False, lw=lw, ec=INK,
                         zorder=10, clip_on=False, **kw)
    fig.patches.append(box)
    return box


def round_image(ax, img, r=0.08, lw=0.8, ec="#9A9A9A", **kw):
    """imshow clipped to a rounded rectangle (axes coordinates)."""
    im = ax.imshow(img, **kw)
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={r}",
                          transform=ax.transAxes, fc="none", ec="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    border = FancyBboxPatch((0, 0), 1, 1, boxstyle=f"round,pad=0,rounding_size={r}",
                            transform=ax.transAxes, fc="none", ec=ec, lw=lw, zorder=5)
    ax.add_patch(border)
    ax.set_axis_off()
    return im


def outlined(lw=2.6, color=INK):
    """Path effect: white line with a dark outline, as in the reference."""
    return [pe.Stroke(linewidth=lw, foreground=color), pe.Normal()]


def save(fig, name, outdir=None):
    # Figures land in $FIG_OUT, by default this directory (figs/).
    outdir = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
