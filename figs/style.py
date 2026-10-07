"""Shared matplotlib style for the paper figures.

Kept in one place so every figure has the same fonts, sizes and palette, and so
the colours are consistent across figures (a reader should be able to learn
"blue = ours" once).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour-blind-safe (Okabe-Ito). Ours is the one that must stand out.
OURS = "#0072B2"
GD = "#D55E00"
GRASP = "#009E73"
CEM = "#CC79A7"
L2 = "#E69F00"
GREY = "#666666"

METHOD_COLOR = {"ours": OURS, "fk": OURS, "gd": GD, "grasp": GRASP,
                "cem": CEM, "mpc_cem": CEM, "l2": L2}
METHOD_LABEL = {"ours": "Ours", "fk": "Ours", "gd": "GD", "grasp": "GRASP",
                "cem": "CEM", "mpc_cem": "CEM", "l2": "$L_2$ tilt"}


def use_paper_style():
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
    })
