# Teaser figure — raw components

Every component of `fig_teaser.pdf` as a standalone file. Nothing is cropped out
of the composed figure: each panel is regenerated from the same source the teaser
reads, so resolution is that of the source rather than of the space the panel
occupies in the layout.

## Components

| file | what it is | source |
|---|---|---|
| `tile_*.png` | environment tiles, 224x224 (cube 320x320) | byte-for-byte copies of `figs/renders/tiles/` |
| `tile_rounded_*.pdf` | the same tiles as the teaser draws them, rounded corners r=0.1 | via `round_image` |
| `roadmap.pdf` / `.png` | the roadmap panel: free-space mask, kNN edges, all 3800 nodes coloured by the heat solution (2-98 percentile clip), gold goal star, y-axis inverted | `pm25_value_s3.npz` + `_roadmap.npz` (600 dpi png) |
| `colorbar_C.pdf` | the near/far colour scale alone | `VALUE_CMAP` |
| `smc_stage1..4.pdf` | the four tempered densities, beta = 0, 0.25, 0.6, 1.0, with 45 particles each and the rounded clip | synthetic, illustrative — NOT data |
| `glyph_proprio.pdf`, `glyph_visual.pdf` | state-vector and patch-token glyphs | drawn |

## Scripts

| file | role |
|---|---|
| `fig_teaser.py` | generates the composed teaser (`fig_teaser.pdf`/`.png`) |
| `extract_teaser_components.py` | generates everything in this folder |
| `iclr_style.py` | shared palette and helpers both scripts import |

Run from `figs/`: `python extract_teaser_components.py`. Output lands here.

## Fidelity

The drawing code in `extract_teaser_components.py` is copied verbatim from
`fig_teaser.py` — same arrays, same offsets, same styling constants — so these
are the teaser's panels rendered larger, not approximations. An earlier version
re-derived them and silently differed (wrong offset sign, no free-space mask or
kNN edges, isotropic instead of anisotropic SMC modes). If `fig_teaser.py`
changes, mirror the change here rather than re-deriving.

One thing does have to change when rendering larger: marker *areas* and line
widths. In the teaser the roadmap occupies 0.9 x 0.9in, so `s=1.2` nodes read as
dense; the same absolute size in a 4in figure turns them into specks and lets the
kNN edges dominate. The extractor therefore scales point areas by `SC^2` and line
widths by `SC`, with `SC = 4.0 / 0.9`, so the panel looks identical, only bigger.

## Note on the SMC panel

`smc_stage1..4` are a synthetic three-mode density tempered from the proposal to
the target. They illustrate the mechanism and are not produced from any run, as
the teaser caption states.
