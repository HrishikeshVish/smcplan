#!/bin/bash
# Rebuild a result of the paper from finished runs. No GPU: these read the
# per-seed result.txt files the sweeps wrote and print the table or write the
# figure. writeup/equal_particle_stats.py is the single source of per-seed
# outcomes behind every table, so a number traces back to the seed directory it
# came from.
#
#   bash scripts/reproduce.sh table1       # Tab. 1, main planner comparison
#   bash scripts/reproduce.sh cube         # Tab. 4, cube under LeFlow's protocol
#   bash scripts/reproduce.sh learned      # LeFlow and GC-IDM, the planners that train a network
#   bash scripts/reproduce.sh ablations    # M-step, budget, SMC and sensitivity
#   bash scripts/reproduce.sh ci           # Wilson intervals and paired McNemar
#   bash scripts/reproduce.sh figures      # every figure the paper includes
#   bash scripts/reproduce.sh all
#
# Tables land in OUT_DIR (this repo by default) as the .tex files the paper
# \inputs; figures land in figs/.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"
[ $# -ge 1 ] || { sed -n 2,16p "$0"; exit 1; }

one() {
  case $1 in
    table1)    "$PYTHON" writeup/make_equal_particle_tables.py ;;
    cube)      "$PYTHON" writeup/make_cube_table.py ;;
    learned)   "$PYTHON" writeup/make_learned_table.py ;;
    ablations) "$PYTHON" writeup/make_ablation_tables.py ;;
    ci)        "$PYTHON" writeup/make_paired_table.py ;;
    figures)
      "$PYTHON" figs/fig_teaser.py              # Fig. 1
      "$PYTHON" figs/fig_success_curves.py      # Fig. 4 panels and legend, and the appendix curves
      "$PYTHON" figs/fig_paired_summary.py      # Fig. 5
      "$PYTHON" figs/fig_cost_structures.py     # Fig. 9, the two cost structures
      "$PYTHON" figs/fig_paired.py              # appendix, per-environment paired panels
      "$PYTHON" figs/fig_kernels.py             # appendix, propagation kernels
      "$PYTHON" figs/fig_abl_knobs.py           # appendix, sensitivity
      "$PYTHON" figs/fig_particles.py           # appendix, particle count
      "$PYTHON" writeup/pde_baseline_stats.py && "$PYTHON" figs/fig_pde_transfer.py
      for e in pointmaze pusht cube; do "$PYTHON" figs/frame_strip.py "$e"; done ;;
    *) echo "unknown target $1"; exit 1 ;;
  esac
}

if [ "$1" = all ]; then for t in table1 cube learned ablations ci figures; do one "$t"; done; else one "$1"; fi
