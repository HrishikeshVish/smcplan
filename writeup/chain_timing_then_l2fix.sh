#!/bin/bash
# 2026-10-07: isolated timing benchmark first (GPU0 must be otherwise idle), then
# the fixed-setting L2 arm (l2fix = pdefix with lambda_v=0) for the PDE-vs-L2
# figure: 3 staggered GPU0 workers on the grid, then cube, then a mop-up pass for
# seeds whose claim was left by an OOM/crash. GPU1 is reserved: never used here.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
# (timing benchmark stopped 2026-10-07 at the user's request: wall clock is not a fair unit across planners; script removed)
for w in 1 2 3; do
  GPU=0 ARM=l2fix LAMBDA_V=0 NEED_MIB=7000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/l2fix_w$w.log 2>&1 &
  sleep 120
done
wait
for w in 1 2; do
  GPU=0 ARM=l2fix NEED_MIB=4000 bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/l2fix_cube_w$w.log 2>&1 &
  sleep 60
done
wait
# mop-up: no worker is alive now, so every remaining claim without a result is stale
for c in sweep_emsmc/table/*/l2fix_H*/seed_*.claim sweep_cube_leflow/l2fix/seed_*.claim; do
  [ -e "$c" ] || continue; s=${c%.claim}
  [ -s "$s/result.txt" ] && continue
  rm -rf "$s" "$c"; echo "cleared $s" >> sweep_emsmc/table/l2fix_mopup.log
done
GPU=0 ARM=l2fix LAMBDA_V=0 NEED_MIB=7000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/l2fix_mopup.log 2>&1
GPU=0 ARM=l2fix NEED_MIB=4000 bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/l2fix_mopup.log 2>&1
echo "=== L2FIX CHAIN DONE $(date +%T) ===" >> sweep_emsmc/table/l2fix_mopup.log
