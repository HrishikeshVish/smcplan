#!/bin/bash
# Report seeds that have a run directory but no result.txt, i.e. runs that died
# (almost always CUDA OOM under contention) and were skipped rather than retried.
#
# Both sweep scripts drop these silently: run_pusht_baselines.sh has no retry
# loop, and run_fk_sweep.sh only retries when the log contains the literal
# string "OutOfMemoryError" -- but PyTorch also raises
# "torch.AcceleratorError: CUDA error: out of memory", which that test misses.
# Failures write no result.txt, so re-running the sweep fills exactly the gaps.
cd /home/hviswan/Documents/grasp
total=0
for arm in sweep/pusht/*_H*/ sweep_fk/table/wall/*_H*/ sweep_pusht_h25/*/; do
  [ -d "$arm" ] || continue
  miss=""
  for d in "$arm"seed_*; do
    [ -d "$d" ] || continue
    [ -s "$d/result.txt" ] || miss="$miss $(basename $d | sed 's/seed_//')"
  done
  n=$(echo $miss | wc -w)
  if [ "$n" -gt 0 ]; then
    printf "%-38s %2d missing:%s\n" "$arm" "$n" "$miss"
    total=$((total + n))
  fi
done
echo "---"
echo "total gaps: $total"
