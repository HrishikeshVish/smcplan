#!/bin/bash
# Per GPU, after its L2-table worker finishes: cube (L2 arm) -> mop-up of any
# table seeds left unfinished (claims cleared beforehand) -> PDE arm of the table.
cd /home/hviswan/Documents/grasp
for g in 0 1; do
  ( until grep -q "WORKER DONE" sweep_emsmc/table/worker_gpu$g.log 2>/dev/null; do sleep 60; done
    GPU=$g ARM=em bash writeup/run_emsmc_cube.sh >> sweep_cube_leflow/em_worker_gpu$g.log 2>&1
    GPU=$g NEED_MIB=12000 bash writeup/run_emsmc_table.sh >> sweep_emsmc/table/worker_gpu$g.log 2>&1
    GPU=$g NEED_MIB=12000 bash writeup/run_emsmc_pde.sh >> sweep_emsmc/table/pde_worker_gpu$g.log 2>&1 ) &
done
wait
