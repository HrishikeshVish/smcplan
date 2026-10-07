#!/bin/bash
# Start the PDE arm on each GPU as soon as that GPU's L2-table worker finishes.
cd /home/hviswan/Documents/grasp
for g in 0 1; do
  ( until grep -q "WORKER DONE" sweep_emsmc/table/worker_gpu$g.log 2>/dev/null; do sleep 60; done
    GPU=$g NEED_MIB=12000 bash writeup/run_emsmc_pde.sh >> sweep_emsmc/table/pde_worker_gpu$g.log 2>&1 ) &
done
wait
