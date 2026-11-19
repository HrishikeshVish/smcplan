#!/bin/bash
# After both GPU0 workers finish, one more pass over the kernel ablation: the
# claim queues skip finished seeds, so this only re-runs seeds whose claims were
# cleared after a failure (e.g. pointmaze kfix_mala H10 #1, OOM in a worker overlap).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
while pgrep -f "queue_final2.sh" > /dev/null || pgrep -f "second_worker2_gpu0.sh" > /dev/null; do sleep 120; done
for K in mala rwm hmc; do
  GPU=0 KERNEL=$K ARM=kfix_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/kfix_mopup.log 2>&1
  GPU=0 KERNEL=$K ARM=kfix_$K bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/kfix_mopup.log 2>&1
done
