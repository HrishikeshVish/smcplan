#!/bin/bash
# Final mop-up for the kernel ablation, run after the two Push-T HMC workers finish
# (replaces mopup_ablation.sh, whose 12 GB-gated pass competed with them).
# Claim queues skip finished seeds, so this only re-runs seeds whose claims were
# cleared after a failure.
cd /home/hviswan/Documents/grasp
while pgrep -f "hmc_pusht_worker" > /dev/null; do sleep 60; done
for K in mala rwm hmc; do
  GPU=0 KERNEL=$K ARM=kfix_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/kfix_mopup.log 2>&1
  GPU=0 KERNEL=$K ARM=kfix_$K bash writeup/run_emsmc_cube.sh >> sweep_emsmc/table/kfix_mopup.log 2>&1
done
echo "=== MOPUP DONE $(date +%T) ===" >> sweep_emsmc/table/kfix_mopup.log
