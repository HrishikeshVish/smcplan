#!/bin/bash
# Kernel ablation: swap the JALA (ULA+Jarzynski) E-step for MALA, RWM and HMC with
# everything else fixed (L2 cost, P=12, adaptive beta<=1, score M-step, 150 iters).
# H=10 on the four table envs (100 paired seeds, same seeds as em_H10) + cube (50).
# One worker per GPU on the shared claim queues; kernels in order mala, rwm, hmc.
cd /home/hviswan/Documents/grasp
for g in 0 1; do
  ( for K in ${KERNELS:-mala rwm hmc}; do
      GPU=$g KERNEL=$K ARM=k_$K HORIZONS=10 NEED_MIB=12000 bash writeup/run_emsmc_table.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
      GPU=$g KERNEL=$K ARM=k_$K bash writeup/run_emsmc_cube.sh \
        >> sweep_emsmc/table/kernel_worker_gpu$g.log 2>&1
    done ) &
done
wait
