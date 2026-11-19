#!/bin/bash
# HMC Wall H10 worker with a low memory gate: a Wall run needs ~1 GB, but the main
# worker's 12 GB gate blocks it whenever a Push-T run (~6 GB) is on GPU0.
# Shares the claim queue, so it only takes seeds nobody has claimed.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=0 KERNEL=hmc ARM=kfix_hmc ENVS=wall HORIZONS=10 NEED_MIB=3000 \
  bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/kfix_hmc_wall_w.log 2>&1
