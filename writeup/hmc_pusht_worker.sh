#!/bin/bash
# Extra GPU0 worker for the HMC ablation on Push-T H10 (shares the claim queue
# with the main worker, so no seed runs twice). Rollout chunk 2 (exact; memory only).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=0 KERNEL=hmc ARM=kfix_hmc ENVS=pusht HORIZONS=10 CHUNK=2 NEED_MIB=${NEED_MIB:-7000} \
  bash writeup/run_emsmc_final.sh >> sweep_emsmc/table/kfix_hmc_pusht_w${W:-1}.log 2>&1
