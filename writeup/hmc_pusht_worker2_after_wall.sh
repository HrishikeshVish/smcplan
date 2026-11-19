#!/bin/bash
# Second extra Push-T HMC worker, started only once the main worker's HMC Wall
# cell is complete (Wall run + one chunk-2 Push-T run leave no room for another).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
until [ "$(ls sweep_emsmc/table/wall/kfix_hmc_H10/seed_*/result.txt 2>/dev/null | wc -l)" -ge 100 ]; do sleep 60; done
W=2 NEED_MIB=7000 bash writeup/hmc_pusht_worker.sh
