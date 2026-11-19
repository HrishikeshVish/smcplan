#!/bin/bash
# Launch the JALA-EM main-table sweep: one worker per GPU on the shared claim
# queue of run_emsmc_table.sh (two DINO-WM runs do not fit on one 16 GB card at
# H>=10). NEED_MIB gates each run on free memory so a worker waits rather than
# OOMing a co-tenant's job. Logs: sweep_emsmc/table/worker_gpu<g>.log
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
mkdir -p sweep_emsmc/table
for g in ${GPUS:-0 1}; do
  GPU=$g NEED_MIB=${NEED_MIB:-12000} ENVS="${ENVS:-pointmaze wall pusht reacher}" \
    setsid nohup bash writeup/run_emsmc_table.sh >> sweep_emsmc/table/worker_gpu$g.log 2>&1 &
done
