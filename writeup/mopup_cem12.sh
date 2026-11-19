#!/bin/bash
# Once both CEM-12 workers have finished their pass, re-run any CEM-12 seed whose
# claim was cleared after a failure (e.g. Push-T H15 #33: torch.hub hang).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
while pgrep -f "run_cem12.sh" > /dev/null; do sleep 60; done
GPU=0 NEED_MIB=4000 ENVS="pusht cube" bash writeup/run_cem12.sh >> sweep/cem12_gpu0_w2.log 2>&1
