#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export CUDA_VISIBLE_DEVICES=${GPU:-0}
while read env k rule c; do
  for st in 250 1000 3000; do
    python writeup/calib_surrogate.py $env $k $rule $c 20 $st 2>&1 | grep -E "^==|rho=|Error|Traceback"
  done
done <<LIST
pointmaze 4 degree 1.0
wall 32 degree 30.0
cube 4 degree 3.0
cube 16 degree 1.0
reacher 16 degree 3.0
pusht 64 degree 10.0
LIST
echo SURROGATE2-DONE
