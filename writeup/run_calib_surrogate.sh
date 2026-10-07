#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export CUDA_VISIBLE_DEVICES=${GPU:-0}
while read env k rule c; do
  python writeup/calib_surrogate.py $env $k $rule $c 30 2>&1 | grep -E "^==|rho=|Error|Traceback"
done <<LIST
pointmaze 4 degree 1.0
pointmaze 64 bandwidth 1.0
reacher 16 degree 3.0
reacher 64 bandwidth 1.0
cube 4 degree 3.0
cube 64 bandwidth 1.0
pusht 64 degree 10.0
pusht 64 bandwidth 1.0
LIST
echo SURROGATE-DONE
