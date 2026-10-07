#!/bin/bash
# Wait for the reacher+pointmaze sweep to finish, then run Push-T.
cd /home/hviswan/Documents/grasp
echo "=== QUEUE: waiting for main sweep ($(date +%T)) ==="
while pgrep -f ".run_sweep_active.sh" > /dev/null; do sleep 60; done
echo "=== QUEUE: main sweep done, starting Push-T ($(date +%T)) ==="
cp writeup/run_sweep.sh writeup/.run_sweep_pusht.sh
bash writeup/.run_sweep_pusht.sh pusht
echo "=== QUEUE: ALL DONE ($(date +%T)) ==="
