#!/bin/bash
# Wait for GPU0 to actually free up, then resume the sweep.
#
# Context: another user's job occupied GPU0, so the sweep's per-run memory gate
# timed out repeatedly and marked seeds ABANDONED (~1/hour) while doing no work.
# Left running it would have churned through all 2,700 seeds as abandonments and
# reported "complete". Abandoned seeds write no result.txt, so resuming simply
# re-runs them -- nothing is lost by stopping and restarting here.
#
# Requires the card to hold enough free memory for a sustained stretch, so we
# don't start into a brief dip between someone else's allocations.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

NEED_MIB=10000        # headroom for a run plus slack
STABLE_CHECKS=10      # consecutive passes required
INTERVAL=60

gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0; }

echo "=== WATCH: waiting for GPU0 >= ${NEED_MIB} MiB for $((STABLE_CHECKS*INTERVAL))s ($(date +%T)) ==="
ok=0
while true; do
    f=$(gpu_free)
    if [ "$f" -ge "$NEED_MIB" ]; then
        ok=$((ok + 1))
        [ $((ok % 5)) -eq 0 ] && echo "=== WATCH: ${ok}/${STABLE_CHECKS} stable (free=${f}) $(date +%T) ==="
        if [ "$ok" -ge "$STABLE_CHECKS" ]; then break; fi
    else
        if [ "$ok" -gt 0 ]; then echo "=== WATCH: reset, free=${f} ($(date +%T)) ==="; fi
        ok=0
    fi
    sleep $INTERVAL
done

echo "=== WATCH: GPU free, resuming sweep ($(date +%T)) ==="
cp writeup/run_sweep.sh writeup/.run_sweep_resumed.sh
bash writeup/.run_sweep_resumed.sh reacher pointmaze pusht
echo "=== WATCH: SWEEP DONE ($(date +%T)) ==="
