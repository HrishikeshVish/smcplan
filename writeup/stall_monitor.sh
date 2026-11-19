#!/bin/bash
# Emits one line per event: a worker with no log progress for STALL_S seconds
# (and again when it recovers), any OOM/CRASH line, and a worker exiting.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
STALL_S=${STALL_S:-1800}
A_LOGS="sweep_emsmc/table/kfix_mopup.log sweep/gds12_gpu0.log sweep_emsmc/table/final_gpu0.log sweep/baselines_ms_gpu0.log sweep/cem12_gpu0.log sweep_emsmc/table/kfix_gpu0.log"
B_LOGS="sweep_emsmc/table/kfix_hmc_wall_w.log sweep_emsmc/table/kfix_hmc_pusht_w1.log sweep_emsmc/table/kfix_hmc_pusht_w2.log"
newest() { local m=0 t; for f in "$@"; do [ -f "$f" ] && t=$(stat -c %Y "$f") && [ "$t" -gt "$m" ] && m=$t; done; echo $m; }
declare -A stalled alive
alive[A]=1; alive[B]=1
for f in $A_LOGS $B_LOGS; do touch -a "$f" 2>/dev/null; done
tail -n0 -F $A_LOGS $B_LOGS 2>/dev/null | grep --line-buffered -E "OOM|CRASH|ABANDONED" | sed -u 's/^/ALERT run failure: /' &
while true; do
  now=$(date +%s)
  for w in A B; do
    if [ $w = A ]; then logs=$A_LOGS; proc="mopup_final.sh"; else logs=$B_LOGS; proc="hmc_pusht_worker"; fi
    if ! pgrep -f "$proc" > /dev/null; then
      [ "${alive[$w]}" = 1 ] && echo "INFO worker $w ($proc) has exited at $(date +%T)"; alive[$w]=0; continue
    fi
    age=$(( now - $(newest $logs) ))
    if [ $age -gt $STALL_S ] && [ "${stalled[$w]}" != 1 ]; then
      stalled[$w]=1
      echo "ALERT stall: worker $w no progress for $((age/60)) min at $(date +%T); GPU0 free MiB: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)"
    elif [ $age -le $STALL_S ] && [ "${stalled[$w]}" = 1 ]; then
      stalled[$w]=0; echo "INFO worker $w recovered at $(date +%T)"
    fi
  done
  sleep 60
done
