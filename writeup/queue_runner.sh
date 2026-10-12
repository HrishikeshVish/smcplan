#!/bin/bash
# Sequential GPU0 job queue for the reviewer round (one job at a time, GPU1 is
# reserved for another project). Runs queue_rev/*.sh in name order; a job is
# renamed .running then .done. New jobs can be dropped in at any time; the
# runner exits when the queue is empty and queue_rev/STOP exists.
cd /home/hviswan/Documents/grasp
Q=$PWD/queue_rev
while true; do
  next=$(ls $Q/*.sh 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then [ -e $Q/STOP ] && break; sleep 60; continue; fi
  base=${next%.sh}; mv "$next" "$base.running"
  echo "[$(date +%F' '%T)] start $(basename $base)" >> $Q/runner.log
  bash "$base.running" >> "$base.log" 2>&1
  echo "[$(date +%F' '%T)] done  $(basename $base) exit=$?" >> $Q/runner.log
  mv "$base.running" "$base.done"
done
echo "[$(date +%F' '%T)] runner exit" >> $Q/runner.log
