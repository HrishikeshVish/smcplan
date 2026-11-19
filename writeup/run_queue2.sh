#!/bin/bash
# Serialise: GPU 1 is shared with other jobs, so one sweep at a time.
while pgrep -f "run_abl_particles.sh" >/dev/null; do sleep 60; done
sleep 20
bash "$(dirname "${BASH_SOURCE[0]}")"/run_abl_freeze.sh
