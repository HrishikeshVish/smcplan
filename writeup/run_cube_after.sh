#!/bin/bash
# Cube co-residency with the point-maze budget sweep OOMs the card, so wait for
# the queue to drain and then run cube alone on the same GPU the queue used.
while pgrep -f "run_abl_stream.sh budget" >/dev/null; do sleep 60; done
sleep 30
GPU=1 bash "$(dirname "${BASH_SOURCE[0]}")"/run_cube_expand.sh
