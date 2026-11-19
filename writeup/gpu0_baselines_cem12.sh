#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
GPU=0 bash writeup/run_baselines_ms.sh >> sweep/baselines_ms_gpu0.log 2>&1
GPU=0 bash writeup/run_cem12.sh >> sweep/cem12_gpu0.log 2>&1
