#!/bin/bash
# Tune JALA-EM on Push-T, on the seeds where GD-12 succeeds and JALA-EM (table
# config) fails. Base = the table config (writeup/run_emsmc_table.sh: ULA/JALA,
# P=12, adaptive beta<=1, score M-step, L2 cost, 150 iters); TAG names the
# variant, OVR holds its overrides. Output sweep_emsmc/tune_pusht/<TAG>/H<H>/seed_s.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd /home/hviswan/Documents/grasp
export DATASET_DIR=/home/hviswan/Documents/grasp/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MUJOCO_GL; export DISPLAY=:0
GPU=${GPU:?}; TAG=${TAG:?}; OVR=${OVR:-}; H=${H:-5}
LAMBDA_V=${LAMBDA_V:-0.0}   # roadmap cost weight; "auto" = scale-matched PDE arm
SEEDS=${SEEDS:?seed list}
NEED_MIB=${NEED_MIB:-9000}
case $H in 5) CH=8;; 10) CH=4;; *) CH=2;; esac
CH=${CHUNK:-$CH}
BASE="++planner.kernel=ula planner.precond=none planner.kernel_step_size=0.18 \
planner.num_particles=12 planner.beta_schedule=adaptive planner.beta_max=1.0 \
planner.m_step=score ++planner.lambda_v=$LAMBDA_V planner.opt_steps=150 planner.eval_every=1 \
planner.stop_on_success=true planner.eval_chunk=$CH"
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU; }
for s in $SEEDS; do
  OUT=$PWD/sweep_emsmc/tune_pusht/$TAG/H$H/seed_$s
  [ -s "$OUT/result.txt" ] && continue
  mkdir -p $(dirname $OUT); mkdir "$OUT.claim" 2>/dev/null || continue
  while [ "$(gpu_free)" -lt "$NEED_MIB" ]; do sleep 20; done
  mkdir -p $OUT; t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU GRASP_RESULTS_DIR=$OUT timeout 5400 python plan.py n_evals=1 \
    seed=$s goal_H=$H save_images=False model_name=pusht planner=emsmc_pusht goal_source=dset \
    $BASE $OVR ckpt_base_path=/home/hviswan/Documents/grasp/checkpoints > $OUT/run.log 2>&1
  dt=$(( $(date +%s) - t0 ))
  if [ -s "$OUT/seed_success_log.txt" ]; then
    read ok it < <(tail -1 $OUT/seed_success_log.txt | awk '{print ($1>=1.0)?1:0, $5}')
    snr=$(grep -o "snr=[0-9.]*" $OUT/run.log | awk -F= '{s+=$2;n++}END{if(n)printf "%.2f",s/n}')
    ess=$(grep -o "ESS=[0-9.]*/" $OUT/run.log | tr -d 'ESS=/' | awk '{s+=$1;n++}END{if(n)printf "%.1f",s/n}')
    rs=$(grep -o "resamples=[0-9]*" $OUT/run.log | tail -1 | cut -d= -f2)
    div=$(grep -c DIVERGED $OUT/run.log)
    echo "$ok $it $dt $snr $ess $rs $div" > $OUT/result.txt
    echo "[$TAG H$H s$s] ok=$ok it=$it snr=$snr ess=$ess resamples=$rs diverged=$div ${dt}s"
    rmdir "$OUT.claim"
  else
    echo "[$TAG H$H s$s] CRASH/OOM ${dt}s"
  fi
done
echo "=== $TAG H$H DONE $(date +%T) ==="
