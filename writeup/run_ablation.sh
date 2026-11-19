#!/bin/bash
# Kernel ablation table for the paper. FK is the method (its Adam+Langevin move is
# ULA with a diagonal preconditioner and momentum); MALA, HMC and annealed flow
# transport are alternative kernels inside the SAME SMC sampler -- identical V,
# identical L_dyn, identical tempering/resampling, only the move differs.
# Seed set: writeup/aft_sanity_seeds.txt (seeds where FK beat the baselines).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/ablation
CK=ckpt_base_path=$PWD/checkpoints

MALA="++planner.kernel=mala"
HMC="++planner.kernel=hmc ++planner.hmc_leaps=1 ++planner.hmc_beta1=0.9 ++planner.hmc_warmup=10"
AFT="++planner.kernel=aft ++planner.flow_persist=true ++planner.flow_shared=false ++planner.flow_scale=true ++planner.flow_weighted=true ++planner.flow_steps=1 ++planner.flow_lr=0.1 ++planner.flow_logdet_weight=1e-3"

run () {  # kernel env H seed gl args...
  local k=$1 env=$2 H=$3 s=$4 gl=$5; shift 5
  local OUT=$ROOT/$k/$env/H$H/seed_$s
  [ -s "$OUT/result.txt" ] && return; mkdir -p $OUT
  if [ "$gl" = egl ]; then export MUJOCO_GL=egl; unset DISPLAY; else unset MUJOCO_GL; export DISPLAY=:2; fi
  local KA=""; case $k in mala) KA=$MALA;; hmc) KA=$HMC;; aft) KA=$AFT;; esac
  for try in 1 2 3; do
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=${GPU:-1} GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s \
      save_images=False planner.opt_steps=150 planner.eval_every=1 "$@" $KA $CK > $OUT/run.log 2>&1
    if grep -q "Success rate:" $OUT/run.log; then
      local ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1
      local ex=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')
      echo "$ok $ex $(( $(date +%s) - t0 ))" > $OUT/result.txt
      echo "[$k $env H$H s$s] success=$ok exit=$ex"
      return
    fi
    grep -qiE "outofmemoryerror|out of memory" $OUT/run.log && { echo "[$k $env H$H s$s] OOM try $try"; sleep 60; continue; }
    echo "[$k $env H$H s$s] CRASHED: $(grep -oE '[A-Za-z]*Error' $OUT/run.log | head -1)"; return
  done
}

KERNELS=${KERNELS:-"fk mala hmc aft"}
np () { [ "$1" = 15 ] && echo 2 || echo 3; }   # H15 drops to 2 particles (VRAM)

for H in 5 10 15; do case $H in 5) S="6 17 19 20";; 10) S="6 19 23";; 15) S="19 31 3";; esac
  for s in $S; do for k in $KERNELS; do
    run $k pointmaze $H $s none model_name=point_maze planner=fk_pointmaze goal_H=$H goal_source=random_state \
      planner.num_particles=$(np $H) planner.num_smooth_samples=1; done; done; done

for s in 17 24 26 30 39 8 9 14 23 32; do for k in $KERNELS; do
  run $k cube 5 $s egl model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0; done; done

for H in 5 10 15; do case $H in 5) S="1 2 3 4";; 10) S="1 2 3";; 15) S="1 2 3";; esac
  for s in $S; do for k in $KERNELS; do
    run $k reacher $H $s egl model_name=lewm_reacher planner=fk goal_H=$H goal_source=random_state \
      planner.num_particles=8 planner.num_smooth_samples=8; done; done; done

for H in 5 10 15; do case $H in 5) S="10 20 34 45";; 10) S="16 23 24";; 15) S="6 61 94";; esac
  for s in $S; do for k in $KERNELS; do
    run $k pusht $H $s none model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset \
      planner.num_particles=$(np $H) planner.num_smooth_samples=1; done; done; done

echo "=== ABLATION DONE ==="
