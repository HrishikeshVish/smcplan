#!/bin/bash
# Second half of the kernel ablation: seeds FK FAILED in the main table sweep
# (sweep_fk/table, sweep_cube_leflow/fk). The first half used seeds picked because
# FK beat the baselines, which conditions the table on FK success; these seeds
# remove that bias, so the comparison is over both regimes.
#
# Seed choice is deterministic: the FK-failure list per env/H, sampled at evenly
# spaced indices (4 at H5, 3 at H10/H15). Cube has only 6 FK failures -- all used.
# NOTE reacher: the table sweep ran with unseeded goals, so "FK failed seed X" may
# not reproduce; the fk arm here re-measures it on the fixed goals, so the fk
# column is the authority, not the old table.
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
export DATASET_DIR=$PWD/data PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=$PWD/sweep_aft/ablation_fkfail
CK=ckpt_base_path=$PWD/checkpoints

MALA="++planner.kernel=mala"
HMC="++planner.kernel=hmc ++planner.hmc_leaps=1 ++planner.hmc_beta1=0.9 ++planner.hmc_warmup=10"
AFT="++planner.kernel=aft ++planner.flow_persist=true ++planner.flow_shared=false ++planner.flow_scale=true ++planner.flow_weighted=true ++planner.flow_steps=1 ++planner.flow_lr=0.1 ++planner.flow_logdet_weight=1e-3"

# wait for the FK-success half to release the GPU
while pgrep -f "run_ablation\.sh" > /dev/null; do sleep 30; done

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
      echo "[FKFAIL $k $env H$H s$s] success=$ok exit=$ex"
      return
    fi
    grep -qiE "outofmemoryerror|out of memory" $OUT/run.log && { echo "[FKFAIL $k $env H$H s$s] OOM try $try"; sleep 60; continue; }
    echo "[FKFAIL $k $env H$H s$s] CRASHED: $(grep -oE '[A-Za-z]*Error' $OUT/run.log | head -1)"; return
  done
}

KERNELS=${KERNELS:-"fk mala hmc aft"}
np () { [ "$1" = 15 ] && echo 2 || echo 3; }

for H in 5 10 15; do case $H in 5) S="15 41 58 98";; 10) S="7 52 100";; 15) S="2 53 97";; esac
  for s in $S; do for k in $KERNELS; do
    run $k pointmaze $H $s none model_name=point_maze planner=fk_pointmaze goal_H=$H goal_source=random_state \
      planner.num_particles=$(np $H) planner.num_smooth_samples=1; done; done; done

for s in 5 21 27 35 38 45; do for k in $KERNELS; do
  run $k cube 5 $s egl model_name=lewm_cube planner=fk_cube goal_H=5 goal_source=dset planner.beta_max=10.0; done; done

for H in 10 15; do case $H in 10) S="1 88 95";; 15) S="4 50 99";; esac
  for s in $S; do for k in $KERNELS; do
    run $k reacher $H $s egl model_name=lewm_reacher planner=fk goal_H=$H goal_source=random_state \
      planner.num_particles=8 planner.num_smooth_samples=8; done; done; done

for H in 5 10 15; do case $H in 5) S="2 25 59 100";; 10) S="1 49 100";; 15) S="1 50 100";; esac
  for s in $S; do for k in $KERNELS; do
    run $k pusht $H $s none model_name=pusht planner=fk_pusht goal_H=$H goal_source=dset \
      planner.num_particles=$(np $H) planner.num_smooth_samples=1; done; done; done

echo "=== ABLATION FKFAIL DONE ==="
