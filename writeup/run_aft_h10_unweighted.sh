#!/bin/bash
# The three AFT H=10 failures, rerun with the per-particle fit UNWEIGHTED (weights
# cannot change a per-particle minimiser; they only starved low-weight particles).
source ~/miniconda3/etc/profile.d/conda.sh; conda activate grasp
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; unset MUJOCO_GL; export DISPLAY=:2
AFT="++planner.kernel=aft ++planner.flow_persist=true ++planner.flow_shared=false ++planner.flow_scale=true ++planner.flow_weighted=false ++planner.flow_steps=1 ++planner.flow_lr=0.1 ++planner.flow_logdet_weight=1e-3"
R=$PWD/sweep_aft/sanity_unweighted
go () { local env=$1 s=$2; shift 2; local OUT=$R/$env/aft_H10/seed_$s; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=1 GRASP_RESULTS_DIR=$OUT timeout 3600 python plan.py n_evals=1 seed=$s save_images=False goal_H=10 \
    planner.opt_steps=150 planner.eval_every=1 planner.num_particles=3 planner.num_smooth_samples=1 "$@" $AFT \
    ckpt_base_path=$PWD/checkpoints > $OUT/run.log 2>&1
  ok=0; grep -q "Success rate:  1.0" $OUT/run.log && ok=1; echo "[$env H10 s$s aft unweighted] success=$ok exit=$(tail -1 $OUT/seed_success_log.txt 2>/dev/null | awk '{print $5}')"; }
go pointmaze 6 model_name=point_maze planner=fk_pointmaze goal_source=random_state
go pointmaze 23 model_name=point_maze planner=fk_pointmaze goal_source=random_state
go pusht 23 model_name=pusht planner=fk_pusht goal_source=dset
echo "=== DONE ==="
