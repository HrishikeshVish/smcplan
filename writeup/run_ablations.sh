#!/bin/bash
# Appendix ablations and sensitivity of our planner (pdefix configuration of
# Table 1), one setting changed per arm, on the 100 seeds of Point maze H=10
# (DINO-WM) and Reacher H=10 (LeWM). Paired with the pdefix arm of
# sweep_emsmc/table. The iteration budget K is not run: no setting of the
# planner depends on K (m_smooth_samples=1, adaptive beta), so success at budget
# K is read off the success-against-iteration curves.
# Usage: GPU=0 CELLS="pointmaze reacher" ARMS="..." bash writeup/run_ablations.sh
cd /home/hviswan/Documents/grasp
GPU=${GPU:?set GPU}
CELLS=${CELLS:-"pointmaze reacher"}
ROOT=$PWD/sweep_emsmc/ablate
arms_for() {  # env -> "name|override" lines
  local e=$1 k c
  case $e in pointmaze) k=4; c=1.0;; reacher) k=16; c=3.0;; esac
  cat <<EOF
N2|planner.num_particles=2
N4|planner.num_particles=4
N8|planner.num_particles=8
N16|planner.num_particles=16
N32|planner.num_particles=32
tau0|planner.resample_frac=0.0
tau025|planner.resample_frac=0.25
tau075|planner.resample_frac=0.75
tau1|planner.resample_frac=1.0
uniformw|++planner.uniform_weights=true
alpha025|++planner.alpha_scale=0.25
alpha05|++planner.alpha_scale=0.5
alpha2|++planner.alpha_scale=2.0
alpha4|++planner.alpha_scale=4.0
eta009|planner.kernel_step_size=0.09
eta036|planner.kernel_step_size=0.36
sigp05|planner.sigma_p=0.5
sigp2|planner.sigma_p=2.0
khalf|planner.graph_k_neighbors=$((k / 2))
kdouble|planner.graph_k_neighbors=$((k * 2))
cthird|++planner.heat_c=$(python3 -c "print($c/3)")
ctriple|++planner.heat_c=$(python3 -c "print($c*3)")
lamv05|++planner.lambda_v_scale=0.5
lamv2|++planner.lambda_v_scale=2.0
ophop|++planner.v_operator=hop
opgreen|++planner.v_operator=green
opsp|++planner.v_operator=shortest_path
EOF
}
for e in $CELLS; do
  arms_for $e | while IFS='|' read -r name ov; do
    [ -n "$ARMS" ] && ! echo " $ARMS " | grep -q " $name " && continue
    ARM=abl_$name EXTRA="$ov" ENVS=$e HORIZONS=10 ROOT=$ROOT GPU=$GPU \
      bash writeup/run_emsmc_final.sh < /dev/null
  done
done
echo "=== ABLATIONS DONE $(date +%T) ==="
