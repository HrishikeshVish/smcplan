#!/bin/bash

# Define the log file
LOG_FILE="seed_success_log.txt"

# Clear the log file if it exists from a previous run
> "$LOG_FILE"

NUM_SEEDS=300
echo "Starting LeWM reacher evaluation across $NUM_SEEDS seeds..."
echo "---------------------------------------"

for seed in $(seq 1 $NUM_SEEDS); do
    echo "Running Seed: $seed..."

    python plan.py model_name=lewm_reacher n_evals=1 planner=cem goal_H=5 goal_source='dset' seed=$seed save_images=False

    if [ -f "$LOG_FILE" ]; then
        LATEST_SCORE=$(tail -n 1 "$LOG_FILE")
        echo "Seed $seed Result: $LATEST_SCORE"
    fi
    echo "---------------------------------------"
done

echo "All $NUM_SEEDS seeds completed."
if [ -s "$LOG_FILE" ]; then
    awk '{ sum += $1; n++ } END { if (n > 0) printf "Average Success Rate (%d Seeds): %.4f\n", n, sum / n; }' "$LOG_FILE"
else
    echo "Error: $LOG_FILE is empty or was not created."
fi
