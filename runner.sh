#!/bin/bash

# Define the log file
LOG_FILE="seed_success_log.txt"

# Clear the log file if it exists from a previous run
> "$LOG_FILE"

echo "Starting evaluation across 100 seeds..."
echo "---------------------------------------"

# Loop through seeds 1 to 100
for seed in {1..100}; do
    echo "Running Seed: $seed..."
    
    # Run the python script
    python plan.py model_name=pusht n_evals=1 planner=gd goal_H=5 goal_source='dset' planner.opt_steps=150 seed=$seed
    
    # Optional: Print the success rate of the run that just finished
    if [ -f "$LOG_FILE" ]; then
        LATEST_SCORE=$(tail -n 1 "$LOG_FILE")
        echo "Seed $seed Max Success: $LATEST_SCORE"
    fi
    echo "---------------------------------------"
done

# Compute the average over all seeds using awk
echo "All 100 seeds completed."
if [ -s "$LOG_FILE" ]; then
    awk '{ sum += $1; n++ } END { if (n > 0) printf "Average Success Rate (100 Seeds): %.4f\n", sum / n; }' "$LOG_FILE"
else
    echo "Error: $LOG_FILE is empty or was not created."
fi