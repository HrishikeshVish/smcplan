# Pre-registration: trajectory efficiency, roadmap vs raw-L2 tilt

Written BEFORE any seed of the cube expansion (38 new seeds, 76 runs) produced a
result. The 6 cube pairs already in sweep_aft/rawl2 are excluded from the test
set; they are pilot data and contributed to choosing the hypothesis.

## Metrics (fixed, four, chosen on physical grounds -- not on which way they leaned)

Evaluated ONLY on pairs where both arms succeed, since below the success
threshold "closer" is not an improvement and above it both runs failed.

1. control effort    sum |a| over executed actions
2. path length       arclength in the components the success test scores
3. time to reach     first frame index with margin < 1, / (T-1)
4. backtracking      sum of positive increments of the margin, / starting margin

Excluded by the same standard: action jerk, tortuosity, dwell, worst excursion,
approach AUC.

## Scoring

Per-pair log ratio log(rawL2 / roadmap); ratio > 1 favours roadmap.
Wilcoxon signed-rank, two-sided, on the 4 metrics with Holm correction.
Success itself is scored separately by the sign test on discordant pairs.

## Predictions

P1  roadmap >= rawL2 on success across the 44 cube pairs.
P2  at least 2 of the 4 metrics show a median ratio > 1 with Holm-corrected p < 0.05.
P3  the effect, if present, does NOT appear when the same test is run on the
    both-succeed stratum of an env where the mechanism should not apply.

## Standing risk, recorded now

On the 5 pilot cube pairs the direction is already AGAINST P2: control effort
0.892 and backtracking 0.226 both favour rawL2. n=5 is not evidence, but this
test is entered with the pilot leaning the wrong way, and that is the honest
prior. If P2 fails, the efficiency claim is dropped -- it is not re-run with a
different metric set.

## Current-data status of the four (main sweep, n=125, for the record)

metric           median ratio   95% CI            Wilcoxon p
control effort   1.0109         [0.9884, 1.0292]  0.687
path length      1.0051         [0.9726, 1.0329]  0.930
time to reach    1.0000         [1.0000, 1.0000]  0.227
backtracking     1.0231         [0.9408, 1.1002]  0.949

All four CIs span 1.0.
