# Methods and reproducibility details

## Participants and risk grouping

The analysis includes 40 participants, 20 high-risk and 20 low-risk. Selection followed Nazari et al. (2024): adjusted BART scores in the top 20% of the distribution defined the high-risk stratum and scores in the bottom 20% defined the low-risk stratum, with random selection from these extreme groups. Twenty participants were selected per group. Data for some participants had been recorded previously within the same research program but were not included in the earlier published study and were included here. The final sample comprised 36 right-handed and four left-handed participants.

## Stacked subject-level cross-validation

The reinforcement-learning model is re-estimated within each outer-training participant fold. Both downstream-training and outer-test latent trajectories are generated from that fold's train-only M6 posterior. Downstream models are then fitted only to outer-training participants and scored on untouched outer-test participants.

## M6 train-fold sampling

For the primary seed-2026 folds, M6 used four NUTS chains, 600 tuning iterations followed by 600 retained draws per chain, four cores, and `target_accept=0.90`. Fold-specific seeds were 2123, 2220, 2317, 2414, and 2511. The aggregate diagnostics are in `results/ddm/actor_control_fold_posterior_diagnostics.csv`.

## Fixed-horizon right-censored DDM

The primary DDM uses a one-boundary first-passage likelihood with a fixed 700-ms analytic horizon and t0 fixed at 30 ms. Presses from 100 through 700 ms contribute the first-passage density. No response by 700 ms contributes survival through the horizon. Presses after 700 ms are retained in the behavioral sequence but treated as right-censored for the current-trial likelihood. Press trials with missing RT or RT below 100 ms are excluded from the DDM likelihood.

Strictly lagged behavioral controls and empirical cue beliefs are constructed from the complete behavioral sequence before likelihood-specific exclusion/censoring, so the observation model does not retroactively erase preceding behavior from later-trial predictors.

## Public versus controlled workflows

The public workflow regenerates figures from aggregate frozen tables. Full RL and DDM refitting additionally requires controlled trial-level inputs and fold-specific posterior objects.
