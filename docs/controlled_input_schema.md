# Controlled input schema

Full refitting requires participant-level inputs that are intentionally absent from the public repository. For an authorized rerun, materialize them below `controlled_inputs/` using the formal paths below or pass equivalent explicit command-line paths where supported.

```text
controlled_inputs/
  behavioral_trial_data/                         Participant trial-level behavioral source files
  rl_crossvalidated_posteriors/                  Fold-specific train-only actor–control RL posterior/CV artifacts
  rl_full_fit/
    actor_control_primary/idata.nc                Full-fit posterior used only for parameter-recovery simulation
  repeated_partition_outputs.csv                 Optional map for completed repeated-partition reruns
```

Required behavioral information includes a stable participant identifier, trial order, cue condition, Go/NoGo target class, response presence, response time, correctness/error variables, and risk-group membership where required by the hierarchical actor–control RL model.

The instructed cue-probability mapping is public in `configs/instructed_cue_probabilities.json`.

The downstream primary pipeline uses fold-specific posterior objects fitted to outer-training participants only. Held-out participant trajectories are generated as population-predictive/new-participant trajectories; participant-specific fitted random effects from the held-out participant must not be used.

Response time is represented in seconds internally in the first-passage implementation, while command-line horizon/threshold arguments use milliseconds. The frozen primary analysis uses a 700-ms analytic horizon and a 100-ms lower RT validity threshold.

## Repeated-partition output map

If repeated controlled reruns are regenerated, create `controlled_inputs/repeated_partition_outputs.csv` with:

```text
seed,out_dir
2026,reproduced/controlled_rerun/repeated_partitions/seed_2026
7777,reproduced/controlled_rerun/repeated_partitions/seed_7777
...
```

The map is an execution locator only. It is not distributed with participant identifiers and does not alter the frozen seed list in `configs/repeated_partition_seeds.csv`.
