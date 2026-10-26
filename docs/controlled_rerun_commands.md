# Controlled rerun commands

These commands are templates for an **authorized controlled-input rerun**. They write to `reproduced/controlled_rerun/` and never overwrite frozen public results under `results/`. Run them from the repository root after materializing the inputs described in `docs/controlled_input_schema.md`.

## 1. Primary action-occurrence benchmark

```bash
python code/01_primary_action/action_occurrence_benchmark.py \
  --nested-script shared/rl/nested_crossvalidated_latent_generation.py \
  --rl-script shared/rl/actor_control_model_family.py \
  --ddm-script shared/ddm/one_boundary_diffusion.py \
  --rl-out controlled_inputs/rl_crossvalidated_posteriors \
  --data-root controlled_inputs/behavioral_trial_data \
  --cue-probabilities configs/instructed_cue_probabilities.json \
  --out reproduced/controlled_rerun/primary_action
```

## 2. Primary fixed-horizon first-passage benchmark

```bash
python code/02_primary_joint_response/joint_response_time_benchmark.py \
  --nested-script shared/rl/nested_crossvalidated_latent_generation.py \
  --rl-script shared/rl/actor_control_model_family.py \
  --ddm-script shared/ddm/one_boundary_diffusion.py \
  --rl-out controlled_inputs/rl_crossvalidated_posteriors \
  --data-root controlled_inputs/behavioral_trial_data \
  --cue-probabilities configs/instructed_cue_probabilities.json \
  --out reproduced/controlled_rerun/primary_joint_response
```

## 3. Repeated outer-partition stability summary

After completing the frozen repeated-partition reruns, create the formal output map described in `docs/controlled_input_schema.md` and run:

```bash
python code/02_primary_joint_response/summarize_repeated_partitions.py \
  --partition-map controlled_inputs/repeated_partition_outputs.csv \
  --out reproduced/controlled_rerun/repeated_partition_summary
```

Repeated partitions reuse the same participants and are interpreted as stability analyses, not independent replications.

## 4. Completed robustness analyses

The robustness entry points write below `reproduced/controlled_rerun/robustness/` when their default output arguments are used:

```bash
python code/03_robustness/absolute_predictive_checks.py
python code/03_robustness/analytic_horizon_sensitivity.py
python code/03_robustness/predictor_mapping_sensitivity.py
python code/03_robustness/richer_history_action_benchmark.py
python code/03_robustness/richer_history_joint_response_benchmark.py
python code/03_robustness/likelihood_component_decomposition.py
```

Parameter recovery additionally requires the full-fit posterior declared in the controlled-input schema:

```bash
python code/03_robustness/actor_control_parameter_recovery.py \
  --source-idata controlled_inputs/rl_full_fit/actor_control_primary/idata.nc
```

## 5. Historical model development

`code/00_model_development/` preserves portable historical source snapshots for candidate screening and historical controls. These scripts document model-development stages; they are not a single recommended current rerun pipeline. Their status and stopping/advancement logic are defined in `docs/analysis_lineage_and_decision_log.md` and `docs/analysis_stage_registry.tsv`.
