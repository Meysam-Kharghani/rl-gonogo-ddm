# Manuscript-to-repository alignment

This document maps the final behavioral-computational manuscript to the frozen public evidence. It exists to prevent a future reporting update from silently drifting away from the analysis package.

## Main narrative

| Manuscript element | Scientific role | Public evidence |
|---|---|---|
| Figure 1 | Stage-gated development, selection, cross-fitting, robustness logic | `docs/analysis_lineage_and_decision_log.md`, `docs/analysis_stage_registry.tsv` |
| Figure 2 | Action-occurrence RL increments under compact versus richer history | `results/action/key_contrasts.csv`, `results/robustness/richer_history_action/richer_history_planned_contrasts.csv` |
| Figure 3 | First-passage RL increments under compact versus richer history | `results/ddm/core_key_contrasts.csv`, `results/robustness/richer_history_joint_response/rich_ddm_planned_contrasts.csv` |
| Figure 4 | Policy increment decomposed into survival/censoring and observed-response density | `results/robustness/likelihood_component_decomposition/stratified_ddm_contrasts.csv` |
| Table 1 | Compact-versus-richer history summary for action and first passage | `results/reporting/table_data/manuscript_table_01_incremental_information.csv` |

## Supplementary figures

| Figure | Role | Public evidence |
|---|---|---|
| Figure S1 | Analytic-horizon sensitivity | `results/robustness/analytic_horizon_sensitivity/horizon_sensitivity_planned_contrasts.csv` |
| Figure S2 | Repeated outer-partition stability | `results/ddm/repeated_cv_seeds.csv`, `results/ddm/repeated_cv_overall.csv` |
| Figure S3 | Predictor-to-first-passage mapping sensitivity | `results/robustness/predictor_mapping_sensitivity/mapping_planned_contrasts.csv` |
| Figure S4 | Participant-level parameter recovery | `results/robustness/actor_control_parameter_recovery/actor_control_parameter_recovery_correlations_by_rep.csv` |
| Figure S5 | Cue-belief prior-strength sensitivity | `results/ddm/prior_strength_sensitivity.csv` |

## Supplementary tables

The publication-facing table exports are stored as `results/reporting/table_data/supplementary_table_01.csv` through `supplementary_table_18.csv`. Machine-readable labels are normalized to the repository naming policy; numerical values and scientific content are unchanged. Their evidence roles are:

1. stage-gated analysis lineage → `docs/analysis_stage_registry.tsv`;
2. conceptual RL candidate family → `docs/model_registry_rl.csv`, `results/model_development/rl_candidate_selection/`;
3. actor–control prior specification → `shared/rl/actor_control_model_family.py`;
4. primary action decomposition → `results/action/key_contrasts.csv`;
5. action sensitivities → `results/action/sensitivity_key_contrasts.csv`;
6. richer-history feature definition → `code/03_robustness/richer_history_action_benchmark.py` and `code/03_robustness/richer_history_joint_response_benchmark.py`;
7. richer-history action contrasts → `results/robustness/richer_history_action/richer_history_planned_contrasts.csv`;
8. conceptual fixed-horizon first-passage models → `docs/model_nomenclature.md`, `docs/model_registry_ddm.csv`;
9. primary first-passage decomposition → `results/ddm/core_key_contrasts.csv`;
10. richer-history first-passage contrasts → `results/robustness/richer_history_joint_response/rich_ddm_planned_contrasts.csv`;
11. lower-RT-threshold and nondecision-time sensitivity → `results/ddm/rtmin_sensitivity.csv`, `results/ddm/t0_sensitivity.csv`;
12. absolute response-by-horizon calibration → `results/robustness/absolute_predictive_checks/overall_event_calibration.csv`;
13. response-time quantile calibration → `results/robustness/absolute_predictive_checks/conditional_rt_quantiles.csv`;
14. likelihood-component and target-stratified contrasts → `results/robustness/likelihood_component_decomposition/stratified_ddm_contrasts.csv`;
15. alternative predictor mappings → `results/robustness/predictor_mapping_sensitivity/mapping_planned_contrasts.csv`;
16. simulation-based recovery summaries → `results/robustness/actor_control_parameter_recovery/actor_control_parameter_recovery_summary.csv`;
17. cue-probability representation contrasts → `results/ddm/prior_key_contrasts.csv`;
18. technical crosswalk → `docs/model_nomenclature.md`.

## Reporting rules

- Technical IDs remain available for computational traceability but are not the primary language of manuscript prose.
- Repeated partitions are stability analyses on the same 40 participants, not independent replications.
- The 700-ms value is an analytic observation horizon, not a verified task-enforced deadline.
- Mapping sensitivity prevents a uniquely mechanistic interpretation of the historical actor/control-to-first-passage parameter placement.
- Richer-history analyses qualify the compact-history primary result and are part of the final interpretation, not optional postscript material.
