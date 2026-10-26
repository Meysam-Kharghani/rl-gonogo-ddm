# Analysis lineage and decision log — behavioral-computational study

## Reading rule

This project used staged model development. **Downstream sensitivity analyses were not intended to be symmetric across every candidate.** A candidate that was weak, unstable, superseded by a cleaner comparator, or irrelevant to the revised scientific question was stopped at that gate. Later robustness analyses were concentrated on the surviving primary/component models. Therefore, “no later sensitivity file” is not by itself evidence of missing analysis.

Statuses used here:

- `ADVANCED`: survived the gate and informed the next stage.
- `STOPPED_BY_DESIGN`: evaluated but not advanced because the preceding evidence did not justify additional computation.
- `SUPERSEDED`: answered an earlier question but was replaced by a stronger/cleaner comparator.
- `PRIMARY`: frozen model/component used for the main analysis.
- `ROBUSTNESS`: post-development stress test; not retroactively a preregistered primary analysis.
- `DEFERRED_DEVELOPMENT`: incomplete or specification-conflicted extension excluded from the paper.

## Stage 1 — behavioral and task canonicalization

The 40-participant Cued Go/NoGo dataset, cue mapping, action labels, response-time convention, and strictly lagged history variables were canonicalized before the final downstream benchmarks. Current-trial design variables and non-anticipatory behavioral history form the reference information set.

**Decision:** advance to computational model development after structural/QC reconciliation.

## Stage 2 — RL candidate family

Technical candidate IDs M0–M7 were fit and diagnostically assessed. PSIS/WAIC diagnostics were retained as diagnostics but not trusted as the final model-ranking criterion because Pareto-k behavior was poor. Subject-level cross-validation was available for the models that were advanced in the historical selection path.

- M1, M4, M5: full-fit developmental candidates; no recovered subject-level CV artifact; **STOPPED_BY_DESIGN** rather than retrospectively supplied with a complete sensitivity ladder.
- M0, M2, M3: baseline/intermediate candidates; evaluated where relevant; **STOPPED_BY_DESIGN** after weaker held-out performance relative to the advanced actor/control models.
- M6 and M7: advanced to repeated held-out comparison.

Evidence: `docs/model_registry_rl.csv`, `docs/rl_model_completeness.tsv`, `results/model_development/rl_candidate_selection/`.

## Stage 3 — actor/control RL selection

The repeated comparison between the actor–control model and the actor–control plus adaptive cue-belief model showed seed-specific ordering. The actor–control model had the better mean two-partition held-out score and was the more parsimonious representation.

**Decision:** actor–control RL = `PRIMARY`; actor–control + adaptive cue-belief = developmental sensitivity. The latter was not given a duplicate full downstream sensitivity program because it did not show a stable incremental advantage.

## Stage 4 — initial one-boundary DDM candidates

Early DDM candidates tested categorical task structure, post-error caution, RL actor/control/policy information, EEG augmentation, adaptive-belief augmentation, and policy ablation.

Key gates:

- ERP augmentation did not improve held-out DDM prediction → **STOPPED_BY_DESIGN** for the behavioral DDM path; neural questions moved to companion EEG-computational study.
- Adaptive-belief augmentation did not show a stable advantage across partitions → retained as historical sensitivity, not advanced.
- Actor/control without policy performed substantially worse than the full RL model → established that policy carried important information in the early DDM family.
- The full RL model strongly outperformed the initial categorical design reference → **ADVANCED**.

Evidence: `docs/model_registry_ddm.csv`, `docs/ddm_model_completeness.csv`, `results/model_development/ddm_initial_candidate_comparisons/`.

## Stage 5 — behavioral-history control and question revision

A stronger within-person lagged behavioral-history DDM reference substantially improved prediction relative to the earlier categorical-design reference. This changed the scientific question from “does RL predict response dynamics?” to the stricter “does RL add information beyond a specified lagged behavioral-history baseline?”

**Decision:** history-only DDM became the primary reference; earlier categorical-only comparisons remained supportive/historical.

Evidence: `results/model_development/ddm_behavioral_history_controls/` and the DDM registry.

## Stage 6 — frozen primary decomposition

The final fixed-horizon right-censored family used four core models:

- lagged-history reference;
- actor/control + history;
- policy + history;
- full RL + history.

The historical primary comparison found a detectable policy increment over the prespecified primary history baseline and no detectable actor/control increment after policy was present. These models were the appropriate targets for the full current sensitivity ladder.

Evidence: `results/ddm/`, `results/reporting/`.

## Stage 7 — cue-probability representation

Categorical cue coding, instructed numerical cue probabilities, and strictly lagged empirical cue beliefs were compared in a dedicated family. Empirical updating did not show a stable advantage under the primary prior-strength specification.

**Decision:** retain as supportive cue-representation analysis; do not reinterpret it as an isolated actor/control test.

## Stage 8 — frozen sensitivity/stability analyses

The selected core models were evaluated for regularization choices, alternate outer partition, response-time filtering, group inclusion, repeated outer partitions, nondecision-time settings, lower RT threshold, later-trial horizons, and cue-prior strength where relevant.

These tests were concentrated on the surviving core models by design.

## Stage 9 — post-development robustness

Completed robustness analyses are public under `results/robustness/`:

1. analytic-horizon sensitivity;
2. absolute fixed-horizon predictive checks;
3. predictor-mapping sensitivity;
4. actor–control parameter recovery;
5. richer-history action benchmark;
6. richer-history joint-response benchmark;
7. likelihood-component decomposition.

These analyses **qualify** rather than replace the historical primary analysis. Richer behavioral history attenuates the RL-specific increments, and the DDM policy gain is concentrated largely in survival/by-horizon occurrence likelihood. Predictor-mapping sensitivity also argues against a strong mechanistic interpretation of the exact actor/control-to-DDM placement.

## Stage 10 — deferred prospect-theory extension

A later prospect-theory RL extension was partially run but the proposal and implemented model naming/specification were not aligned and the repeated run was incomplete.

**Decision:** `DEFERRED_DEVELOPMENT`; excluded from the scientific evidence package and from the active code/results tree. The decision is preserved here so its absence cannot be mistaken for an omitted completed analysis.
