# behavioral-computational study model-development evidence

The public package preserves the candidate-selection record separately from the frozen primary and robustness analyses.

## RL family

Historical candidates M0–M7 have frozen full-fit summaries, diagnostics, posterior-predictive summaries, and LOO/WAIC diagnostic files. PSIS/WAIC is retained as a diagnostic rather than the model-selection criterion because Pareto-k diagnostics were poor.

Subject-level cross-validation exists for the candidates that were advanced through the historical selection path. M1, M4, and M5 were fit but were not advanced into a recovered subject-level CV branch. The project does **not** manufacture a retrospective symmetric CV/sensitivity ladder for candidates that were stopped at an earlier gate.

The repeated held-out comparison that informed the final actor–control versus actor–control-plus-adaptive-belief decision is distributed under `results/model_development/rl_candidate_selection/`.

## DDM family

`docs/ddm_model_completeness.csv` and `docs/model_registry_ddm.csv` document D0–D15 and D6A. Historical candidate comparisons, behavioral-history controls, transition analyses, and the final fixed-horizon decomposition are kept as distinct stages.

Sensitivity analyses are intentionally concentrated on models that survived the staged development process and directly support the final scientific contrasts. Weak, unstable, targeted-control, or superseded models are retained as development evidence but are not assigned an artificial full sensitivity ladder.

## Additional supportive analyses

Compact outputs are retained for response-time predictive-fit checks, conditional DDM model recovery, joint-model validation, and exploratory entropy-related comparisons. These are supporting/developmental evidence and do not replace the primary fold-aligned analyses.

Historical executed-source hashes are recorded under `provenance/`; exact historical source bytes remain in the controlled scientific archive, while the public tree uses formal portable source names.
