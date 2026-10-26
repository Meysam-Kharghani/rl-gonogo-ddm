# Results index — behavioral-computational study

## Historical primary held-out action benchmark

Against the prespecified primary lagged-history reference, actor/control states produced a small positive held-out increment for action occurrence. Policy probability did not add a detectable increment after actor/control plus the primary history baseline.

These results are the historical primary analysis and remain unchanged under `results/action/` and `results/reporting/`.

## Historical primary fixed-horizon joint response analysis

Against the prespecified primary lagged-history diffusion reference, policy probability improved the joint fixed-horizon occurrence-and-timing likelihood, while actor/control did not add a detectable increment after policy and history were included.

The analysis uses a 700-ms analytic observation horizon, not a task-enforced response deadline.

## Post-development robustness qualification

The final interpretation must also incorporate `results/robustness/`:

- **richer-history action:** actor/control- and policy-specific increments attenuate when the behavioral-history comparator is made substantially richer; the full RL block retains only a small positive increment;
- **richer-history joint response:** policy, actor/control, and full-RL increments are not detectably positive against the richer DDM history comparator;
- **likelihood decomposition:** the historical policy advantage is concentrated mainly in survival/by-horizon occurrence contributions rather than observed event-time density;
- **predictor mapping:** exact actor/control placement within drift versus boundary distance is not uniquely supported;
- **absolute predictive checks:** event-by-horizon calibration is broadly reasonable, while conditional NoGo RT calibration is weaker;
- **analytic horizon:** the historical policy-vs-primary-history result remains stable across nearby horizons.

The manuscript should therefore describe the primary findings as conditional on the prespecified primary lagged-history baseline and avoid strong mechanistic interpretation of DDM parameter placement.

## Cue-prior representation

Strictly lagged empirical cue-probability updating does not show a stable advantage over instructed cue probabilities across the tested prior-strength range.

## Repeated partitions

Repeated outer partitions assess stability to train/test composition within the same 40 participants; they are not independent replication.
