# Evidence hierarchy — behavioral-computational study

## 1. Historical primary analyses

The historical primary analyses remain the fixed subject-level outer-fold action benchmark and the fixed-horizon right-censored one-boundary DDM decomposition. Their numerical values are preserved under `results/action/`, `results/ddm/`, and `results/reporting/`.

The correct claim is **conditional on the prespecified primary lagged-history baseline**. The public package should not generalize this to every plausible behavioral-history representation.

## 2. Post-development robustness that materially qualifies the primary result

### Richer-history action comparator

Actor/control alone and policy alone no longer show intervals excluding zero beyond the richer behavioral-history comparator. The full RL block retains only a small positive increment.

### Richer-history joint-response comparator

Policy, actor/control, and the full RL block do not show intervals excluding zero against the richer DDM history comparator.

**Interpretation:** the primary RL increments are sensitive to the strength of the behavioral-history baseline and should not be described as information that necessarily exceeds all observable-history representations.

## 3. Robustness that supports parts of the primary result

The policy-vs-primary-history contrast is stable across nearby analytic horizons. Absolute event-by-horizon calibration is broadly reasonable, although conditional RT calibration is notably worse for NoGo responses than for Go responses.

## 4. Likelihood-component interpretation

The policy advantage over the primary history reference is concentrated mainly in censored/survival or by-horizon occurrence contributions rather than a clear improvement in event-time density among observed responses. The manuscript should therefore describe the DDM as a **joint fixed-horizon occurrence-and-timing likelihood**, while explaining where the predictive gain arises.

## 5. Mapping and parameter-recovery limits

Predictor-mapping sensitivity does not uniquely support the historical actor-to-drift/control-to-distance mapping. Parameter recovery is heterogeneous, with weaker recovery for some learning-rate variance parameters. Exact DDM placement and weakly recovered hierarchical parameters must not be treated as strong mechanistic evidence.

## 6. Developmental extensions excluded from evidence

The incomplete prospect-theory RL extension is not part of the manuscript evidence hierarchy.
