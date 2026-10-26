# Model nomenclature

Technical IDs are retained only where they are needed for computational traceability. Manuscript prose should use the conceptual labels below.

| Technical ID | Conceptual manuscript label | Role |
|---|---|---|
| M6 | actor–control reinforcement-learning model | Frozen RL representation used for primary downstream analyses |
| M7 | actor–control plus adaptive cue-belief model | Competitive developmental sensitivity model; not the primary generator |
| D6 | lagged-history diffusion reference | Primary fixed-horizon reference model |
| D14 | actor/control + lagged-history diffusion model | Tests actor/control increment without policy probability |
| D15 | policy + lagged-history diffusion model | Tests policy-probability increment without actor/control states |
| D9 | full-RL + lagged-history diffusion model | Maximal primary diffusion model |
| D0 | categorical-design diffusion reference | Cue-coding/prior-family reference |
| D10 | instructed-probability diffusion model | Numerical instructed cue-probability representation |
| D11 | empirical cue-belief diffusion model | Strictly lagged empirical cue-probability representation |

Historical IDs D1–D5, D6A, D7, D8, D12, and D13 remain in registries to preserve the model-development record. They should not be introduced into the manuscript unless a specific historical/sensitivity result requires them.
