# Change log

## 1.0.0 — 2026-10-25

- added verified manuscript-author metadata and `CITATION.cff` for GitHub citation support;

- synchronized the repository with the rewritten manuscript **Incremental predictive value of cross-fitted reinforcement-learning states beyond behavioral history in cued Go/NoGo**;
- rebuilt the reporting layer around the final four main figures and five supplementary figures;
- added publication-facing manuscript/Supplement table exports, normalized to formal public labels, and a manuscript-to-evidence crosswalk;
- preserved the complete stage-gated project lineage, including candidates stopped by design and superseded controls;
- integrated richer-history, analytic-horizon, predictor-mapping, absolute-predictive-check, parameter-recovery, and likelihood-component robustness evidence;
- formalized public code/result naming and removed internal development-family labels, article-ordinal labels, nonportable absolute paths, and informal source nicknames;
- formalized the repeated-partition summarizer so it consumes an explicit partition-output map rather than inferring historical private directory names;
- removed participant-level identifiers and participant-level result vectors from the public boundary;
- added MIT licensing;
- added release versioning and publication-facing citation guidance;
- retained historical executed-source SHA-256 values by scientific source role;
- strengthened repository validation for public naming, privacy, syntax, structured files, reporting alignment, and release-manifest integrity;
- kept figure reproduction non-destructive under `reproduced/` with exact SHA-256 verification against frozen references.
- aligned public controlled-input QA with the privacy boundary: absent participant-level inputs are **NOT APPLICABLE** in the public release, while a separate strict preflight is reserved for authorized reruns.
