# Incremental predictive value of cross-fitted reinforcement-learning states beyond behavioral history in cued Go/NoGo

Public analysis repository for the behavioral-computational study **“Incremental predictive value of cross-fitted reinforcement-learning states beyond behavioral history in cued Go/NoGo.”**

**Release:** 1.0.0 (2026-10-25)  
**Repository name:** `rl-gonogo-ddm`  
**License:** MIT  
**Cohort:** 40 participants (20 high-risk-propensity, 20 low-risk-propensity), 380 trials per participant  
**Public-data boundary:** participant-level behavioral data and participant-specific posterior objects are not redistributed.

## Scientific question

The repository evaluates whether cross-fitted reinforcement-learning (RL) states retain held-out predictive information beyond observable behavioral history, and whether any advantage in a fixed-horizon one-boundary first-passage likelihood reflects response occurrence/survival, observed-response timing, or both.

The final evidence hierarchy is deliberately stage-gated:

1. candidate RL and first-passage models were developed and screened;
2. a parsimonious actor–control RL representation was selected for strict downstream evaluation;
3. primary action-occurrence and fixed-horizon first-passage comparisons were evaluated under participant-level cross-fitting;
4. later robustness analyses challenged the primary findings with richer behavioral history, alternative analytic horizons and predictor mappings, absolute predictive checks, simulation-based recovery, and likelihood-component decomposition.

Candidate branches that failed or became irrelevant at an earlier decision gate were **stopped by design**. They were not retrospectively given the full sensitivity program applied to surviving models. The complete decision path is documented in `docs/analysis_lineage_and_decision_log.md` and `docs/analysis_stage_registry.tsv`.

## Main conclusion represented by this release

Against the prespecified compact lagged-history comparator, selected RL quantities showed modest held-out increments. Those component-specific increments attenuated against a richer strictly lagged behavioral-history representation. In the first-passage analysis, the historical policy-probability advantage was concentrated mainly in censored/survival contributions rather than in the density of observed response times. The exact mapping of RL variables to first-passage parameters was not uniquely supported.

This repository therefore supports a **predictive representation** interpretation rather than a strong mechanistic claim.

## Repository structure

```text
code/
  00_model_development/          Stage-gated historical and supportive model-development source
  01_primary_action/             Binary action-occurrence benchmark
  02_primary_joint_response/    Fixed-horizon right-censored one-boundary first-passage benchmark
  03_robustness/                Completed post-development robustness analyses
  04_reporting/                 Final manuscript figure generation

shared/
  rl/                            Actor–control RL model and cross-fitted latent generation
  ddm/                           One-boundary first-passage support code

results/
  model_development/             Candidate-screening and stage-selection evidence
  action/                        Frozen action-occurrence results
  ddm/                           Frozen first-passage results and fold-level fit parameters
  robustness/                    Completed robustness results
  reporting/                     Final manuscript source data and frozen figures

configs/                         Machine-readable analysis settings
controlled_inputs/               Public placeholder for non-redistributed inputs
provenance/                      Historical source hashes and release manifest
docs/                            Lineage, nomenclature, QA, data-access, and manuscript crosswalk
```

## Reproduction levels

### 1. Public manuscript-figure reproduction

No participant-level data are required.

```bash
python -m pip install -r requirements_reporting.txt
make figures
```

The generated assets are compared by exact SHA-256 identity with the frozen reference figures in `results/reporting/figures/`.

### 2. Repository validation

```bash
make validate
```

Validation checks required paths, Python syntax, JSON/CSV/TSV integrity, prohibited internal naming, nonportable absolute paths, participant-identifier leakage, and the frozen release manifest.

### 3. Full controlled rerun

A full rerun requires authorized participant-level behavioral inputs and fold-specific train-only RL posterior objects that are intentionally absent from the public repository. See:

- `docs/controlled_input_schema.md`
- `docs/controlled_rerun_commands.md`
- `make controlled-preflight-required` after authorized controlled inputs have been materialized

The public release contains no participant-level controlled inputs. `make controlled-preflight` therefore reports the controlled-data check as **NOT APPLICABLE** and exits successfully; it returns an error only if a partial controlled-data materialization is detected.

Controlled reruns write below `reproduced/controlled_rerun/` and do not overwrite frozen public results.

## Manuscript-facing versus technical model names

Technical model IDs such as `M6`, `D6`, `D14`, and `D15` are retained only where computational traceability requires them. Manuscript-facing interpretation uses conceptual labels such as **actor–control RL model**, **compact lagged-history reference**, **policy + history**, and **full RL + history**. See `docs/model_nomenclature.md`.

## Manuscript/reporting alignment

The final manuscript contains four main figures and the Supplement contains five figures. Their exact public source tables and generating script are indexed in:

- `docs/manuscript_alignment.md`
- `results/reporting/source_data_map.tsv`
- `results/reporting/figures/FIGURE_INDEX.md`

## Privacy and data availability

The public release excludes participant-level trial rows, participant identifiers, participant-level predictive-score vectors, fold manifests containing participant IDs, and participant-specific posterior objects. Aggregate results sufficient for the public reporting layer are included. See `docs/data_availability.md` and `docs/privacy_and_exclusion_policy.md`.

## License

Repository-authored software and documentation are released under the **MIT License**; see `LICENSE`. The license does not grant access to controlled participant-level data or override third-party data/software terms.

## Citation

See `CITATION.md` and `CITATION.cff` for the verified author and citation metadata.
