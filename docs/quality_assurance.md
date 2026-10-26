# Quality assurance

`tools/validate_repository.py` is a standard-library-only release validator. It checks:

- release version, MIT license, and verified machine-readable author/citation metadata;
- required repository structure and decision-lineage documentation;
- the exact final manuscript reporting contract (Figures 1–4, Figures S1–S5, Table 1, Tables S1–S18);
- Python syntax compilation;
- JSON and delimited-table parsing;
- formal public naming, nonportable-path exclusion, and unresolved release-placeholder exclusion;
- participant-identifier leakage at the public boundary;
- SHA-256 and byte-size agreement with `provenance/release_manifest.tsv`.

Scientific QA is distinct from repository QA. The public evidence preserves candidate screening, model-selection/stopping decisions, primary action and first-passage analyses, repeated-partition stability summaries, and completed robustness analyses. Candidates stopped at an earlier gate are not assigned artificial downstream sensitivity analyses.

## Reporting QA

`make source-data` regenerates the final figure source tables and numeric main-table source from frozen aggregate analysis results into `reproduced/reporting_source_data/`, then checks exact SHA-256 identity against the frozen publication-facing source tables.

`make figures` regenerates publication figures into `reproduced/figures/` and compares every generated PNG/PDF/SVG plus the figure index with `results/reporting/figures/` using exact SHA-256 identity.

`make quality-check` runs validation, source-data regeneration/verification, and figure regeneration/verification.

The GitHub Actions workflow performs the same public-boundary validation and reporting reproduction in a clean hosted environment using the pinned reporting requirements.

## Controlled-input QA

`make controlled-preflight` validates the controlled-data boundary without requiring participant-level data to be public. When no controlled inputs are materialized, as expected in the public release, it reports **NOT APPLICABLE** and exits successfully. If some controlled inputs are present but the materialization is incomplete, it returns an error to prevent an incomplete rerun. In an authorized controlled-data environment, `make controlled-preflight-required` requires the complete input set before a full rerun; add `--include-parameter-recovery` when validating the optional recovery input.

## Source provenance

Historical executed analysis sources are represented publicly by formal portable code and conceptual source roles. `provenance/historical_source_hashes.tsv` and `docs/model_development_source_hashes.tsv` preserve SHA-256 identifiers for historical executed/development sources. Public source normalization may change path strings, documentation, or publication-facing terminology without altering the frozen numerical evidence.
