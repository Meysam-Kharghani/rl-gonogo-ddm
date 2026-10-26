# Public naming policy

The public repository uses publication-facing, concept-based names for paths, modules, result directories, documentation, and generated reporting assets.

## Allowed technical identifiers

Historical computational IDs such as `M6`, `D6`, `D14`, and `D15` may appear where exact traceability requires them. They are identifiers, not manuscript-facing scientific names. Their conceptual meanings are defined in `docs/model_nomenclature.md`.

## Prohibited public naming

Public paths and textual content must not contain:

- nonportable absolute user paths;
- internal project-root or migration labels;
- article-ordinal directory labels;
- informal development nicknames;
- source-family version labels that are not part of the scientific model definition;
- historical private filenames when a conceptual source-role description is sufficient.

Examples of prohibited internal tokens are enforced programmatically by `tools/validate_repository.py`. Historical executed-source hashes are retained by source **role** in `provenance/historical_source_hashes.tsv` rather than by exposing private filesystem names.

This policy does not alter numerical results or technical model IDs needed for scientific cross-reference.
