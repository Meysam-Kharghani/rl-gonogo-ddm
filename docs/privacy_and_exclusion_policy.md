# Public-data boundary and exclusion policy

The public repository intentionally excludes participant-level behavioral rows, participant identifiers, per-participant predictive-score vectors, fold manifests that enumerate participant IDs, and participant-specific posterior objects.

Aggregate counts such as `n_subjects`, numbers of positive participant contrasts, and bootstrap summaries are permitted because they do not identify individuals.

`docs/excluded_public_artifacts.tsv` records participant-level artifacts removed from the public repository. Exclusion does not erase their provenance: the aggregate analyses and model-development decision record remain public, while controlled participant-level artifacts remain in the authorized project workspace.
