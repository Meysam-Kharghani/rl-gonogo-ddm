# Reproducibility workflows

## Public release verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_reporting.txt
make quality-check
```

This requires no participant-level data. It validates the repository, regenerates publication-facing source data from the frozen aggregate results, and reproduces all final figures under `reproduced/` without altering the frozen references.

## Controlled downstream rerun

Authorized users can materialize behavioral trial data and fold-specific cross-validated actor–control RL posterior objects according to `docs/controlled_input_schema.md`, then run the scripts under `code/01_primary_action/`, `code/02_primary_joint_response/`, and `code/03_robustness/` with explicit input/output arguments.

## RL posterior regeneration

Regenerating fold-specific actor–control RL posterior objects additionally requires the controlled behavioral data and the PyMC/ArviZ environment. The portable model and nested cross-validation modules are under `shared/rl/`. Participant-level data and posterior objects are intentionally absent from the public release.

## Historical model-development source

`code/00_model_development/` documents stage-gated model development. These files are evidence of project lineage, not a single recommended end-to-end rerun target. Current primary and robustness entry points are listed in `docs/controlled_rerun_commands.md`.
