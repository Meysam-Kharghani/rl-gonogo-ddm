#!/usr/bin/env python3
"""Check the controlled-input materialization boundary without reading participant data."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check controlled inputs required for behavioral-computational study reruns without "
            "reading participant data. In a public release with no controlled inputs materialized, "
            "the check is not applicable and exits successfully."
        )
    )
    parser.add_argument(
        "--include-parameter-recovery",
        action="store_true",
        help="Also require the controlled full-fit posterior used for parameter recovery.",
    )
    parser.add_argument(
        "--require-present",
        action="store_true",
        help=(
            "Require the complete controlled input set. Use this only in an authorized "
            "controlled-data environment before a full rerun."
        ),
    )
    args = parser.parse_args()

    checks: list[tuple[str, Path, bool]] = []

    def req_dir(rel: str, label: str) -> Path:
        path = ROOT / rel
        present = path.is_dir() and any(path.rglob("*"))
        checks.append((label, path, present))
        return path

    def req_file(rel: str, label: str) -> Path:
        path = ROOT / rel
        present = path.is_file()
        checks.append((label, path, present))
        return path

    beh = req_dir("controlled_inputs/behavioral_trial_data", "behavioral trial-data directory")
    rl = req_dir("controlled_inputs/rl_crossvalidated_posteriors", "cross-validated RL posterior directory")
    if args.include_parameter_recovery:
        req_file(
            "controlled_inputs/rl_full_fit/actor_control_primary/idata.nc",
            "full-fit posterior for parameter recovery",
        )

    # A pristine public release intentionally contains none of these participant-level inputs.
    present_count = sum(present for _, _, present in checks)
    if not args.require_present and present_count == 0:
        print("CONTROLLED-INPUT PREFLIGHT: NOT APPLICABLE")
        print("No controlled participant-level inputs are materialized in this public release.")
        return 0

    # Once controlled inputs are materialized, validate the expected RL posterior structure as well.
    if rl.is_dir() and any(rl.rglob("*")):
        req_file(
            "controlled_inputs/rl_crossvalidated_posteriors/artifacts/arrays__actor_control_model_family.npz",
            "RL artifact array",
        )
        cvs = list(rl.rglob("cv_kfold_subject.json"))
        idatas = list(rl.rglob("idata.nc"))
        checks.append(("CV summary files", rl, len(cvs) > 0))
        checks.append(("at least five fold posterior netCDF files", rl, len(idatas) >= 5))

    missing = [(label, path) for label, path, present in checks if not present]
    for label, path, present in checks:
        print(("OK     " if present else "MISSING"), label, ":", path.relative_to(ROOT))

    if missing:
        print("\nCONTROLLED-INPUT PREFLIGHT: INCOMPLETE CONTROLLED MATERIALIZATION")
        for label, path in missing:
            print("MISSING:", label, ":", path.relative_to(ROOT))
        return 1

    print("\nCONTROLLED-INPUT PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
