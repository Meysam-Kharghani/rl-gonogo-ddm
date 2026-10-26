#!/usr/bin/env python3
"""Verify regenerated manuscript reporting source data against frozen public references."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    a = p.parse_args()

    expected = []
    expected.extend(sorted((a.reference / "figure_data").glob("*.csv")))
    expected.append(a.reference / "table_data/manuscript_table_01_incremental_information_numeric.csv")
    errors = []
    for ref in expected:
        rel = ref.relative_to(a.reference)
        gen = a.generated / rel
        if not gen.exists():
            errors.append(f"missing regenerated source-data file: {rel}")
            continue
        if gen.stat().st_size != ref.stat().st_size or sha256(gen) != sha256(ref):
            errors.append(f"regenerated source data differ from frozen reference: {rel}")

    generated_files = sorted(p for p in a.generated.rglob("*") if p.is_file())
    expected_rel = {p.relative_to(a.reference).as_posix() for p in expected}
    extras = [p.relative_to(a.generated).as_posix() for p in generated_files if p.relative_to(a.generated).as_posix() not in expected_rel]
    if extras:
        errors.append("unexpected regenerated source-data files: " + ", ".join(extras))

    print(f"REPORTING SOURCE FILES CHECKED: {len(expected)}")
    if errors:
        for e in errors:
            print("ERROR:", e)
        return 1
    print("REPORTING SOURCE-DATA REPRODUCTION PASS (exact SHA-256 identity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
