#!/usr/bin/env python3
"""Validate the public behavioral-computational reproducibility release."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import py_compile
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance/release_manifest.tsv"
EXPECTED_VERSION = "1.0.0"
TEXT_EXT = {".md", ".txt", ".py", ".json", ".csv", ".tsv", ".yml", ".yaml", ".toml", ".sh", ".m"}

# Generic publication-facing naming checks. These patterns reject development-version
# labels and article-ordinal labels without embedding retired project-specific names.
PROHIBITED_TEXT_PATTERNS = [
    ("development-version RL label", re.compile(r"(?i)\brl(?:core)?[_-]?v\d+(?:[_.-]\d+)*\b")),
    ("article-ordinal label", re.compile(r"(?i)\bpaper[ _-]?[12]\b")),
    ("informal joint-model nickname", re.compile(r"(?i)\bjoint[-_ ]lite\b")),
]
NONPORTABLE_PATH_PATTERNS = [
    re.compile("/" + "home" + r"/[A-Za-z0-9_.-]+/"),
    re.compile("/" + "Users" + r"/[A-Za-z0-9_.-]+/"),
    re.compile("/" + "mnt" + "/" + "data" + r"(?:/|\b)"),
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile("sandbox" + r":/", re.I),
]
PROHIBITED_INFORMAL_PHRASES = [
    re.compile("quick" + r"\s+validation", re.I),
    re.compile("work" + "station", re.I),
    re.compile("machine" + r"[- ]specific", re.I),
    re.compile("portability" + r"[- ]normalized", re.I),
    re.compile(r"\bD2[- ]like\b", re.I),
    re.compile("public" + r"\s+candidate\s+package", re.I),
]
PLACEHOLDER_PATTERNS = [
    re.compile(r"\[(?:AUTHORS?|AFFILIATIONS?|NAME|EMAIL|POSTAL ADDRESS)\]"),
    re.compile(r"<repo(?:sitory)?[-_ ]?url>", re.I),
    re.compile(r"<release[-_ ]?version>", re.I),
    re.compile(r"\b(?:TO" + "DO|T" + "BD|FIX" + "ME)\b", re.I),
]
PARTICIPANT_KEYS = {
    "subject_id", "subject_id_norm", "subj_id", "subj_id_ddm", "participant_id",
    "participant", "subject", "subject_code", "subject_label", "left_out_subject",
    "test_subject_ids", "train_subject_ids",
}
ID_RE = re.compile(r"(?<![A-Za-z0-9])[HL](?:[1-9][0-9]{0,2})(?![A-Za-z0-9])")

EXPECTED_FIGURE_STEMS = [
    "figure_01_analysis_logic",
    "figure_02_action_history_comparison",
    "figure_03_first_passage_history_comparison",
    "figure_04_likelihood_decomposition",
    "supplementary_figure_01_horizon_sensitivity",
    "supplementary_figure_02_repeated_partitions",
    "supplementary_figure_03_mapping_sensitivity",
    "supplementary_figure_04_parameter_recovery",
    "supplementary_figure_05_prior_strength",
]
EXPECTED_SOURCE_DATA = [
    "figure_01_analysis_logic_source_data.csv",
    "figure_02_action_history_comparison_source_data.csv",
    "figure_03_first_passage_history_comparison_source_data.csv",
    "figure_04_likelihood_decomposition_source_data.csv",
    "supplementary_figure_01_horizon_sensitivity_source_data.csv",
    "supplementary_figure_02_repeated_partitions_source_data.csv",
    "supplementary_figure_03_mapping_sensitivity_source_data.csv",
    "supplementary_figure_04_parameter_recovery_source_data.csv",
    "supplementary_figure_05_prior_strength_source_data.csv",
]


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks: list[str] = []

    def ok(self, msg: str) -> None:
        self.checks.append(msg)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_release_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in {"reproduced", ".git"}:
            continue
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if rel.as_posix() == "provenance/release_manifest.tsv":
            continue
        yield path


def check_release_metadata(report: Report) -> None:
    version = ROOT / "VERSION"
    license_file = ROOT / "LICENSE"
    citation = ROOT / "CITATION.md"
    if not version.exists():
        report.err("missing VERSION")
    elif version.read_text(encoding="utf-8").strip() != EXPECTED_VERSION:
        report.err(f"VERSION must equal {EXPECTED_VERSION}")
    if not license_file.exists():
        report.err("missing LICENSE")
    elif not license_file.read_text(encoding="utf-8", errors="replace").startswith("MIT License\n"):
        report.err("LICENSE is not the MIT License text")
    if not citation.exists():
        report.err("missing CITATION.md")
    cff = ROOT / "CITATION.cff"
    if not cff.exists():
        report.err("missing CITATION.cff")
    else:
        cff_text = cff.read_text(encoding="utf-8")
        for author in ("Meysam Kharghani", "Abolfazl Jafarian", "Farhad Farkhondeh Tale Navi", "Leila Mehdizadeh Fanid"):
            parts = author.split()
            if parts[0] not in cff_text or parts[-1] not in cff_text:
                report.err(f"CITATION.cff missing verified author metadata for {author}")
    report.ok("release metadata, verified citation metadata, and MIT license checked")


def check_required(report: Report) -> None:
    required = [
        "README.md", "VERSION", "LICENSE", "CITATION.md", "CITATION.cff", "Makefile",
        ".github/workflows/validate.yml",
        "docs/analysis_lineage_and_decision_log.md",
        "docs/analysis_stage_registry.tsv",
        "docs/model_nomenclature.md",
        "docs/evidence_hierarchy.md",
        "docs/privacy_and_exclusion_policy.md",
        "docs/excluded_public_artifacts.tsv",
        "docs/controlled_input_schema.md",
        "docs/data_availability.md",
        "docs/robustness_results_index.md",
        "docs/public_naming_policy.md",
        "docs/controlled_rerun_commands.md",
        "docs/release_quality_report.md",
        "docs/manuscript_alignment.md",
        "docs/publication_metadata.md",
        "docs/model_registry_rl.csv",
        "docs/model_registry_ddm.csv",
        "docs/rl_model_completeness.tsv",
        "docs/ddm_model_completeness.csv",
        "code/00_model_development/README.md",
        "code/01_primary_action/action_occurrence_benchmark.py",
        "code/02_primary_joint_response/joint_response_time_benchmark.py",
        "code/02_primary_joint_response/summarize_repeated_partitions.py",
        "code/03_robustness",
        "code/04_reporting/build_manuscript_source_data.py",
        "code/04_reporting/make_publication_figures.py",
        "shared/rl/actor_control_model_family.py",
        "shared/rl/nested_crossvalidated_latent_generation.py",
        "shared/ddm/one_boundary_diffusion.py",
        "requirements_reporting.txt", "requirements_analysis.txt",
        "environment/environment.yml",
        "provenance/README.md", "provenance/historical_source_hashes.tsv",
        "results/model_development", "results/action", "results/ddm",
        "results/robustness", "results/reporting",
        "controlled_inputs/README.md",
    ]
    for item in required:
        if not (ROOT / item).exists():
            report.err(f"missing required path: {item}")
    report.ok("required-path inventory checked")


def check_reporting_contract(report: Report) -> None:
    fig_dir = ROOT / "results/reporting/figures"
    source_dir = ROOT / "results/reporting/figure_data"
    table_dir = ROOT / "results/reporting/table_data"
    expected_figures = {f"{stem}.{ext}" for stem in EXPECTED_FIGURE_STEMS for ext in ("png", "pdf", "svg")}
    expected_figures.add("FIGURE_INDEX.md")
    if fig_dir.exists():
        actual = {path.name for path in fig_dir.iterdir() if path.is_file()}
        missing = sorted(expected_figures - actual)
        extra = sorted(actual - expected_figures)
        if missing:
            report.err("missing final reporting figure asset(s): " + ", ".join(missing))
        if extra:
            report.err("unexpected reporting figure asset(s): " + ", ".join(extra))
    for name in EXPECTED_SOURCE_DATA:
        if not (source_dir / name).exists():
            report.err(f"missing reporting source-data file: results/reporting/figure_data/{name}")
    table_expected = {
        "manuscript_table_01_incremental_information.csv",
        "manuscript_table_01_incremental_information_numeric.csv",
        *{f"supplementary_table_{index:02d}.csv" for index in range(1, 19)},
    }
    if table_dir.exists():
        actual_tables = {path.name for path in table_dir.glob("*.csv")}
        missing_tables = sorted(table_expected - actual_tables)
        if missing_tables:
            report.err("missing manuscript/supplement table export(s): " + ", ".join(missing_tables))
    report.ok("final manuscript reporting contract checked")


def check_names_and_text(report: Report) -> None:
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        for label, pattern in PROHIBITED_TEXT_PATTERNS:
            if pattern.search(rel):
                report.err(f"{label} in public path: {rel}")
        if path.suffix.lower() not in TEXT_EXT or path.stat().st_size > 20_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            report.err(f"cannot read text file {rel}: {exc}")
            continue
        for label, pattern in PROHIBITED_TEXT_PATTERNS:
            if pattern.search(text):
                report.err(f"{label} in public content: {rel}")
        for pattern in NONPORTABLE_PATH_PATTERNS:
            if pattern.search(text):
                report.err(f"nonportable absolute path in content: {rel}")
        for pattern in PROHIBITED_INFORMAL_PHRASES:
            if pattern.search(text):
                report.err(f"informal development phrase in public content: {rel}")
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                report.err(f"unresolved publication/release placeholder in content: {rel}")
    report.ok("formal public naming, nonportable-path, informal-phrase, and placeholder policy checked")


def _bad_header_fields(header: list[str]) -> list[str]:
    bad = []
    for field in header:
        key = field.strip().lower()
        if key in PARTICIPANT_KEYS:
            bad.append(field)
            continue
        if re.search(r"(?:^|_)(?:subject|subj|participant)(?:_|-)?(?:id|ids|code|label)(?:_|$)", key):
            bad.append(field)
    return bad


def check_privacy(report: Report) -> None:
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size < 20_000_000:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                with path.open(encoding="utf-8", errors="replace", newline="") as handle:
                    header = next(csv.reader(handle, delimiter=sep), [])
                bad = _bad_header_fields(header)
                if bad:
                    report.err(f"participant identifier-like column(s) in public table {rel}: {bad}")
            except Exception as exc:
                report.err(f"cannot inspect table header {rel}: {exc}")
        if path.name.lower().endswith(".csv.gz") and path.stat().st_size < 50_000_000:
            try:
                with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
                    header = next(csv.reader(handle), [])
                bad = _bad_header_fields(header)
                if bad:
                    report.err(f"participant identifier-like column(s) in compressed public table {rel}: {bad}")
            except Exception as exc:
                report.err(f"cannot inspect compressed table header {rel}: {exc}")
        if path.suffix.lower() == ".json" and path.stat().st_size < 20_000_000:
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))

                def walk(value):
                    if isinstance(value, dict):
                        for key, child in value.items():
                            key_low = str(key).lower()
                            if key_low in PARTICIPANT_KEYS or re.search(
                                r"(?:^|_)(?:subject|subj|participant)(?:_|-)?(?:id|ids|code|label)(?:_|$)", key_low
                            ):
                                return True
                            if walk(child):
                                return True
                    elif isinstance(value, list):
                        return any(walk(child) for child in value)
                    return False

                if walk(obj):
                    report.err(f"participant identifier-like key in public JSON: {rel}")
            except json.JSONDecodeError:
                pass
        if path.suffix.lower() in TEXT_EXT and path.stat().st_size < 10_000_000:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"(?i)\b[HL][0-9]{1,3}[^\n,]*\.set\b", text):
                report.err(f"participant-coded EEG filename in public content: {rel}")
            if rel.startswith("results/") and path.suffix.lower() in {".csv", ".tsv", ".json"} and ID_RE.search(text):
                report.err(f"participant-coded H/L identifier token in public result artifact: {rel}")
    report.ok("participant-level public-boundary checks completed")


def check_structured(report: Report) -> None:
    for path in iter_release_files():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                report.err(f"invalid JSON {rel}: {exc}")
        if path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size < 30_000_000:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                with path.open(encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle, delimiter=sep)
                    header = next(reader, None)
                    if header is None:
                        report.err(f"empty table: {rel}")
            except Exception as exc:
                report.err(f"invalid delimited file {rel}: {exc}")
    report.ok("JSON/CSV/TSV parse checks completed")


def check_python(report: Report) -> None:
    with tempfile.TemporaryDirectory(prefix="repository-python-compile-") as temp_dir:
        index = 0
        for base in [ROOT / "code", ROOT / "shared", ROOT / "tools"]:
            for path in base.rglob("*.py"):
                try:
                    target = Path(temp_dir) / f"module_{index:04d}.pyc"
                    py_compile.compile(str(path), cfile=str(target), doraise=True)
                    index += 1
                except Exception as exc:
                    report.err(f"Python compile failure {path.relative_to(ROOT)}: {exc}")
    report.ok("Python syntax compilation completed")


def check_manifest(report: Report) -> None:
    if not MANIFEST.exists():
        report.err("missing provenance/release_manifest.tsv")
        return
    expected: dict[str, tuple[str, int]] = {}
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            expected[row["path"]] = (row["sha256"], int(row["size_bytes"]))
    actual = {path.relative_to(ROOT).as_posix(): path for path in iter_release_files()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        report.err("manifest entries missing from release: " + ", ".join(missing[:20]))
    if extra:
        report.err("files missing from manifest: " + ", ".join(extra[:20]))
    for rel, path in actual.items():
        if rel not in expected:
            continue
        expected_hash, expected_size = expected[rel]
        if path.stat().st_size != expected_size:
            report.err(f"manifest size mismatch: {rel}")
        elif sha256(path) != expected_hash:
            report.err(f"manifest SHA256 mismatch: {rel}")
    report.ok("release-manifest hashes checked")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksums-only", action="store_true")
    args = parser.parse_args()
    report = Report()
    if args.checksums_only:
        check_manifest(report)
    else:
        check_release_metadata(report)
        check_required(report)
        check_reporting_contract(report)
        check_names_and_text(report)
        check_privacy(report)
        check_structured(report)
        check_python(report)
        check_manifest(report)
    print(f"REPOSITORY: {ROOT.name}")
    print(f"CHECKS: {len(report.checks)}  WARNINGS: {len(report.warnings)}  ERRORS: {len(report.errors)}")
    for item in report.warnings:
        print("WARNING:", item)
    for item in report.errors:
        print("ERROR:", item)
    if report.errors:
        return 1
    print("VALIDATION PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
