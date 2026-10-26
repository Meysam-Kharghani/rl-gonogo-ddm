#!/usr/bin/env python3
"""Summarize repeated outer-partition stability for policy + history versus compact history.

Each row of the input mapping identifies one completed controlled rerun. Because the same
participants recur across partitions, seed-level results are treated as stability diagnostics,
not independent replications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--partition-map", type=Path, required=True,
                   help="CSV with columns seed,out_dir for completed repeated-partition reruns.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for repeated-partition summaries.")
    p.add_argument("--n-subject-bootstrap", type=int, default=50_000)
    p.add_argument("--n-crossed-bootstrap", type=int, default=20_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260806)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def qinterval(x: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


def subject_bootstrap(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return qinterval(x[idx].mean(axis=1))


def crossed_bootstrap(matrix: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n_subjects, n_partitions = matrix.shape
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        si = rng.integers(0, n_subjects, size=n_subjects)
        pi = rng.integers(0, n_partitions, size=n_partitions)
        vals[b] = float(matrix[np.ix_(si, pi)].mean())
    return qinterval(vals)


def planned_policy_contrast(path: Path) -> tuple[float, float, float, int] | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    q = d[(d["model"] == "D15") & (d["reference"] == "D6")]
    if q.empty:
        return None
    r = q.iloc[0]
    return float(r["mean_subject_delta_elpd"]), float(r["bootstrap_ci_low"]), float(r["bootstrap_ci_high"]), int(r["positive_subjects"])


def main() -> int:
    a = parse_args()
    mapping = pd.read_csv(a.partition_map)
    required_cols = {"seed", "out_dir"}
    if not required_cols.issubset(mapping.columns):
        raise ValueError(f"partition map must contain {sorted(required_cols)}")
    out = a.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    required = [
        "strict_ddm_subject_metrics.csv",
        "crossvalidated_ddm_fold_scores.csv",
        "crossvalidated_ddm_optimizer_summary.csv",
        "analysis_settings.json",
    ]
    seed_rows, subject_parts, fold_parts, warnings = [], [], [], []

    for _, m in mapping.iterrows():
        seed = int(m["seed"])
        run = Path(m["out_dir"]).expanduser().resolve()
        missing = [name for name in required if not (run / name).exists()]
        if missing:
            msg = f"seed={seed}: missing {missing} under {run}"
            if a.allow_incomplete:
                warnings.append(msg)
                continue
            raise FileNotFoundError(msg)

        settings = json.loads((run / "analysis_settings.json").read_text(encoding="utf-8"))
        if int(settings.get("seed", seed)) != seed:
            raise ValueError(f"seed mismatch for {run}")

        opt = pd.read_csv(run / "crossvalidated_ddm_optimizer_summary.csv")
        opt["spread"] = opt["worst_start_nll"] - opt["best_start_nll"]

        subj = pd.read_csv(run / "strict_ddm_subject_metrics.csv")
        subj["subject_id"] = subj["subject_id"].astype(str)
        wide_s = subj.pivot_table(index="subject_id", columns="variant", values="elpd", aggfunc="sum")
        if not {"D6", "D15"}.issubset(wide_s.columns):
            raise ValueError(f"seed={seed}: required compact-history and policy variants are missing")
        sdelta = (wide_s["D15"] - wide_s["D6"]).rename("delta_elpd").reset_index()
        sdelta.insert(0, "seed", seed)
        subject_parts.append(sdelta)

        fold = pd.read_csv(run / "crossvalidated_ddm_fold_scores.csv")
        if "horizon_min_prior_trials" in fold.columns:
            fold = fold[fold["horizon_min_prior_trials"] == 0].copy()
        wide_f = fold.pivot(index="fold", columns="variant", values="elpd")
        if not {"D6", "D15"}.issubset(wide_f.columns):
            raise ValueError(f"seed={seed}: required compact-history and policy variants are missing from fold scores")
        fdelta = (wide_f["D15"] - wide_f["D6"]).rename("delta_elpd").reset_index()
        fdelta.insert(0, "seed", seed)
        fold_parts.append(fdelta)

        planned = planned_policy_contrast(run / "crossvalidated_ddm_planned_contrasts.csv")
        if planned is None:
            mean, lo, hi, pos = float(sdelta.delta_elpd.mean()), np.nan, np.nan, int((sdelta.delta_elpd > 0).sum())
        else:
            mean, lo, hi, pos = planned

        seed_rows.append({
            "seed": seed,
            "n_subjects": int(len(sdelta)),
            "mean_subject_delta_elpd": mean,
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "positive_subjects": pos,
            "positive_folds": int((fdelta.delta_elpd > 0).sum()),
            "n_folds": int(len(fdelta)),
            "optimizer_all_success": bool(opt["success"].astype(bool).all()),
            "optimizer_max_start_spread": float(opt["spread"].max()),
        })

    if not seed_rows:
        raise RuntimeError("No complete repeated-partition reruns were found.")

    seed_df = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    subject_df = pd.concat(subject_parts, ignore_index=True)
    fold_df = pd.concat(fold_parts, ignore_index=True)
    matrix = subject_df.pivot(index="subject_id", columns="seed", values="delta_elpd")
    if matrix.isna().any().any():
        raise ValueError("subject-by-partition matrix is incomplete")

    subject_mean = matrix.mean(axis=1)
    subj_lo, subj_hi = subject_bootstrap(subject_mean.to_numpy(), a.n_subject_bootstrap, a.bootstrap_seed)
    cross_lo, cross_hi = crossed_bootstrap(matrix.to_numpy(), a.n_crossed_bootstrap, a.bootstrap_seed + 1)
    seed_means = seed_df.mean_subject_delta_elpd.to_numpy(float)

    overall = pd.DataFrame([{
        "n_partitions": int(len(seed_df)),
        "n_subjects": int(matrix.shape[0]),
        "mean_of_partition_means": float(seed_means.mean()),
        "median_partition_mean": float(np.median(seed_means)),
        "min_partition_mean": float(seed_means.min()),
        "max_partition_mean": float(seed_means.max()),
        "positive_partition_count": int((seed_means > 0).sum()),
        "partition_ci_above_zero_count": int((seed_df.bootstrap_ci_low > 0).sum()),
        "positive_fold_count": int((fold_df.delta_elpd > 0).sum()),
        "total_fold_count": int(len(fold_df)),
        "partition_averaged_subject_mean": float(subject_mean.mean()),
        "subject_bootstrap_ci_low": subj_lo,
        "subject_bootstrap_ci_high": subj_hi,
        "crossed_stability_interval_low": cross_lo,
        "crossed_stability_interval_high": cross_hi,
    }])

    seed_df.to_csv(out / "repeated_partition_seed_summary.csv", index=False)
    overall.to_csv(out / "repeated_partition_overall_summary.csv", index=False)

    # Participant-level and fold-level outputs are useful for controlled QA but remain inside the authorized rerun directory.
    subject_df.to_csv(out / "repeated_partition_subject_deltas.csv", index=False)
    fold_df.to_csv(out / "repeated_partition_fold_deltas.csv", index=False)

    lines = [
        "# Repeated outer-partition stability report",
        "",
        "Target contrast: policy probability beyond the compact lagged-history first-passage reference.",
        "",
        "Repeated partitions reuse the same participants and therefore quantify stability rather than independent replication.",
        "",
        "## Overall summary",
        "",
        overall.to_markdown(index=False),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in warnings] if warnings else ["- None."])
    (out / "repeated_partition_stability_report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
