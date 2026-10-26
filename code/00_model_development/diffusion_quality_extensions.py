#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

# -*- coding: utf-8 -*-
"""
diffusion_quality_extensions.py

Six quality-extension analyses for the reinforcement-learning-to-first-passage development program.

Run from:
    cd <REPOSITORY_ROOT>/controlled_inputs

Recommended:
    python code/00_model_development/diffusion_quality_extensions.py all \
      --base . \
      --ddm_script shared/ddm/one_boundary_diffusion.py \
      --out results/model_development/diffusion_quality_extensions \
      --latents controlled_inputs/rl_latents/actor_control_seed_2026.csv \
      --data_root data/raw/csv

The six stages:
1) Data provenance / subject-count audit.
2) DDM predictor audit table.
3) RT predictive-fit / RT quantile check for nested D2.
4) Lagged ERN/Pe post-error exploratory analysis.
5) Small DDM model-recovery / simulation sanity check.
6) Exploratory joint reinforcement-learning/first-passage model reinforcement-learning-to-first-passage maximum-likelihood branch.

Important:
- Stages 1-5 are intended for manuscript/supplementary use if they pass QC.
- Stage 6 is explicitly exploratory. It is a joint reinforcement-learning/first-passage model MLE branch, not the
  primary hierarchical Bayesian RL model and not a replacement for the current
  leakage-controlled nested reinforcement-learning-to-first-passage pipeline unless it converges and is validated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


EPS = 1e-12


# =============================================================================
# Generic helpers
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(p: Path, obj: Any) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(p: Path, default=None):
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def safe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def import_module_from_path(path: Path, name: str = "ddm_mod"):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_latents(base: Path, preferred: Optional[Path] = None, variant: str = "M6", seed: int = 2026) -> Path:
    candidates = []
    if preferred is not None:
        candidates.append(Path(preferred))
    candidates += [
        base / "outputs" / "ddm_latents" / f"trial_latents_for_ddm_{variant}_seed{seed}.csv",
        base / "results" / "actor_control_model_family_summary" / f"seed_{seed}" / "results" / f"{variant}_posterior" / "trial_latents_for_ddm.csv",
        base / "results" / "actor_control_model_family_cv_posteriors" / f"seed_{seed}" / "results" / f"{variant}_posterior" / "trial_latents_for_ddm.csv",
        base / f"trial_latents_for_ddm_{variant}_seed{seed}.csv",
        base / "trial_latents_for_ddm.csv",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    # Generic fallback for controlled archives: search by scientific artifact name and
    # technical model ID rather than by a historical directory suffix.
    for c in base.rglob("trial_latents_for_ddm.csv"):
        if variant.lower() in c.as_posix().lower():
            return c
    raise FileNotFoundError(
        "Could not find latent CSV. Pass --latents explicitly. Tried:\n" +
        "\n".join(str(c) for c in candidates)
    )


def load_ddm_data(ddm, latents: Path, data_root: Optional[Path], deadline_ms: float,
                  rt_min_ms: float, rt_max_ms: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    raw = ddm.load_latents(Path(latents), deadline_ms=deadline_ms, data_root=Path(data_root) if data_root else None)
    filt, info = ddm.filter_ddm_trials(raw, rt_min_ms=rt_min_ms, rt_max_ms=rt_max_ms)
    return raw, filt, info


def standardize_series(x: pd.Series) -> pd.Series:
    z = pd.to_numeric(x, errors="coerce").astype(float)
    sd = z.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros(len(z)), index=z.index)
    return (z - z.mean()) / sd


def bootstrap_mean_ci(vals: np.ndarray, seed: int = 2026, n_boot: int = 5000) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def sem(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size <= 1:
        return np.nan
    return float(vals.std(ddof=1) / np.sqrt(vals.size))


# =============================================================================
# Stage 1: data provenance / subject-count audit
# =============================================================================

def infer_group_from_name(path: Path) -> Optional[int]:
    s = str(path).lower()
    stem = path.stem.lower()
    parts = [p.lower() for p in path.parts]
    if "csv_high" in parts or re.match(r"^h\d+", stem) or "high" in stem:
        return 1
    if "csv_low" in parts or re.match(r"^l\d+", stem) or "low" in stem:
        return 0
    return None


def subject_id_from_file(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_metadata_features", "")
    return stem


def audit_raw_files(data_root: Path) -> pd.DataFrame:
    rows = []
    if not data_root or not Path(data_root).exists():
        return pd.DataFrame()
    for p in sorted(Path(data_root).rglob("*_metadata_features.xlsx")):
        lname = p.name.lower()
        if "merged" in lname or "all_" in lname or lname.startswith("all"):
            continue
        rows.append({
            "subject_id_from_filename": subject_id_from_file(p),
            "group_inferred": infer_group_from_name(p),
            "path": str(p),
            "size_bytes": int(p.stat().st_size) if p.exists() else None,
        })
    return pd.DataFrame(rows)


def stage1_data_provenance(args) -> None:
    base = Path(args.base).resolve()
    out = ensure_dir(Path(args.out).resolve() / "01_data_provenance")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage1")

    latents = find_latents(base, Path(args.latents) if args.latents else None, variant=args.variant, seed=args.seed)
    raw, d, filter_info = load_ddm_data(ddm, latents, Path(args.data_root) if args.data_root else None,
                                        args.deadline_ms, args.rt_min_ms, args.rt_max_ms)

    subj = (
        d.groupby("subj_id_ddm")
        .agg(
            group_high=("group_high", "first"),
            n_trials=("press", "size"),
            n_go=("is_nogo", lambda x: int((x == 0).sum())),
            n_nogo=("is_nogo", lambda x: int((x == 1).sum())),
            n_press=("press", "sum"),
            n_hit=("hit", "sum"),
            n_fa=("fa", "sum"),
            n_miss=("miss", "sum"),
            n_cr=("cr", "sum"),
            mean_rt_press_sec=("rt_sec", "mean"),
            cnv_nonzero=("cnv_z", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).abs() > 1e-12).sum()) if "cnv_z" in d.columns else 0),
            p300_nonzero=("p300_z", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).abs() > 1e-12).sum()) if "p300_z" in d.columns else 0),
            ern_nonzero=("ern_strength_z", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).abs() > 1e-12).sum()) if "ern_strength_z" in d.columns else 0),
            pe_nonzero=("pe_strength_z", lambda x: int((pd.to_numeric(x, errors="coerce").fillna(0).abs() > 1e-12).sum()) if "pe_strength_z" in d.columns else 0),
        )
        .reset_index()
        .sort_values(["group_high", "subj_id_ddm"])
    )
    subj["group_label"] = np.where(subj["group_high"] == 1, "high", "low")
    subj.to_csv(out / "audit_subject_counts_from_latents.csv", index=False)

    by_group = (
        subj.groupby(["group_high", "group_label"])
        .agg(
            n_subjects=("subj_id_ddm", "nunique"),
            total_trials=("n_trials", "sum"),
            total_go=("n_go", "sum"),
            total_nogo=("n_nogo", "sum"),
            total_hit=("n_hit", "sum"),
            total_fa=("n_fa", "sum"),
            total_miss=("n_miss", "sum"),
            total_cr=("n_cr", "sum"),
        )
        .reset_index()
    )
    by_group["hit_rate_go"] = by_group["total_hit"] / by_group["total_go"].replace(0, np.nan)
    by_group["fa_rate_nogo"] = by_group["total_fa"] / by_group["total_nogo"].replace(0, np.nan)
    by_group.to_csv(out / "audit_trial_counts_by_group.csv", index=False)

    raw_files = audit_raw_files(Path(args.data_root)) if args.data_root else pd.DataFrame()
    if len(raw_files) > 0:
        raw_files.to_csv(out / "audit_raw_excel_files.csv", index=False)
        raw_subs = set(raw_files["subject_id_from_filename"].astype(str))
    else:
        raw_subs = set()

    latent_subs = set(subj["subj_id_ddm"].astype(str))
    overlap_rows = []
    if raw_subs:
        overlap_rows.append({"set": "latent_subjects", "n": len(latent_subs)})
        overlap_rows.append({"set": "raw_file_subjects", "n": len(raw_subs)})
        overlap_rows.append({"set": "intersection", "n": len(latent_subs & raw_subs)})
        overlap_rows.append({"set": "latent_not_in_raw", "n": len(latent_subs - raw_subs), "subjects": ";".join(sorted(latent_subs - raw_subs))})
        overlap_rows.append({"set": "raw_not_in_latent", "n": len(raw_subs - latent_subs), "subjects": ";".join(sorted(raw_subs - latent_subs))})
    pd.DataFrame(overlap_rows).to_csv(out / "audit_latent_raw_subject_overlap.csv", index=False)

    report = []
    report.append("# Data provenance and subject-count audit")
    report.append("")
    report.append(f"- Time: {now_str()}")
    report.append(f"- Latent file: `{latents}`")
    report.append(f"- Data root: `{args.data_root}`")
    report.append(f"- Raw latent rows: {len(raw)}")
    report.append(f"- Filtered DDM rows: {len(d)}")
    report.append(f"- Filter info: `{json.dumps(filter_info, ensure_ascii=False)}`")
    report.append(f"- Subjects in filtered DDM table: {subj['subj_id_ddm'].nunique()}")
    report.append(f"- Subject counts by group: {by_group[['group_label','n_subjects']].to_dict('records')}")
    report.append("")
    report.append("## Critical provenance note")
    report.append("")
    report.append(
        "The source article reports that 16 high-risk and 16 low-risk participants were selected for EEG recording "
        "after BART screening. If the current computational dataset contains 40 subjects, this must be described "
        "as the available trial-level dataset used for reanalysis, or the analysis should be repeated on the exact "
        "32-subject subset if that subset is required for strict comparability."
    )
    report.append("")
    report.append("## Output files")
    for name in [
        "audit_subject_counts_from_latents.csv",
        "audit_trial_counts_by_group.csv",
        "audit_raw_excel_files.csv",
        "audit_latent_raw_subject_overlap.csv",
    ]:
        if (out / name).exists():
            report.append(f"- `{name}`")
    (out / "DATA_PROVENANCE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[stage1] wrote {out}")


# =============================================================================
# Stage 2: DDM predictor audit
# =============================================================================

def stage2_predictor_audit(args) -> None:
    out = ensure_dir(Path(args.out).resolve() / "02_ddm_predictor_audit")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage2")
    rows = []

    source_map = getattr(ddm, "CONTINUOUS_SOURCE_MAP", {})
    for vk, var in ddm.VARIANTS.items():
        for component, cols in [
            ("drift", var.drift_cols),
            ("distance_to_bound", var.dist_cols),
            ("nondecision_time", getattr(var, "t0_cols", ("intercept",))),
        ]:
            for c in cols:
                rows.append({
                    "variant": vk,
                    "variant_name": var.name,
                    "component": component,
                    "predictor": c,
                    "source_column_if_standardized": source_map.get(c, ""),
                    "is_erp_direct": c in {"cnv_z_global", "p300_z_global", "ern_strength_z", "pe_strength_z"},
                    "is_rl_latent": c in {"control_z", "actor_qdiff_z", "p_press_z", "belief_z", "assoc_z"},
                    "is_behavior_history": c in {"prev_error", "prev_fa", "prev_press", "prior_press_rate_z", "prior_fa_rate_z"},
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / "ddm_predictor_audit_by_variant.csv", index=False)

    erp = pd.DataFrame([
        {
            "erp_feature": "CNV",
            "latent_column": "cnv_z",
            "direct_ddm_predictor": True,
            "ddm_predictor_name": "cnv_z_global",
            "ddm_model": "D3",
            "component": "distance_to_bound",
            "timing_rationale": "Cue/S1-locked preparatory signal; temporally suitable for response preparation/caution.",
        },
        {
            "erp_feature": "P300",
            "latent_column": "p300_z",
            "direct_ddm_predictor": True,
            "ddm_predictor_name": "p300_z_global",
            "ddm_model": "D3",
            "component": "drift",
            "timing_rationale": "Target/S2-locked peri-decisional stimulus-evaluation signal; treated as a drift covariate.",
        },
        {
            "erp_feature": "ERN",
            "latent_column": "ern_strength_z",
            "direct_ddm_predictor": False,
            "ddm_predictor_name": "",
            "ddm_model": "",
            "component": "",
            "timing_rationale": "Response-locked post-error component; retained for interpretation/lagged post-error analyses, not same-trial DDM prediction.",
        },
        {
            "erp_feature": "Pe",
            "latent_column": "pe_strength_z",
            "direct_ddm_predictor": False,
            "ddm_predictor_name": "",
            "ddm_model": "",
            "component": "",
            "timing_rationale": "Response-locked post-error component; retained for interpretation/lagged post-error analyses, not same-trial DDM prediction.",
        },
    ])
    erp.to_csv(out / "erp_predictor_status.csv", index=False)

    md = []
    md.append("# DDM predictor audit")
    md.append("")
    md.append("Main correction: D3 uses CNV and P300 as direct ERP covariates. ERN and Pe are not direct same-trial DDM predictors.")
    md.append("")
    md.append("- P300 enters the drift component in D3.")
    md.append("- CNV enters the distance-to-bound component in D3.")
    md.append("- ERN and Pe are response-locked error-monitoring features; they are retained for interpretation and lagged exploratory analyses.")
    (out / "DDM_PREDICTOR_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stage2] wrote {out}")


# =============================================================================
# Stage 3: RT predictive fit / RT quantile check
# =============================================================================

def find_cv_fold_jsons(nested_root: Path, config_contains: str = "main", variant: str = "D2") -> List[Path]:
    nested_root = Path(nested_root)
    if not nested_root.exists():
        return []
    paths = sorted(nested_root.rglob(f"cv/{variant}/fold_*.json"))
    if config_contains:
        # Prefer exact main folder, but do not match sensitivity/main duplicate if absent.
        filt = []
        for p in paths:
            s = str(p)
            if config_contains == "main":
                if "_main" in s and "sensitivity" not in s:
                    filt.append(p)
            elif config_contains in s:
                filt.append(p)
        if filt:
            return filt
    return paths


def predict_rt_quantiles_for_fold(ddm, df_all: pd.DataFrame, fold_json: Path,
                                  t0_min: float, t0_max: float, dist_min: float,
                                  q_probs=(0.1, 0.25, 0.5, 0.75, 0.9),
                                  group_cols=("group_label", "trial_type")) -> List[Dict]:
    obj = read_json(fold_json)
    if obj is None:
        return []
    fit = obj["fit"]
    train_subs = [str(x) for x in obj["train_subjects"]]
    test_subs = [str(x) for x in obj["test_subjects"]]
    train = df_all[df_all["subj_id_ddm"].astype(str).isin(train_subs)].reset_index(drop=True)
    test = df_all[df_all["subj_id_ddm"].astype(str).isin(test_subs)].reset_index(drop=True)

    variant = ddm.VARIANTS[fit["variant"]]
    Xv, Xd, Xt0, _ = ddm.prepare_designs(train, test, variant)
    theta = np.asarray(fit["theta"], dtype=float)
    v, d, t0 = ddm.predict_components(theta, Xv, Xd, Xt0, t0_min, t0_max, dist_min)

    test = test.copy()
    test["group_label"] = np.where(test["group_high"].astype(int) == 1, "high", "low")
    test["trial_type"] = np.where(test["is_nogo"].astype(int) == 1, "NoGo", "Go")
    test["fold"] = obj["fold"]
    test["_v"] = v
    test["_d"] = d
    test["_t0"] = t0

    rows = []

    # Overall plus group/type combinations.
    group_specs = [(("all",), pd.Series(True, index=test.index), {"group_label": "all", "trial_type": "all"})]
    for gl in ["low", "high"]:
        m = test["group_label"].eq(gl)
        group_specs.append(((gl,), m, {"group_label": gl, "trial_type": "all"}))
    for tt in ["Go", "NoGo"]:
        m = test["trial_type"].eq(tt)
        group_specs.append(((tt,), m, {"group_label": "all", "trial_type": tt}))
    for gl in ["low", "high"]:
        for tt in ["Go", "NoGo"]:
            m = test["group_label"].eq(gl) & test["trial_type"].eq(tt)
            group_specs.append(((gl, tt), m, {"group_label": gl, "trial_type": tt}))

    for _, mask, lab in group_specs:
        sub = test.loc[mask].copy()
        if len(sub) == 0:
            continue
        obs_press = sub[(sub["press"] == 1) & np.isfinite(sub["rt_sec"])]["rt_sec"].to_numpy(float)
        if len(obs_press) < 5:
            continue

        # Pooled conditional predicted CDF for response RTs.
        deadline = sub["deadline_sec"].to_numpy(float)
        vv = sub["_v"].to_numpy(float)
        dd = sub["_d"].to_numpy(float)
        tt0 = sub["_t0"].to_numpy(float)
        denom = ddm.firstpass_cdf(deadline - tt0, dd, vv)
        denom_sum = float(np.sum(denom))
        if denom_sum <= 1e-9:
            continue

        grid = np.linspace(max(0.02, np.nanmin(tt0) + 0.001), float(np.nanmax(deadline)), 500)
        pooled = []
        for rt in grid:
            T = rt - tt0
            F = ddm.firstpass_cdf(T, dd, vv)
            F = np.where(T > 0, F, 0.0)
            pooled.append(float(np.sum(F) / denom_sum))
        pooled = np.clip(np.asarray(pooled), 0, 1)

        for q in q_probs:
            obs_q = float(np.quantile(obs_press, q))
            try:
                pred_q = float(np.interp(q, pooled, grid))
            except Exception:
                pred_q = np.nan
            rows.append({
                "fold": obj["fold"],
                "variant": fit["variant"],
                "group_label": lab["group_label"],
                "trial_type": lab["trial_type"],
                "q": q,
                "observed_rt_sec": obs_q,
                "predicted_rt_sec": pred_q,
                "error_sec": pred_q - obs_q if np.isfinite(pred_q) else np.nan,
                "n_observed_press": int(len(obs_press)),
                "n_trials": int(len(sub)),
                "predicted_response_mass": denom_sum / len(sub),
            })
    return rows


def stage3_rt_predictive_fit(args) -> None:
    base = Path(args.base).resolve()
    out = ensure_dir(Path(args.out).resolve() / "03_rt_predictive_fit")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage3")

    all_rows = []
    roots = []
    if args.nested_roots:
        roots = [Path(r) for r in args.nested_roots]
    else:
        roots = sorted((base / "results" / "nested_rl_ddm").glob("M6_seed*"))

    for root in roots:
        root = Path(root)
        lat = root / "nested_latents" / f"{root.name}_nested_trial_latents_for_ddm.csv"
        if not lat.exists():
            print(f"[stage3][skip] missing nested latents: {lat}")
            continue
        _, df, _ = load_ddm_data(ddm, lat, Path(args.data_root) if args.data_root else None,
                                 args.deadline_ms, args.rt_min_ms, args.rt_max_ms)
        folds = find_cv_fold_jsons(root, config_contains=args.config, variant=args.rt_variant)
        if not folds:
            print(f"[stage3][skip] no fold jsons for {root} variant={args.rt_variant} config={args.config}")
            continue
        for fj in folds:
            rows = predict_rt_quantiles_for_fold(ddm, df, fj, args.t0_min, args.t0_max, args.dist_min)
            for r in rows:
                r["root"] = root.name
                r["fold_json"] = str(fj)
            all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    res.to_csv(out / "rt_quantile_fit_by_fold.csv", index=False)

    if len(res) > 0:
        summ = (
            res.groupby(["root", "variant", "group_label", "trial_type", "q"], dropna=False)
            .agg(
                mean_observed_rt_sec=("observed_rt_sec", "mean"),
                mean_predicted_rt_sec=("predicted_rt_sec", "mean"),
                mean_error_sec=("error_sec", "mean"),
                se_error_sec=("error_sec", sem),
                n_folds=("fold", "nunique"),
                total_observed_press=("n_observed_press", "sum"),
            )
            .reset_index()
        )
        summ.to_csv(out / "rt_quantile_fit_summary.csv", index=False)

        # Plot overall quantiles per root.
        plt.figure(figsize=(7.5, 6))
        plotdf = summ[(summ["group_label"] == "all") & (summ["trial_type"] == "all")].copy()
        for root, g in plotdf.groupby("root"):
            plt.plot(g["mean_observed_rt_sec"], g["mean_predicted_rt_sec"], marker="o", label=root)
        mn = min(plotdf["mean_observed_rt_sec"].min(), plotdf["mean_predicted_rt_sec"].min())
        mx = max(plotdf["mean_observed_rt_sec"].max(), plotdf["mean_predicted_rt_sec"].max())
        plt.plot([mn, mx], [mn, mx], linestyle="--")
        plt.xlabel("Observed RT quantile (s)")
        plt.ylabel("Predicted RT quantile (s)")
        plt.title(f"RT predictive fit, {args.rt_variant}, nested {args.config}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "rt_quantile_fit_overall.png", dpi=300)
        plt.close()

    md = [
        "# RT predictive-fit / RT quantile check",
        "",
        f"- Variant: {args.rt_variant}",
        f"- Nested config: {args.config}",
        "- Output tables: `rt_quantile_fit_by_fold.csv`, `rt_quantile_fit_summary.csv`",
        "- Figure: `rt_quantile_fit_overall.png`",
    ]
    (out / "RT_PREDICTIVE_FIT_README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stage3] wrote {out}")


# =============================================================================
# Stage 4: lagged ERN/Pe post-error analysis
# =============================================================================

def fit_ols_beta(y: np.ndarray, X: np.ndarray, beta_index: int = 1) -> float:
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y2 = y[ok]
    X2 = X[ok]
    if len(y2) <= X2.shape[1] + 2:
        return np.nan
    try:
        b = np.linalg.lstsq(X2, y2, rcond=None)[0]
        return float(b[beta_index])
    except Exception:
        return np.nan


def cluster_boot_ols(df: pd.DataFrame, y_col: str, x_col: str, control_cols: List[str],
                     subj_col: str = "subj_id_ddm", n_boot: int = 3000, seed: int = 2026) -> Dict[str, float]:
    d = df.copy()
    needed = [y_col, x_col, subj_col] + control_cols
    d = d.dropna(subset=needed)
    if len(d) < 10 or d[x_col].std(ddof=0) < 1e-12:
        return {"beta": np.nan, "se_boot": np.nan, "ci025": np.nan, "ci975": np.nan, "n": int(len(d)), "n_subjects": int(d[subj_col].nunique())}

    X = np.column_stack([np.ones(len(d)), d[x_col].to_numpy(float)] + [d[c].to_numpy(float) for c in control_cols])
    y = d[y_col].to_numpy(float)
    beta = fit_ols_beta(y, X, beta_index=1)

    rng = np.random.default_rng(seed)
    subs = d[subj_col].unique()
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(subs, size=len(subs), replace=True)
        bd = pd.concat([d[d[subj_col] == s] for s in sampled], ignore_index=True)
        Xb = np.column_stack([np.ones(len(bd)), bd[x_col].to_numpy(float)] + [bd[c].to_numpy(float) for c in control_cols])
        yb = bd[y_col].to_numpy(float)
        boots.append(fit_ols_beta(yb, Xb, beta_index=1))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    return {
        "beta": beta,
        "se_boot": float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan,
        "ci025": float(np.percentile(boots, 2.5)) if len(boots) else np.nan,
        "ci975": float(np.percentile(boots, 97.5)) if len(boots) else np.nan,
        "n": int(len(d)),
        "n_subjects": int(d[subj_col].nunique()),
    }


def stage4_lagged_erp(args) -> None:
    base = Path(args.base).resolve()
    out = ensure_dir(Path(args.out).resolve() / "04_lagged_ern_pe")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage4")
    latents = find_latents(base, Path(args.latents) if args.latents else None, variant=args.variant, seed=args.seed)
    _, d, _ = load_ddm_data(ddm, latents, Path(args.data_root) if args.data_root else None,
                            args.deadline_ms, args.rt_min_ms, args.rt_max_ms)

    d = d.sort_values(["subj_id_ddm", "trial_index"]).copy()
    for col in ["press", "fa", "error", "is_nogo", "group_high", "rt_sec", "ern_strength_z", "pe_strength_z"]:
        if col not in d.columns:
            d[col] = np.nan

    next_cols = ["press", "fa", "error", "is_nogo", "rt_sec", "cue_up", "cue_bi"]
    for col in next_cols:
        d[f"next_{col}"] = d.groupby("subj_id_ddm")[col].shift(-1)
    d["next_trial_exists"] = d.groupby("subj_id_ddm")["trial_index"].shift(-1).notna().astype(int)

    fa_trials = d[(d["fa"] == 1) & (d["next_trial_exists"] == 1)].copy()
    fa_trials["ern_lag_z"] = standardize_series(fa_trials["ern_strength_z"])
    fa_trials["pe_lag_z"] = standardize_series(fa_trials["pe_strength_z"])
    fa_trials["next_rt_press_sec"] = np.where(fa_trials["next_press"] == 1, fa_trials["next_rt_sec"], np.nan)
    fa_trials["next_log_rt"] = np.log(fa_trials["next_rt_press_sec"])

    # Linear probability models for binary next-trial outcomes and OLS for log RT.
    outcomes = [
        ("next_press", "Next-trial press probability"),
        ("next_error", "Next-trial error probability"),
        ("next_fa", "Next-trial false-alarm indicator"),
        ("next_log_rt", "Next-trial log RT among presses"),
    ]
    predictors = [("ern_lag_z", "ERN on previous false alarm"), ("pe_lag_z", "Pe on previous false alarm")]
    controls = ["group_high", "next_is_nogo", "next_cue_up", "next_cue_bi"]

    rows = []
    for xcol, xlab in predictors:
        for ycol, ylab in outcomes:
            res = cluster_boot_ols(fa_trials, y_col=ycol, x_col=xcol, control_cols=controls,
                                   n_boot=args.bootstrap, seed=args.seed)
            rows.append({
                "predictor": xcol,
                "predictor_label": xlab,
                "outcome": ycol,
                "outcome_label": ylab,
                **res,
                "note": "Exploratory lagged analysis over false-alarm trials; coefficient from OLS/LPM with cluster bootstrap by subject.",
            })
    outdf = pd.DataFrame(rows)
    outdf.to_csv(out / "lagged_ern_pe_posterror_results.csv", index=False)
    fa_trials.to_csv(out / "lagged_ern_pe_analysis_dataset.csv", index=False)

    # Plot coefficients.
    plot = outdf.copy()
    plt.figure(figsize=(9, 5))
    y = np.arange(len(plot))
    x = plot["beta"].to_numpy(float)
    lo = plot["ci025"].to_numpy(float)
    hi = plot["ci975"].to_numpy(float)
    xerr = np.vstack([x - lo, hi - x])
    labels = plot["predictor"] + " → " + plot["outcome"]
    plt.axvline(0, linewidth=1)
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Lagged coefficient with cluster-bootstrap 95% CI")
    plt.title("Exploratory lagged ERN/Pe post-error analyses")
    plt.tight_layout()
    plt.savefig(out / "lagged_ern_pe_coefficients.png", dpi=300)
    plt.close()

    md = [
        "# Lagged ERN/Pe post-error exploratory analysis",
        "",
        "This analysis avoids same-trial post-decisional leakage by using ERN/Pe measured on false-alarm trial t to predict behavior on trial t+1.",
        "",
        f"- False-alarm trials with a following trial: {len(fa_trials)}",
        f"- Subjects represented: {fa_trials['subj_id_ddm'].nunique() if len(fa_trials) else 0}",
        "- Results: `lagged_ern_pe_posterror_results.csv`",
    ]
    (out / "LAGGED_ERP_README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stage4] wrote {out}")


# =============================================================================
# Stage 5: model recovery / simulation sanity check
# =============================================================================

def find_full_fit(base: Path, variant: str = "D2") -> Optional[Path]:
    """Locate a compatible full-fit artifact without assuming historical private directory names."""
    hits = sorted((base / "results").rglob(f"models/{variant}/fit.json"))
    return hits[0] if hits else None


def simulate_ddm_dataset(ddm, df: pd.DataFrame, fit: Dict, variant_key: str,
                         t0_min: float, t0_max: float, dist_min: float,
                         dt: float = 0.004, seed: int = 2026) -> pd.DataFrame:
    variant = ddm.VARIANTS[variant_key]
    Xv, Xd, Xt0, _ = ddm.prepare_designs(df, df, variant)
    theta = np.asarray(fit["theta"], dtype=float)
    v, dist, t0 = ddm.predict_components(theta, Xv, Xd, Xt0, t0_min, t0_max, dist_min)
    deadline = df["deadline_sec"].to_numpy(float)
    max_steps = int(np.ceil(np.nanmax(deadline - t0) / dt)) + 2
    n = len(df)
    rng = np.random.default_rng(seed)

    x = np.zeros(n, dtype=float)
    hit = np.zeros(n, dtype=bool)
    hit_time = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)

    sqrt_dt = math.sqrt(dt)
    for step in range(1, max_steps + 1):
        t = step * dt
        active = (~hit) & ((t + t0) <= deadline)
        if not np.any(active):
            break
        idx = np.where(active)[0]
        x[idx] += v[idx] * dt + sqrt_dt * rng.standard_normal(len(idx))
        h = idx[x[idx] >= dist[idx]]
        if len(h) > 0:
            hit[h] = True
            hit_time[h] = t + t0[h]

    sim = df.copy()
    sim["press"] = hit.astype(int)
    sim["rt_sec"] = hit_time
    sim.loc[sim["press"] == 0, "rt_sec"] = np.nan
    sim["hit"] = ((sim["is_nogo"] == 0) & (sim["press"] == 1)).astype(int)
    sim["fa"] = ((sim["is_nogo"] == 1) & (sim["press"] == 1)).astype(int)
    sim["miss"] = ((sim["is_nogo"] == 0) & (sim["press"] == 0)).astype(int)
    sim["cr"] = ((sim["is_nogo"] == 1) & (sim["press"] == 0)).astype(int)
    sim["correct"] = (((sim["is_nogo"] == 0) & (sim["press"] == 1)) | ((sim["is_nogo"] == 1) & (sim["press"] == 0))).astype(int)
    sim["error"] = 1 - sim["correct"]
    sim = sim.sort_values(["subj_id_ddm", "trial_index"]).reset_index(drop=True)
    sim["prev_error"] = sim.groupby("subj_id_ddm")["error"].shift(1).fillna(0).astype(int)
    sim["prev_fa"] = sim.groupby("subj_id_ddm")["fa"].shift(1).fillna(0).astype(int)
    sim["prev_press"] = sim.groupby("subj_id_ddm")["press"].shift(1).fillna(0).astype(int)
    return sim


def stage5_model_recovery(args) -> None:
    base = Path(args.base).resolve()
    out = ensure_dir(Path(args.out).resolve() / "05_model_recovery")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage5")
    latents = find_latents(base, Path(args.latents) if args.latents else None, variant=args.variant, seed=args.seed)
    _, d, _ = load_ddm_data(ddm, latents, Path(args.data_root) if args.data_root else None,
                            args.deadline_ms, args.rt_min_ms, args.rt_max_ms)

    fit_path = find_full_fit(base, variant=args.recovery_source_variant, root_hint=f"{args.variant}_seed{args.seed}")
    if fit_path and fit_path.exists():
        source_fit = read_json(fit_path)
        print(f"[stage5] using source fit {fit_path}")
    else:
        print("[stage5] no source full fit found; fitting source D2 now")
        source_fit = ddm.fit_one_model(
            d, ddm.VARIANTS[args.recovery_source_variant],
            deadline_ms=args.deadline_ms,
            n_starts=args.n_starts,
            seed=args.seed,
            l2=args.l2,
            t0_min=args.t0_min,
            t0_max=args.t0_max,
            dist_min=args.dist_min,
        )
        write_json(out / "source_fit_generated.json", source_fit)

    rows = []
    for rep in range(args.recovery_reps):
        print(f"[stage5] simulation recovery rep {rep+1}/{args.recovery_reps}")
        sim = simulate_ddm_dataset(ddm, d, source_fit, args.recovery_source_variant,
                                   args.t0_min, args.t0_max, args.dist_min,
                                   dt=args.sim_dt, seed=args.seed + 10000 + rep)
        sim.to_csv(out / f"simulated_dataset_rep_{rep:02d}.csv.gz", index=False, compression="gzip")

        for vk in args.recovery_fit_variants:
            fit = ddm.fit_one_model(
                sim, ddm.VARIANTS[vk],
                deadline_ms=args.deadline_ms,
                n_starts=args.n_starts,
                seed=args.seed + 1000*rep + len(vk),
                l2=args.l2,
                t0_min=args.t0_min,
                t0_max=args.t0_max,
                dist_min=args.dist_min,
            )
            rows.append({
                "rep": rep,
                "fit_variant": vk,
                "fit_variant_name": ddm.VARIANTS[vk].name,
                "source_variant": args.recovery_source_variant,
                "loglik": fit["loglik"],
                "nll": fit["nll"],
                "aic": fit["aic"],
                "bic": fit["bic"],
                "n_params": fit["n_params"],
                "success": fit["success"],
                "sim_press_rate": float(sim["press"].mean()),
                "sim_hit_rate": float(sim.loc[sim["is_nogo"] == 0, "press"].mean()),
                "sim_fa_rate": float(sim.loc[sim["is_nogo"] == 1, "press"].mean()),
            })
    res = pd.DataFrame(rows)
    res.to_csv(out / "model_recovery_fit_results.csv", index=False)

    if len(res) > 0:
        ranks = res.copy()
        ranks["rank_aic"] = ranks.groupby("rep")["aic"].rank(method="first", ascending=True)
        rank_summary = (
            ranks.groupby("fit_variant")
            .agg(
                n_reps=("rep", "nunique"),
                mean_aic=("aic", "mean"),
                mean_rank_aic=("rank_aic", "mean"),
                wins_aic=("rank_aic", lambda x: int((x == 1).sum())),
            )
            .reset_index()
            .sort_values("mean_rank_aic")
        )
        rank_summary.to_csv(out / "model_recovery_rank_summary.csv", index=False)

    md = [
        "# Model recovery / simulation sanity check",
        "",
        f"- Source model: {args.recovery_source_variant}",
        f"- Refit variants: {args.recovery_fit_variants}",
        f"- Repetitions: {args.recovery_reps}",
        "",
        "This is a conditional DDM recovery check using the original trial-level covariates and RL latents.",
        "It is not a full recovery of the upstream Bayesian RL model.",
    ]
    (out / "MODEL_RECOVERY_README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stage5] wrote {out}")


# =============================================================================
# Stage 6: exploratory joint reinforcement-learning/first-passage maximum-likelihood branch
# =============================================================================

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def firstpass_logpdf_np(t, d, v):
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.full_like(t, -1e12, dtype=float)
    ok = (t > 1e-6) & (d > 1e-6) & np.isfinite(t) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        tt, dd, vv = t[ok], d[ok], v[ok]
        out[ok] = np.log(dd) - 0.5*np.log(2*np.pi) - 1.5*np.log(tt) - ((dd - vv*tt)**2)/(2*tt)
    return out


def firstpass_cdf_np(T, d, v):
    T = np.asarray(T, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    F = np.zeros_like(T, dtype=float)
    ok = (T > 1e-6) & (d > 1e-6) & np.isfinite(T) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        TT, dd, vv = T[ok], d[ok], v[ok]
        sqrtT = np.sqrt(TT)
        term1 = norm.cdf((vv*TT - dd)/sqrtT)
        expo = np.exp(np.clip(2*vv*dd, -50, 50))
        term2 = expo * norm.cdf(-(vv*TT + dd)/sqrtT)
        F[ok] = np.clip(term1 + term2, EPS, 1-EPS)
    return np.clip(F, EPS, 1-EPS)


def joint_forward_states(df: pd.DataFrame, par: Dict[str, float], use_actor=True, use_control=True) -> Dict[str, np.ndarray]:
    d = df.sort_values(["subj_id_ddm", "trial_index"]).copy()
    n = len(d)
    control = np.zeros(n)
    qdiff = np.zeros(n)
    p_rl = np.zeros(n)

    # Cue index: down=0, bi=1, up=2
    cue_idx = np.where(d["cue_up"].to_numpy(int) == 1, 2, np.where(d["cue_bi"].to_numpy(int) == 1, 1, 0))
    action = d["press"].to_numpy(float)
    correct = d["correct"].to_numpy(float)
    error = d["error"].to_numpy(float)
    nogo = d["is_nogo"].to_numpy(float)
    prev_error = d["prev_error"].to_numpy(float)
    group = d["group_high"].to_numpy(float)
    subjects = d["subj_id_ddm"].astype(str).to_numpy()

    alpha_actor = expit(par.get("alpha_actor_raw", 0.0))
    alpha_ctrl = expit(par.get("alpha_ctrl_raw", 0.0))
    beta_actor = np.exp(par.get("log_beta_actor", 0.0))
    w_control = par.get("w_control", 0.0)

    bias = par.get("rl_bias", 0.0)
    b_nogo = par.get("rl_b_nogo", 0.0)
    b_prev_error = par.get("rl_b_prev_error", 0.0)
    b_group = par.get("rl_b_group", 0.0)
    b_cue_bi = par.get("rl_b_cue_bi", 0.0)
    b_cue_up = par.get("rl_b_cue_up", 0.0)

    # State per subject.
    Q: Dict[str, np.ndarray] = {}
    C: Dict[str, float] = {}
    for i in range(n):
        s = subjects[i]
        if s not in Q:
            Q[s] = np.zeros((3, 2), dtype=float)
            C[s] = 0.0
        ci = cue_idx[i]
        qd = Q[s][ci, 1] - Q[s][ci, 0]
        cprev = C[s]
        qdiff[i] = qd
        control[i] = cprev

        logit = (
            bias
            + b_nogo * nogo[i]
            + b_prev_error * prev_error[i]
            + b_group * group[i]
            + b_cue_bi * (cue_idx[i] == 1)
            + b_cue_up * (cue_idx[i] == 2)
        )
        if use_actor:
            logit += beta_actor * qd
        if use_control:
            logit += w_control * cprev
        p = np.clip(expit(logit), 1e-9, 1-1e-9)
        p_rl[i] = p

        # teacher-forced state updates using observed/simulated behavior
        if use_control:
            C[s] = np.clip(cprev + alpha_ctrl * (error[i] - cprev), 0, 1)
        if use_actor:
            chosen = int(action[i] > 0.5)
            delta = correct[i] - Q[s][ci, chosen]
            Q[s][ci, chosen] += alpha_actor * delta
            Q[s] = np.clip(Q[s], -5, 5)

    return {"control": control, "qdiff": qdiff, "p_rl": p_rl}


JOINT_PARAM_NAMES = [
    "rl_bias", "rl_b_nogo", "rl_b_prev_error", "rl_b_group", "rl_b_cue_bi", "rl_b_cue_up",
    "alpha_actor_raw", "log_beta_actor", "alpha_ctrl_raw", "w_control",
    "v_int", "v_nogo", "v_group", "v_cue_bi", "v_cue_up", "v_qdiff", "v_prl",
    "d_int", "d_nogo", "d_group", "d_cue_bi", "d_cue_up", "d_prev_error", "d_control",
    "t0_raw",
]


def vec_to_par(x: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(JOINT_PARAM_NAMES, x)}


def joint_nll(x: np.ndarray, df: pd.DataFrame, use_actor=True, use_control=True,
              action_weight: float = 1.0, t0_min: float = 0.03, t0_max: float = 0.08,
              dist_min: float = 0.05, l2: float = 1e-4) -> float:
    par = vec_to_par(x)
    st = joint_forward_states(df, par, use_actor=use_actor, use_control=use_control)

    press = df["press"].to_numpy(float)
    rt = df["rt_sec"].to_numpy(float)
    deadline = df["deadline_sec"].to_numpy(float)
    nogo = df["is_nogo"].to_numpy(float)
    group = df["group_high"].to_numpy(float)
    cue_bi = df["cue_bi"].to_numpy(float)
    cue_up = df["cue_up"].to_numpy(float)
    prev_error = df["prev_error"].to_numpy(float)

    qdiff = standardize_series(pd.Series(st["qdiff"])).to_numpy(float)
    prl = standardize_series(pd.Series(st["p_rl"])).to_numpy(float)
    control = standardize_series(pd.Series(st["control"])).to_numpy(float)

    # RL action likelihood.
    p = np.clip(st["p_rl"], 1e-9, 1 - 1e-9)
    ll_action = np.sum(press * np.log(p) + (1 - press) * np.log(1 - p))

    # DDM likelihood.
    v = (
        par["v_int"] + par["v_nogo"]*nogo + par["v_group"]*group
        + par["v_cue_bi"]*cue_bi + par["v_cue_up"]*cue_up
        + (par["v_qdiff"]*qdiff if use_actor else 0)
        + (par["v_prl"]*prl if use_actor else 0)
    )
    dist = softplus(
        par["d_int"] + par["d_nogo"]*nogo + par["d_group"]*group
        + par["d_cue_bi"]*cue_bi + par["d_cue_up"]*cue_up
        + par["d_prev_error"]*prev_error
        + (par["d_control"]*control if use_control else 0)
    ) + dist_min
    t0 = t0_min + (t0_max - t0_min) * expit(par["t0_raw"])

    is_press = press > 0.5
    ll = np.zeros(len(df), dtype=float)
    if np.any(is_press):
        ll[is_press] = firstpass_logpdf_np(rt[is_press] - t0, dist[is_press], v[is_press])
    if np.any(~is_press):
        F = firstpass_cdf_np(deadline[~is_press] - t0, dist[~is_press], v[~is_press])
        ll[~is_press] = np.log(np.clip(1 - F, EPS, 1))
    if not np.all(np.isfinite(ll)):
        return 1e30

    penalty = l2 * float(np.sum(np.asarray(x)**2))
    return -float(action_weight * ll_action + np.sum(ll)) + penalty


def fit_joint_model(df: pd.DataFrame, use_actor=True, use_control=True, n_starts=5, seed=2026,
                    maxiter=500, action_weight=1.0, t0_min=0.03, t0_max=0.08,
                    dist_min=0.05, l2=1e-4) -> Dict:
    rng = np.random.default_rng(seed)
    npar = len(JOINT_PARAM_NAMES)
    base = np.zeros(npar)
    # reasonable DDM initial values
    base[JOINT_PARAM_NAMES.index("v_int")] = 0.8
    base[JOINT_PARAM_NAMES.index("d_int")] = 0.0
    base[JOINT_PARAM_NAMES.index("log_beta_actor")] = 0.0

    bounds = [(-5, 5)] * npar
    starts = [base] + [base + rng.normal(0, 0.25, npar) for _ in range(max(0, n_starts - 1))]

    best = None
    hist = []
    for si, x0 in enumerate(starts):
        res = minimize(
            joint_nll,
            x0=x0,
            args=(df, use_actor, use_control, action_weight, t0_min, t0_max, dist_min, l2),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 30},
        )
        hist.append({"start": si, "success": bool(res.success), "nll": float(res.fun), "message": str(res.message)})
        if best is None or res.fun < best.fun:
            best = res

    k = npar
    n = len(df)
    return {
        "success": bool(best.success),
        "message": str(best.message),
        "nll": float(best.fun),
        "loglik_joint": -float(best.fun),
        "aic_joint": float(2*k + 2*best.fun),
        "bic_joint": float(k*np.log(n) + 2*best.fun),
        "n_params": k,
        "n_trials": n,
        "use_actor": use_actor,
        "use_control": use_control,
        "param_names": JOINT_PARAM_NAMES,
        "theta": best.x.tolist(),
        "params": vec_to_par(best.x),
        "history": hist,
    }


def stage6_joint_model(args) -> None:
    base = Path(args.base).resolve()
    out = ensure_dir(Path(args.out).resolve() / "06_joint_model_exploratory")
    ddm = import_module_from_path(Path(args.ddm_script), "ddm_stage6")
    latents = find_latents(base, Path(args.latents) if args.latents else None, variant=args.variant, seed=args.seed)
    _, d, _ = load_ddm_data(ddm, latents, Path(args.data_root) if args.data_root else None,
                            args.deadline_ms, args.rt_min_ms, args.rt_max_ms)

    # Optional speed subset by subjects.
    if args.joint_max_subjects and args.joint_max_subjects > 0:
        subs = sorted(d["subj_id_ddm"].unique())[:args.joint_max_subjects]
        d = d[d["subj_id_ddm"].isin(subs)].reset_index(drop=True)

    models = [
        ("J0_baseline_no_actor_no_control", False, False),
        ("J1_control_only", False, True),
        ("J2_actor_control", True, True),
    ]
    rows = []
    for name, use_actor, use_control in models:
        print(f"[stage6] fitting {name}")
        fit = fit_joint_model(
            d,
            use_actor=use_actor,
            use_control=use_control,
            n_starts=args.joint_starts,
            seed=args.seed,
            maxiter=args.joint_maxiter,
            action_weight=args.joint_action_weight,
            t0_min=args.t0_min,
            t0_max=args.t0_max,
            dist_min=args.dist_min,
            l2=args.l2,
        )
        write_json(out / f"{name}.json", fit)
        rows.append({
            "model": name,
            "success": fit["success"],
            "nll_joint": fit["nll"],
            "aic_joint": fit["aic_joint"],
            "bic_joint": fit["bic_joint"],
            "n_params": fit["n_params"],
            "n_trials": fit["n_trials"],
            "use_actor": use_actor,
            "use_control": use_control,
        })
    res = pd.DataFrame(rows).sort_values("aic_joint")
    res.to_csv(out / "joint_model_model_comparison.csv", index=False)

    md = [
        "# Exploratory joint reinforcement-learning/first-passage branch",
        "",
        "This is a global MLE exploratory branch that jointly optimizes an action likelihood and a censored one-boundary DDM likelihood.",
        "It is not the same as the primary hierarchical Bayesian RL model and should not replace the leakage-controlled nested analysis unless it is separately validated.",
        "",
        "Models:",
        "- J0: baseline task/history terms, no actor/control states.",
        "- J1: control state only.",
        "- J2: actor-control states.",
    ]
    (out / "JOINT_MODEL_DEVELOPMENT_README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stage6] wrote {out}")


# =============================================================================
# Runner
# =============================================================================

def run_all(args) -> None:
    stage1_data_provenance(args)
    stage2_predictor_audit(args)
    stage3_rt_predictive_fit(args)
    stage4_lagged_erp(args)
    stage5_model_recovery(args)
    stage6_joint_model(args)
    print(f"[all] done. Outputs in {args.out}")


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp):
        sp.add_argument("--base", default=".")
        sp.add_argument("--out", default="results/model_development/diffusion_quality_extensions")
        sp.add_argument("--ddm_script", default="shared/ddm/one_boundary_diffusion.py")
        sp.add_argument("--latents", default=None)
        sp.add_argument("--data_root", default="data/raw/csv")
        sp.add_argument("--variant", default="M6")
        sp.add_argument("--seed", type=int, default=2026)
        sp.add_argument("--deadline_ms", type=float, default=700)
        sp.add_argument("--rt_min_ms", type=float, default=100)
        sp.add_argument("--rt_max_ms", type=float, default=1000)
        sp.add_argument("--t0_min", type=float, default=0.03)
        sp.add_argument("--t0_max", type=float, default=0.08)
        sp.add_argument("--dist_min", type=float, default=0.05)
        sp.add_argument("--l2", type=float, default=1e-4)
        sp.add_argument("--n_starts", type=int, default=4)

        # stage 3
        sp.add_argument("--nested_roots", nargs="*", default=None)
        sp.add_argument("--rt_variant", default="D2")
        sp.add_argument("--config", default="main")

        # stage 4
        sp.add_argument("--bootstrap", type=int, default=3000)

        # stage 5
        sp.add_argument("--recovery_reps", type=int, default=5)
        sp.add_argument("--recovery_source_variant", default="D2")
        sp.add_argument("--recovery_fit_variants", nargs="+", default=["D0", "D2", "D5"])
        sp.add_argument("--sim_dt", type=float, default=0.004)

        # stage 6
        sp.add_argument("--joint_starts", type=int, default=5)
        sp.add_argument("--joint_maxiter", type=int, default=500)
        sp.add_argument("--joint_action_weight", type=float, default=1.0)
        sp.add_argument("--joint_max_subjects", type=int, default=0,
                        help="Optional restricted subset for validation. 0 means all subjects.")

    for name, func in [
        ("all", run_all),
        ("data_audit", stage1_data_provenance),
        ("predictor_audit", stage2_predictor_audit),
        ("rt_fit", stage3_rt_predictive_fit),
        ("lagged_erp", stage4_lagged_erp),
        ("model_recovery", stage5_model_recovery),
        ("joint_model", stage6_joint_model),
    ]:
        sp = sub.add_parser(name)
        add_common(sp)
        sp.set_defaults(func=func)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
