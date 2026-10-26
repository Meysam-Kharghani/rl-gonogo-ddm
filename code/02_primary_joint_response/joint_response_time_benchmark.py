#!/usr/bin/env python3
"""Strict outer-fold RL -> fixed-horizon right-censored one-boundary DDM benchmark.

For every RL outer fold, both outer-training and outer-test latent trajectories
are generated from that fold's train-only RL posterior. DDM parameters are fit
only on outer-training subjects and evaluated only on untouched outer-test
subjects. Subject-level ELPD is retained for clustered bootstrap contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def import_py(path: str | Path, name: str):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((base + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def jsonable(x: Any):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(jsonable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def safe_logit(x, eps: float = 1e-6):
    z = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(z / (1.0 - z))


def cue_probability_map(path: str | Path) -> dict[str, float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k).strip().lower(): float(v) for k, v in raw.items()}


def add_beliefs(df: pd.DataFrame, probs: dict[str, float], strength: float) -> pd.DataFrame:
    out = df.copy()
    trial_col = "trial_index" if "trial_index" in out.columns else "trial_index_numeric"
    if trial_col not in out.columns:
        out["trial_index_numeric"] = out.groupby("subj_id_ddm").cumcount() + 1
        trial_col = "trial_index_numeric"
    out = out.sort_values(["subj_id_ddm", trial_col]).reset_index(drop=True)
    cue = out["cue_label_ddm"].astype(str).str.strip().str.lower()
    out["instructed_nogo_prob"] = cue.map(probs)
    if out["instructed_nogo_prob"].isna().any():
        bad = sorted(cue[out["instructed_nogo_prob"].isna()].unique())
        raise ValueError(f"Missing instructed probability for cues: {bad}")
    out["instructed_nogo_logit"] = safe_logit(out["instructed_nogo_prob"])
    out["_nogo"] = pd.to_numeric(out["is_nogo"], errors="coerce").fillna(0.0)
    grp = out.groupby(["subj_id_ddm", "cue_label_ddm"], sort=False)
    prior_n = grp.cumcount().astype(float)
    prior_ng = grp["_nogo"].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    a0 = strength * out["instructed_nogo_prob"]
    b0 = strength * (1.0 - out["instructed_nogo_prob"])
    out["empirical_nogo_prob"] = (a0 + prior_ng) / (a0 + b0 + prior_n)
    out["empirical_nogo_logit"] = safe_logit(out["empirical_nogo_prob"])
    if "trial_index" not in out.columns:
        out["trial_index"] = out[trial_col]
    return out.drop(columns=["_nogo"])


def register_variants(ddm) -> None:
    V = ddm.DDMVariant
    ddm.CONTINUOUS_SOURCE_MAP.update({
        "instructed_nogo_prob_z": "instructed_nogo_prob",
        "instructed_nogo_logit_z": "instructed_nogo_logit",
        "empirical_nogo_prob_z": "empirical_nogo_prob",
        "empirical_nogo_logit_z": "empirical_nogo_logit",
    })
    # Historical variant names are retained for compatibility. D2, D12, and D13
    # include the full RL block: policy probability, actor Q-difference, and control.
    ddm.VARIANTS.update({
        "D9": V(
            key="D9", name="actor_control_plus_compact_lagged_history",
            drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                        "actor_qdiff_z", "p_press_z", "prev_press",
                        "prior_press_rate_z", "prior_fa_rate_z"),
            dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                       "prev_error", "prev_fa", "control_z",
                       "prior_press_rate_z", "prior_fa_rate_z"),
        ),
        "D10": V(
            key="D10", name="instructed_numeric_prior",
            drift_cols=("intercept", "is_nogo", "group_high", "instructed_nogo_logit_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "instructed_nogo_prob_z"),
        ),
        "D11": V(
            key="D11", name="lagged_empirical_cue_belief",
            drift_cols=("intercept", "is_nogo", "group_high", "empirical_nogo_logit_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "empirical_nogo_prob_z"),
        ),
        "D12": V(
            key="D12", name="instructed_prior_plus_actor_control",
            drift_cols=("intercept", "is_nogo", "group_high", "instructed_nogo_logit_z",
                        "actor_qdiff_z", "p_press_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error",
                       "instructed_nogo_prob_z", "control_z"),
        ),
        "D13": V(
            key="D13", name="empirical_cue_belief_plus_actor_control",
            drift_cols=("intercept", "is_nogo", "group_high", "empirical_nogo_logit_z",
                        "actor_qdiff_z", "p_press_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error",
                       "empirical_nogo_prob_z", "control_z"),
        ),
        "D14": V(
            key="D14", name="actor_control_plus_compact_lagged_history",
            drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                        "actor_qdiff_z", "prev_press", "prior_press_rate_z", "prior_fa_rate_z"),
            dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                       "prev_error", "prev_fa", "control_z",
                       "prior_press_rate_z", "prior_fa_rate_z"),
        ),
        "D15": V(
            key="D15", name="policy_plus_compact_lagged_history",
            drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                        "p_press_z", "prev_press", "prior_press_rate_z", "prior_fa_rate_z"),
            dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                       "prev_error", "prev_fa", "prior_press_rate_z", "prior_fa_rate_z"),
        ),
    })


def prepare_frame(ddm, rows: list[dict], data_root: Path, deadline_ms: float,
                  rt_min_ms: float, rt_max_ms: float,
                  probs: dict[str, float], prior_strength: float) -> tuple[pd.DataFrame, dict]:
    """Prepare the fixed-horizon DDM frame without erasing behavioral history.

    Instructed/empirical cue representations are computed on the complete
    pre-filter trial sequence. Presses after the analytic horizon are retained
    but right-censored at the horizon; only press trials with missing RT or RT
    below the lower validity threshold are excluded from the DDM likelihood.
    """
    if not np.isclose(float(rt_max_ms), float(deadline_ms)):
        raise ValueError(
            "For fixed-horizon censoring, rt_max_ms must equal deadline_ms; "
            f"got rt_max_ms={rt_max_ms}, deadline_ms={deadline_ms}."
        )

    with tempfile.TemporaryDirectory(prefix="strict_ddm_latents_") as td:
        fp = Path(td) / "latents.csv"
        pd.DataFrame(rows).to_csv(fp, index=False)
        raw = ddm.load_latents(fp, deadline_ms=deadline_ms, data_root=data_root)

    # Build strictly lagged cue beliefs on the COMPLETE behavioral sequence.
    raw = add_beliefs(raw.reset_index(drop=True).copy(), probs, prior_strength)

    rt_min = float(rt_min_ms) / 1000.0
    deadline = float(deadline_ms) / 1000.0
    press_obs = pd.to_numeric(raw["press"], errors="coerce").fillna(0).astype(int).eq(1)
    rt_obs = pd.to_numeric(raw["rt_sec"], errors="coerce")

    invalid_press = press_obs & (~np.isfinite(rt_obs) | (rt_obs < rt_min))
    keep = ~invalid_press

    raw["press_observed"] = press_obs.astype(int)
    raw["rt_observed_sec"] = rt_obs
    raw["ddm_event_by_deadline"] = (press_obs & np.isfinite(rt_obs) & (rt_obs >= rt_min) & (rt_obs <= deadline)).astype(int)
    raw["ddm_late_press_censored"] = (press_obs & np.isfinite(rt_obs) & (rt_obs > deadline)).astype(int)
    raw["ddm_right_censored"] = (raw["ddm_event_by_deadline"].eq(0) & keep).astype(int)

    d = raw.loc[keep].copy().reset_index(drop=True)
    d["press"] = d["ddm_event_by_deadline"].astype(int)
    d.loc[d["press"].eq(0), "rt_sec"] = np.nan

    info = {
        "rt_min_ms": float(rt_min_ms),
        "rt_max_ms": float(rt_max_ms),
        "deadline_ms": float(deadline_ms),
        "n_raw": int(len(raw)),
        "n_kept": int(len(d)),
        "n_removed": int(invalid_press.sum()),
        "n_removed_press_missing_or_anticipatory_rt": int(invalid_press.sum()),
        "n_events_by_deadline": int(d["ddm_event_by_deadline"].sum()),
        "n_right_censored": int(d["ddm_right_censored"].sum()),
        "n_late_press_right_censored": int(d["ddm_late_press_censored"].sum()),
        "late_press_handling": "right_censored_at_fixed_analytic_horizon",
    }
    return d, info


def bootstrap_ci(vals, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        means[b] = np.mean(rng.choice(x, size=len(x), replace=True))
    return float(np.mean(x)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nested-script", required=True)
    ap.add_argument("--rl-script", required=True)
    ap.add_argument("--ddm-script", required=True)
    ap.add_argument("--rl-out", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--cue-probabilities", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rl-variant", default="M6")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--folds", nargs="+", type=int, default=None,
                    help="Optional subset of existing outer-fold indices for restricted validation runs.")
    ap.add_argument("--max-rl-draws", type=int, default=300)
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--variants", nargs="+",
                    default=["D0", "D2", "D5", "D6", "D9", "D10", "D11", "D12", "D13", "D14", "D15"])
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--deadline-ms", type=float, default=700.0)
    ap.add_argument("--rt-min-ms", type=float, default=100.0)
    ap.add_argument("--rt-max-ms", type=float, default=700.0)
    ap.add_argument("--t0-min", type=float, default=0.03)
    ap.add_argument("--t0-max", type=float, default=0.03)
    ap.add_argument("--dist-min", type=float, default=0.05)
    ap.add_argument("--horizons", nargs="+", type=int, default=[0, 20, 50, 100, 200])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    nested = import_py(args.nested_script, "strict_ddm_nested_support")
    rlmod = import_py(args.rl_script, "strict_ddm_rl_model")
    ddm = import_py(args.ddm_script, "strict_ddm_model")
    register_variants(ddm)

    unknown = [v for v in args.variants if v not in ddm.VARIANTS]
    if unknown:
        raise ValueError(f"Unknown DDM variants: {unknown}")

    probs = cue_probability_map(args.cue_probabilities)
    arr = rlmod.load_artifacts(Path(args.rl_out))
    meta = rlmod.get_meta(arr)
    n_subj = int(meta["n_subj"])
    all_idx = set(range(n_subj))

    cv_path = nested.cv_summary_path(Path(args.rl_out), args.seed, args.rl_variant)
    cv_obj = nested.read_json(cv_path)
    fold_meta = cv_obj.get("folds", [])
    if not fold_meta:
        raise ValueError(f"No fold metadata in {cv_path}")
    if len(fold_meta) != args.k:
        raise ValueError(f"Requested k={args.k}, but metadata contains {len(fold_meta)} folds")
    selected = set(args.folds) if args.folds is not None else {int(fr["fold"]) for fr in fold_meta}

    fold_rows: list[dict] = []
    subject_rows: list[dict] = []
    fit_rows: list[dict] = []
    qc_rows: list[dict] = []

    for fr in fold_meta:
        fi = int(fr["fold"])
        if fi not in selected:
            continue
        test_idx = [int(x) for x in fr.get("test_subject_indices", [])]
        if not test_idx:
            fr2 = nested.read_json(nested.fold_result_path(Path(args.rl_out), args.seed, args.rl_variant, fi), default={})
            test_idx = [int(x) for x in fr2.get("test_subject_indices", [])]
        train_idx = sorted(all_idx.difference(test_idx))
        print(f"[outer fold {fi}] train={len(train_idx)} test={len(test_idx)}", flush=True)

        idata = nested.load_rl_fold_idata(Path(args.rl_out), args.seed, args.rl_variant, fi)
        arr_train = rlmod.subset_subjects(arr, train_idx)
        arr_test = rlmod.subset_subjects(arr, test_idx)
        lat_train = nested.posterior_draws_mean_latents(
            rlmod, arr_train, args.rl_variant, idata, args.max_rl_draws,
            stable_seed(args.seed, "ddm_train", fi),
        )
        lat_test = nested.posterior_draws_mean_latents(
            rlmod, arr_test, args.rl_variant, idata, args.max_rl_draws,
            stable_seed(args.seed, "ddm_test", fi),
        )
        rows_train = nested.arr_to_nested_latent_rows(rlmod, arr_train, lat_train, fi, args.seed, args.rl_variant)
        rows_test = nested.arr_to_nested_latent_rows(rlmod, arr_test, lat_test, fi, args.seed, args.rl_variant)
        train, train_qc = prepare_frame(
            ddm, rows_train, Path(args.data_root), args.deadline_ms,
            args.rt_min_ms, args.rt_max_ms, probs, args.prior_strength,
        )
        test, test_qc = prepare_frame(
            ddm, rows_test, Path(args.data_root), args.deadline_ms,
            args.rt_min_ms, args.rt_max_ms, probs, args.prior_strength,
        )
        qc_rows.append({"fold": fi, "split": "train", **train_qc})
        qc_rows.append({"fold": fi, "split": "test", **test_qc})

        for vk in args.variants:
            variant = ddm.VARIANTS[vk]
            print(f"[outer fold {fi}] fit {vk}: {variant.name}", flush=True)
            fit = ddm.fit_one_model(
                train, variant, deadline_ms=args.deadline_ms,
                n_starts=args.n_starts,
                seed=stable_seed(args.seed, "ddm_fit", fi, vk),
                l2=args.l2, t0_min=args.t0_min, t0_max=args.t0_max,
                dist_min=args.dist_min,
            )
            write_json(out / "fits" / f"fold_{fi:02d}" / vk / "fit.json", fit)
            history = fit.get("optimizer_history", [])
            fit_rows.append({
                "fold": fi, "variant": vk, "variant_name": variant.name,
                "success": bool(fit.get("success")), "message": fit.get("message", ""),
                "nll_train": fit.get("nll"), "n_params": fit.get("n_params"),
                "n_successful_starts": int(sum(bool(h.get("success")) for h in history)),
                "n_starts": int(len(history)),
                "best_start_nll": float(min((h.get("fun", np.inf) for h in history), default=np.nan)),
                "worst_start_nll": float(max((h.get("fun", -np.inf) for h in history), default=np.nan)),
            })

            for horizon in sorted(set(args.horizons)):
                target = test[pd.to_numeric(test["trial_n_prior"], errors="coerce").fillna(0) >= horizon].reset_index(drop=True)
                if target.empty:
                    continue
                sc = ddm.score_model(target, train, fit, args.t0_min, args.t0_max, args.dist_min)
                ppc = sc.get("ppc", {})
                fold_rows.append({
                    "fold": fi, "horizon_min_prior_trials": int(horizon),
                    "variant": vk, "variant_name": variant.name,
                    "n_test_trials": int(len(target)),
                    "n_test_subjects": int(target["subj_id_ddm"].nunique()),
                    "elpd": float(sc["elpd"]), "nll": float(sc["nll"]),
                    "elpd_per_trial": float(sc["elpd"] / len(target)),
                    "observed_press": ppc.get("observed_press"),
                    "predicted_press": ppc.get("predicted_press"),
                    "observed_fa": ppc.get("observed_fa"),
                    "predicted_fa": ppc.get("predicted_fa"),
                    "observed_hit": ppc.get("observed_hit"),
                    "predicted_hit": ppc.get("predicted_hit"),
                })

            # Primary subject-clustered inference uses horizon 0.
            for sid, g in test.groupby("subj_id_ddm", sort=False):
                scs = ddm.score_model(g.reset_index(drop=True), train, fit,
                                      args.t0_min, args.t0_max, args.dist_min)
                subject_rows.append({
                    "fold": fi, "subject_id": str(sid),
                    "variant": vk, "variant_name": variant.name,
                    "n_trials": int(len(g)), "elpd": float(scs["elpd"]),
                    "elpd_per_trial": float(scs["elpd"] / len(g)),
                })

    fold_df = pd.DataFrame(fold_rows)
    subj_df = pd.DataFrame(subject_rows)
    fit_df = pd.DataFrame(fit_rows)
    qc_df = pd.DataFrame(qc_rows)
    fold_df.to_csv(out / "crossvalidated_ddm_fold_scores.csv", index=False)
    subj_df.to_csv(out / "strict_ddm_subject_metrics.csv", index=False)
    fit_df.to_csv(out / "crossvalidated_ddm_optimizer_summary.csv", index=False)
    qc_df.to_csv(out / "crossvalidated_ddm_filter_qc.csv", index=False)

    summary = (
        fold_df.groupby(["horizon_min_prior_trials", "variant", "variant_name"], as_index=False)
        .agg(n_folds=("fold", "nunique"), total_elpd=("elpd", "sum"),
             mean_elpd_per_trial=("elpd_per_trial", "mean"),
             mean_observed_press=("observed_press", "mean"),
             mean_predicted_press=("predicted_press", "mean"),
             mean_observed_fa=("observed_fa", "mean"),
             mean_predicted_fa=("predicted_fa", "mean"),
             mean_observed_hit=("observed_hit", "mean"),
             mean_predicted_hit=("predicted_hit", "mean"))
    )
    summary["mean_abs_press_error"] = (summary["mean_predicted_press"] - summary["mean_observed_press"]).abs()
    summary["mean_abs_fa_error"] = (summary["mean_predicted_fa"] - summary["mean_observed_fa"]).abs()
    summary["mean_abs_hit_error"] = (summary["mean_predicted_hit"] - summary["mean_observed_hit"]).abs()
    summary = summary.sort_values(["horizon_min_prior_trials", "total_elpd"], ascending=[True, False])
    summary.to_csv(out / "crossvalidated_ddm_summary.csv", index=False)

    planned = [
        ("D14", "D6", "actor/control beyond history without p_press"),
        ("D9", "D14", "p_press beyond actor/control plus history"),
        ("D9", "D15", "actor/control beyond p_press plus history"),
        ("D15", "D6", "p_press beyond history"),
        ("D9", "D6", "total RL information beyond history"),
        ("D10", "D0", "instructed numeric prior beyond categorical design"),
        ("D11", "D0", "strictly lagged empirical cue belief beyond categorical design"),
        ("D11", "D10", "adaptive empirical cue belief beyond instructed prior"),
        ("D12", "D2", "instructed numeric vs categorical cue coding within full RL family"),
        ("D13", "D2", "empirical vs categorical cue coding within full RL family"),
        ("D13", "D12", "empirical vs instructed cue coding within full RL family"),
        ("D12", "D10", "full RL block beyond instructed prior"),
        ("D13", "D11", "full RL block beyond empirical cue belief"),
        ("D5", "D0", "actor/control without p_press beyond design"),
        ("D2", "D0", "full RL block beyond categorical design"),
    ]
    wide = subj_df.pivot_table(index="subject_id", columns="variant", values="elpd", aggfunc="sum")
    contrast_rows = []
    for i, (model, ref, label) in enumerate(planned):
        if model not in wide.columns or ref not in wide.columns:
            continue
        vals = (wide[model] - wide[ref]).dropna().to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci(vals, args.n_bootstrap, stable_seed(args.seed, "contrast", i, model, ref))
        contrast_rows.append({
            "model": model, "reference": ref, "contrast": label,
            "n_subjects": int(len(vals)), "mean_subject_delta_elpd": mean,
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
            "positive_subjects": int(np.sum(vals > 0)),
            "total_delta_elpd": float(np.sum(vals)),
        })
    contrast_df = pd.DataFrame(contrast_rows)
    contrast_df.to_csv(out / "crossvalidated_ddm_planned_contrasts.csv", index=False)

    settings = {
        "nested_script": str(Path(args.nested_script).resolve()),
        "rl_script": str(Path(args.rl_script).resolve()),
        "ddm_script": str(Path(args.ddm_script).resolve()),
        "rl_out": str(Path(args.rl_out).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "cue_probabilities": str(Path(args.cue_probabilities).resolve()),
        "out": str(out), "rl_variant": args.rl_variant, "seed": args.seed,
        "k": args.k, "folds_run": sorted(selected), "max_rl_draws": args.max_rl_draws,
        "variants": args.variants, "n_starts": args.n_starts, "l2": args.l2,
        "deadline_ms": args.deadline_ms, "rt_min_ms": args.rt_min_ms,
        "rt_max_ms": args.rt_max_ms,
        "prior_strength": args.prior_strength,
        "late_press_handling": (
            "Presses after the fixed analytic horizon are retained and right-censored; "
            "only press trials with missing RT or RT below the lower validity threshold are excluded."
        ),
        "empirical_belief_construction": (
            "Instructed and empirical cue representations are computed on the complete pre-filter "
            "trial sequence, so likelihood-specific RT exclusions/censoring do not erase preceding "
            "cue/target observations from the strictly lagged update."
        ),
        "t0_min": args.t0_min,
        "t0_max": args.t0_max, "dist_min": args.dist_min,
        "horizons": args.horizons, "n_bootstrap": args.n_bootstrap,
        "n_subjects_total": n_subj,
        "strict_outer_fold_definition": (
            "For each RL outer fold, both train and test latent trajectories were generated "
            "from that fold's train-only RL posterior. DDM parameters were fitted on outer-train "
            "subjects and evaluated only on untouched outer-test subjects."
        ),
    }
    write_json(out / "analysis_settings.json", settings)

    lines = [
        "# Strict stacked RL-to-DDM benchmark", "",
        "## Main cumulative-horizon summary", "",
        summary.head(80).to_markdown(index=False), "",
        "## Planned subject-clustered contrasts", "",
        contrast_df.to_markdown(index=False) if not contrast_df.empty else "No planned contrasts available.", "",
        "## Optimizer summary", "",
        fit_df.to_markdown(index=False) if not fit_df.empty else "No fits available.", "",
        "## Interpretation rules", "",
        "- D14 > D6 tests actor-value/control information beyond online behavioral history without p_press.",
        "- D9 > D14 tests policy probability beyond actor-value/control plus history.",
        "- D9 > D15 tests actor-value/control beyond policy probability plus history.",
        "- D15 > D6 tests policy probability beyond history.",
        "- D11 > D10 tests strictly lagged empirical updating beyond instructed cue probabilities.",
        "- Use horizon 0 as the primary analysis; later cumulative horizons are warm-start sensitivities.",
        "- Inspect optimizer success and multi-start stability before interpreting ELPD contrasts.",
    ]
    (out / "CROSSVALIDATED_DDM_BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nPlanned contrasts")
    print(contrast_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
