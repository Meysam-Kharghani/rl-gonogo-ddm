#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

"""Extend the existing one-boundary DDM with the missing reviewer-facing controls.

Added models
------------
D9  : actor-control + causal online behavioral history (hybrid)
D10 : constrained numerical instructed-prior model
D11 : causal empirical cue-belief model initialized at instructed probabilities
D12 : instructed prior + actor-control states
D13 : empirical cue belief + actor-control states

The script imports the project's finalized DDM implementation, so likelihood,
filtering, optimization, and censoring are identical to the existing analyses.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_utils import (  # noqa: E402
    add_instructed_and_empirical_cue_beliefs,
    bootstrap_mean_ci,
    cue_probability_map,
    ensure_dir,
    import_module,
    stratified_subject_folds,
    write_json,
)


def register_variants(ddm) -> None:
    V = ddm.DDMVariant
    ddm.CONTINUOUS_SOURCE_MAP.update({
        "instructed_nogo_prob_z": "instructed_nogo_prob",
        "instructed_nogo_logit_z": "instructed_nogo_logit",
        "empirical_nogo_prob_z": "empirical_nogo_prob",
        "empirical_nogo_logit_z": "empirical_nogo_logit",
    })
    ddm.VARIANTS.update({
        "D9": V(
            key="D9", name="actor_control_plus_compact_lagged_history",
            drift_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high",
                        "actor_qdiff_z", "p_press_z", "prev_press", "prior_press_rate_z", "prior_fa_rate_z"),
            dist_cols=("intercept", "is_nogo", "cue_up", "cue_bi", "group_high", "prev_error", "prev_fa",
                       "control_z", "prior_press_rate_z", "prior_fa_rate_z"),
        ),
        "D10": V(
            key="D10", name="instructed_numeric_prior",
            drift_cols=("intercept", "is_nogo", "group_high", "instructed_nogo_logit_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "instructed_nogo_prob_z"),
        ),
        "D11": V(
            key="D11", name="causal_empirical_cue_belief",
            drift_cols=("intercept", "is_nogo", "group_high", "empirical_nogo_logit_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "empirical_nogo_prob_z"),
        ),
        "D12": V(
            key="D12", name="instructed_prior_plus_actor_control",
            drift_cols=("intercept", "is_nogo", "group_high", "instructed_nogo_logit_z", "actor_qdiff_z", "p_press_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "instructed_nogo_prob_z", "control_z"),
        ),
        "D13": V(
            key="D13", name="empirical_cue_belief_plus_actor_control",
            drift_cols=("intercept", "is_nogo", "group_high", "empirical_nogo_logit_z", "actor_qdiff_z", "p_press_z"),
            dist_cols=("intercept", "is_nogo", "group_high", "prev_error", "empirical_nogo_prob_z", "control_z"),
        ),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddm-script", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--cue-probabilities", default=str(HERE / "config" / "instructed_cue_probabilities.json"))
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["D0", "D2", "D6", "D9", "D10", "D11", "D12", "D13"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-starts", type=int, default=12)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--deadline-ms", type=float, default=700.0)
    ap.add_argument("--rt-min-ms", type=float, default=100.0)
    ap.add_argument("--rt-max-ms", type=float, default=1000.0)
    ap.add_argument("--t0-min", type=float, default=0.03)
    ap.add_argument("--t0-max", type=float, default=0.08)
    ap.add_argument("--dist-min", type=float, default=0.05)
    ap.add_argument("--horizons", type=int, nargs="+", default=[0, 20, 50, 100, 200])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    ddm = import_module(args.ddm_script, "hybrid_cue_prior_development")
    register_variants(ddm)

    raw = ddm.load_latents(Path(args.latents), deadline_ms=args.deadline_ms,
                           data_root=Path(args.data_root) if args.data_root else None)
    d, filter_info = ddm.filter_ddm_trials(raw, rt_min_ms=args.rt_min_ms, rt_max_ms=args.rt_max_ms)
    target_col = "trial_class" if "trial_class" in d.columns else "raw_trial_class"
    trial_col = "trial_index" if "trial_index" in d.columns else "trial_index_numeric"
    if trial_col not in d.columns:
        d["trial_index_numeric"] = d.groupby("subj_id_ddm").cumcount() + 1
        trial_col = "trial_index_numeric"
    d = add_instructed_and_empirical_cue_beliefs(
        d, "subj_id_ddm", "cue_label_ddm", target_col, trial_col,
        cue_probability_map(args.cue_probabilities), args.prior_strength,
    )
    if "trial_index" not in d.columns:
        d["trial_index"] = d[trial_col]

    unknown = [v for v in args.variants if v not in ddm.VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    folds = stratified_subject_folds(d, "subj_id_ddm", "group_high", args.k, args.seed)
    rows: list[dict] = []
    fit_rows: list[dict] = []

    for fi, test_subs in enumerate(folds):
        train = d[~d["subj_id_ddm"].astype(str).isin(test_subs)].reset_index(drop=True)
        test = d[d["subj_id_ddm"].astype(str).isin(test_subs)].reset_index(drop=True)
        for vk in args.variants:
            variant = ddm.VARIANTS[vk]
            print(f"[fold {fi+1}/{args.k}] {vk} {variant.name}", flush=True)
            fit = ddm.fit_one_model(
                train, variant, deadline_ms=args.deadline_ms, n_starts=args.n_starts,
                seed=args.seed + fi * 1000 + sum(map(ord, vk)), l2=args.l2,
                t0_min=args.t0_min, t0_max=args.t0_max, dist_min=args.dist_min,
            )
            fit_rows.append({
                "fold": fi, "variant": vk, "variant_name": variant.name,
                "success": fit.get("success"), "nll_train": fit.get("nll"),
                "n_params": fit.get("n_params"), "message": fit.get("message", ""),
            })
            write_json(out / "fits" / f"fold_{fi:02d}" / vk / "fit.json", fit)

            for horizon in sorted(set(args.horizons)):
                target = test[pd.to_numeric(test["trial_n_prior"], errors="coerce").fillna(0) >= horizon].reset_index(drop=True)
                if target.empty:
                    continue
                sc = ddm.score_model(target, train, fit, args.t0_min, args.t0_max, args.dist_min)
                rows.append({
                    "fold": fi, "variant": vk, "variant_name": variant.name,
                    "horizon_min_prior_trials": int(horizon), "n_test_trials": len(target),
                    "n_test_subjects": int(target["subj_id_ddm"].nunique()),
                    "elpd": float(sc["elpd"]), "nll": float(sc["nll"]),
                    "elpd_per_trial": float(sc["elpd"] / len(target)),
                    "observed_press": sc.get("ppc", {}).get("observed_press"),
                    "predicted_press": sc.get("ppc", {}).get("predicted_press"),
                    "observed_fa": sc.get("ppc", {}).get("observed_fa"),
                    "predicted_fa": sc.get("ppc", {}).get("predicted_fa"),
                    "observed_hit": sc.get("ppc", {}).get("observed_hit"),
                    "predicted_hit": sc.get("ppc", {}).get("predicted_hit"),
                })

    fold_df = pd.DataFrame(rows)
    fit_df = pd.DataFrame(fit_rows)
    fold_df.to_csv(out / "hybrid_ddm_fold_scores.csv", index=False)
    fit_df.to_csv(out / "hybrid_ddm_optimizer_summary.csv", index=False)

    summary = (
        fold_df.groupby(["horizon_min_prior_trials", "variant", "variant_name"], as_index=False)
        .agg(n_folds=("fold", "nunique"), total_elpd=("elpd", "sum"), mean_elpd_per_trial=("elpd_per_trial", "mean"),
             mean_abs_press_calibration=("predicted_press", lambda x: np.nan))
    )
    # Recompute calibration error cleanly from fold rows.
    cal = fold_df.assign(
        press_abs=lambda x: (x["predicted_press"] - x["observed_press"]).abs(),
        fa_abs=lambda x: (x["predicted_fa"] - x["observed_fa"]).abs(),
        hit_abs=lambda x: (x["predicted_hit"] - x["observed_hit"]).abs(),
    ).groupby(["horizon_min_prior_trials", "variant"], as_index=False).agg(
        mean_press_abs_error=("press_abs", "mean"), mean_fa_abs_error=("fa_abs", "mean"), mean_hit_abs_error=("hit_abs", "mean")
    )
    summary = summary.drop(columns=["mean_abs_press_calibration"]).merge(cal, on=["horizon_min_prior_trials", "variant"], how="left")
    summary = summary.sort_values(["horizon_min_prior_trials", "total_elpd"], ascending=[True, False])
    summary.to_csv(out / "hybrid_ddm_summary.csv", index=False)

    # Fold-paired comparisons. Five folds are not independent biological units;
    # these CIs are descriptive sensitivity summaries, not the primary inference.
    pair_rows = []
    for horizon, gh in fold_df.groupby("horizon_min_prior_trials"):
        w = gh.pivot_table(index="fold", columns="variant", values="elpd", aggfunc="sum")
        for model in w.columns:
            for ref in ["D0", "D2", "D6"]:
                if model == ref or ref not in w.columns:
                    continue
                vals = (w[model] - w[ref]).dropna().to_numpy()
                mean, lo, hi = bootstrap_mean_ci(vals, args.n_bootstrap, args.seed + abs(hash((horizon, model, ref))) % 100000)
                pair_rows.append({
                    "horizon_min_prior_trials": int(horizon), "model": model, "reference": ref,
                    "n_folds": len(vals), "mean_fold_delta_elpd": mean,
                    "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                    "positive_folds": int(np.sum(vals > 0)),
                })
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(out / "hybrid_ddm_pairwise_fold_deltas.csv", index=False)

    write_json(out / "analysis_settings.json", {
        "latents": str(Path(args.latents).resolve()), "n_subjects": int(d["subj_id_ddm"].nunique()),
        "n_trials": len(d), "filter": filter_info, "variants": args.variants,
        "cue_probabilities": cue_probability_map(args.cue_probabilities), "prior_strength": args.prior_strength,
        "k": args.k, "seed": args.seed, "n_starts": args.n_starts,
        "t0_min": args.t0_min, "t0_max": args.t0_max, "horizons": args.horizons,
    })

    lines = [
        "# Hybrid DDM and instructed-prior controls", "", "## Summary", "",
        summary.head(40).to_markdown(index=False), "", "## Pairwise fold deltas", "",
        pair.head(50).to_markdown(index=False), "", "## Decision rules", "",
        "- D9 > D6 supports incremental actor-control value beyond personalized online behavioral history.",
        "- D11 > D10 supports causal adaptation to experienced cue frequencies beyond the instructed prior.",
        "- D12/D13 > D2 tests whether explicit cue-belief representations improve the actor-control bridge.",
        "- Use horizon 0 as confirmatory. Treat later horizons as a warm-start sensitivity analysis.",
        "- Do not treat multiple horizons, seeds, or t0 settings as independent replications.",
    ]
    (out / "HYBRID_DDM_PRIOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
