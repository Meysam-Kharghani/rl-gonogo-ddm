#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

"""
DDM entropy diagnostics and posterior-predictive-style figures.

This script does not fit new DDM models. It visualizes and audits the outputs
from entropy_diffusion_exploration.py.

Main questions
--------------
1. Did entropy models improve held-out NLL over D2_equivalent?
2. Are entropy/confidence-related DDM components interpretable?
3. Are predicted response probabilities calibrated across probability bins?
4. Are entropy models improving calibration or mostly overfitting full-data NLL?

Inputs
------
results/model_development/ddm_entropy_exploratory/
  ddm_entropy_cv_scores.csv
  ddm_entropy_cv_pairwise.csv
  ddm_entropy_final_coefficients.csv
  ddm_entropy_final_model_summary.csv
  ddm_entropy_final_trial_predictions.csv
  ddm_entropy_component_quantile_summary.csv

Outputs
-------
results/model_development/ddm_entropy_exploratory_diagnostics/
  Figure_DDM_Entropy_CV_ModelComparison.png
  Figure_DDM_Entropy_Components_ByPolicyEntropy.png
  Figure_DDM_Entropy_Calibration.png
  Figure_DDM_Entropy_Coefficients.png
  ddm_entropy_calibration_bins.csv
  DDM_ENTROPY_DIAGNOSTICS_REPORT.md

Example
-------
python code/00_model_development/entropy_diffusion_diagnostics.py \
  --ddm_dir results/model_development/ddm_entropy_exploratory \
  --out results/model_development/ddm_entropy_exploratory_diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ORDER = [
    "D2_equivalent",
    "E1_policy_entropy_drift",
    "E2_policy_entropy_bound",
    "E3_policy_entropy_both",
    "E4_policy_confidence_both",
    "E5_uncertainty_confidence_both",
]


def safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pretty_model_name(x: str) -> str:
    mapping = {
        "D2_equivalent": "D2-equivalent",
        "E1_policy_entropy_drift": "Entropy→drift",
        "E2_policy_entropy_bound": "Entropy→bound",
        "E3_policy_entropy_both": "Entropy→both",
        "E4_policy_confidence_both": "Confidence→both",
        "E5_uncertainty_confidence_both": "Entropy+confidence→both",
    }
    return mapping.get(x, x)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    if "model_name" not in df.columns:
        return df
    temp = df.copy()
    temp["model_order"] = temp["model_name"].map({m: i for i, m in enumerate(MODEL_ORDER)}).fillna(999)
    return temp.sort_values(["model_order", "model_name"]).drop(columns=["model_order"])


def make_cv_model_figure(scores: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    if scores.empty:
        ax.text(0.5, 0.5, "Missing CV scores", ha="center", va="center")
        ax.set_axis_off()
    else:
        agg = scores.groupby("model_name").agg(
            mean_test=("test_nll_per_trial", "mean"),
            sd_test=("test_nll_per_trial", "std"),
            n=("test_nll_per_trial", "count"),
        ).reset_index()
        agg = ordered(agg)
        x = np.arange(len(agg))
        ax.bar(x, agg["mean_test"].values)
        ax.errorbar(x, agg["mean_test"].values, yerr=agg["sd_test"].fillna(0).values, fmt="none", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_model_name(m) for m in agg["model_name"]], rotation=25, ha="right")
        ax.set_ylabel("Held-out NLL per trial")
        ax.set_title("Cross-validated DDM entropy model comparison")
        ax.spines[["top", "right"]].set_visible(False)

        for i, row in agg.iterrows():
            ax.text(i, row["mean_test"] + 0.002, f"{row['mean_test']:.4f}", ha="center", va="bottom", fontsize=8)

        if not pairwise.empty:
            txt = "Pairwise vs D2-equivalent:\n"
            for _, r in pairwise.iterrows():
                delta = r.get("mean_delta_test_nll_per_trial", np.nan)
                q = r.get("q_value_fdr_bh", np.nan)
                txt += f"{pretty_model_name(r['model_name'])}: Δ={delta:.4f}, q={q:.3g}\n"
            ax.text(1.02, 0.98, txt.strip(), transform=ax.transAxes, va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "Figure_DDM_Entropy_CV_ModelComparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_component_figure(qsum: pd.DataFrame, out_dir: Path) -> None:
    # Use policy_entropy quantiles for D2_equivalent, E4, E5 if available.
    models = ["D2_equivalent", "E4_policy_confidence_both", "E5_uncertainty_confidence_both"]
    metrics = [
        ("mean_drift_v", "Drift v"),
        ("mean_bound_a", "Bound a"),
        ("mean_pred_response_prob", "Predicted response probability"),
        ("observed_response_rate", "Observed response rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    if qsum.empty:
        axes.flat[0].text(0.5, 0.5, "Missing quantile summary", ha="center", va="center")
        for ax in axes.flat:
            ax.set_axis_off()
    else:
        for ax, (metric, title) in zip(axes.flat, metrics):
            for model in models:
                sub = qsum[
                    (qsum["model_name"] == model)
                    & (qsum["quantile_variable"] == "policy_entropy")
                ].copy()
                if sub.empty or metric not in sub.columns:
                    continue
                sub["qnum"] = sub["quantile"].map({"Q1_low": 1, "Q2": 2, "Q3": 3, "Q4_high": 4})
                sub = sub.sort_values("qnum")
                ax.plot(sub["qnum"].values, sub[metric].values, marker="o", label=pretty_model_name(model))
            ax.set_title(title)
            ax.set_xlabel("Policy entropy quartile")
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xticklabels(["Q1 low", "Q2", "Q3", "Q4 high"])
            ax.spines[["top", "right"]].set_visible(False)
        axes.flat[0].legend(fontsize=8)

    fig.suptitle("DDM components across policy-entropy quartiles", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_DDM_Entropy_Components_ByPolicyEntropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def calibration_bins(pred: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame(rows)

    for model, g in pred.groupby("model_name"):
        p = pd.to_numeric(g["pred_response_prob_by_deadline"], errors="coerce")
        y = pd.to_numeric(g["is_response"], errors="coerce")
        mask = np.isfinite(p) & np.isfinite(y)
        gg = g.loc[mask].copy()
        if len(gg) < n_bins:
            continue

        # Use quantile bins to avoid sparse bins.
        try:
            gg["prob_bin"] = pd.qcut(
                pd.to_numeric(gg["pred_response_prob_by_deadline"], errors="coerce").rank(method="first"),
                q=n_bins,
                labels=False,
                duplicates="drop",
            )
        except Exception:
            continue

        for b, h in gg.groupby("prob_bin"):
            rows.append({
                "model_name": model,
                "bin": int(b),
                "n_trials": int(len(h)),
                "mean_predicted_prob": float(pd.to_numeric(h["pred_response_prob_by_deadline"], errors="coerce").mean()),
                "observed_response_rate": float(pd.to_numeric(h["is_response"], errors="coerce").mean()),
                "mean_policy_entropy": float(pd.to_numeric(h.get("policy_entropy", pd.Series(dtype=float)), errors="coerce").mean()),
                "mean_trial_nll": float(pd.to_numeric(h["trial_nll"], errors="coerce").mean()),
            })

    return pd.DataFrame(rows)


def make_calibration_figure(calib: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    if calib.empty:
        ax.text(0.5, 0.5, "Missing calibration bins", ha="center", va="center")
        ax.set_axis_off()
    else:
        for model in MODEL_ORDER:
            sub = calib[calib["model_name"] == model].sort_values("mean_predicted_prob")
            if sub.empty:
                continue
            ax.plot(sub["mean_predicted_prob"], sub["observed_response_rate"], marker="o", label=pretty_model_name(model))
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("Mean predicted response probability")
        ax.set_ylabel("Observed response rate")
        ax.set_title("DDM response-probability calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "Figure_DDM_Entropy_Calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_coeff_figure(coefs: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    if coefs.empty:
        ax.text(0.5, 0.5, "Missing coefficients", ha="center", va="center")
        ax.set_axis_off()
    else:
        ent = coefs[coefs["feature"].isin(["policy_entropy", "policy_confidence"])].copy()
        if ent.empty:
            ax.text(0.5, 0.5, "No entropy coefficients retained", ha="center", va="center")
            ax.set_axis_off()
        else:
            ent["label"] = ent["model_name"].map(pretty_model_name) + "\n" + ent["component"] + "\n" + ent["feature"]
            ent = ent.sort_values(["model_name", "component", "feature"])
            x = np.arange(len(ent))
            ax.bar(x, ent["coefficient"].values)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(ent["label"], rotation=65, ha="right", fontsize=8)
            ax.set_ylabel("Coefficient")
            ax.set_title("Entropy coefficients in full-data DDM fits")
            ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "Figure_DDM_Entropy_Coefficients.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
    final_summary: pd.DataFrame,
    coefs: pd.DataFrame,
    calib: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# DDM entropy diagnostics report")
    lines.append("")
    lines.append("This report visualizes the entropy-augmented DDM analysis and supports manuscript decisions.")
    lines.append("")

    lines.append("## Cross-validated decision")
    lines.append("")
    if pairwise.empty:
        lines.append("- Pairwise CV comparison was unavailable.")
    else:
        best = pairwise.sort_values("mean_delta_test_nll_per_trial").iloc[0]
        lines.append(
            f"- Best mean held-out delta versus D2-equivalent: `{best['comparison']}`, "
            f"Δ NLL/trial = {best['mean_delta_test_nll_per_trial']:.6f}, "
            f"q = {best['q_value_fdr_bh']:.3g}, improved folds = "
            f"{int(best['folds_improved_negative_delta'])}/{int(best['n_folds'])}."
        )
        if float(best["q_value_fdr_bh"]) < 0.05 and float(best["mean_delta_test_nll_per_trial"]) < 0:
            lines.append("- Interpretation: entropy provides a CV-supported DDM improvement.")
        else:
            lines.append("- Interpretation: entropy did not provide a robust held-out DDM improvement over D2-equivalent.")

    lines.append("")
    lines.append("## Full-data fit")
    lines.append("")
    if not final_summary.empty:
        fs = final_summary.sort_values("nll_per_trial").iloc[0]
        lines.append(
            f"- Best full-data NLL/trial: `{fs['model_name']}` = {fs['nll_per_trial']:.6f}."
        )
        lines.append("- Full-data improvements should not be treated as confirmatory unless they agree with held-out CV.")

    lines.append("")
    lines.append("## Entropy coefficients")
    lines.append("")
    ent = coefs[coefs["feature"].isin(["policy_entropy", "policy_confidence"])].copy() if not coefs.empty else pd.DataFrame()
    if ent.empty:
        lines.append("- No entropy coefficients were retained.")
    else:
        lines.append("| Model | Component | Feature | Coefficient |")
        lines.append("|---|---|---|---:|")
        for _, r in ent.iterrows():
            lines.append(
                f"| {r['model_name']} | {r['component']} | {r['feature']} | {r['coefficient']:.4f} |"
            )

    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    if calib.empty:
        lines.append("- Calibration bins were not generated.")
    else:
        # A simple absolute calibration error by model.
        cal = calib.copy()
        cal["abs_error"] = (cal["observed_response_rate"] - cal["mean_predicted_prob"]).abs()
        err = cal.groupby("model_name")["abs_error"].mean().reset_index().sort_values("abs_error")
        lines.append("| Model | Mean absolute calibration error |")
        lines.append("|---|---:|")
        for _, r in err.iterrows():
            lines.append(f"| {r['model_name']} | {r['abs_error']:.4f} |")

    lines.append("")
    lines.append("## Recommended wording")
    lines.append("")
    lines.append(
        "Entropy-augmented DDM variants showed interpretable component-level patterns, "
        "but they did not reliably improve held-out likelihood over the D2-equivalent actor-control baseline. "
        "Accordingly, entropy effects are best framed as neural/behavioral characterization of uncertainty-related "
        "error monitoring rather than as an additional core DDM component."
    )
    lines.append("")

    (out_dir / "DDM_ENTROPY_DIAGNOSTICS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddm_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ddm_dir = Path(args.ddm_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = safe_read(ddm_dir / "ddm_entropy_cv_scores.csv")
    pairwise = safe_read(ddm_dir / "ddm_entropy_cv_pairwise.csv")
    coefs = safe_read(ddm_dir / "ddm_entropy_final_coefficients.csv")
    final_summary = safe_read(ddm_dir / "ddm_entropy_final_model_summary.csv")
    pred = safe_read(ddm_dir / "ddm_entropy_final_trial_predictions.csv")
    qsum = safe_read(ddm_dir / "ddm_entropy_component_quantile_summary.csv")

    calib = calibration_bins(pred, n_bins=10)
    calib.to_csv(out_dir / "ddm_entropy_calibration_bins.csv", index=False)

    make_cv_model_figure(scores, pairwise, out_dir)
    make_component_figure(qsum, out_dir)
    make_calibration_figure(calib, out_dir)
    make_coeff_figure(coefs, out_dir)

    write_report(out_dir, scores, pairwise, final_summary, coefs, calib)

    print(f"Wrote DDM entropy diagnostics to: {out_dir}")
    print((out_dir / "DDM_ENTROPY_DIAGNOSTICS_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
