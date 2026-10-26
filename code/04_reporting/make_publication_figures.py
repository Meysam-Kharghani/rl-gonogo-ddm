#!/usr/bin/env python3
"""Generate final manuscript and supplementary figures from public reporting source data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_TAG = "rl-behavioral-history-first-passage-reproducibility"
FIXED_DATE = datetime(2026, 10, 24, tzinfo=timezone.utc)

plt.rcParams["svg.hashsalt"] = PROJECT_TAG
plt.rcParams["font.family"] = "DejaVu Sans"


def save(fig: plt.Figure, out: Path, stem: str, dpi: int = 220) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        if ext == "png":
            meta = {"Software": PROJECT_TAG}
            fig.savefig(out / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight", metadata=meta)
        elif ext == "pdf":
            meta = {"Creator": PROJECT_TAG, "Producer": "matplotlib", "CreationDate": FIXED_DATE, "ModDate": FIXED_DATE}
            fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", metadata=meta)
        else:
            meta = {"Date": "2026-10-24", "Creator": PROJECT_TAG}
            fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", metadata=meta)
    plt.close(fig)


def forest(ax, df: pd.DataFrame, label_col: str, title: str, xlabel: str) -> None:
    y = np.arange(len(df))[::-1]
    x = df["mean_delta_elpd"].astype(float).to_numpy()
    lo = df["ci_low"].astype(float).to_numpy()
    hi = df["ci_high"].astype(float).to_numpy()
    ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3, linewidth=1.2)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col].astype(str).tolist())
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)


def figure_01(data_dir: Path, out: Path) -> None:
    # The flow diagram is intentionally generated from a compact conceptual source table.
    pd.read_csv(data_dir / "figure_01_analysis_logic_source_data.csv")
    fig, ax = plt.subplots(figsize=(8.1, 5.9))
    ax.set_axis_off()

    def box(x, y, w, h, text, fs=10.5):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fill=False, linewidth=1.2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
        return (x, y, w, h)

    def arrow(a, b, start="bottom", end="top"):
        x, y, w, h = a
        X, Y, W, H = b
        pts = {
            "top": (x + w / 2, y + h),
            "bottom": (x + w / 2, y),
            "left": (x, y + h / 2),
            "right": (x + w, y + h / 2),
        }
        pts2 = {
            "top": (X + W / 2, Y + H),
            "bottom": (X + W / 2, Y),
            "left": (X, Y + H / 2),
            "right": (X + W, Y + H / 2),
        }
        ax.add_patch(FancyArrowPatch(pts[start], pts2[end], arrowstyle="-|>", mutation_scale=12, linewidth=1.0))

    b1 = box(0.07, 0.82, 0.27, 0.10, "Cued Go/NoGo behavior\n40 participants, 15,200 trials")
    b2 = box(0.07, 0.61, 0.27, 0.13, "Stage-gated RL development\nactor–control model selected\nfor downstream evaluation")
    b3 = box(0.39, 0.61, 0.32, 0.13, "Subject-level cross-fitting\ntrain-fold RL fit → new-participant\nlatent trajectories")
    b4 = box(0.045, 0.37, 0.32, 0.14, "Action occurrence\nlogistic prediction\nprimary history → richer history")
    b5 = box(0.395, 0.37, 0.32, 0.14, "Fixed-horizon first passage\nresponse by 700 ms + timing\nprimary history → richer history")
    b6 = box(0.745, 0.37, 0.215, 0.14, "Likelihood\ndecomposition\nevent vs survival")
    b7 = box(0.29, 0.09, 0.45, 0.14, "Final inference\nRL increment is modest and baseline-dependent;\nfirst-passage gain is concentrated in response occurrence/survival", fs=10)

    arrow(b1, b2)
    arrow(b2, b3, start="right", end="left")
    arrow(b3, b4, start="bottom", end="top")
    arrow(b3, b5)
    arrow(b5, b6, start="right", end="left")
    arrow(b4, b7)
    arrow(b5, b7)
    arrow(b6, b7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, out, "figure_01_analysis_logic")


def figure_02(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "figure_02_action_history_comparison_source_data.csv")
    d["plot_label"] = d["history_comparator"] + ": " + d["rl_component"]
    fig, ax = plt.subplots(figsize=(8.7, 5.15))
    forest(ax, d, "plot_label", "Action occurrence: incremental RL information depends on history comparator", "Mean held-out ΔELPD per participant")
    save(fig, out, "figure_02_action_history_comparison")


def figure_03(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "figure_03_first_passage_history_comparison_source_data.csv")
    d["plot_label"] = d["history_comparator"] + ": " + d["rl_component"]
    fig, ax = plt.subplots(figsize=(8.45, 5.15))
    forest(ax, d, "plot_label", "Fixed-horizon first-passage likelihood: incremental RL information", "Mean held-out ΔELPD per participant")
    save(fig, out, "figure_03_first_passage_history_comparison")


def figure_04(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "figure_04_likelihood_decomposition_source_data.csv")
    fig, ax = plt.subplots(figsize=(8.45, 5.15))
    forest(ax, d, "likelihood_component", "Policy increment decomposed by likelihood contribution", "Mean held-out ΔELPD per participant")
    save(fig, out, "figure_04_likelihood_decomposition")


def supplementary_01(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "supplementary_figure_01_horizon_sensitivity_source_data.csv")
    x = d["analytic_horizon_ms"].to_numpy(float)
    y = d["mean_subject_delta_elpd"].to_numpy(float)
    lo = d["bootstrap_ci_low"].to_numpy(float)
    hi = d["bootstrap_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", linewidth=1.2, capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("Analytic horizon (ms)")
    ax.set_ylabel("Mean ΔELPD / participant")
    ax.set_title("Policy increment across analytic horizons")
    ax.grid(alpha=0.18)
    save(fig, out, "supplementary_figure_01_horizon_sensitivity")


def supplementary_02(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "supplementary_figure_02_repeated_partitions_source_data.csv")
    y = np.arange(len(d))[::-1]
    x = d["mean_subject_delta_elpd"].to_numpy(float)
    lo = d["bootstrap_ci_low"].to_numpy(float)
    hi = d["bootstrap_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.0, 5.15))
    ax.errorbar(x, y, xerr=np.vstack([x - lo, hi - x]), fmt="o", capsize=3, linewidth=1.2)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["seed"].astype(str))
    ax.set_xlabel("Policy beyond primary history: mean ΔELPD / participant")
    ax.set_ylabel("Outer-partition seed")
    ax.set_title("Repeated outer-partition stability")
    ax.grid(axis="x", alpha=0.18)
    save(fig, out, "supplementary_figure_02_repeated_partitions")


def supplementary_03(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "supplementary_figure_03_mapping_sensitivity_source_data.csv")
    fig, ax = plt.subplots(figsize=(8.45, 5.15))
    forest(ax, d, "mapping_label", "Alternative predictor-to-first-passage mappings", "Alternative mapping − primary mapping, mean ΔELPD / participant")
    save(fig, out, "supplementary_figure_03_mapping_sensitivity")


def supplementary_04(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "supplementary_figure_04_parameter_recovery_source_data.csv")
    order = ["Actor learning rate", "Control learning rate", "Actor sensitivity", "Response bias"]
    means = d.groupby("parameter_label")["pearson_r"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7.25, 4.5))
    ax.bar(np.arange(len(order)), means.to_numpy(float))
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=22, ha="center")
    ax.set_ylabel("Mean simulated–recovered correlation")
    ax.set_title("Participant-level parameter recovery across 10 simulations")
    ax.grid(axis="y", alpha=0.18)
    save(fig, out, "supplementary_figure_04_parameter_recovery")


def supplementary_05(data_dir: Path, out: Path) -> None:
    d = pd.read_csv(data_dir / "supplementary_figure_05_prior_strength_source_data.csv")
    x = d["prior_strength"].to_numpy(float)
    y = d["mean_subject_delta_elpd"].to_numpy(float)
    lo = d["bootstrap_ci_low"].to_numpy(float)
    hi = d["bootstrap_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.35, 4.5))
    ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]), marker="o", linewidth=1.2, capsize=3)
    ax.axhline(0, linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("Pseudo-count prior strength (s)")
    ax.set_ylabel("Empirical belief − instructed prior, mean ΔELPD / participant")
    ax.set_title("Cue-belief prior-strength sensitivity")
    ax.grid(alpha=0.18)
    save(fig, out, "supplementary_figure_05_prior_strength")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-data", type=Path, default=Path("results/reporting/figure_data"))
    p.add_argument("--out", type=Path, default=Path("results/reporting/figures"))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for old in a.out.iterdir():
        if old.is_file():
            old.unlink()

    figure_01(a.source_data, a.out)
    figure_02(a.source_data, a.out)
    figure_03(a.source_data, a.out)
    figure_04(a.source_data, a.out)
    supplementary_01(a.source_data, a.out)
    supplementary_02(a.source_data, a.out)
    supplementary_03(a.source_data, a.out)
    supplementary_04(a.source_data, a.out)
    supplementary_05(a.source_data, a.out)

    index = """# Figure index\n\n## Main manuscript\n- Figure 1: analysis logic and final evidential hierarchy.\n- Figure 2: action-occurrence RL increments under compact and richer behavioral-history comparators.\n- Figure 3: fixed-horizon first-passage RL increments under compact and richer behavioral-history comparators.\n- Figure 4: policy-probability increment decomposed into overall, censored/survival, observed-response, and target-specific contributions.\n\n## Supplementary material\n- Figure S1: analytic-horizon sensitivity.\n- Figure S2: repeated outer-partition stability.\n- Figure S3: alternative predictor-to-first-passage mappings.\n- Figure S4: participant-level parameter recovery.\n- Figure S5: cue-belief prior-strength sensitivity.\n\n## Interpretation constraints\n- Repeated outer partitions reuse the same 40 participants and are stability analyses, not independent replications.\n- Positive predictive contrasts do not identify a unique causal or cognitive mechanism.\n- The 700-ms value is an analytic observation horizon, not a verified task-enforced response deadline.\n- The historical first-passage policy increment is concentrated mainly in censored/survival contributions rather than observed-response density.\n"""
    (a.out / "FIGURE_INDEX.md").write_text(index, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
