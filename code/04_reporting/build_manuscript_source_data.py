#!/usr/bin/env python3
"""Build final manuscript figure source data and the main results table from frozen aggregate results.

This script uses only public aggregate results. It does not require participant-level data.
Technical model identifiers are used only to select historically frozen rows; all generated
reporting tables use conceptual publication-facing labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def row(df: pd.DataFrame, **conds) -> pd.Series:
    q = df.copy()
    for k, v in conds.items():
        q = q[q[k] == v]
    if len(q) != 1:
        raise ValueError(f"Expected one row for {conds}; found {len(q)}")
    return q.iloc[0]


def build(results_root: Path, out_root: Path) -> None:
    fig = out_root / "figure_data"
    tab = out_root / "table_data"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            (1, "Cued Go/NoGo behavior", "INPUT", "40 participants; 15,200 trials"),
            (2, "Stage-gated RL development", "DEVELOPMENT", "Actor–control representation selected for downstream evaluation"),
            (3, "Participant-level cross-fitting", "PRIMARY_PIPELINE", "Train-fold RL estimation and new-participant latent generation"),
            (4, "Action occurrence", "PRIMARY_AND_ROBUSTNESS", "Compact history followed by richer history"),
            (5, "Fixed-horizon first passage", "PRIMARY_AND_ROBUSTNESS", "Response by 700 ms plus observed-response timing"),
            (6, "Likelihood decomposition", "ROBUSTNESS", "Observed-response density versus censored/survival contribution"),
            (7, "Final inference", "INTERPRETATION", "RL increment is modest and baseline-dependent; first-passage gain is concentrated in response occurrence/survival"),
        ],
        columns=["order", "stage", "status", "role"],
    ).to_csv(fig / "figure_01_analysis_logic_source_data.csv", index=False)

    action = pd.read_csv(results_root / "action/key_contrasts.csv")
    action_rich = pd.read_csv(results_root / "robustness/richer_history_action/richer_history_planned_contrasts.csv")
    action_rows = []
    for comparator, component, conds in [
        ("Primary history", "actor/control", dict(model="actor_control_plus_compact_lagged_history", reference="compact_lagged_history")),
        ("Primary history", "policy", dict(model="policy_plus_compact_lagged_history", reference="compact_lagged_history")),
        ("Primary history", "full RL block", dict(model="full_rl_plus_compact_lagged_history", reference="compact_lagged_history")),
    ]:
        r = row(action, **conds)
        action_rows.append((comparator, component, r.mean_subject_delta_elpd, r.bootstrap_ci_low, r.bootstrap_ci_high, int(r.positive_subjects), 40))
    for comparator, component, conds in [
        ("Richer history", "actor/control", dict(model="richer_lagged_history_plus_actor_control", reference="richer_lagged_history")),
        ("Richer history", "policy", dict(model="richer_lagged_history_plus_policy", reference="richer_lagged_history")),
        ("Richer history", "full RL block", dict(model="richer_lagged_history_plus_full_rl", reference="richer_lagged_history")),
    ]:
        r = row(action_rich, **conds)
        action_rows.append((comparator, component, r.mean_subject_delta_elpd, r.ci_low, r.ci_high, int(r.positive_subjects), 40))
    action_df = pd.DataFrame(action_rows, columns=["history_comparator", "rl_component", "mean_delta_elpd", "ci_low", "ci_high", "positive_participants", "n_participants"])
    action_df.to_csv(fig / "figure_02_action_history_comparison_source_data.csv", index=False)

    joint = pd.read_csv(results_root / "ddm/core_key_contrasts.csv")
    joint_rich = pd.read_csv(results_root / "robustness/richer_history_joint_response/rich_ddm_planned_contrasts.csv")
    joint_rows = []
    for comparator, component, conds in [
        ("Primary history", "actor/control", dict(model="D14", reference="D6")),
        ("Primary history", "policy", dict(model="D15", reference="D6")),
        ("Primary history", "full RL block", dict(model="D9", reference="D6")),
    ]:
        r = row(joint, **conds)
        joint_rows.append((comparator, component, r.mean_subject_delta_elpd, r.bootstrap_ci_low, r.bootstrap_ci_high, int(r.positive_subjects), 40))
    for comparator, component, conds in [
        ("Richer history", "actor/control", dict(model="HR14", reference="HR6")),
        ("Richer history", "policy", dict(model="HR15", reference="HR6")),
        ("Richer history", "full RL block", dict(model="HR9", reference="HR6")),
    ]:
        r = row(joint_rich, **conds)
        joint_rows.append((comparator, component, r.mean_subject_delta_elpd, r.ci_low, r.ci_high, int(r.positive_subjects), 40))
    joint_df = pd.DataFrame(joint_rows, columns=["history_comparator", "rl_component", "mean_delta_elpd", "ci_low", "ci_high", "positive_participants", "n_participants"])
    joint_df.to_csv(fig / "figure_03_first_passage_history_comparison_source_data.csv", index=False)

    strat = pd.read_csv(results_root / "robustness/likelihood_component_decomposition/stratified_ddm_contrasts.csv")
    decomp_rows = []
    for family, stratum, label in [
        ("overall", "ALL", "Overall"),
        ("likelihood_type", "censored", "Survival / censored contribution"),
        ("likelihood_type", "event", "Observed-response density"),
        ("target_x_likelihood", "Go | censored", "Go: survival / censored"),
        ("target_x_likelihood", "Go | event", "Go: observed-response density"),
        ("target_x_likelihood", "NoGo | censored", "NoGo: survival / censored"),
        ("target_x_likelihood", "NoGo | event", "NoGo: observed-response density"),
    ]:
        r = row(strat, stratum_family=family, stratum=stratum, model="D15", reference="D6")
        decomp_rows.append((label, r.mean_subject_delta_elpd, r.ci_low, r.ci_high, int(r.positive_subjects), 40))
    pd.DataFrame(decomp_rows, columns=["likelihood_component", "mean_delta_elpd", "ci_low", "ci_high", "positive_participants", "n_participants"]).to_csv(fig / "figure_04_likelihood_decomposition_source_data.csv", index=False)

    horizon = pd.read_csv(results_root / "robustness/analytic_horizon_sensitivity/horizon_sensitivity_planned_contrasts.csv")
    horizon = horizon[(horizon.model == "D15") & (horizon.reference == "D6")].sort_values("analytic_horizon_ms")
    horizon[["analytic_horizon_ms", "mean_subject_delta_elpd", "bootstrap_ci_low", "bootstrap_ci_high", "positive_subjects", "n_subjects"]].to_csv(fig / "supplementary_figure_01_horizon_sensitivity_source_data.csv", index=False)

    repeated = pd.read_csv(results_root / "ddm/repeated_cv_seeds.csv").sort_values("mean_subject_delta_elpd", ascending=False)
    repeated[["seed", "mean_subject_delta_elpd", "bootstrap_ci_low", "bootstrap_ci_high", "positive_subjects"]].to_csv(fig / "supplementary_figure_02_repeated_partitions_source_data.csv", index=False)

    mapping = pd.read_csv(results_root / "robustness/predictor_mapping_sensitivity/mapping_planned_contrasts.csv")
    labels = {
        "D14S": "Actor/control: swapped mapping vs primary",
        "D15S": "Policy: distance mapping vs primary drift mapping",
        "D9S": "Full RL: swapped mapping vs primary",
        "D9X": "Full RL: expanded mapping vs primary",
    }
    mapping["mapping_label"] = mapping["alternative"].map(labels)
    mapping_out = mapping[["mapping_label", "mean_subject_delta_elpd", "ci_low", "ci_high", "positive_subjects", "n_subjects"]].rename(columns={"mean_subject_delta_elpd":"mean_delta_elpd"})
    mapping_out.to_csv(fig / "supplementary_figure_03_mapping_sensitivity_source_data.csv", index=False)

    recovery = pd.read_csv(results_root / "robustness/actor_control_parameter_recovery/actor_control_parameter_recovery_correlations_by_rep.csv")
    recovery = recovery[recovery.parameter.isin(["alpha_actor", "alpha_ctrl", "beta_actor", "bias"])].copy()
    recovery["parameter_label"] = recovery.parameter.map({
        "alpha_actor": "Actor learning rate",
        "alpha_ctrl": "Control learning rate",
        "beta_actor": "Actor sensitivity",
        "bias": "Response bias",
    })
    recovery[["rep", "parameter_label", "pearson_r", "n"]].to_csv(fig / "supplementary_figure_04_parameter_recovery_source_data.csv", index=False)

    prior = pd.read_csv(results_root / "ddm/prior_strength_sensitivity.csv")
    prior = prior[(prior.model == "D11") & (prior.reference == "D10")].sort_values("prior_strength")
    prior[["prior_strength", "mean_subject_delta_elpd", "bootstrap_ci_low", "bootstrap_ci_high", "positive_subjects", "n_subjects"]].to_csv(fig / "supplementary_figure_05_prior_strength_source_data.csv", index=False)

    # Main manuscript Table 1, with numeric fields kept numeric for reuse.
    combined = []
    for task, d in [("Action", action_df), ("First passage", joint_df)]:
        for _, r in d.iterrows():
            combined.append({
                "outcome": task,
                "history_comparator": r.history_comparator,
                "rl_component": r.rl_component,
                "mean_delta_elpd": float(r.mean_delta_elpd),
                "ci_low": float(r.ci_low),
                "ci_high": float(r.ci_high),
                "positive_participants": int(r.positive_participants),
                "n_participants": int(r.n_participants),
            })
    pd.DataFrame(combined).to_csv(tab / "manuscript_table_01_incremental_information_numeric.csv", index=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args()
    build(a.results_root, a.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
