#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: CODE_ONLY_DEVELOPMENT_PRECURSOR; SUPERSEDED_BY_COMPLETED_ROBUSTNESS_ANALYSIS
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

"""Simulation-based parameter recovery for the M6 actor-control RL model.

The existing project contains DDM and joint reinforcement-learning/first-passage model-recovery checks, but not a
full recovery of the upstream hierarchical Bayesian M6 model. This script:
1. loads the real task design and a fitted M6 posterior;
2. simulates new actions sequentially from source posterior-mean parameters;
3. refits M6 with the project's original PyMC implementation;
4. reports subject-parameter correlations and 95% interval coverage.

This is computationally expensive. Start with 2 reps and modest draws, then run at
least 10 reps for the supplement if diagnostics are acceptable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common_utils import ensure_dir, import_module, write_json  # noqa: E402


def simulate_m6(rl, arr: dict, params: dict, seed: int) -> tuple[dict, pd.DataFrame]:
    meta = rl.get_meta(arr)
    S = int(meta["n_subj"])
    T = int(meta["n_trials_per_subj"])
    C = len(meta["cue_levels"])
    cue = arr["cue_idx"].astype(int)
    nogo = arr["is_nogo"].astype(float)

    def par(name, default):
        return np.asarray(params.get(name, default), dtype=float)

    bias = par("bias", np.zeros(S)).reshape(S)
    b_cue = par("b_cue", np.zeros(C)).reshape(C)
    b_nogo = float(par("b_nogo", 0.0))
    b_prev_error = float(par("b_prev_error", 0.0))
    lapse = float(par("lapse_eps", 0.0))
    alpha_ctrl = par("alpha_ctrl", np.zeros(S)).reshape(S)
    w_control = float(par("w_control", 0.0))
    alpha_actor = par("alpha_actor", np.zeros(S)).reshape(S)
    beta_actor = par("beta_actor", np.zeros(S)).reshape(S)

    rng = np.random.default_rng(seed)
    control = np.zeros(S)
    Q = np.zeros((S, C, 2))
    prev_error = np.zeros(S)

    action = np.zeros((S, T), dtype=np.int8)
    correct = np.zeros((S, T), dtype=np.int8)
    error = np.zeros((S, T), dtype=np.int8)
    p_all = np.zeros((S, T), dtype=float)
    control_all = np.zeros((S, T), dtype=float)
    qdiff_all = np.zeros((S, T), dtype=float)

    for t in range(T):
        ci = cue[:, t]
        qdiff = Q[np.arange(S), ci, 1] - Q[np.arange(S), ci, 0]
        control_all[:, t] = control
        qdiff_all[:, t] = qdiff
        logit = bias + b_cue[ci] + b_nogo * nogo[:, t] + b_prev_error * prev_error
        logit += w_control * control + beta_actor * qdiff
        p = rl.sigmoid_np(logit)
        p = np.clip((1 - lapse) * p + lapse * 0.5, 1e-8, 1 - 1e-8)
        a = rng.binomial(1, p).astype(np.int8)
        c = (((nogo[:, t] == 0) & (a == 1)) | ((nogo[:, t] == 1) & (a == 0))).astype(np.int8)
        e = (1 - c).astype(np.int8)

        action[:, t] = a
        correct[:, t] = c
        error[:, t] = e
        p_all[:, t] = p

        control = np.clip(control + alpha_ctrl * (e - control), 0, 1)
        chosen_q = np.where(a == 1, Q[np.arange(S), ci, 1], Q[np.arange(S), ci, 0])
        delta = c - chosen_q
        press = a == 1
        Q[np.arange(S)[press], ci[press], 1] += alpha_actor[press] * delta[press]
        Q[np.arange(S)[~press], ci[~press], 0] += alpha_actor[~press] * delta[~press]
        Q = np.clip(Q, -5, 5)
        prev_error = e.astype(float)

    sim = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in arr.items()}
    sim["action"] = action
    sim["correct"] = correct
    sim["error"] = error
    sim["prev_error"] = np.zeros_like(error)
    sim["prev_error"][:, 1:] = error[:, :-1]
    sim["hit"] = ((nogo == 0) & (action == 1)).astype(np.int8)
    sim["fa"] = ((nogo == 1) & (action == 1)).astype(np.int8)
    sim["miss"] = ((nogo == 0) & (action == 0)).astype(np.int8)
    sim["cr"] = ((nogo == 1) & (action == 0)).astype(np.int8)
    # RT/EEG arrays are irrelevant to M6 and remain unchanged.

    rows = []
    for s, sid in enumerate(meta["subj_ids"]):
        rows.append({
            "subject_id": sid, "sim_press_rate": float(action[s].mean()),
            "sim_fa_rate": float(action[s][nogo[s] == 1].mean()),
            "sim_hit_rate": float(action[s][nogo[s] == 0].mean()),
            "mean_generating_p": float(p_all[s].mean()),
        })
    return sim, pd.DataFrame(rows)


def hdi_bounds(az, da, prob: float = 0.95):
    h = az.hdi(da, hdi_prob=prob)
    # ArviZ returns an hdi dimension with lower/higher coordinates.
    dim = [d for d in h.dims if d.lower() == "hdi"][-1]
    return np.asarray(h.sel({dim: "lower"})), np.asarray(h.sel({dim: "higher"}))


def compare_parameter(az, source: np.ndarray, recovered_da, name: str, rep: int) -> list[dict]:
    mean = np.asarray(recovered_da.mean(dim=("chain", "draw")))
    lo, hi = hdi_bounds(az, recovered_da, 0.95)
    src = np.asarray(source, dtype=float)
    src, mean, lo, hi = np.broadcast_arrays(src, mean, lo, hi)
    rows = []
    for idx in np.ndindex(src.shape if src.shape else (1,)):
        j = () if src.shape == () else idx
        rows.append({
            "rep": rep, "parameter": name, "element": "scalar" if src.shape == () else ":".join(map(str, idx)),
            "source_value": float(src[j]), "recovered_mean": float(mean[j]),
            "hdi95_low": float(lo[j]), "hdi95_high": float(hi[j]),
            "covered": bool(lo[j] <= src[j] <= hi[j]),
            "absolute_error": float(abs(mean[j] - src[j])),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl-script", required=True, help="Path to scripts/01_fit_rl_actor_control.py")
    ap.add_argument("--project-out", required=True, help="RL project output root containing artifacts/arrays__ACTOR_CONTROL_MODEL_FAMILY.npz")
    ap.add_argument("--source-idata", required=True, help="Fitted M6 idata.nc used as the generator")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = ensure_dir(args.out)
    rl = import_module(args.rl_script, "rl_m6_recovery")
    _, az = rl.require_bayes()
    arr = rl.load_artifacts(Path(args.project_out))
    source_idata = az.from_netcdf(str(Path(args.source_idata).expanduser().resolve()))
    source_params = rl.posterior_mean_params(source_idata)

    compare_names = [
        "mu_bias", "delta_bias_group", "sigma_bias", "bias", "b_cue", "b_nogo", "b_prev_error", "lapse_eps",
        "alpha_ctrl", "w_control", "alpha_actor", "beta_actor",
        "mu_alpha_ctrl", "delta_alpha_ctrl_group", "sigma_alpha_ctrl",
        "mu_alpha_actor", "delta_alpha_actor_group", "sigma_alpha_actor",
        "mu_logbeta_actor", "delta_logbeta_actor_group", "sigma_logbeta_actor",
    ]
    compare_names = [n for n in compare_names if n in source_idata.posterior]

    param_rows: list[dict] = []
    diag_rows: list[dict] = []
    sim_rows: list[pd.DataFrame] = []

    cfg = rl.FitConfig(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores,
                       seed=args.seed, target_accept=args.target_accept)

    for rep in range(args.reps):
        rep_dir = ensure_dir(out / f"rep_{rep:02d}")
        sim_arr, sim_summary = simulate_m6(rl, arr, source_params, args.seed + 10000 + rep)
        sim_summary["rep"] = rep
        sim_rows.append(sim_summary)
        rl.save_npz(rep_dir / "simulated_arrays_M6.npz", sim_arr)

        # fit_nuts writes to <seed_out>/results/M6_posterior/idata.nc
        cfg_rep = rl.FitConfig(**{**cfg.__dict__, "seed": args.seed + rep})
        idata_path = rl.fit_nuts(rep_dir, "M6", sim_arr, cfg_rep, force_refit=args.force)
        rec = az.from_netcdf(str(idata_path))
        diag = rl.diagnostics_summary(rec)
        diag_rows.append({"rep": rep, **diag})

        for name in compare_names:
            source = np.asarray(source_params[name])
            param_rows.extend(compare_parameter(az, source, rec.posterior[name], name, rep))

    params_df = pd.DataFrame(param_rows)
    diag_df = pd.DataFrame(diag_rows)
    sims_df = pd.concat(sim_rows, ignore_index=True)
    params_df.to_csv(out / "m6_parameter_recovery_elements.csv", index=False)
    diag_df.to_csv(out / "m6_parameter_recovery_diagnostics.csv", index=False)
    sims_df.to_csv(out / "m6_simulation_behavior_summary.csv", index=False)

    summary = params_df.groupby("parameter", as_index=False).agg(
        n_elements=("covered", "size"), coverage_95=("covered", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        source_sd=("source_value", "std"), recovered_sd=("recovered_mean", "std"),
    )
    corr_rows = []
    for (rep, par), g in params_df.groupby(["rep", "parameter"]):
        if len(g) >= 5 and g["source_value"].std() > 1e-10 and g["recovered_mean"].std() > 1e-10:
            corr_rows.append({"rep": rep, "parameter": par, "pearson_r": g["source_value"].corr(g["recovered_mean"]), "n": len(g)})
    corr = pd.DataFrame(corr_rows)
    corr_summary = corr.groupby("parameter", as_index=False).agg(mean_recovery_r=("pearson_r", "mean"), min_recovery_r=("pearson_r", "min"), n_reps=("rep", "nunique")) if not corr.empty else pd.DataFrame()
    if not corr_summary.empty:
        summary = summary.merge(corr_summary, on="parameter", how="left")
    summary.to_csv(out / "m6_parameter_recovery_summary.csv", index=False)
    corr.to_csv(out / "m6_parameter_recovery_correlations_by_rep.csv", index=False)

    write_json(out / "settings.json", vars(args))
    lines = [
        "# M6 hierarchical RL parameter recovery", "", "## Parameter summary", "",
        summary.to_markdown(index=False), "", "## Sampler diagnostics", "", diag_df.to_markdown(index=False), "",
        "## Interpretation guardrails", "",
        "- Recovery is strongest when subject-level correlations are high and 95% interval coverage is near 0.95.",
        "- Poor recovery of a parameter means it should not carry a strong mechanistic claim, even if predictive performance is good.",
        "- This simulation conditions on the observed cue/target sequence and source posterior-mean parameters.",
    ]
    (out / "M6_PARAMETER_RECOVERY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
