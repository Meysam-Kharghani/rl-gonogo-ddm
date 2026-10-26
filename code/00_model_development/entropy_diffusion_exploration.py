#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

"""
Entropy-augmented one-boundary DDM analysis for the reinforcement-learning-to-first-passage project.

Purpose
-------
This is a targeted DDM extension after the entropy analyses. It asks whether
predecision computational uncertainty features improve a DDM-style likelihood
over a D2-equivalent RL actor-control bridge.

Important guardrail
-------------------
This script is a self-contained DDM reimplementation for model comparison within
this entropy analysis. It is not intended to reproduce the exact NLL values from
the original D0-D8 pipeline unless the original parameterization was identical.

Use the outputs as a targeted sensitivity/extension:
- Does policy entropy enter drift?
- Does policy entropy enter boundary/distance-to-bound?
- Does confidence produce the opposite pattern?
- Does adding uncertainty improve held-out DDM likelihood over a D2-equivalent baseline?

Model family
------------
One-boundary/censored hitting-time likelihood:
- response trials: inverse-Gaussian first-passage density at RT - t0
- no-response trials: survival probability to deadline - t0

Components:
- drift v_i = exp(X_v beta), clipped to [v_min, v_max]
- bound a_i = exp(X_a gamma), clipped to [a_min, a_max]
- t0 is a scalar constrained between t0_min and t0_max

Candidate models:
- D2_equivalent:
    drift: design + p_press + actor_qdiff_current
    bound: design + control_current
- E1_policy_entropy_drift:
    D2_equivalent + policy_entropy in drift
- E2_policy_entropy_bound:
    D2_equivalent + policy_entropy in bound
- E3_policy_entropy_both:
    D2_equivalent + policy_entropy in drift and bound
- E4_policy_confidence_both:
    D2_equivalent + policy_confidence in drift and bound
- E5_uncertainty_confidence_both:
    D2_equivalent + policy_entropy and policy_confidence in drift and bound

Inputs
------
Recommended input:
controlled_inputs/entropy_trial_features/entropy_trial_features.csv

Outputs
-------
- ddm_entropy_cv_scores.csv
- ddm_entropy_cv_pairwise.csv
- ddm_entropy_final_coefficients.csv
- ddm_entropy_final_model_summary.csv
- ddm_entropy_final_trial_predictions.csv
- ddm_entropy_component_quantile_summary.csv
- DDM_ENTROPY_COMPONENT_ANALYSIS_REPORT.md

Example
-------
python code/00_model_development/entropy_diffusion_exploration.py \
  --entropy_trial_features controlled_inputs/entropy_trial_features/entropy_trial_features.csv \
  --out results/model_development/ddm_entropy_exploratory \
  --maxiter 350
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.model_selection import StratifiedGroupKFold


MODEL_SPECS = {
    "D2_equivalent": {
        "drift_extra": [],
        "bound_extra": [],
    },
    "E1_policy_entropy_drift": {
        "drift_extra": ["policy_entropy"],
        "bound_extra": [],
    },
    "E2_policy_entropy_bound": {
        "drift_extra": [],
        "bound_extra": ["policy_entropy"],
    },
    "E3_policy_entropy_both": {
        "drift_extra": ["policy_entropy"],
        "bound_extra": ["policy_entropy"],
    },
    "E4_policy_confidence_both": {
        "drift_extra": ["policy_confidence"],
        "bound_extra": ["policy_confidence"],
    },
    "E5_uncertainty_confidence_both": {
        "drift_extra": ["policy_entropy", "policy_confidence"],
        "bound_extra": ["policy_entropy", "policy_confidence"],
    },
}

DRIFT_BASE_NUMERIC = ["p_press", "actor_qdiff_current"]
BOUND_BASE_NUMERIC = ["control_current"]
DESIGN_CATEGORICAL = ["cue_type_norm", "trial_class_norm", "group"]
DESIGN_NUMERIC = ["is_nogo_numeric"]


@dataclass
class TransformInfo:
    numeric_means: Dict[str, float]
    numeric_sds: Dict[str, float]
    categorical_columns: List[str]
    feature_columns: List[str]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    ok = np.isfinite(pvals)
    if ok.sum() == 0:
        return out
    p = pvals[ok]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    tmp = np.empty(n)
    tmp[order] = q
    out[ok] = tmp
    return out


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def load_trials(path: Path, rt_min: float, rt_max: float) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "subject_id_norm" not in df.columns and "subject_id" in df.columns:
        df["subject_id_norm"] = df["subject_id"].astype(str)

    if "group" in df.columns:
        df["group"] = df["group"].map(norm_text)
    else:
        df["group"] = df["subject_id_norm"].astype(str).map(lambda s: "high" if s.upper().startswith("H") else "low")

    if "trial_class_norm" not in df.columns:
        if "trial_class" in df.columns:
            df["trial_class_norm"] = df["trial_class"].map(norm_text)
        elif "is_nogo" in df.columns:
            df["trial_class_norm"] = np.where(pd.to_numeric(df["is_nogo"], errors="coerce") == 1, "nogo", "go")
        else:
            df["trial_class_norm"] = ""

    if "cue_type_norm" not in df.columns:
        if "cue_type" in df.columns:
            df["cue_type_norm"] = df["cue_type"].map(norm_text)
        else:
            df["cue_type_norm"] = ""

    if "response_present_numeric" not in df.columns:
        if "response_present" in df.columns:
            df["response_present_numeric"] = pd.to_numeric(df["response_present"], errors="coerce").fillna(0).astype(int)
        elif "action_press" in df.columns:
            df["response_present_numeric"] = pd.to_numeric(df["action_press"], errors="coerce").fillna(0).astype(int)
        else:
            raise ValueError("Could not find response_present/action_press column.")

    # RT in seconds.
    if "rt_ms" in df.columns:
        rt = pd.to_numeric(df["rt_ms"], errors="coerce")
    elif "rt_response_from_target_ms" in df.columns:
        rt = pd.to_numeric(df["rt_response_from_target_ms"], errors="coerce")
    elif "rt" in df.columns:
        rt = pd.to_numeric(df["rt"], errors="coerce")
    else:
        rt = pd.Series(np.nan, index=df.index)

    # Infer units.
    med = rt.dropna().median()
    if pd.notna(med) and med > 10:
        df["rt_sec"] = rt / 1000.0
    else:
        df["rt_sec"] = rt

    df["is_response"] = pd.to_numeric(df["response_present_numeric"], errors="coerce").fillna(0).astype(int)
    df["is_nogo_numeric"] = (df["trial_class_norm"] == "nogo").astype(float)

    # Keep all no-response trials; keep response trials only in RT range.
    valid_noresp = df["is_response"] == 0
    valid_resp = (df["is_response"] == 1) & np.isfinite(df["rt_sec"]) & (df["rt_sec"] >= rt_min) & (df["rt_sec"] <= rt_max)
    out = df.loc[valid_noresp | valid_resp].copy()

    # Drop missing required D2-equivalent columns.
    required = ["p_press", "actor_qdiff_current", "control_current"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required D2-equivalent columns: {missing}")

    for c in required + ["policy_entropy", "policy_confidence"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.reset_index(drop=True)
    return out


def usable_numeric_columns(df: pd.DataFrame, candidates: List[str], min_nonmissing: int = 50) -> List[str]:
    cols = []
    for c in candidates:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= min_nonmissing and x.nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def build_design_train(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> Tuple[np.ndarray, TransformInfo]:
    parts = []

    means, sds = {}, {}
    for c in numeric_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        m = float(x.mean(skipna=True))
        sd = float(x.std(skipna=True))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        means[c] = m
        sds[c] = sd
        parts.append(((x - m) / sd).rename(c))

    cat_used = [c for c in categorical_cols if c in df.columns]
    if cat_used:
        dummies = pd.get_dummies(df[cat_used].astype(str), prefix=cat_used, drop_first=True, dtype=float)
    else:
        dummies = pd.DataFrame(index=df.index)

    # Remove any zero-variance dummy columns.
    cat_cols = []
    for c in dummies.columns:
        if dummies[c].nunique(dropna=True) > 1:
            cat_cols.append(c)
    if cat_cols:
        parts.append(dummies[cat_cols])

    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X.insert(0, "Intercept", 1.0)

    info = TransformInfo(
        numeric_means=means,
        numeric_sds=sds,
        categorical_columns=cat_cols,
        feature_columns=list(X.columns),
    )
    return X.values.astype(float), info


def build_design_apply(df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str], info: TransformInfo) -> np.ndarray:
    parts = []

    for c in numeric_cols:
        if c not in info.numeric_means:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        parts.append(((x - info.numeric_means[c]) / info.numeric_sds[c]).rename(c))

    cat_used = [c for c in categorical_cols if c in df.columns]
    if cat_used:
        dummies = pd.get_dummies(df[cat_used].astype(str), prefix=cat_used, drop_first=True, dtype=float)
        dummies = dummies.reindex(columns=info.categorical_columns, fill_value=0.0)
    else:
        dummies = pd.DataFrame(index=df.index, columns=info.categorical_columns).fillna(0.0)

    if len(info.categorical_columns):
        parts.append(dummies)

    if parts:
        X = pd.concat(parts, axis=1)
    else:
        X = pd.DataFrame(index=df.index)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X.insert(0, "Intercept", 1.0)
    X = X.reindex(columns=info.feature_columns, fill_value=0.0)
    return X.values.astype(float)


def get_model_columns(df: pd.DataFrame, spec: Dict[str, List[str]]) -> Tuple[List[str], List[str], List[str]]:
    drift_candidates = DRIFT_BASE_NUMERIC + spec["drift_extra"] + DESIGN_NUMERIC
    bound_candidates = BOUND_BASE_NUMERIC + spec["bound_extra"] + DESIGN_NUMERIC

    drift_numeric = usable_numeric_columns(df, drift_candidates)
    bound_numeric = usable_numeric_columns(df, bound_candidates)

    cat = [c for c in DESIGN_CATEGORICAL if c in df.columns and df[c].astype(str).nunique() > 1]
    return drift_numeric, bound_numeric, cat


def ig_logpdf_logsf(t: np.ndarray, v: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse-Gaussian first-passage density and survival.
    SciPy parameterization:
        invgauss(mu=mean/scale, scale=scale)
    For DDM first-passage:
        mean = a / v
        lambda = a^2
        shape = mean / lambda = 1 / (a*v)
        scale = lambda = a^2
    """
    from scipy.stats import invgauss

    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)

    mu_shape = 1.0 / np.clip(a * v, 1e-8, np.inf)
    scale = np.clip(a * a, 1e-8, np.inf)

    logpdf = invgauss.logpdf(t, mu=mu_shape, scale=scale)
    logsf = invgauss.logsf(t, mu=mu_shape, scale=scale)

    return logpdf, logsf


def unpack_params(theta: np.ndarray, nv: int, na: int, Xv: np.ndarray, Xa: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, float]:
    beta = theta[:nv]
    gamma = theta[nv:nv+na]
    t0_raw = theta[nv+na]

    log_v = Xv @ beta
    log_a = Xa @ gamma

    v = np.exp(np.clip(log_v, np.log(args.v_min), np.log(args.v_max)))
    a = np.exp(np.clip(log_a, np.log(args.a_min), np.log(args.a_max)))
    t0 = args.t0_min + (args.t0_max - args.t0_min) * float(sigmoid(np.array([t0_raw]))[0])

    return v, a, t0


def nll(theta: np.ndarray, Xv: np.ndarray, Xa: np.ndarray, is_resp: np.ndarray, rt_sec: np.ndarray, args, ridge: float = 1e-4) -> float:
    nv = Xv.shape[1]
    na = Xa.shape[1]
    v, a, t0 = unpack_params(theta, nv, na, Xv, Xa, args)

    loglik = np.empty(len(is_resp), dtype=float)
    loglik.fill(np.nan)

    # Response trials.
    resp = is_resp == 1
    if resp.any():
        t = rt_sec[resp] - t0
        bad = ~np.isfinite(t) | (t <= 1e-5)
        lp = np.full(resp.sum(), -1e12, dtype=float)
        if (~bad).any():
            logpdf, _ = ig_logpdf_logsf(t[~bad], v[resp][~bad], a[resp][~bad])
            lp[~bad] = logpdf
        loglik[resp] = lp

    # No-response trials.
    noresp = is_resp == 0
    if noresp.any():
        t_deadline = args.deadline_sec - t0
        if t_deadline <= 1e-5:
            loglik[noresp] = -1e12
        else:
            _, lsf = ig_logpdf_logsf(np.full(noresp.sum(), t_deadline), v[noresp], a[noresp])
            loglik[noresp] = lsf

    loglik = np.where(np.isfinite(loglik), loglik, -1e12)
    penalty = ridge * float(np.sum(theta[1:nv] ** 2) + np.sum(theta[nv+1:nv+na] ** 2))
    val = -float(np.sum(loglik)) + penalty
    if not np.isfinite(val):
        return 1e30
    return val


def evaluate(theta: np.ndarray, Xv: np.ndarray, Xa: np.ndarray, is_resp: np.ndarray, rt_sec: np.ndarray, args) -> Dict[str, float]:
    total_nll = nll(theta, Xv, Xa, is_resp, rt_sec, args, ridge=0.0)
    return {
        "n_trials": int(len(is_resp)),
        "n_response": int(np.sum(is_resp == 1)),
        "n_no_response": int(np.sum(is_resp == 0)),
        "nll_total": float(total_nll),
        "nll_per_trial": float(total_nll / max(1, len(is_resp))),
    }


def fit_model(train_df: pd.DataFrame, model_name: str, args) -> Dict[str, object]:
    spec = MODEL_SPECS[model_name]
    drift_numeric, bound_numeric, cat_cols = get_model_columns(train_df, spec)

    Xv, vinfo = build_design_train(train_df, drift_numeric, cat_cols)
    Xa, ainfo = build_design_train(train_df, bound_numeric, cat_cols)

    is_resp = train_df["is_response"].astype(int).values
    rt_sec = pd.to_numeric(train_df["rt_sec"], errors="coerce").values.astype(float)

    nv = Xv.shape[1]
    na = Xa.shape[1]
    theta0 = np.zeros(nv + na + 1, dtype=float)
    theta0[0] = np.log(1.5)       # drift intercept
    theta0[nv] = np.log(1.0)      # bound intercept
    theta0[-1] = 0.0              # t0 raw -> midpoint

    bounds = [(-6, 6)] * (nv + na) + [(-8, 8)]

    opt = optimize.minimize(
        nll,
        theta0,
        args=(Xv, Xa, is_resp, rt_sec, args),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(args.maxiter), "ftol": 1e-7, "maxls": 30},
    )

    theta = opt.x if opt.x is not None else theta0
    train_eval = evaluate(theta, Xv, Xa, is_resp, rt_sec, args)

    return {
        "model_name": model_name,
        "theta": theta,
        "success": bool(opt.success),
        "message": str(opt.message),
        "fun": float(opt.fun) if np.isfinite(opt.fun) else np.nan,
        "drift_numeric": drift_numeric,
        "bound_numeric": bound_numeric,
        "cat_cols": cat_cols,
        "vinfo": vinfo,
        "ainfo": ainfo,
        "train_eval": train_eval,
    }


def apply_model(fit: Dict[str, object], df: pd.DataFrame, args) -> Tuple[Dict[str, float], pd.DataFrame]:
    Xv = build_design_apply(df, fit["drift_numeric"], fit["cat_cols"], fit["vinfo"])
    Xa = build_design_apply(df, fit["bound_numeric"], fit["cat_cols"], fit["ainfo"])
    is_resp = df["is_response"].astype(int).values
    rt_sec = pd.to_numeric(df["rt_sec"], errors="coerce").values.astype(float)

    ev = evaluate(fit["theta"], Xv, Xa, is_resp, rt_sec, args)

    nv = Xv.shape[1]
    na = Xa.shape[1]
    v, a, t0 = unpack_params(fit["theta"], nv, na, Xv, Xa, args)

    _, logsf_deadline = ig_logpdf_logsf(
        np.full(len(df), args.deadline_sec - t0),
        v,
        a,
    )
    pred_response_prob = 1.0 - np.exp(np.clip(logsf_deadline, -700, 0))

    pred = pd.DataFrame({
        "subject_id_norm": df["subject_id_norm"].values,
        "trial_index_numeric": df["trial_index_numeric"].values if "trial_index_numeric" in df.columns else np.arange(len(df)),
        "model_name": fit["model_name"],
        "is_response": is_resp,
        "rt_sec": rt_sec,
        "drift_v": v,
        "bound_a": a,
        "t0_sec": t0,
        "pred_response_prob_by_deadline": pred_response_prob,
    })

    # Add trial-wise NLL.
    loglik = np.empty(len(df), dtype=float)
    resp = is_resp == 1
    if resp.any():
        t = rt_sec[resp] - t0
        lp = np.full(resp.sum(), -1e12, dtype=float)
        ok = np.isfinite(t) & (t > 1e-5)
        if ok.any():
            logpdf, _ = ig_logpdf_logsf(t[ok], v[resp][ok], a[resp][ok])
            lp[ok] = logpdf
        loglik[resp] = lp

    noresp = is_resp == 0
    if noresp.any():
        _, lsf = ig_logpdf_logsf(np.full(noresp.sum(), args.deadline_sec - t0), v[noresp], a[noresp])
        loglik[noresp] = lsf
    pred["trial_nll"] = -np.where(np.isfinite(loglik), loglik, -1e12)

    for c in ["policy_entropy", "policy_confidence", "p_press", "control_current", "actor_qdiff_current", "group", "trial_class_norm", "cue_type_norm", "outcome"]:
        if c in df.columns:
            pred[c] = df[c].values

    return ev, pred


def coefficients_table(fit: Dict[str, object], args) -> pd.DataFrame:
    theta = fit["theta"]
    nv = len(fit["vinfo"].feature_columns)
    na = len(fit["ainfo"].feature_columns)

    rows = []
    for i, name in enumerate(fit["vinfo"].feature_columns):
        rows.append({
            "model_name": fit["model_name"],
            "component": "drift_log_v",
            "feature": name,
            "coefficient": float(theta[i]),
        })
    for j, name in enumerate(fit["ainfo"].feature_columns):
        rows.append({
            "model_name": fit["model_name"],
            "component": "bound_log_a",
            "feature": name,
            "coefficient": float(theta[nv+j]),
        })
    t0 = args.t0_min + (args.t0_max - args.t0_min) * float(sigmoid(np.array([theta[nv+na]]))[0])
    rows.append({
        "model_name": fit["model_name"],
        "component": "t0",
        "feature": "scalar_t0_sec",
        "coefficient": float(t0),
    })
    return pd.DataFrame(rows)


def cv_analysis(df: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    # Stratify by response/no-response; group by subject.
    y = df["is_response"].astype(int).values
    groups = df["subject_id_norm"].astype(str).values

    cv = StratifiedGroupKFold(n_splits=int(args.n_splits), shuffle=True, random_state=int(args.seed))

    for fold, (train_idx, test_idx) in enumerate(cv.split(df, y, groups), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        for model_name in MODEL_SPECS:
            print(f"CV fold {fold}: fitting {model_name} ...", flush=True)
            try:
                fit = fit_model(train_df, model_name, args)
                train_eval = fit["train_eval"]
                test_eval, _ = apply_model(fit, test_df, args)
                rows.append({
                    "fold": int(fold),
                    "model_name": model_name,
                    "success": bool(fit["success"]),
                    "message": fit["message"],
                    "n_train": int(train_eval["n_trials"]),
                    "n_test": int(test_eval["n_trials"]),
                    "train_nll_total": float(train_eval["nll_total"]),
                    "train_nll_per_trial": float(train_eval["nll_per_trial"]),
                    "test_nll_total": float(test_eval["nll_total"]),
                    "test_nll_per_trial": float(test_eval["nll_per_trial"]),
                    "n_drift_features": int(len(fit["vinfo"].feature_columns)),
                    "n_bound_features": int(len(fit["ainfo"].feature_columns)),
                    "drift_numeric": ",".join(fit["drift_numeric"]),
                    "bound_numeric": ",".join(fit["bound_numeric"]),
                })
            except Exception as exc:
                rows.append({
                    "fold": int(fold),
                    "model_name": model_name,
                    "success": False,
                    "message": f"ERROR: {exc}",
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "train_nll_total": np.nan,
                    "train_nll_per_trial": np.nan,
                    "test_nll_total": np.nan,
                    "test_nll_per_trial": np.nan,
                    "n_drift_features": np.nan,
                    "n_bound_features": np.nan,
                    "drift_numeric": "",
                    "bound_numeric": "",
                })

    scores = pd.DataFrame(rows)

    pair_rows = []
    if not scores.empty:
        wide = scores.pivot_table(index="fold", columns="model_name", values="test_nll_per_trial", aggfunc="mean")
        baseline = "D2_equivalent"
        for model_name in MODEL_SPECS:
            if model_name == baseline or model_name not in wide.columns or baseline not in wide.columns:
                continue
            delta = (wide[model_name] - wide[baseline]).dropna().values.astype(float)
            # Negative delta means model improves held-out NLL.
            if len(delta) >= 2:
                t, p = stats.ttest_1samp(delta, 0.0)
                ci = stats.t.interval(0.95, df=len(delta)-1, loc=float(np.mean(delta)), scale=float(stats.sem(delta)))
                pair_rows.append({
                    "comparison": f"{model_name}_minus_{baseline}",
                    "model_name": model_name,
                    "baseline": baseline,
                    "n_folds": int(len(delta)),
                    "mean_delta_test_nll_per_trial": float(np.mean(delta)),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "folds_improved_negative_delta": int(np.sum(delta < 0)),
                    "t_value": float(t),
                    "p_value": float(p),
                })

    pairwise = pd.DataFrame(pair_rows)
    if not pairwise.empty:
        pairwise["q_value_fdr_bh"] = fdr_bh(pairwise["p_value"].values)
        pairwise = pairwise.sort_values(["mean_delta_test_nll_per_trial", "p_value"]).reset_index(drop=True)

    return scores, pairwise


def component_quantile_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Use final model prediction table from each model.
    for model_name, g0 in pred.groupby("model_name"):
        for var in ["policy_entropy", "policy_confidence", "p_press", "control_current"]:
            if var not in g0.columns:
                continue
            x = pd.to_numeric(g0[var], errors="coerce")
            if x.notna().sum() < 20 or x.nunique(dropna=True) < 4:
                continue
            try:
                q = pd.qcut(x.rank(method="first"), q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
            except Exception:
                continue
            g = g0.copy()
            g["quantile"] = q
            for quant, h in g.groupby("quantile"):
                rows.append({
                    "model_name": model_name,
                    "quantile_variable": var,
                    "quantile": str(quant),
                    "n_trials": int(len(h)),
                    "mean_drift_v": float(pd.to_numeric(h["drift_v"], errors="coerce").mean()),
                    "mean_bound_a": float(pd.to_numeric(h["bound_a"], errors="coerce").mean()),
                    "mean_pred_response_prob": float(pd.to_numeric(h["pred_response_prob_by_deadline"], errors="coerce").mean()),
                    "observed_response_rate": float(pd.to_numeric(h["is_response"], errors="coerce").mean()),
                    "mean_rt_sec_response_trials": float(pd.to_numeric(h.loc[h["is_response"] == 1, "rt_sec"], errors="coerce").mean()),
                    "mean_trial_nll": float(pd.to_numeric(h["trial_nll"], errors="coerce").mean()),
                })
    return pd.DataFrame(rows)


def final_fit_all(df: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coef_tables = []
    summary_rows = []
    pred_tables = []

    for model_name in MODEL_SPECS:
        print(f"Final fit: {model_name} ...", flush=True)
        fit = fit_model(df, model_name, args)
        ev, pred = apply_model(fit, df, args)

        coef_tables.append(coefficients_table(fit, args))
        pred_tables.append(pred)

        summary_rows.append({
            "model_name": model_name,
            "success": bool(fit["success"]),
            "message": fit["message"],
            "n_trials": int(ev["n_trials"]),
            "n_response": int(ev["n_response"]),
            "n_no_response": int(ev["n_no_response"]),
            "nll_total": float(ev["nll_total"]),
            "nll_per_trial": float(ev["nll_per_trial"]),
            "n_drift_features": int(len(fit["vinfo"].feature_columns)),
            "n_bound_features": int(len(fit["ainfo"].feature_columns)),
            "drift_numeric": ",".join(fit["drift_numeric"]),
            "bound_numeric": ",".join(fit["bound_numeric"]),
        })

    coefs = pd.concat(coef_tables, ignore_index=True) if coef_tables else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    pred_all = pd.concat(pred_tables, ignore_index=True) if pred_tables else pd.DataFrame()
    qsum = component_quantile_summary(pred_all)
    return coefs, summary, pred_all, qsum


def fmt(x, digits=4) -> str:
    try:
        x = float(x)
        if not np.isfinite(x):
            return "NA"
        return f"{x:.{digits}f}"
    except Exception:
        return "NA"


def write_report(out_dir: Path, df: pd.DataFrame, scores: pd.DataFrame, pairwise: pd.DataFrame, final_summary: pd.DataFrame, coefs: pd.DataFrame, qsum: pd.DataFrame) -> None:
    lines = []
    lines.append("# DDM entropy-component analysis report")
    lines.append("")
    lines.append("This report tests entropy-augmented one-boundary DDM variants against a D2-equivalent RL actor-control baseline.")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Trials used: {len(df)}")
    lines.append(f"- Subjects: {df['subject_id_norm'].nunique() if 'subject_id_norm' in df.columns else 'NA'}")
    lines.append(f"- Response trials: {int((df['is_response'] == 1).sum())}")
    lines.append(f"- No-response trials: {int((df['is_response'] == 0).sum())}")

    lines.append("")
    lines.append("## Cross-validated model comparison")
    lines.append("")
    if scores.empty:
        lines.append("- No CV scores generated.")
    else:
        agg = scores.groupby("model_name").agg(
            n_folds=("fold", "count"),
            mean_test_nll_per_trial=("test_nll_per_trial", "mean"),
            sd_test_nll_per_trial=("test_nll_per_trial", "std"),
            mean_train_nll_per_trial=("train_nll_per_trial", "mean"),
            success_rate=("success", "mean"),
        ).reset_index().sort_values("mean_test_nll_per_trial")
        lines.append("| Model | folds | test NLL/trial | train NLL/trial | success rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, row in agg.iterrows():
            lines.append(
                f"| {row['model_name']} | {int(row['n_folds'])} | "
                f"{row['mean_test_nll_per_trial']:.6f} ± {row['sd_test_nll_per_trial']:.6f} | "
                f"{row['mean_train_nll_per_trial']:.6f} | {row['success_rate']:.2f} |"
            )

    lines.append("")
    lines.append("## Pairwise improvement over D2_equivalent")
    lines.append("")
    lines.append("Negative delta means the entropy model has lower held-out NLL than D2_equivalent.")
    if pairwise.empty:
        lines.append("- No pairwise rows generated.")
    else:
        lines.append("")
        lines.append("| Comparison | folds | Δ test NLL/trial | 95% CI | improved folds | p | q |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in pairwise.iterrows():
            lines.append(
                f"| {row['comparison']} | {int(row['n_folds'])} | "
                f"{row['mean_delta_test_nll_per_trial']:.6f} | "
                f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
                f"{int(row['folds_improved_negative_delta'])}/{int(row['n_folds'])} | "
                f"{row['p_value']:.3g} | {row['q_value_fdr_bh']:.3g} |"
            )

    lines.append("")
    lines.append("## Final full-data fit")
    lines.append("")
    if not final_summary.empty:
        lines.append("| Model | NLL/trial | drift predictors | bound predictors |")
        lines.append("|---|---:|---|---|")
        for _, row in final_summary.sort_values("nll_per_trial").iterrows():
            lines.append(
                f"| {row['model_name']} | {row['nll_per_trial']:.6f} | "
                f"{row['drift_numeric']} | {row['bound_numeric']} |"
            )

    lines.append("")
    lines.append("## Entropy coefficients from full-data fits")
    lines.append("")
    if coefs.empty:
        lines.append("- No coefficients generated.")
    else:
        ent = coefs[coefs["feature"].isin(["policy_entropy", "policy_confidence"])].copy()
        if ent.empty:
            lines.append("- No entropy coefficients retained; entropy columns may have been missing or zero-variance.")
        else:
            lines.append("| Model | Component | Feature | Coefficient |")
            lines.append("|---|---|---|---:|")
            for _, row in ent.iterrows():
                lines.append(f"| {row['model_name']} | {row['component']} | {row['feature']} | {row['coefficient']:.4f} |")

    lines.append("")
    lines.append("## Component quantile summary")
    lines.append("")
    if qsum.empty:
        lines.append("- No quantile summary generated.")
    else:
        keep = qsum[qsum["quantile_variable"].isin(["policy_entropy", "policy_confidence"])].copy()
        lines.append("| Model | Variable | Quantile | n | drift | bound | predicted press | observed press | RT |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, row in keep.head(60).iterrows():
            lines.append(
                f"| {row['model_name']} | {row['quantile_variable']} | {row['quantile']} | "
                f"{int(row['n_trials'])} | {row['mean_drift_v']:.4f} | {row['mean_bound_a']:.4f} | "
                f"{row['mean_pred_response_prob']:.4f} | {row['observed_response_rate']:.4f} | "
                f"{row['mean_rt_sec_response_trials']:.4f} |"
            )

    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append("- This is a self-contained DDM extension and should be compared internally against its D2_equivalent baseline.")
    lines.append("- A robust entropy-DDM claim requires lower held-out NLL than D2_equivalent, preferably across most folds and after FDR correction.")
    lines.append("- If entropy changes full-data coefficients but does not improve held-out likelihood, frame it as descriptive/component-level sensitivity, not predictive DDM improvement.")
    lines.append("- Use only predecision entropy features in DDM. Do not enter action_surprise because it uses the observed response.")
    lines.append("")

    (out_dir / "DDM_ENTROPY_COMPONENT_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entropy_trial_features", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--deadline_sec", type=float, default=0.700)
    parser.add_argument("--rt_min", type=float, default=0.100)
    parser.add_argument("--rt_max", type=float, default=1.000)
    parser.add_argument("--t0_min", type=float, default=0.030)
    parser.add_argument("--t0_max", type=float, default=0.080)
    parser.add_argument("--v_min", type=float, default=0.05)
    parser.add_argument("--v_max", type=float, default=8.0)
    parser.add_argument("--a_min", type=float, default=0.20)
    parser.add_argument("--a_max", type=float, default=5.0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=350)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = load_trials(Path(args.entropy_trial_features).expanduser().resolve(), rt_min=float(args.rt_min), rt_max=float(args.rt_max))

        scores, pairwise = cv_analysis(df, args)
        coefs, final_summary, pred_all, qsum = final_fit_all(df, args)

    scores.to_csv(out_dir / "ddm_entropy_cv_scores.csv", index=False)
    pairwise.to_csv(out_dir / "ddm_entropy_cv_pairwise.csv", index=False)
    coefs.to_csv(out_dir / "ddm_entropy_final_coefficients.csv", index=False)
    final_summary.to_csv(out_dir / "ddm_entropy_final_model_summary.csv", index=False)
    pred_all.to_csv(out_dir / "ddm_entropy_final_trial_predictions.csv", index=False)
    qsum.to_csv(out_dir / "ddm_entropy_component_quantile_summary.csv", index=False)

    write_report(out_dir, df, scores, pairwise, final_summary, coefs, qsum)

    print(f"Wrote DDM entropy-component analysis to: {out_dir}")
    print((out_dir / "DDM_ENTROPY_COMPONENT_ANALYSIS_REPORT.md").read_text(encoding="utf-8")[:7000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
