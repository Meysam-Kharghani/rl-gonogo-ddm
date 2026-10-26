#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

# -*- coding: utf-8 -*-
"""
joint_supportive_validation.py

Stronger validation branch for the exploratory joint reinforcement-learning/first-passage model.

Purpose
-------
This script does NOT replace the main Bayesian RL -> leakage-controlled nested DDM
pipeline. It is an additional sensitivity branch intended to make the joint reinforcement-learning/first-passage model
analysis more defensible.

It adds:
1) Subject-wise cross-validation for J0/J1/J2.
2) Multiple optimizer/fold seeds.
3) Sensitivity configs for RT window and t0 bounds.
4) Held-out predictive comparison.
5) Optimizer stability diagnostics.
6) Posterior/predictive-style simulation checks for choices and RT quantiles.
7) Small model-recovery check from J2-simulated data.

Run from:
    cd <REPOSITORY_ROOT>/controlled_inputs

Restricted validation run:
    python code/00_model_development/joint_supportive_validation.py \
      --base . \
      --ddm_script shared/ddm/one_boundary_diffusion.py \
      --latents controlled_inputs/rl_latents/actor_control_seed_2026.csv \
      --out results/model_development/joint_model_supportive_validation_subset \
      --configs main \
      --outer_seeds 2026 \
      --n_folds 3 \
      --n_starts 2 \
      --maxiter 120 \
      --max_subjects 8 \
      --recovery_reps 1

Fuller run:
    python code/00_model_development/joint_supportive_validation.py \
      --base . \
      --ddm_script shared/ddm/one_boundary_diffusion.py \
      --latents controlled_inputs/rl_latents/actor_control_seed_2026.csv \
      --out results/model_development/joint_model_supportive_validation \
      --configs main higher_rtmin_150 conservative_t0 wider_t0 \
      --outer_seeds 2026 7777 \
      --n_folds 5 \
      --n_starts 8 \
      --maxiter 700 \
      --recovery_reps 10

Outputs
-------
results/model_development/joint_model_supportive_validation/
    joint_model_crossvalidated_fold_assignments.csv
    joint_model_crossvalidated_results_by_fold.csv
    joint_model_crossvalidated_summary.csv
    optimizer_stability_by_fit.csv
    joint_model_predictive_check_by_fold.csv
    joint_model_predictive_check_summary.csv
    model_recovery_results.csv
    model_recovery_summary.csv
    figures/
        fig_joint_model_crossvalidated_delta_elpd.png
        fig_joint_model_predictive_check_rt_quantiles.png
    JOINT_MODEL_VALIDATION_REPORT.md
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
# General utilities
# =============================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def import_module_from_path(path: Path, module_name: str):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing module: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def sem(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(arr.std(ddof=1) / np.sqrt(len(arr)))


def bootstrap_ci_mean(x, seed=2026, n_boot=5000) -> Tuple[float, float]:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    b = arr[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def softplus(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def zscore(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if not np.isfinite(s) or s < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - m) / s


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# DDM first-passage functions
# =============================================================================

def firstpass_logpdf_np(t, d, v):
    """
    Log density for first passage to upper boundary d for Brownian motion:
        X(t) = v t + W(t), boundary d > 0, start at 0.
    """
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    out = np.full_like(t, -1e12, dtype=float)
    ok = (t > 1e-6) & (d > 1e-6) & np.isfinite(t) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        tt, dd, vv = t[ok], d[ok], v[ok]
        out[ok] = np.log(dd) - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(tt) - ((dd - vv * tt) ** 2) / (2 * tt)
    return out


def firstpass_cdf_np(T, d, v):
    """
    CDF for first passage to upper boundary d by time T.
    """
    T = np.asarray(T, dtype=float)
    d = np.asarray(d, dtype=float)
    v = np.asarray(v, dtype=float)
    F = np.zeros_like(T, dtype=float)
    ok = (T > 1e-6) & (d > 1e-6) & np.isfinite(T) & np.isfinite(d) & np.isfinite(v)
    if np.any(ok):
        TT, dd, vv = T[ok], d[ok], v[ok]
        sqrtT = np.sqrt(TT)
        term1 = norm.cdf((vv * TT - dd) / sqrtT)
        expo = np.exp(np.clip(2 * vv * dd, -50, 50))
        term2 = expo * norm.cdf(-(vv * TT + dd) / sqrtT)
        F[ok] = np.clip(term1 + term2, EPS, 1 - EPS)
    return np.clip(F, EPS, 1 - EPS)


# =============================================================================
# Data loading and splits
# =============================================================================

CONFIGS = {
    "main": {
        "deadline_ms": 700.0,
        "rt_min_ms": 100.0,
        "rt_max_ms": 1000.0,
        "t0_min": 0.03,
        "t0_max": 0.08,
    },
    "higher_rtmin_150": {
        "deadline_ms": 700.0,
        "rt_min_ms": 150.0,
        "rt_max_ms": 1000.0,
        "t0_min": 0.03,
        "t0_max": 0.08,
    },
    "conservative_t0": {
        "deadline_ms": 700.0,
        "rt_min_ms": 100.0,
        "rt_max_ms": 1000.0,
        "t0_min": 0.05,
        "t0_max": 0.15,
    },
    "wider_t0": {
        "deadline_ms": 700.0,
        "rt_min_ms": 100.0,
        "rt_max_ms": 1000.0,
        "t0_min": 0.02,
        "t0_max": 0.20,
    },
}


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "trial_index" not in d.columns:
        d["trial_index"] = d.groupby("subj_id_ddm").cumcount()

    # Required binary columns.
    for col in ["press", "is_nogo", "group_high", "cue_up", "cue_bi"]:
        if col not in d.columns:
            d[col] = 0
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

    if "deadline_sec" not in d.columns:
        d["deadline_sec"] = 0.700

    if "rt_sec" not in d.columns:
        d["rt_sec"] = np.nan
    d["rt_sec"] = pd.to_numeric(d["rt_sec"], errors="coerce")

    # Outcome columns.
    d["hit"] = ((d["is_nogo"] == 0) & (d["press"] == 1)).astype(int)
    d["fa"] = ((d["is_nogo"] == 1) & (d["press"] == 1)).astype(int)
    d["miss"] = ((d["is_nogo"] == 0) & (d["press"] == 0)).astype(int)
    d["cr"] = ((d["is_nogo"] == 1) & (d["press"] == 0)).astype(int)
    d["correct"] = ((d["hit"] == 1) | (d["cr"] == 1)).astype(int)
    d["error"] = 1 - d["correct"]

    d = d.sort_values(["subj_id_ddm", "trial_index"]).reset_index(drop=True)
    d["prev_error"] = d.groupby("subj_id_ddm")["error"].shift(1).fillna(0).astype(int)
    d["prev_fa"] = d.groupby("subj_id_ddm")["fa"].shift(1).fillna(0).astype(int)
    d["prev_press"] = d.groupby("subj_id_ddm")["press"].shift(1).fillna(0).astype(int)
    return d


def load_filtered_data(ddm_mod, latents: Path, data_root: Optional[Path], cfg: Dict) -> pd.DataFrame:
    raw = ddm_mod.load_latents(
        Path(latents),
        deadline_ms=cfg["deadline_ms"],
        data_root=Path(data_root) if data_root else None,
    )
    filt, info = ddm_mod.filter_ddm_trials(
        raw,
        rt_min_ms=cfg["rt_min_ms"],
        rt_max_ms=cfg["rt_max_ms"],
    )
    d = prepare_columns(filt)
    return d


def make_stratified_subject_folds(df: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    sub = (
        df.groupby("subj_id_ddm")
        .agg(group_high=("group_high", "first"), n_trials=("press", "size"))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignments = []
    for group_val, g in sub.groupby("group_high"):
        subs = g["subj_id_ddm"].astype(str).to_numpy()
        rng.shuffle(subs)
        for i, s in enumerate(subs):
            assignments.append({
                "subj_id_ddm": s,
                "group_high": int(group_val),
                "fold": int(i % n_folds),
                "outer_seed": int(seed),
            })
    return pd.DataFrame(assignments)


# =============================================================================
# Supportive joint model
# =============================================================================

PARAM_NAMES = [
    # RL action policy
    "rl_bias",
    "rl_b_nogo",
    "rl_b_prev_error",
    "rl_b_group",
    "rl_b_cue_bi",
    "rl_b_cue_up",

    # Actor/control update parameters
    "alpha_actor_raw",
    "log_beta_actor",
    "alpha_ctrl_raw",
    "w_control",

    # DDM drift
    "v_int",
    "v_nogo",
    "v_group",
    "v_cue_bi",
    "v_cue_up",
    "v_prev_error",
    "v_qdiff",
    "v_prl",

    # DDM distance-to-bound
    "d_int",
    "d_nogo",
    "d_group",
    "d_cue_bi",
    "d_cue_up",
    "d_prev_error",
    "d_control",

    # DDM non-decision time
    "t0_raw",
]


def vec_to_par(x: np.ndarray) -> Dict[str, float]:
    return {k: float(v) for k, v in zip(PARAM_NAMES, x)}


def param_mask_for_model(model_key: str) -> Dict[str, bool]:
    if model_key == "J0":
        return {"use_actor": False, "use_control": False}
    if model_key == "J1":
        return {"use_actor": False, "use_control": True}
    if model_key == "J2":
        return {"use_actor": True, "use_control": True}
    raise ValueError(model_key)


def joint_forward_states(df: pd.DataFrame, par: Dict[str, float], use_actor: bool, use_control: bool) -> Dict[str, np.ndarray]:
    """
    Teacher-forced online states.
    For trial t, state uses only previous trials of the same subject.
    After likelihood for t, observed action/outcome at t updates state for t+1.
    """
    d = df.sort_values(["subj_id_ddm", "trial_index"]).copy()
    n = len(d)

    control = np.zeros(n, dtype=float)
    qdiff = np.zeros(n, dtype=float)
    p_rl = np.zeros(n, dtype=float)

    cue_idx = np.where(d["cue_up"].to_numpy(int) == 1, 2, np.where(d["cue_bi"].to_numpy(int) == 1, 1, 0))
    action = d["press"].to_numpy(float)
    correct = d["correct"].to_numpy(float)
    error = d["error"].to_numpy(float)
    nogo = d["is_nogo"].to_numpy(float)
    prev_error = d["prev_error"].to_numpy(float)
    group = d["group_high"].to_numpy(float)
    cue_bi = d["cue_bi"].to_numpy(float)
    cue_up = d["cue_up"].to_numpy(float)
    subjects = d["subj_id_ddm"].astype(str).to_numpy()

    alpha_actor = expit(par.get("alpha_actor_raw", 0.0))
    alpha_ctrl = expit(par.get("alpha_ctrl_raw", 0.0))
    beta_actor = np.exp(np.clip(par.get("log_beta_actor", 0.0), -5, 5))
    w_control = par.get("w_control", 0.0)

    Q: Dict[str, np.ndarray] = {}
    C: Dict[str, float] = {}

    for i in range(n):
        s = subjects[i]
        if s not in Q:
            Q[s] = np.zeros((3, 2), dtype=float)
            C[s] = 0.0

        ci = int(cue_idx[i])
        qd = Q[s][ci, 1] - Q[s][ci, 0]
        cprev = C[s]
        qdiff[i] = qd
        control[i] = cprev

        logit = (
            par["rl_bias"]
            + par["rl_b_nogo"] * nogo[i]
            + par["rl_b_prev_error"] * prev_error[i]
            + par["rl_b_group"] * group[i]
            + par["rl_b_cue_bi"] * cue_bi[i]
            + par["rl_b_cue_up"] * cue_up[i]
        )
        if use_actor:
            logit += beta_actor * qd
        if use_control:
            logit += w_control * cprev

        p = float(np.clip(expit(logit), 1e-9, 1 - 1e-9))
        p_rl[i] = p

        # Update after trial t.
        if use_control:
            C[s] = float(np.clip(cprev + alpha_ctrl * (error[i] - cprev), 0.0, 1.0))

        if use_actor:
            chosen = int(action[i] > 0.5)
            delta = correct[i] - Q[s][ci, chosen]
            Q[s][ci, chosen] += alpha_actor * delta
            Q[s] = np.clip(Q[s], -5, 5)

    return {"control": control, "qdiff": qdiff, "p_rl": p_rl}


def compute_components(df: pd.DataFrame, par: Dict[str, float], states: Dict[str, np.ndarray],
                       use_actor: bool, use_control: bool,
                       t0_min: float, t0_max: float, dist_min: float) -> Dict[str, np.ndarray]:
    nogo = df["is_nogo"].to_numpy(float)
    group = df["group_high"].to_numpy(float)
    cue_bi = df["cue_bi"].to_numpy(float)
    cue_up = df["cue_up"].to_numpy(float)
    prev_error = df["prev_error"].to_numpy(float)

    qdiff_z = zscore(states["qdiff"])
    prl_z = zscore(states["p_rl"])
    control_z = zscore(states["control"])

    v = (
        par["v_int"]
        + par["v_nogo"] * nogo
        + par["v_group"] * group
        + par["v_cue_bi"] * cue_bi
        + par["v_cue_up"] * cue_up
        + par["v_prev_error"] * prev_error
    )
    if use_actor:
        v = v + par["v_qdiff"] * qdiff_z + par["v_prl"] * prl_z

    d_lin = (
        par["d_int"]
        + par["d_nogo"] * nogo
        + par["d_group"] * group
        + par["d_cue_bi"] * cue_bi
        + par["d_cue_up"] * cue_up
        + par["d_prev_error"] * prev_error
    )
    if use_control:
        d_lin = d_lin + par["d_control"] * control_z

    dist = softplus(d_lin) + dist_min
    t0 = t0_min + (t0_max - t0_min) * expit(par["t0_raw"])

    return {
        "v": np.asarray(v, dtype=float),
        "dist": np.asarray(dist, dtype=float),
        "t0": np.full(len(df), float(t0), dtype=float),
        "qdiff_z": qdiff_z,
        "p_rl_z": prl_z,
        "control_z": control_z,
    }


def joint_loglik_parts(theta: np.ndarray, df: pd.DataFrame, model_key: str,
                       cfg: Dict, action_weight: float = 1.0) -> Dict[str, float]:
    par = vec_to_par(theta)
    mask = param_mask_for_model(model_key)
    use_actor = mask["use_actor"]
    use_control = mask["use_control"]

    d = df.sort_values(["subj_id_ddm", "trial_index"]).copy()
    st = joint_forward_states(d, par, use_actor=use_actor, use_control=use_control)
    comp = compute_components(
        d, par, st, use_actor=use_actor, use_control=use_control,
        t0_min=cfg["t0_min"], t0_max=cfg["t0_max"], dist_min=cfg.get("dist_min", 0.05)
    )

    press = d["press"].to_numpy(float)
    rt = d["rt_sec"].to_numpy(float)
    deadline = d["deadline_sec"].to_numpy(float)

    p = np.clip(st["p_rl"], 1e-9, 1 - 1e-9)
    ll_action_vec = press * np.log(p) + (1 - press) * np.log(1 - p)

    v = comp["v"]
    dist = comp["dist"]
    t0 = comp["t0"]

    is_press = press > 0.5
    ll_ddm = np.zeros(len(d), dtype=float)
    if np.any(is_press):
        ll_ddm[is_press] = firstpass_logpdf_np(rt[is_press] - t0[is_press], dist[is_press], v[is_press])
    if np.any(~is_press):
        F = firstpass_cdf_np(deadline[~is_press] - t0[~is_press], dist[~is_press], v[~is_press])
        ll_ddm[~is_press] = np.log(np.clip(1 - F, EPS, 1))

    if not np.all(np.isfinite(ll_ddm)):
        return {"ll_action": -1e30, "ll_ddm": -1e30, "ll_joint": -1e30}

    ll_action = float(np.sum(ll_action_vec))
    ll_ddm_sum = float(np.sum(ll_ddm))
    ll_joint = float(action_weight * ll_action + ll_ddm_sum)
    return {"ll_action": ll_action, "ll_ddm": ll_ddm_sum, "ll_joint": ll_joint}


def joint_nll(theta: np.ndarray, df: pd.DataFrame, model_key: str,
              cfg: Dict, action_weight: float, l2: float) -> float:
    parts = joint_loglik_parts(theta, df, model_key=model_key, cfg=cfg, action_weight=action_weight)
    if not np.isfinite(parts["ll_joint"]):
        return 1e30

    # Freeze actor/control effect parameters for reduced models to avoid meaningless penalties.
    par = vec_to_par(theta)
    if model_key == "J0":
        freeze = ["alpha_actor_raw", "log_beta_actor", "alpha_ctrl_raw", "w_control",
                  "v_qdiff", "v_prl", "d_control"]
    elif model_key == "J1":
        freeze = ["alpha_actor_raw", "log_beta_actor", "v_qdiff", "v_prl"]
    else:
        freeze = []
    penal_vec = np.asarray([v for k, v in par.items() if k not in freeze], dtype=float)
    penalty = l2 * float(np.sum(penal_vec ** 2))
    return -parts["ll_joint"] + penalty


def initial_theta(model_key: str, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(len(PARAM_NAMES), dtype=float)
    # Reasonable starting values.
    x[PARAM_NAMES.index("rl_bias")] = 0.0
    x[PARAM_NAMES.index("rl_b_nogo")] = -2.0
    x[PARAM_NAMES.index("alpha_actor_raw")] = 0.0
    x[PARAM_NAMES.index("alpha_ctrl_raw")] = -0.5
    x[PARAM_NAMES.index("log_beta_actor")] = 0.0
    x[PARAM_NAMES.index("v_int")] = 0.8
    x[PARAM_NAMES.index("v_nogo")] = -1.0
    x[PARAM_NAMES.index("d_int")] = 0.0
    x[PARAM_NAMES.index("t0_raw")] = 0.0

    # Make reduced models stable.
    if model_key == "J0":
        for k in ["alpha_actor_raw", "log_beta_actor", "alpha_ctrl_raw", "w_control", "v_qdiff", "v_prl", "d_control"]:
            x[PARAM_NAMES.index(k)] = 0.0
    elif model_key == "J1":
        for k in ["alpha_actor_raw", "log_beta_actor", "v_qdiff", "v_prl"]:
            x[PARAM_NAMES.index(k)] = 0.0

    return x + rng.normal(0, 0.15, len(PARAM_NAMES))


def bounds_for_model(model_key: str) -> List[Tuple[float, float]]:
    b = [(-6, 6)] * len(PARAM_NAMES)
    # keep learning/log-beta bounded.
    for k in ["alpha_actor_raw", "alpha_ctrl_raw"]:
        b[PARAM_NAMES.index(k)] = (-5, 5)
    b[PARAM_NAMES.index("log_beta_actor")] = (-4, 4)
    b[PARAM_NAMES.index("t0_raw")] = (-6, 6)

    # Freeze irrelevant terms by setting exact bounds.
    if model_key == "J0":
        for k in ["alpha_actor_raw", "log_beta_actor", "alpha_ctrl_raw", "w_control", "v_qdiff", "v_prl", "d_control"]:
            b[PARAM_NAMES.index(k)] = (0.0, 0.0)
    elif model_key == "J1":
        for k in ["alpha_actor_raw", "log_beta_actor", "v_qdiff", "v_prl"]:
            b[PARAM_NAMES.index(k)] = (0.0, 0.0)
    return b


def fit_joint_model(train_df: pd.DataFrame, model_key: str, cfg: Dict,
                    n_starts: int, seed: int, maxiter: int,
                    action_weight: float, l2: float) -> Dict:
    rng = np.random.default_rng(seed)
    bounds = bounds_for_model(model_key)
    starts = [initial_theta(model_key, rng) for _ in range(n_starts)]

    best = None
    history = []
    for si, x0 in enumerate(starts):
        # Clip x0 into bounds.
        x0c = np.asarray(x0, dtype=float).copy()
        for j, (lo, hi) in enumerate(bounds):
            x0c[j] = min(max(x0c[j], lo), hi)
        res = minimize(
            joint_nll,
            x0=x0c,
            args=(train_df, model_key, cfg, action_weight, l2),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 50},
        )
        history.append({
            "start": si,
            "success": bool(res.success),
            "nll": float(res.fun),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", -1)),
        })
        if best is None or float(res.fun) < float(best.fun):
            best = res

    k_free = sum(1 for lo, hi in bounds if hi > lo)
    train_parts = joint_loglik_parts(best.x, train_df, model_key=model_key, cfg=cfg, action_weight=action_weight)
    n = len(train_df)
    fit = {
        "model_key": model_key,
        "success": bool(best.success),
        "message": str(best.message),
        "theta": [float(z) for z in best.x],
        "params": vec_to_par(best.x),
        "n_params_free": int(k_free),
        "n_train_trials": int(n),
        "train_ll_action": train_parts["ll_action"],
        "train_ll_ddm": train_parts["ll_ddm"],
        "train_ll_joint": train_parts["ll_joint"],
        "train_aic_joint": float(2 * k_free - 2 * train_parts["ll_joint"]),
        "train_bic_joint": float(k_free * np.log(max(n, 1)) - 2 * train_parts["ll_joint"]),
        "history": history,
    }
    return fit


def evaluate_fit(fit: Dict, test_df: pd.DataFrame, cfg: Dict, action_weight: float) -> Dict:
    theta = np.asarray(fit["theta"], dtype=float)
    parts = joint_loglik_parts(theta, test_df, model_key=fit["model_key"], cfg=cfg, action_weight=action_weight)
    return {
        "test_ll_action": parts["ll_action"],
        "test_ll_ddm": parts["ll_ddm"],
        "test_ll_joint": parts["ll_joint"],
        "test_elpd_per_trial": parts["ll_joint"] / max(len(test_df), 1),
    }


# =============================================================================
# Simulation and predictive checks
# =============================================================================

def simulate_from_fit(fit: Dict, df: pd.DataFrame, cfg: Dict, seed: int, dt: float = 0.004) -> pd.DataFrame:
    """
    Simulate choices and RTs from the fitted joint reinforcement-learning/first-passage model.
    State recursion uses simulated actions/outcomes, not observed actions after t.
    """
    rng = np.random.default_rng(seed)
    par = fit["params"]
    model_key = fit["model_key"]
    mask = param_mask_for_model(model_key)
    use_actor = mask["use_actor"]
    use_control = mask["use_control"]

    d = df.sort_values(["subj_id_ddm", "trial_index"]).copy().reset_index(drop=True)
    n = len(d)

    cue_idx = np.where(d["cue_up"].to_numpy(int) == 1, 2, np.where(d["cue_bi"].to_numpy(int) == 1, 1, 0))
    nogo = d["is_nogo"].to_numpy(float)
    group = d["group_high"].to_numpy(float)
    cue_bi = d["cue_bi"].to_numpy(float)
    cue_up = d["cue_up"].to_numpy(float)
    subjects = d["subj_id_ddm"].astype(str).to_numpy()
    deadline = d["deadline_sec"].to_numpy(float)

    alpha_actor = expit(par.get("alpha_actor_raw", 0.0))
    alpha_ctrl = expit(par.get("alpha_ctrl_raw", 0.0))
    beta_actor = np.exp(np.clip(par.get("log_beta_actor", 0.0), -5, 5))
    w_control = par.get("w_control", 0.0)

    Q: Dict[str, np.ndarray] = {}
    C: Dict[str, float] = {}

    press_sim = np.zeros(n, dtype=int)
    rt_sim = np.full(n, np.nan, dtype=float)
    correct_sim = np.zeros(n, dtype=int)
    error_sim = np.zeros(n, dtype=int)

    # Need online arrays for DDM component.
    qdiff_arr = np.zeros(n, dtype=float)
    prl_arr = np.zeros(n, dtype=float)
    control_arr = np.zeros(n, dtype=float)

    for i in range(n):
        s = subjects[i]
        if s not in Q:
            Q[s] = np.zeros((3, 2), dtype=float)
            C[s] = 0.0

        ci = int(cue_idx[i])
        qd = Q[s][ci, 1] - Q[s][ci, 0]
        cprev = C[s]

        prev_err_i = 0
        if i > 0 and subjects[i-1] == s:
            prev_err_i = error_sim[i-1]

        logit = (
            par["rl_bias"]
            + par["rl_b_nogo"] * nogo[i]
            + par["rl_b_prev_error"] * prev_err_i
            + par["rl_b_group"] * group[i]
            + par["rl_b_cue_bi"] * cue_bi[i]
            + par["rl_b_cue_up"] * cue_up[i]
        )
        if use_actor:
            logit += beta_actor * qd
        if use_control:
            logit += w_control * cprev
        p_rl = float(np.clip(expit(logit), 1e-9, 1-1e-9))

        qdiff_arr[i] = qd
        prl_arr[i] = p_rl
        control_arr[i] = cprev

        # Trial-level z-scoring is not available online; use current unstandardized state directly for a sequence-level approximation.
        # For generating RT, use current states in the same transformation as fitting but within the complete simulated sequence
        # is computed below by a one-pass approximation. To keep simulation stable, compute DDM directly here with raw states.
        v = (
            par["v_int"]
            + par["v_nogo"] * nogo[i]
            + par["v_group"] * group[i]
            + par["v_cue_bi"] * cue_bi[i]
            + par["v_cue_up"] * cue_up[i]
            + par["v_prev_error"] * prev_err_i
        )
        if use_actor:
            v += par["v_qdiff"] * qd + par["v_prl"] * p_rl

        d_lin = (
            par["d_int"]
            + par["d_nogo"] * nogo[i]
            + par["d_group"] * group[i]
            + par["d_cue_bi"] * cue_bi[i]
            + par["d_cue_up"] * cue_up[i]
            + par["d_prev_error"] * prev_err_i
        )
        if use_control:
            d_lin += par["d_control"] * cprev

        dist = float(softplus(d_lin) + cfg.get("dist_min", 0.05))
        t0 = float(cfg["t0_min"] + (cfg["t0_max"] - cfg["t0_min"]) * expit(par["t0_raw"]))

        max_steps = int(max(1, np.ceil((deadline[i] - t0) / dt)))
        x = 0.0
        hit_boundary = False
        for step in range(1, max_steps + 1):
            x += v * dt + math.sqrt(dt) * rng.standard_normal()
            if x >= dist:
                hit_boundary = True
                rt_sim[i] = t0 + step * dt
                break

        press_i = int(hit_boundary)
        press_sim[i] = press_i
        correct_i = int((nogo[i] == 0 and press_i == 1) or (nogo[i] == 1 and press_i == 0))
        correct_sim[i] = correct_i
        error_i = 1 - correct_i
        error_sim[i] = error_i

        if use_control:
            C[s] = float(np.clip(cprev + alpha_ctrl * (error_i - cprev), 0.0, 1.0))
        if use_actor:
            delta = correct_i - Q[s][ci, press_i]
            Q[s][ci, press_i] += alpha_actor * delta
            Q[s] = np.clip(Q[s], -5, 5)

    sim = d.copy()
    sim["press"] = press_sim
    sim["rt_sec"] = rt_sim
    sim["hit"] = ((sim["is_nogo"] == 0) & (sim["press"] == 1)).astype(int)
    sim["fa"] = ((sim["is_nogo"] == 1) & (sim["press"] == 1)).astype(int)
    sim["miss"] = ((sim["is_nogo"] == 0) & (sim["press"] == 0)).astype(int)
    sim["cr"] = ((sim["is_nogo"] == 1) & (sim["press"] == 0)).astype(int)
    sim["correct"] = ((sim["hit"] == 1) | (sim["cr"] == 1)).astype(int)
    sim["error"] = 1 - sim["correct"]
    sim = prepare_columns(sim)
    return sim


def summarize_ppc(obs: pd.DataFrame, sim: pd.DataFrame, model_key: str, cfg_name: str, fold: int, outer_seed: int) -> List[Dict]:
    rows = []
    specs = [("all", np.ones(len(obs), dtype=bool))]
    specs += [("low", obs["group_high"].to_numpy(int) == 0), ("high", obs["group_high"].to_numpy(int) == 1)]
    specs += [("go", obs["is_nogo"].to_numpy(int) == 0), ("nogo", obs["is_nogo"].to_numpy(int) == 1)]

    for label, mask in specs:
        o = obs.loc[mask]
        s = sim.loc[mask]
        if len(o) == 0:
            continue
        row = {
            "config": cfg_name,
            "outer_seed": outer_seed,
            "fold": fold,
            "model_key": model_key,
            "group": label,
            "n_trials": int(len(o)),
            "obs_press_rate": float(o["press"].mean()),
            "sim_press_rate": float(s["press"].mean()),
            "obs_hit_rate": float(o.loc[o["is_nogo"] == 0, "press"].mean()) if int((o["is_nogo"] == 0).sum()) > 0 else np.nan,
            "sim_hit_rate": float(s.loc[s["is_nogo"] == 0, "press"].mean()) if int((s["is_nogo"] == 0).sum()) > 0 else np.nan,
            "obs_fa_rate": float(o.loc[o["is_nogo"] == 1, "press"].mean()) if int((o["is_nogo"] == 1).sum()) > 0 else np.nan,
            "sim_fa_rate": float(s.loc[s["is_nogo"] == 1, "press"].mean()) if int((s["is_nogo"] == 1).sum()) > 0 else np.nan,
        }
        oq = o.loc[(o["press"] == 1) & np.isfinite(o["rt_sec"]), "rt_sec"].to_numpy(float)
        sq = s.loc[(s["press"] == 1) & np.isfinite(s["rt_sec"]), "rt_sec"].to_numpy(float)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            row[f"obs_rt_q{int(q*100)}"] = float(np.quantile(oq, q)) if len(oq) >= 5 else np.nan
            row[f"sim_rt_q{int(q*100)}"] = float(np.quantile(sq, q)) if len(sq) >= 5 else np.nan
        rows.append(row)
    return rows


# =============================================================================
# Main analyses
# =============================================================================

MODEL_NAMES = {
    "J0": "baseline",
    "J1": "control-only",
    "J2": "actor-control",
}


def run_cv(args, ddm_mod, out: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_cv_rows = []
    all_fold_rows = []
    all_start_rows = []
    all_ppc_rows = []

    for cfg_name in args.configs:
        if cfg_name not in CONFIGS:
            raise ValueError(f"Unknown config {cfg_name}. Available: {list(CONFIGS)}")
        cfg = CONFIGS[cfg_name].copy()
        cfg["dist_min"] = args.dist_min

        print(f"\n[cv] config={cfg_name}")
        df = load_filtered_data(ddm_mod, Path(args.latents), Path(args.data_root) if args.data_root else None, cfg)

        if args.max_subjects and args.max_subjects > 0:
            subs = sorted(df["subj_id_ddm"].astype(str).unique())[:args.max_subjects]
            df = df[df["subj_id_ddm"].astype(str).isin(subs)].reset_index(drop=True)
            print(f"[cv] using max_subjects={len(subs)} for restricted-subset evaluation")

        for outer_seed in args.outer_seeds:
            folds = make_stratified_subject_folds(df, n_folds=args.n_folds, seed=int(outer_seed))
            folds["config"] = cfg_name
            all_fold_rows.append(folds)

            for fold in range(args.n_folds):
                test_subs = set(folds.loc[folds["fold"] == fold, "subj_id_ddm"].astype(str))
                train_df = df[~df["subj_id_ddm"].astype(str).isin(test_subs)].reset_index(drop=True)
                test_df = df[df["subj_id_ddm"].astype(str).isin(test_subs)].reset_index(drop=True)

                print(f"[cv] seed={outer_seed} fold={fold} train={len(train_df)} test={len(test_df)}")

                for model_key in ["J0", "J1", "J2"]:
                    fit_seed = int(outer_seed) + 1000 * fold + {"J0": 0, "J1": 101, "J2": 202}[model_key]
                    fit = fit_joint_model(
                        train_df, model_key=model_key, cfg=cfg,
                        n_starts=args.n_starts, seed=fit_seed, maxiter=args.maxiter,
                        action_weight=args.action_weight, l2=args.l2,
                    )
                    test_eval = evaluate_fit(fit, test_df, cfg=cfg, action_weight=args.action_weight)

                    fold_dir = ensure_dir(out / "fits" / cfg_name / f"seed_{outer_seed}" / f"fold_{fold}" / model_key)
                    write_json(fold_dir / "fit.json", fit)

                    all_cv_rows.append({
                        "config": cfg_name,
                        "outer_seed": int(outer_seed),
                        "fold": int(fold),
                        "model_key": model_key,
                        "model_name": MODEL_NAMES[model_key],
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "train_ll_joint": fit["train_ll_joint"],
                        "train_ll_action": fit["train_ll_action"],
                        "train_ll_ddm": fit["train_ll_ddm"],
                        "test_ll_joint": test_eval["test_ll_joint"],
                        "test_ll_action": test_eval["test_ll_action"],
                        "test_ll_ddm": test_eval["test_ll_ddm"],
                        "test_elpd_per_trial": test_eval["test_elpd_per_trial"],
                        "n_params_free": fit["n_params_free"],
                        "success": fit["success"],
                    })

                    hist = pd.DataFrame(fit["history"])
                    if len(hist) > 0:
                        best = hist["nll"].min()
                        hist["delta_from_best_nll"] = hist["nll"] - best
                        hist["config"] = cfg_name
                        hist["outer_seed"] = int(outer_seed)
                        hist["fold"] = int(fold)
                        hist["model_key"] = model_key
                        all_start_rows.extend(hist.to_dict("records"))

                    if model_key == "J2" and args.ppc_sims > 0:
                        for sim_i in range(args.ppc_sims):
                            sim = simulate_from_fit(
                                fit, test_df, cfg=cfg,
                                seed=int(outer_seed) + 100000 * (fold + 1) + sim_i,
                                dt=args.sim_dt,
                            )
                            all_ppc_rows.extend(summarize_ppc(test_df, sim, model_key, cfg_name, fold, int(outer_seed)))

    cv = pd.DataFrame(all_cv_rows)
    folds_df = pd.concat(all_fold_rows, ignore_index=True) if all_fold_rows else pd.DataFrame()
    starts_df = pd.DataFrame(all_start_rows)
    ppc_df = pd.DataFrame(all_ppc_rows)

    cv.to_csv(out / "joint_model_crossvalidated_results_by_fold.csv", index=False)
    folds_df.to_csv(out / "joint_model_crossvalidated_fold_assignments.csv", index=False)
    starts_df.to_csv(out / "optimizer_stability_by_start.csv", index=False)
    ppc_df.to_csv(out / "joint_model_predictive_check_by_fold.csv", index=False)

    return cv, starts_df, ppc_df


def summarize_cv(cv: pd.DataFrame, out: Path) -> pd.DataFrame:
    if len(cv) == 0:
        return pd.DataFrame()

    rows = []
    for (cfg, outer_seed), g in cv.groupby(["config", "outer_seed"]):
        # Per fold deltas for J2 vs controls.
        wide = g.pivot_table(index="fold", columns="model_key", values="test_ll_joint", aggfunc="first")
        for comp in ["J0", "J1"]:
            if "J2" not in wide.columns or comp not in wide.columns:
                continue
            delta = (wide["J2"] - wide[comp]).to_numpy(float)
            ci_lo, ci_hi = bootstrap_ci_mean(delta, seed=int(outer_seed))
            rows.append({
                "config": cfg,
                "outer_seed": int(outer_seed),
                "comparison": f"J2_minus_{comp}",
                "mean_delta_test_elpd": float(np.nanmean(delta)),
                "total_delta_test_elpd": float(np.nansum(delta)),
                "se_delta": sem(delta),
                "ci025_boot_mean_delta": ci_lo,
                "ci975_boot_mean_delta": ci_hi,
                "positive_folds": int(np.sum(delta > 0)),
                "n_folds": int(np.sum(np.isfinite(delta))),
            })

    # Also pooled across outer seeds/configs.
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "joint_model_crossvalidated_summary_by_seed.csv", index=False)

    pooled_rows = []
    for (cfg, comp), g in summary.groupby(["config", "comparison"]):
        vals = []
        # reconstruct fold-level deltas pooled across seeds
        for outer_seed in g["outer_seed"].unique():
            sub = cv[(cv["config"] == cfg) & (cv["outer_seed"] == outer_seed)]
            wide = sub.pivot_table(index="fold", columns="model_key", values="test_ll_joint", aggfunc="first")
            base = comp.replace("J2_minus_", "")
            vals.extend((wide["J2"] - wide[base]).to_numpy(float))
        vals = np.asarray(vals, dtype=float)
        ci_lo, ci_hi = bootstrap_ci_mean(vals, seed=2026)
        pooled_rows.append({
            "config": cfg,
            "comparison": comp,
            "mean_delta_test_elpd": float(np.nanmean(vals)),
            "total_delta_test_elpd": float(np.nansum(vals)),
            "se_delta": sem(vals),
            "ci025_boot_mean_delta": ci_lo,
            "ci975_boot_mean_delta": ci_hi,
            "positive_folds": int(np.sum(vals > 0)),
            "n_folds_total": int(np.sum(np.isfinite(vals))),
        })

    pooled = pd.DataFrame(pooled_rows).sort_values(["config", "comparison"])
    pooled.to_csv(out / "joint_model_crossvalidated_summary.csv", index=False)
    return pooled


def summarize_optimizer(starts: pd.DataFrame, out: Path) -> pd.DataFrame:
    if len(starts) == 0:
        return pd.DataFrame()

    def second_best_delta(x):
        xs = np.sort(np.asarray(x, dtype=float))
        if len(xs) < 2:
            return np.nan
        return float(xs[1] - xs[0])

    summ = (
        starts.groupby(["config", "outer_seed", "fold", "model_key"])
        .agg(
            n_starts=("start", "count"),
            n_success=("success", "sum"),
            best_nll=("nll", "min"),
            worst_nll=("nll", "max"),
            second_best_delta_nll=("nll", second_best_delta),
            median_delta_from_best=("delta_from_best_nll", "median"),
        )
        .reset_index()
    )
    summ.to_csv(out / "optimizer_stability_by_fit.csv", index=False)
    return summ


def summarize_ppc_table(ppc: pd.DataFrame, out: Path) -> pd.DataFrame:
    if len(ppc) == 0:
        return pd.DataFrame()

    # Errors.
    p = ppc.copy()
    for m in ["press_rate", "hit_rate", "fa_rate"]:
        p[f"{m}_error"] = p[f"sim_{m}"] - p[f"obs_{m}"]
        p[f"{m}_abs_error"] = p[f"{m}_error"].abs()

    for q in [10, 25, 50, 75, 90]:
        p[f"rt_q{q}_error"] = p[f"sim_rt_q{q}"] - p[f"obs_rt_q{q}"]
        p[f"rt_q{q}_abs_error"] = p[f"rt_q{q}_error"].abs()

    summ = (
        p.groupby(["config", "group", "model_key"])
        .agg(
            n_rows=("n_trials", "count"),
            mean_press_rate_abs_error=("press_rate_abs_error", "mean"),
            mean_hit_rate_abs_error=("hit_rate_abs_error", "mean"),
            mean_fa_rate_abs_error=("fa_rate_abs_error", "mean"),
            mean_rt_q50_abs_error_sec=("rt_q50_abs_error", "mean"),
            mean_rt_q90_abs_error_sec=("rt_q90_abs_error", "mean"),
        )
        .reset_index()
    )
    summ.to_csv(out / "joint_model_predictive_check_summary.csv", index=False)
    p.to_csv(out / "joint_model_predictive_check_by_fold_with_errors.csv", index=False)
    return summ


def run_model_recovery(args, ddm_mod, out: Path) -> pd.DataFrame:
    if args.recovery_reps <= 0:
        return pd.DataFrame()

    rows = []
    rec_dir = ensure_dir(out / "model_recovery")

    # Use main config for recovery unless user requests all configs.
    cfg_names = args.recovery_configs if args.recovery_configs else ["main"]

    for cfg_name in cfg_names:
        cfg = CONFIGS[cfg_name].copy()
        cfg["dist_min"] = args.dist_min

        df = load_filtered_data(ddm_mod, Path(args.latents), Path(args.data_root) if args.data_root else None, cfg)
        if args.max_subjects and args.max_subjects > 0:
            subs = sorted(df["subj_id_ddm"].astype(str).unique())[:args.max_subjects]
            df = df[df["subj_id_ddm"].astype(str).isin(subs)].reset_index(drop=True)

        print(f"\n[recovery] fitting source J2 on full data config={cfg_name}")
        source_fit = fit_joint_model(
            df, model_key="J2", cfg=cfg,
            n_starts=args.n_starts, seed=args.outer_seeds[0] + 999,
            maxiter=args.maxiter, action_weight=args.action_weight, l2=args.l2,
        )
        write_json(rec_dir / f"source_J2_{cfg_name}.json", source_fit)

        for rep in range(args.recovery_reps):
            print(f"[recovery] config={cfg_name} rep={rep+1}/{args.recovery_reps}")
            sim = simulate_from_fit(source_fit, df, cfg=cfg, seed=args.outer_seeds[0] + 20000 + rep, dt=args.sim_dt)
            sim.to_csv(rec_dir / f"sim_{cfg_name}_rep_{rep:02d}.csv.gz", index=False, compression="gzip")

            for model_key in ["J0", "J1", "J2"]:
                fit = fit_joint_model(
                    sim, model_key=model_key, cfg=cfg,
                    n_starts=max(2, args.n_starts // 2),
                    seed=args.outer_seeds[0] + 30000 + rep * 100 + {"J0": 0, "J1": 10, "J2": 20}[model_key],
                    maxiter=args.maxiter,
                    action_weight=args.action_weight,
                    l2=args.l2,
                )
                rows.append({
                    "config": cfg_name,
                    "rep": int(rep),
                    "source_model": "J2",
                    "fit_model": model_key,
                    "fit_model_name": MODEL_NAMES[model_key],
                    "train_ll_joint": fit["train_ll_joint"],
                    "aic_joint": fit["train_aic_joint"],
                    "bic_joint": fit["train_bic_joint"],
                    "success": fit["success"],
                    "n_params_free": fit["n_params_free"],
                    "sim_press_rate": float(sim["press"].mean()),
                    "sim_fa_rate": float(sim.loc[sim["is_nogo"] == 1, "press"].mean()),
                    "sim_hit_rate": float(sim.loc[sim["is_nogo"] == 0, "press"].mean()),
                })

    res = pd.DataFrame(rows)
    if len(res) > 0:
        res["rank_aic"] = res.groupby(["config", "rep"])["aic_joint"].rank(method="first", ascending=True)
        res.to_csv(out / "model_recovery_results.csv", index=False)
        summ = (
            res.groupby(["config", "fit_model"])
            .agg(
                n_reps=("rep", "nunique"),
                mean_aic=("aic_joint", "mean"),
                mean_rank_aic=("rank_aic", "mean"),
                wins_aic=("rank_aic", lambda x: int((x == 1).sum())),
            )
            .reset_index()
            .sort_values(["config", "mean_rank_aic"])
        )
        summ.to_csv(out / "model_recovery_summary.csv", index=False)
    return res


# =============================================================================
# Figures and report
# =============================================================================

def make_figures(cv_summary: pd.DataFrame, ppc_summary: pd.DataFrame, out: Path) -> None:
    fig_dir = ensure_dir(out / "figures")

    if len(cv_summary) > 0:
        plot = cv_summary.copy()
        labels = plot["config"] + "\n" + plot["comparison"].str.replace("J2_minus_", "J2−")
        x = np.arange(len(plot))
        y = plot["mean_delta_test_elpd"].to_numpy(float)
        lo = plot["ci025_boot_mean_delta"].to_numpy(float)
        hi = plot["ci975_boot_mean_delta"].to_numpy(float)
        err = np.vstack([y - lo, hi - y])

        plt.figure(figsize=(max(8, 0.7 * len(plot)), 5.8))
        plt.axhline(0, linewidth=1)
        plt.errorbar(x, y, yerr=err, fmt="o", capsize=4)
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
        plt.ylabel("Held-out ΔELPD per fold, J2 minus control")
        plt.title("Supportive joint model: subject-wise CV actor–control advantage")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_joint_model_crossvalidated_delta_elpd.png", dpi=300)
        plt.close()

    if len(ppc_summary) > 0:
        plot = ppc_summary[(ppc_summary["group"] == "all") & (ppc_summary["model_key"] == "J2")].copy()
        if len(plot) > 0:
            x = np.arange(len(plot))
            plt.figure(figsize=(8, 4.8))
            plt.bar(x, plot["mean_rt_q50_abs_error_sec"] * 1000)
            plt.xticks(x, plot["config"], rotation=35, ha="right")
            plt.ylabel("Median RT absolute error (ms)")
            plt.title("Supportive joint model: J2 predictive check")
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_joint_model_predictive_check_rt_quantiles.png", dpi=300)
            plt.close()


def write_report(args, out: Path, cv_summary: pd.DataFrame, opt_summary: pd.DataFrame,
                 ppc_summary: pd.DataFrame, recovery: pd.DataFrame) -> None:
    lines = []
    lines.append("# Supportive joint-model validation report")
    lines.append("")
    lines.append(f"- Generated: {now_str()}")
    lines.append(f"- Latent file: `{args.latents}`")
    lines.append(f"- Configs: `{args.configs}`")
    lines.append(f"- Outer seeds: `{args.outer_seeds}`")
    lines.append(f"- Folds: `{args.n_folds}`")
    lines.append(f"- Starts per fit: `{args.n_starts}`")
    lines.append("")
    lines.append("## Interpretation guardrail")
    lines.append("")
    lines.append(
        "This branch is a exploratory joint reinforcement-learning/first-passage maximum-likelihood sensitivity analysis. "
        "It should not replace the primary Bayesian RL plus leakage-controlled nested DDM pipeline. "
        "If the J2 actor-control model consistently outperforms J0/J1 on held-out subjects, it supports "
        "the qualitative robustness of the actor-control account."
    )
    lines.append("")

    if len(cv_summary) > 0:
        lines.append("## Held-out CV summary")
        lines.append("")
        for _, r in cv_summary.iterrows():
            lines.append(
                f"- {r['config']} {r['comparison']}: mean ΔELPD/fold = "
                f"{r['mean_delta_test_elpd']:.3f}, 95% bootstrap CI "
                f"[{r['ci025_boot_mean_delta']:.3f}, {r['ci975_boot_mean_delta']:.3f}], "
                f"positive folds {int(r['positive_folds'])}/{int(r['n_folds_total'])}."
            )
        lines.append("")

    if len(ppc_summary) > 0:
        lines.append("## Predictive checks")
        lines.append("")
        lines.append("See `joint_model_predictive_check_summary.csv` and `figures/fig_joint_model_predictive_check_rt_quantiles.png`.")
        lines.append("")

    if len(recovery) > 0:
        lines.append("## Model recovery")
        lines.append("")
        lines.append("See `model_recovery_summary.csv`. A defensible result is J2 winning most/all AIC recoveries when data are simulated from J2.")
        lines.append("")

    lines.append("## Output files")
    lines.append("")
    for name in [
        "joint_model_crossvalidated_fold_assignments.csv",
        "joint_model_crossvalidated_results_by_fold.csv",
        "joint_model_crossvalidated_summary.csv",
        "optimizer_stability_by_fit.csv",
        "joint_model_predictive_check_summary.csv",
        "model_recovery_summary.csv",
    ]:
        if (out / name).exists():
            lines.append(f"- `{name}`")
    lines.append("- `figures/`")

    (out / "JOINT_MODEL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=".")
    p.add_argument("--ddm_script", default="shared/ddm/one_boundary_diffusion.py")
    p.add_argument("--latents", required=True)
    p.add_argument("--data_root", default="data/raw/csv")
    p.add_argument("--out", default="results/model_development/joint_model_supportive_validation")

    p.add_argument("--configs", nargs="+", default=["main"], choices=list(CONFIGS.keys()))
    p.add_argument("--outer_seeds", nargs="+", type=int, default=[2026, 7777])
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_starts", type=int, default=6)
    p.add_argument("--maxiter", type=int, default=500)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--action_weight", type=float, default=1.0)
    p.add_argument("--dist_min", type=float, default=0.05)
    p.add_argument("--sim_dt", type=float, default=0.004)

    p.add_argument("--ppc_sims", type=int, default=1, help="Simulated PPC replicates per J2 fold fit.")
    p.add_argument("--recovery_reps", type=int, default=5)
    p.add_argument("--recovery_configs", nargs="*", default=["main"])

    p.add_argument("--max_subjects", type=int, default=0, help="Debug only. 0 = all subjects.")
    p.add_argument("--skip_cv", action="store_true")
    p.add_argument("--skip_recovery", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(args.base).resolve()
    ddm_script = (base / args.ddm_script).resolve() if not Path(args.ddm_script).is_absolute() else Path(args.ddm_script)
    latents = (base / args.latents).resolve() if not Path(args.latents).is_absolute() else Path(args.latents)
    data_root = (base / args.data_root).resolve() if args.data_root and not Path(args.data_root).is_absolute() else Path(args.data_root)

    args.ddm_script = str(ddm_script)
    args.latents = str(latents)
    args.data_root = str(data_root) if args.data_root else None

    out = ensure_dir((base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out))
    print(f"[start] out={out}")
    print(f"[start] latents={latents}")
    print(f"[start] ddm_script={ddm_script}")

    ddm_mod = import_module_from_path(ddm_script, "ddm_for_joint_validation")

    cv = pd.DataFrame()
    opt = pd.DataFrame()
    ppc = pd.DataFrame()

    if not args.skip_cv:
        cv, starts, ppc = run_cv(args, ddm_mod, out)
        cv_summary = summarize_cv(cv, out)
        opt_summary = summarize_optimizer(starts, out)
        ppc_summary = summarize_ppc_table(ppc, out)
    else:
        cv_summary = pd.DataFrame()
        opt_summary = pd.DataFrame()
        ppc_summary = pd.DataFrame()

    if args.skip_recovery:
        recovery = pd.DataFrame()
    else:
        recovery = run_model_recovery(args, ddm_mod, out)

    make_figures(cv_summary, ppc_summary, out)
    write_report(args, out, cv_summary, opt_summary, ppc_summary, recovery)

    print(f"[done] outputs written to {out}")


if __name__ == "__main__":
    main()
