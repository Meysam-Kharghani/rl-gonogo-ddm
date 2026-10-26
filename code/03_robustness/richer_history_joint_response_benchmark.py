#!/usr/bin/env python3
"""behavioral-computational study post-development robustness: richer-history DDM decomposition.

Purpose
-------
Challenge the final D6/D14/D15/D9 decomposition with a stronger, strictly causal
behavioral-history baseline. Rich history is constructed on the COMPLETE
pre-filter trial sequence, before DDM-specific RT exclusions/censoring.

Primary variants
----------------
HR6  = rich history only
HR14 = rich history + actor_qdiff/control
HR15 = rich history + policy p_press
HR9  = rich history + policy + actor_qdiff/control

All 12 rich-history variables are allowed in BOTH drift and distance components.
This deliberately makes the generic behavioral baseline flexible and avoids
claiming that a particular history feature belongs uniquely to one DDM component.

The historical D6/D14/D15/D9 models remain the historical primary family; this is
a prospective robustness analysis.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def imp(path, name):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stable_seed(base, *parts):
    d = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int((base + int.from_bytes(d[:4], "little")) % (2**32 - 1))


def safe_logit(x, eps=1e-6):
    z = np.clip(np.asarray(x, float), eps, 1-eps)
    return np.log(z/(1-z))


RICH_RAW = [
    "press_lag2","press_lag3","press_lag4","press_lag5",
    "error_lag2","error_lag3","error_lag4","error_lag5",
    "press_ewma_a10","press_ewma_a30","fa_ewma_a10","fa_ewma_a30",
]
RICH_Z = [x + "_z" for x in RICH_RAW]


def add_rich_history_complete(d):
    """Add strictly lagged history using original behavior on the full sequence."""
    d = d.copy()
    trial = "trial_index" if "trial_index" in d.columns else "trial_index_numeric"
    if trial not in d.columns:
        d["trial_index_numeric"] = d.groupby("subj_id_ddm").cumcount() + 1
        trial = "trial_index_numeric"
    d = d.sort_values(["subj_id_ddm", trial]).reset_index(drop=True)

    # IMPORTANT: this is called before DDM recoding, so `press` is the original action.
    press = pd.to_numeric(d["press"], errors="coerce").fillna(0).astype(float)
    error = pd.to_numeric(d.get("error", 0), errors="coerce").fillna(0).astype(float)
    nogo = pd.to_numeric(d["is_nogo"], errors="coerce").fillna(0).astype(int)

    for k in [2,3,4,5]:
        d[f"press_lag{k}"] = press.groupby(d["subj_id_ddm"]).shift(k).fillna(0.0)
        d[f"error_lag{k}"] = error.groupby(d["subj_id_ddm"]).shift(k).fillna(0.0)

    for alpha in [.10, .30]:
        p_out = np.zeros(len(d), float)
        fa_out = np.zeros(len(d), float)
        for _, g in d.groupby("subj_id_ddm", sort=False):
            idx = g.index.to_numpy()
            p = press.loc[idx].to_numpy()
            e = error.loc[idx].to_numpy()
            ng = nogo.loc[idx].to_numpy()
            mp, mf = .5, .2
            for j, row_idx in enumerate(idx):
                p_out[row_idx] = mp
                fa_out[row_idx] = mf
                mp = (1-alpha)*mp + alpha*p[j]
                if ng[j] == 1:
                    fa = 1.0 if e[j] > 0 else 0.0
                    mf = (1-alpha)*mf + alpha*fa
        d[f"press_ewma_a{int(alpha*100)}"] = p_out
        d[f"fa_ewma_a{int(alpha*100)}"] = fa_out
    return d


def prepare_frame_rich(h54, ddm, rows, data_root, deadline_ms, rt_min_ms, probs, prior_strength):
    """Copy the 700-ms fixed-horizon semantics, inserting rich history before RT filtering."""
    with tempfile.TemporaryDirectory(prefix="rich_ddm_latents_") as td:
        fp = Path(td) / "latents.csv"
        pd.DataFrame(rows).to_csv(fp, index=False)
        raw = ddm.load_latents(fp, deadline_ms=deadline_ms, data_root=Path(data_root))

    raw = h54.add_beliefs(raw.reset_index(drop=True), probs, prior_strength)
    raw = add_rich_history_complete(raw)

    press_obs = pd.to_numeric(raw["press"], errors="coerce").fillna(0).astype(int)
    rt_obs = pd.to_numeric(raw["rt_sec"], errors="coerce")
    rt_min_sec = float(rt_min_ms)/1000.0
    deadline_sec = float(deadline_ms)/1000.0

    bad = (press_obs.eq(1) & (~np.isfinite(rt_obs) | (rt_obs < rt_min_sec)))
    d = raw.loc[~bad].copy().reset_index(drop=True)

    d["press_observed"] = pd.to_numeric(d["press"], errors="coerce").fillna(0).astype(int)
    d["rt_observed_sec"] = pd.to_numeric(d["rt_sec"], errors="coerce")
    obs = d["press_observed"].eq(1)
    finite = np.isfinite(d["rt_observed_sec"])
    ontime = obs & finite & (d["rt_observed_sec"] >= rt_min_sec) & (d["rt_observed_sec"] <= deadline_sec)
    late = obs & finite & (d["rt_observed_sec"] > deadline_sec)

    d["press"] = ontime.astype(int)
    d["rt_sec"] = np.where(ontime, d["rt_observed_sec"], np.nan)
    d["ddm_event_by_deadline"] = d["press"].astype(int)
    d["ddm_right_censored"] = (~ontime).astype(int)
    d["ddm_late_press_censored"] = late.astype(int)

    info = {
        "n_raw": int(len(raw)), "n_kept": int(len(d)), "n_removed": int(bad.sum()),
        "n_late_press_right_censored": int(late.sum()),
        "rich_history_computed_pre_filter": True,
        "deadline_ms": float(deadline_ms), "rt_min_ms": float(rt_min_ms),
    }
    return d, info


def register_rich_variants(ddm):
    V = ddm.DDMVariant
    for raw, z in zip(RICH_RAW, RICH_Z):
        ddm.CONTINUOUS_SOURCE_MAP[z] = raw

    base_drift = ("intercept","is_nogo","cue_up","cue_bi","group_high",
                  "prev_press","prior_press_rate_z","prior_fa_rate_z")
    base_dist = ("intercept","is_nogo","cue_up","cue_bi","group_high",
                 "prev_error","prev_fa","prior_press_rate_z","prior_fa_rate_z")
    # Strong generic baseline: every new history summary may contribute to both
    # drift and distance. This avoids relying on an unverified component mapping.
    hd = base_drift + tuple(RICH_Z)
    hb = base_dist + tuple(RICH_Z)

    ddm.VARIANTS.update({
        "HR6": V("HR6","rich_history", hd, hb),
        "HR14": V("HR14","rich_history_plus_actor_control",
                  hd + ("actor_qdiff_z",), hb + ("control_z",)),
        "HR15": V("HR15","rich_history_plus_policy",
                  hd + ("p_press_z",), hb),
        "HR9": V("HR9","rich_history_plus_full_rl",
                 hd + ("actor_qdiff_z","p_press_z"), hb + ("control_z",)),
    })


def boot(x, n, seed):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.quantile(b,.025)), float(np.quantile(b,.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-rl-draws", type=int, default=300)
    ap.add_argument("--deadline-ms", type=float, default=700.0)
    ap.add_argument("--rt-min-ms", type=float, default=100.0)
    ap.add_argument("--prior-strength", type=float, default=10.0)
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    a = ap.parse_args()

    root = Path(a.project_root).expanduser().resolve()
    out = Path(a.out).expanduser().resolve() if a.out else root/"reproduced/controlled_rerun/robustness/richer_history_joint_response"
    out.mkdir(parents=True, exist_ok=True)

    nested = imp(root/"shared/rl/nested_crossvalidated_latent_generation.py","rhddm_nested")
    rl = imp(root/"shared/rl/actor_control_model_family.py","rhddm_rl")
    ddm = imp(root/"shared/ddm/one_boundary_diffusion.py","rhddm_ddm")
    h54 = imp(root/"code/02_primary_joint_response/joint_response_time_benchmark.py","rhddm_h54")
    h54.register_variants(ddm)
    register_rich_variants(ddm)

    rlout = root/"controlled_inputs/rl_crossvalidated_posteriors"
    data_root = root/"controlled_inputs/behavioral_trial_data"
    probs = h54.cue_probability_map(root/"configs/instructed_cue_probabilities.json")
    arr = rl.load_artifacts(rlout)
    meta = rl.get_meta(arr)
    all_idx = set(range(int(meta["n_subj"])))
    cv = nested.read_json(nested.cv_summary_path(rlout,a.seed,"M6"))

    variants = ["HR6","HR14","HR15","HR9"]
    fold_rows=[]; subj_rows=[]; fit_rows=[]; qc_rows=[]

    for fr in cv["folds"]:
        fi = int(fr["fold"])
        test_idx = [int(x) for x in fr.get("test_subject_indices",[])]
        if not test_idx:
            fr2 = nested.read_json(nested.fold_result_path(rlout,a.seed,"M6",fi), default={})
            test_idx = [int(x) for x in fr2.get("test_subject_indices",[])]
        train_idx = sorted(all_idx-set(test_idx))
        print(f"[fold {fi}] train={len(train_idx)} test={len(test_idx)}", flush=True)

        idata = nested.load_rl_fold_idata(rlout,a.seed,"M6",fi)
        atr, ate = rl.subset_subjects(arr,train_idx), rl.subset_subjects(arr,test_idx)
        ltr = nested.posterior_draws_mean_latents(rl,atr,"M6",idata,a.max_rl_draws,stable_seed(a.seed,"rhddm_tr",fi))
        lte = nested.posterior_draws_mean_latents(rl,ate,"M6",idata,a.max_rl_draws,stable_seed(a.seed,"rhddm_te",fi))
        rtr = nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,a.seed,"M6")
        rte = nested.arr_to_nested_latent_rows(rl,ate,lte,fi,a.seed,"M6")

        tr,qtr = prepare_frame_rich(h54,ddm,rtr,data_root,a.deadline_ms,a.rt_min_ms,probs,a.prior_strength)
        te,qte = prepare_frame_rich(h54,ddm,rte,data_root,a.deadline_ms,a.rt_min_ms,probs,a.prior_strength)
        qc_rows += [{"fold":fi,"split":"train",**qtr},{"fold":fi,"split":"test",**qte}]

        for vk in variants:
            var=ddm.VARIANTS[vk]
            print(f"[fold {fi}] fit {vk} p={len(var.drift_cols)+len(var.dist_cols)+len(var.t0_cols)}",flush=True)
            fit=ddm.fit_one_model(
                tr,var,deadline_ms=a.deadline_ms,n_starts=a.n_starts,
                seed=stable_seed(a.seed,"rhddm_fit",fi,vk),l2=a.l2,
                t0_min=.03,t0_max=.03,dist_min=.05
            )
            fdir=out/"fits"/f"fold_{fi:02d}"/vk
            fdir.mkdir(parents=True,exist_ok=True)
            (fdir/"fit.json").write_text(json.dumps(fit,indent=2),encoding="utf-8")

            sc=ddm.score_model(te,tr,fit,.03,.03,.05)
            fold_rows.append({"fold":fi,"variant":vk,"variant_name":var.name,
                              "n_test_trials":len(te),"elpd":sc["elpd"],
                              "elpd_per_trial":sc["elpd"]/len(te),**sc["ppc"]})
            hist=fit.get("optimizer_history",[])
            fit_rows.append({"fold":fi,"variant":vk,"success":fit.get("success"),
                             "nll_train":fit.get("nll"),"n_params":fit.get("n_params"),
                             "n_successful_starts":sum(bool(x.get("success")) for x in hist),
                             "n_starts":len(hist)})
            for sid,g in te.groupby("subj_id_ddm",sort=False):
                ss=ddm.score_model(g.reset_index(drop=True),tr,fit,.03,.03,.05)
                subj_rows.append({"fold":fi,"subject_id":str(sid),"variant":vk,
                                  "n_trials":len(g),"elpd":ss["elpd"]})

    f=pd.DataFrame(fold_rows); s=pd.DataFrame(subj_rows)
    f.to_csv(out/"rich_ddm_fold_scores.csv",index=False)
    s.to_csv(out/"rich_ddm_subject_scores.csv",index=False)
    pd.DataFrame(fit_rows).to_csv(out/"rich_ddm_optimizer_summary.csv",index=False)
    pd.DataFrame(qc_rows).to_csv(out/"rich_ddm_filter_qc.csv",index=False)

    planned=[
        ("HR14","HR6","actor/control beyond rich history"),
        ("HR15","HR6","policy beyond rich history"),
        ("HR9","HR14","policy beyond actor/control + rich history"),
        ("HR9","HR15","actor/control beyond policy + rich history"),
        ("HR9","HR6","total RL beyond rich history"),
    ]
    wide=s.pivot_table(index=["fold","subject_id"],columns="variant",values="elpd")
    cr=[]
    for i,(m,r,label) in enumerate(planned):
        x=(wide[m]-wide[r]).dropna().to_numpy()
        mean,lo,hi=boot(x,a.n_bootstrap,stable_seed(a.seed,"contrast",i))
        cr.append({"model":m,"reference":r,"label":label,"n_subjects":len(x),
                   "mean_subject_delta_elpd":mean,"ci_low":lo,"ci_high":hi,
                   "positive_subjects":int((x>0).sum()),"total_delta_elpd":float(x.sum())})
    c=pd.DataFrame(cr)
    c.to_csv(out/"rich_ddm_planned_contrasts.csv",index=False)

    settings={
        "status":"post-development robustness analysis",
        "seed":a.seed,"max_rl_draws":a.max_rl_draws,"deadline_ms":a.deadline_ms,
        "rt_min_ms":a.rt_min_ms,"prior_strength":a.prior_strength,
        "n_starts":a.n_starts,"l2":a.l2,
        "rich_history_features":RICH_RAW,
        "rich_history_mapping":"all 12 standardized rich-history covariates enter both drift and distance",
        "rich_history_timing":"computed on complete pre-filter original behavioral sequence",
        "variants":variants,"planned_contrasts":planned
    }
    (out/"analysis_settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    (out/"RICHER_HISTORY_DDM_REPORT.md").write_text(
        "# Richer-history joint-response robustness analysis\n\n## Planned contrasts\n\n"+c.to_markdown(index=False)+
        "\n\nInterpretation: this is a prospective robustness family. Historical D6/D14/D15/D9 remain the historical primary models.\n",
        encoding="utf-8")
    print(c.to_string(index=False))
    print("Wrote",out)


if __name__=="__main__":
    main()
