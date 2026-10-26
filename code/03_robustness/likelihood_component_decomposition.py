#!/usr/bin/env python3
"""behavioral-computational study post-development robustness: stratified held-out DDM likelihood decomposition.

Uses the already-frozen primary 700-ms fits generated with 300 posterior draws per fold (no refitting) and decomposes
pointwise held-out log likelihood by:
- Go vs NoGo
- observed event vs right-censored
- their 2x2 combinations
- censor subtype (event / late-press-censored / original no-response)

This tests whether D9/D15 gains are broadly distributed or concentrated in the
NoGo/censored strata where absolute RT adequacy is weaker.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd


def imp(path,name):
    path=Path(path).expanduser().resolve()
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod


def stable_seed(base,*parts):
    d=hashlib.sha256("|".join(map(str,parts)).encode()).digest()
    return int((base+int.from_bytes(d[:4],"little"))%(2**32-1))


def pointwise_ll(ddm, test, train, fit, t0_min=.03, t0_max=.03, dist_min=.05):
    var=ddm.VARIANTS[fit["variant"]]
    Xv,Xd,Xt0,_=ddm.prepare_designs(train,test,var)
    theta=np.asarray(fit["theta"],float)
    v,d,t0=ddm.predict_components(theta,Xv,Xd,Xt0,t0_min,t0_max,dist_min)
    press=test["press"].to_numpy(int).astype(bool)
    rt=test["rt_sec"].to_numpy(float)
    deadline=test["deadline_sec"].to_numpy(float)
    ll=np.zeros(len(test),float)
    if press.any():
        td=rt[press]-t0[press]
        ll[press]=ddm.firstpass_logpdf(td,d[press],v[press])
    nr=~press
    if nr.any():
        T=deadline[nr]-t0[nr]
        F=ddm.firstpass_cdf(T,d[nr],v[nr])
        ll[nr]=np.log(np.clip(1-F,ddm.EPS,1))
    if not np.all(np.isfinite(ll)):
        raise ValueError("Non-finite pointwise log likelihood")
    return ll


def boot(x,n,seed):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed)
    b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)])
    return float(x.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--final-fit-root",default=None)
    ap.add_argument("--out",default=None)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--max-rl-draws",type=int,default=300)
    ap.add_argument("--n-bootstrap",type=int,default=10000)
    a=ap.parse_args()

    root=Path(a.project_root).expanduser().resolve()
    fitroot=Path(a.final_fit_root).expanduser().resolve() if a.final_fit_root else root/"results/ddm/primary_fit_artifacts"
    out=Path(a.out).expanduser().resolve() if a.out else root/"reproduced/controlled_rerun/robustness/likelihood_component_decomposition"
    out.mkdir(parents=True,exist_ok=True)

    nested=imp(root/"shared/rl/nested_crossvalidated_latent_generation.py","sll_nested")
    rl=imp(root/"shared/rl/actor_control_model_family.py","sll_rl")
    ddm=imp(root/"shared/ddm/one_boundary_diffusion.py","sll_ddm")
    h54=imp(root/"code/02_primary_joint_response/joint_response_time_benchmark.py","sll_h54")
    h54.register_variants(ddm)

    rlout=root/"controlled_inputs/rl_crossvalidated_posteriors"
    data_root=root/"controlled_inputs/behavioral_trial_data"
    probs=h54.cue_probability_map(root/"configs/instructed_cue_probabilities.json")
    arr=rl.load_artifacts(rlout); meta=rl.get_meta(arr); all_idx=set(range(int(meta["n_subj"])))
    cv=nested.read_json(nested.cv_summary_path(rlout,a.seed,"M6"))
    variants=["D6","D14","D15","D9"]

    trial_rows=[]
    for fr in cv["folds"]:
        fi=int(fr["fold"]); test_idx=[int(x) for x in fr.get("test_subject_indices",[])]
        if not test_idx:
            fr2=nested.read_json(nested.fold_result_path(rlout,a.seed,"M6",fi),default={})
            test_idx=[int(x) for x in fr2.get("test_subject_indices",[])]
        train_idx=sorted(all_idx-set(test_idx))
        idata=nested.load_rl_fold_idata(rlout,a.seed,"M6",fi)
        atr,ate=rl.subset_subjects(arr,train_idx),rl.subset_subjects(arr,test_idx)
        ltr=nested.posterior_draws_mean_latents(rl,atr,"M6",idata,a.max_rl_draws,stable_seed(a.seed,"ddm_train",fi))
        lte=nested.posterior_draws_mean_latents(rl,ate,"M6",idata,a.max_rl_draws,stable_seed(a.seed,"ddm_test",fi))
        rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,a.seed,"M6")
        rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,a.seed,"M6")
        tr,_=h54.prepare_frame(ddm,rtr,data_root,700,100,700,probs,10.0)
        te,_=h54.prepare_frame(ddm,rte,data_root,700,100,700,probs,10.0)

        # Stable trial-level identifiers for merging models.
        trial_col="trial_index" if "trial_index" in te.columns else "trial_index_numeric"
        for vk in variants:
            fp=fitroot/"fits"/f"fold_{fi:02d}"/vk/"fit.json"
            if not fp.exists():
                raise FileNotFoundError(fp)
            fit=json.loads(fp.read_text())
            ll=pointwise_ll(ddm,te,tr,fit)
            for j,row in te.reset_index(drop=True).iterrows():
                event=int(row["press"])
                late=int(row.get("ddm_late_press_censored",0))
                if event:
                    ctype="event"
                elif late:
                    ctype="late_press_censored"
                else:
                    ctype="original_no_response"
                trial_rows.append({
                    "fold":fi,"subject_id":str(row["subj_id_ddm"]),
                    "trial_index":row.get(trial_col,j),
                    "variant":vk,"loglik":float(ll[j]),
                    "target_class":"NoGo" if int(row["is_nogo"])==1 else "Go",
                    "likelihood_type":"event" if event else "censored",
                    "censor_type":ctype,
                    "late_press_censored":late,
                })

    trials=pd.DataFrame(trial_rows)
    trials.to_csv(out/"stratified_trial_loglik.csv",index=False)

    # Per-subject sums within strata preserve participant as bootstrap unit.
    grouping_defs={
        "overall":[],
        "target_class":["target_class"],
        "likelihood_type":["likelihood_type"],
        "target_x_likelihood":["target_class","likelihood_type"],
        "censor_type":["censor_type"],
    }
    subject_parts=[]
    for family,cols in grouping_defs.items():
        keys=["fold","subject_id","variant"]+cols
        z=trials.groupby(keys,as_index=False).agg(elpd=("loglik","sum"),n_trials=("loglik","size"))
        z["stratum_family"]=family
        if cols:
            z["stratum"]=" | ".join(cols)  # placeholder overwritten below
            z["stratum"]=z[cols].astype(str).agg(" | ".join,axis=1)
        else:
            z["stratum"]="ALL"
        subject_parts.append(z[["fold","subject_id","variant","stratum_family","stratum","elpd","n_trials"]])
    subj=pd.concat(subject_parts,ignore_index=True)
    subj.to_csv(out/"stratified_subject_elpd.csv",index=False)

    planned=[
        ("D14","D6","actor/control beyond history"),
        ("D15","D6","policy beyond history"),
        ("D9","D14","policy beyond actor/control+history"),
        ("D9","D15","actor/control beyond policy+history"),
        ("D9","D6","total RL beyond history"),
    ]
    cr=[]
    for fam in subj.stratum_family.unique():
        for stratum in subj.loc[subj.stratum_family==fam,"stratum"].unique():
            d=subj[(subj.stratum_family==fam)&(subj.stratum==stratum)]
            wide=d.pivot_table(index=["fold","subject_id"],columns="variant",values="elpd",aggfunc="sum")
            for i,(m,r,label) in enumerate(planned):
                if m not in wide or r not in wide: continue
                x=(wide[m]-wide[r]).dropna().to_numpy()
                mean,lo,hi=boot(x,a.n_bootstrap,stable_seed(a.seed,fam,stratum,i))
                cr.append({"stratum_family":fam,"stratum":stratum,"model":m,"reference":r,
                           "label":label,"n_subjects":len(x),"mean_subject_delta_elpd":mean,
                           "ci_low":lo,"ci_high":hi,"positive_subjects":int((x>0).sum()),
                           "total_delta_elpd":float(x.sum())})
    contrasts=pd.DataFrame(cr)
    contrasts.to_csv(out/"stratified_ddm_contrasts.csv",index=False)

    # Share of total contrast gain attributable to each target×likelihood stratum.
    tx=contrasts[contrasts.stratum_family=="target_x_likelihood"].copy()
    overall=contrasts[contrasts.stratum_family=="overall"][["model","reference","total_delta_elpd"]].rename(columns={"total_delta_elpd":"overall_total_delta"})
    tx=tx.merge(overall,on=["model","reference"],how="left")
    tx["share_of_overall_delta"]=tx["total_delta_elpd"]/tx["overall_total_delta"].replace(0,np.nan)
    tx.to_csv(out/"target_likelihood_contrast_decomposition.csv",index=False)

    settings={
        "status":"no-refit diagnostic using frozen primary 700-ms fits generated with 300 posterior draws per fold",
        "fit_root":str(fitroot),"seed":a.seed,"max_rl_draws":a.max_rl_draws,
        "variants":variants,"strata":list(grouping_defs),"n_bootstrap":a.n_bootstrap,
    }
    (out/"analysis_settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    (out/"STRATIFIED_DDM_LIKELIHOOD_REPORT.md").write_text(
        "# Stratified DDM held-out likelihood decomposition\n\n"
        "This analysis does not refit or tune any DDM. It decomposes held-out pointwise log likelihood from the frozen final fits.\n\n"
        "## Target × likelihood contrasts\n\n"+tx.to_markdown(index=False),
        encoding="utf-8")
    print(tx.to_string(index=False))
    print("Wrote",out)


if __name__=="__main__":
    main()
