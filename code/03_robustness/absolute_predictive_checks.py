#!/usr/bin/env python3
"""Absolute predictive checks for the frozen primary 700-ms D6/D14/D15/D9 fits.

No refitting or model tuning is performed. The script reconstructs each held-out
fold exactly, loads the saved fit.json, and evaluates event-by-horizon calibration,
time-to-event calibration, and conditional RT quantiles.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

VARIANTS=['D6','D14','D15','D9']; TIMES_MS=[300,400,500,600,700]

def imp(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def safe_logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); return np.log(p/(1-p))

def calibration(y,p):
    y=np.asarray(y,float); x=safe_logit(p)
    def nll(ab):
        z=ab[0]+ab[1]*x; q=1/(1+np.exp(-np.clip(z,-40,40)))
        return -np.sum(y*np.log(np.clip(q,1e-12,1))+(1-y)*np.log(np.clip(1-q,1e-12,1)))
    r=minimize(nll,[0.,1.],method='BFGS')
    return float(r.x[0]),float(r.x[1])

def mixture_quantile(ddm,v,d,t0,deadline_s,q):
    Fd=ddm.firstpass_cdf(np.maximum(deadline_s-t0,0),d,v); denom=float(Fd.sum())
    if denom<=1e-12:return np.nan
    lo=max(0.0,float(np.nanmin(t0))); hi=deadline_s
    for _ in range(60):
        mid=(lo+hi)/2; F=ddm.firstpass_cdf(np.maximum(mid-t0,0),d,v); c=float(F.sum()/denom)
        if c<q: lo=mid
        else: hi=mid
    return 1000*(lo+hi)/2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--final-run',default=None); ap.add_argument('--out',default=None)
    a=ap.parse_args(); root=Path(a.project_root).expanduser().resolve()
    final=Path(a.final_run).expanduser().resolve() if a.final_run else root/'results/ddm/primary_fit_artifacts'
    out=Path(a.out).expanduser().resolve() if a.out else root/'reproduced/controlled_rerun/robustness/absolute_predictive_checks'; out.mkdir(parents=True,exist_ok=True)
    settings=json.loads((final/'analysis_settings.json').read_text())
    s54=imp(root/'code/02_primary_joint_response/joint_response_time_benchmark.py','h700v2')
    nested=imp(root/'shared/rl/nested_crossvalidated_latent_generation.py','nested'); rl=imp(root/'shared/rl/actor_control_model_family.py','rl'); ddm=imp(root/'shared/ddm/one_boundary_diffusion.py','ddm'); s54.register_variants(ddm)
    rl_out=root/'controlled_inputs/rl_crossvalidated_posteriors'; data_root=root/'controlled_inputs/behavioral_trial_data'; probs=s54.cue_probability_map(root/'configs/instructed_cue_probabilities.json'); arr=rl.load_artifacts(rl_out); meta=rl.get_meta(arr); all_idx=set(range(int(meta['n_subj'])))
    cv=nested.read_json(nested.cv_summary_path(rl_out,settings['seed'],settings['rl_variant'])); folds=cv['folds']
    predrows=[]; timerows=[]; rtrows=[]; subjrows=[]
    for fr in folds:
        fi=int(fr['fold']); test_idx=[int(x) for x in fr.get('test_subject_indices',[])]
        if not test_idx:
            fr2=nested.read_json(nested.fold_result_path(rl_out,settings['seed'],settings['rl_variant'],fi),default={}); test_idx=[int(x) for x in fr2['test_subject_indices']]
        train_idx=sorted(all_idx-set(test_idx)); idata=nested.load_rl_fold_idata(rl_out,settings['seed'],settings['rl_variant'],fi)
        atr=rl.subset_subjects(arr,train_idx); ate=rl.subset_subjects(arr,test_idx)
        ltr=nested.posterior_draws_mean_latents(rl,atr,settings['rl_variant'],idata,settings['max_rl_draws'],s54.stable_seed(settings['seed'],'ddm_train',fi))
        lte=nested.posterior_draws_mean_latents(rl,ate,settings['rl_variant'],idata,settings['max_rl_draws'],s54.stable_seed(settings['seed'],'ddm_test',fi))
        rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,settings['seed'],settings['rl_variant']); rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,settings['seed'],settings['rl_variant'])
        tr,_=s54.prepare_frame(ddm,rtr,data_root,700,100,700,probs,10.0); te,_=s54.prepare_frame(ddm,rte,data_root,700,100,700,probs,10.0)
        for vk in VARIANTS:
            fit=json.loads((final/'fits'/f'fold_{fi:02d}'/vk/'fit.json').read_text()); variant=ddm.VARIANTS[vk]
            Xv,Xd,Xt0,_=ddm.prepare_designs(tr,te,variant); theta=np.asarray(fit['theta'],float); v,d,t0=ddm.predict_components(theta,Xv,Xd,Xt0,settings['t0_min'],settings['t0_max'],settings['dist_min'])
            p700=ddm.firstpass_cdf(np.maximum(.700-t0,0),d,v); y=te['press'].to_numpy(int); ci,cs=calibration(y,p700)
            for i,row in te.reset_index(drop=True).iterrows():
                predrows.append({'fold':fi,'variant':vk,'subject_id':str(row['subj_id_ddm']),'trial_index':row.get('trial_index',np.nan),'is_nogo':int(row['is_nogo']),'cue':row.get('cue_label_ddm',''),
                                 'observed_event_700':int(y[i]),'predicted_event_700':float(p700[i]),'rt_observed_ms':float(row['rt_observed_sec']*1000) if np.isfinite(row['rt_observed_sec']) else np.nan})
            for tm in TIMES_MS:
                pp=ddm.firstpass_cdf(np.maximum(tm/1000-t0,0),d,v); obs=((te['press_observed'].to_numpy(int)==1)&(te['rt_observed_sec'].to_numpy(float)<=tm/1000)&np.isfinite(te['rt_observed_sec'].to_numpy(float))).astype(int)
                timerows.append({'fold':fi,'variant':vk,'time_ms':tm,'observed_event_rate':float(obs.mean()),'predicted_event_rate':float(pp.mean()),'brier':float(brier_score_loss(obs,pp))})
            for label,mask in [('overall',np.ones(len(te),bool)),('go',te['is_nogo'].to_numpy(int)==0),('nogo',te['is_nogo'].to_numpy(int)==1)]:
                obsrt=te.loc[mask & te['press'].eq(1),'rt_observed_sec'].to_numpy(float)*1000
                for q in [.1,.25,.5,.75,.9]:
                    rtrows.append({'fold':fi,'variant':vk,'stratum':label,'quantile':q,'observed_rt_ms':float(np.quantile(obsrt,q)) if len(obsrt) else np.nan,
                                   'predicted_rt_ms':mixture_quantile(ddm,v[mask],d[mask],t0[mask],.700,q)})
            sid_arr=te['subj_id_ddm'].astype(str).to_numpy()
            for sid in pd.unique(sid_arr):
                mask=sid_arr==sid; yy=y[mask]; pp=p700[mask]; evrt=te.loc[mask & te['press'].eq(1),'rt_observed_sec'].to_numpy(float)*1000
                subjrows.append({'fold':fi,'variant':vk,'subject_id':sid,'n_trials':int(mask.sum()),'observed_event_rate':float(yy.mean()),'predicted_event_rate':float(pp.mean()),'brier':float(brier_score_loss(yy,pp)),'median_observed_event_rt_ms':float(np.median(evrt)) if len(evrt) else np.nan,'median_predicted_event_rt_ms':mixture_quantile(ddm,v[mask],d[mask],t0[mask],.700,.5)})
            predrows.append({'fold':fi,'variant':vk,'subject_id':'__CALIBRATION__','trial_index':np.nan,'is_nogo':-1,'cue':'','observed_event_700':np.nan,'predicted_event_700':np.nan,'rt_observed_ms':np.nan,'calibration_intercept':ci,'calibration_slope':cs})
    pred=pd.DataFrame(predrows); pred.to_csv(out/'trial_predictions_and_calibration.csv.gz',index=False,compression='gzip')
    pd.DataFrame(timerows).to_csv(out/'time_to_event_calibration.csv',index=False); pd.DataFrame(rtrows).to_csv(out/'conditional_rt_quantiles.csv',index=False); pd.DataFrame(subjrows).to_csv(out/'subject_predictive_diagnostics.csv',index=False)
    # Decile calibration from real prediction rows only
    x=pred[pred.subject_id!='__CALIBRATION__'].copy(); dec=[]
    for vk,g in x.groupby('variant'):
        g=g.copy(); g['bin']=pd.qcut(g.predicted_event_700,10,labels=False,duplicates='drop')
        for b,z in g.groupby('bin'): dec.append({'variant':vk,'bin':int(b),'n':len(z),'mean_predicted':z.predicted_event_700.mean(),'mean_observed':z.observed_event_700.mean()})
    pd.DataFrame(dec).to_csv(out/'event_calibration_deciles.csv',index=False)
    overall=[]
    for vk,g in x.groupby('variant'):
        ci,cs=calibration(g.observed_event_700.to_numpy(int),g.predicted_event_700.to_numpy(float))
        overall.append({'variant':vk,'n_trials':len(g),'observed_event_rate':g.observed_event_700.mean(),'predicted_event_rate':g.predicted_event_700.mean(),'brier':brier_score_loss(g.observed_event_700,g.predicted_event_700),'calibration_intercept':ci,'calibration_slope':cs})
    pd.DataFrame(overall).to_csv(out/'overall_event_calibration.csv',index=False)
    sr=pd.DataFrame(subjrows); sr['median_rt_discrepancy_ms']=sr['median_predicted_event_rt_ms']-sr['median_observed_event_rt_ms']; sr.to_csv(out/'subject_predictive_diagnostics.csv',index=False)
    print('Wrote',out)
if __name__=='__main__': main()
