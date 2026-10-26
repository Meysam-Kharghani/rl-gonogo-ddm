#!/usr/bin/env python3
"""Richer strictly lagged non-RL history baseline for the action-occurrence result.

Tests whether actor/control increment survives a more flexible behavioral-history
benchmark. All added history variables are constructed from trials strictly before
the current trial. No DDM or EEG outcome information is used.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

def imp(p,n):
 s=importlib.util.spec_from_file_location(n,str(p)); m=importlib.util.module_from_spec(s); sys.modules[n]=m; s.loader.exec_module(m); return m

def add_history(d):
 d=d.copy(); trial='trial_index' if 'trial_index' in d.columns else 'trial_index_numeric'; d=d.sort_values(['subj_id_ddm',trial]).reset_index(drop=True); press=pd.to_numeric(d['press'],errors='coerce').fillna(0).astype(float); err=pd.to_numeric(d['error'] if 'error' in d.columns else pd.Series(0,index=d.index),errors='coerce').fillna(0).astype(float); nogo=pd.to_numeric(d['is_nogo'],errors='coerce').fillna(0).astype(int)
 for k in [2,3,4,5]: d[f'press_lag{k}']=press.groupby(d.subj_id_ddm).shift(k).fillna(0); d[f'error_lag{k}']=err.groupby(d.subj_id_ddm).shift(k).fillna(0)
 for alpha in [.10,.30]:
  vals=[]; fav=[]
  for sid,g in d.groupby('subj_id_ddm',sort=False):
   idx=g.index; p=press.loc[idx].to_numpy(); e=err.loc[idx].to_numpy(); ng=nogo.loc[idx].to_numpy(); ep=np.zeros(len(g)); ef=np.zeros(len(g)); mp=.5; mf=.2
   for i in range(len(g)):
    ep[i]=mp; ef[i]=mf; mp=(1-alpha)*mp+alpha*p[i]
    if ng[i]==1: mf=(1-alpha)*mf+alpha*(1.0 if e[i]>0 else 0.0)
   vals.extend(zip(idx,ep)); fav.extend(zip(idx,ef))
  for idx,v in vals:d.loc[idx,f'press_ewma_a{int(alpha*100)}']=v
  for idx,v in fav:d.loc[idx,f'fa_ewma_a{int(alpha*100)}']=v
 return d

def boot(x,n,seed):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); b=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)]); return x.mean(),np.quantile(b,.025),np.quantile(b,.975)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--out',default=None); ap.add_argument('--max-rl-draws',type=int,default=300); ap.add_argument('--seed',type=int,default=2026); ap.add_argument('--n-bootstrap',type=int,default=10000)
 a=ap.parse_args(); root=Path(a.project_root).resolve(); out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/richer_history_action'; out.mkdir(parents=True,exist_ok=True)
 act=imp(root/'code/01_primary_action/action_occurrence_benchmark.py','act'); nested=imp(root/'shared/rl/nested_crossvalidated_latent_generation.py','nested'); rl=imp(root/'shared/rl/actor_control_model_family.py','rl'); ddm=imp(root/'shared/ddm/one_boundary_diffusion.py','ddm')
 rlout=root/'controlled_inputs/rl_crossvalidated_posteriors'; probs=act.cue_map(root/'configs/instructed_cue_probabilities.json'); arr=rl.load_artifacts(rlout); meta=rl.get_meta(arr); allidx=set(range(int(meta['n_subj']))); cv=nested.read_json(nested.cv_summary_path(rlout,a.seed,'M6')); rows=[]; sub=[]; fitrows=[]
 base=['is_nogo','cue_up','cue_bi','prev_error']; current_hist=['prev_press','prior_press_rate','prior_fa_rate']; rich=['press_lag2','press_lag3','press_lag4','press_lag5','error_lag2','error_lag3','error_lag4','error_lag5','press_ewma_a10','press_ewma_a30','fa_ewma_a10','fa_ewma_a30']; AC=['actor_qdiff_current','control_current']; P=['logit_p_press']
 specs={'compact_lagged_history':base+current_hist,'richer_lagged_history':base+current_hist+rich,'richer_lagged_history_plus_actor_control':base+current_hist+rich+AC,'richer_lagged_history_plus_policy':base+current_hist+rich+P,'richer_lagged_history_plus_full_rl':base+current_hist+rich+P+AC}
 Cs=[.0003,.001,.003,.01,.03,.1,.3,1,3,10]
 for fr in cv['folds']:
  fi=int(fr['fold']); test=[int(x) for x in fr.get('test_subject_indices',[])];
  if not test:
   fr2=nested.read_json(nested.fold_result_path(rlout,a.seed,'M6',fi),default={}); test=[int(x) for x in fr2.get('test_subject_indices',[])]
  train=sorted(allidx-set(test)); idata=nested.load_rl_fold_idata(rlout,a.seed,'M6',fi); atr=rl.subset_subjects(arr,train); ate=rl.subset_subjects(arr,test); ltr=nested.posterior_draws_mean_latents(rl,atr,'M6',idata,a.max_rl_draws,act.stable_seed(a.seed,'rhtr',fi)); lte=nested.posterior_draws_mean_latents(rl,ate,'M6',idata,a.max_rl_draws,act.stable_seed(a.seed,'rhte',fi)); rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,a.seed,'M6'); rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,a.seed,'M6'); tr,_=act.prepare_frame(ddm,rtr,root/'controlled_inputs/behavioral_trial_data',700,'all',100,700,probs,10.0); te,_=act.prepare_frame(ddm,rte,root/'controlled_inputs/behavioral_trial_data',700,'all',100,700,probs,10.0); tr['logit_p_press']=act.safe_logit(tr.p_press); te['logit_p_press']=act.safe_logit(te.p_press); tr=add_history(tr); te=add_history(te)
  for name,cols in specs.items():
   C=act.tune_C(tr,cols,Cs,4,act.stable_seed(a.seed,'rhC',fi,name)); model=act.make_model(C,act.stable_seed(a.seed,'rhfit',fi,name)); model.fit(tr[cols],tr.press.astype(int)); p=model.predict_proba(te[cols])[:,1]; met=act.metrics(te.press.astype(int),p); rows.append({'fold':fi,'model':name,'C':C,**met}); fitrows.append({'fold':fi,'model':name,'C':C})
   lp=te.press.to_numpy()*np.log(np.clip(p,1e-9,1))+(1-te.press.to_numpy())*np.log(np.clip(1-p,1e-9,1))
   for sid,z in pd.DataFrame({'sid':te.subj_id_ddm.astype(str),'lp':lp}).groupby('sid'): sub.append({'fold':fi,'subject_id':sid,'model':name,'elpd':z.lp.sum()})
 f=pd.DataFrame(rows); s=pd.DataFrame(sub); f.to_csv(out/'richer_history_fold_scores.csv',index=False); s.to_csv(out/'richer_history_subject_scores.csv',index=False); pd.DataFrame(fitrows).to_csv(out/'richer_history_tuning.csv',index=False)
 comps=[('richer_lagged_history','compact_lagged_history'),('richer_lagged_history_plus_actor_control','richer_lagged_history'),('richer_lagged_history_plus_policy','richer_lagged_history'),('richer_lagged_history_plus_full_rl','richer_lagged_history'),('richer_lagged_history_plus_full_rl','richer_lagged_history_plus_policy')]; w=s.pivot_table(index=['fold','subject_id'],columns='model',values='elpd'); cr=[]
 for i,(x,y) in enumerate(comps):
  z=(w[x]-w[y]).dropna().to_numpy(); m,lo,hi=boot(z,a.n_bootstrap,a.seed+i); cr.append({'model':x,'reference':y,'n_subjects':len(z),'mean_subject_delta_elpd':m,'ci_low':lo,'ci_high':hi,'positive_subjects':int((z>0).sum()),'total_delta_elpd':z.sum()})
 pd.DataFrame(cr).to_csv(out/'richer_history_planned_contrasts.csv',index=False); (out/'analysis_settings.json').write_text(json.dumps({'history_features':rich,'C_grid':Cs,'strictly_lagged':True,'max_rl_draws':a.max_rl_draws,'planned_contrasts':comps},indent=2),encoding='utf-8'); print('Wrote',out)
if __name__=='__main__': main()
