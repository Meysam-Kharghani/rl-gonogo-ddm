#!/usr/bin/env python3
"""Claim-dependent DDM component-mapping sensitivity at the final 700-ms estimand.

Compares current D14/D15/D9 with prespecified alternative RL-to-drift/distance
mappings. This is a sensitivity analysis, not a new model-selection search.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd

def imp(path,name):
 spec=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def boot(x,n,seed):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); bm=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)]); return x.mean(),np.quantile(bm,.025),np.quantile(bm,.975)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2])); ap.add_argument('--out',default=None); ap.add_argument('--max-rl-draws',type=int,default=100); ap.add_argument('--n-starts',type=int,default=6); ap.add_argument('--n-bootstrap',type=int,default=10000); ap.add_argument('--seed',type=int,default=2026)
 a=ap.parse_args(); root=Path(a.project_root).resolve(); out=Path(a.out).resolve() if a.out else root/'reproduced/controlled_rerun/robustness/predictor_mapping_sensitivity'; out.mkdir(parents=True,exist_ok=True)
 s=imp(root/'code/02_primary_joint_response/joint_response_time_benchmark.py','s54'); nested=imp(root/'shared/rl/nested_crossvalidated_latent_generation.py','nested'); rl=imp(root/'shared/rl/actor_control_model_family.py','rl'); ddm=imp(root/'shared/ddm/one_boundary_diffusion.py','ddm'); s.register_variants(ddm); V=ddm.DDMVariant
 # Alternatives preserve history terms and change only RL component loading.
 ddm.VARIANTS.update({
  'D14S':V('D14S','actor_control_history_swapped',('intercept','is_nogo','cue_up','cue_bi','group_high','control_z','prev_press','prior_press_rate_z','prior_fa_rate_z'),('intercept','is_nogo','cue_up','cue_bi','group_high','prev_error','prev_fa','actor_qdiff_z','prior_press_rate_z','prior_fa_rate_z')),
  'D15S':V('D15S','policy_history_policy_in_distance',('intercept','is_nogo','cue_up','cue_bi','group_high','prev_press','prior_press_rate_z','prior_fa_rate_z'),('intercept','is_nogo','cue_up','cue_bi','group_high','prev_error','prev_fa','p_press_z','prior_press_rate_z','prior_fa_rate_z')),
  'D9S':V('D9S','full_hybrid_swapped_actor_control',('intercept','is_nogo','cue_up','cue_bi','group_high','p_press_z','control_z','prev_press','prior_press_rate_z','prior_fa_rate_z'),('intercept','is_nogo','cue_up','cue_bi','group_high','prev_error','prev_fa','actor_qdiff_z','prior_press_rate_z','prior_fa_rate_z')),
  'D9X':V('D9X','full_hybrid_crossloaded',('intercept','is_nogo','cue_up','cue_bi','group_high','actor_qdiff_z','p_press_z','control_z','prev_press','prior_press_rate_z','prior_fa_rate_z'),('intercept','is_nogo','cue_up','cue_bi','group_high','prev_error','prev_fa','control_z','actor_qdiff_z','p_press_z','prior_press_rate_z','prior_fa_rate_z'))})
 variants=['D6','D14','D14S','D15','D15S','D9','D9S','D9X']; rlout=root/'controlled_inputs/rl_crossvalidated_posteriors'; probs=s.cue_probability_map(root/'configs/instructed_cue_probabilities.json'); arr=rl.load_artifacts(rlout); meta=rl.get_meta(arr); allidx=set(range(int(meta['n_subj']))); cv=nested.read_json(nested.cv_summary_path(rlout,a.seed,'M6'))
 subj=[]; folds=[]
 for fr in cv['folds']:
  fi=int(fr['fold']); test=[int(x) for x in fr.get('test_subject_indices',[])];
  if not test:
   fr2=nested.read_json(nested.fold_result_path(rlout,a.seed,'M6',fi),default={}); test=[int(x) for x in fr2.get('test_subject_indices',[])]
  train=sorted(allidx-set(test)); idata=nested.load_rl_fold_idata(rlout,a.seed,'M6',fi); atr=rl.subset_subjects(arr,train); ate=rl.subset_subjects(arr,test)
  ltr=nested.posterior_draws_mean_latents(rl,atr,'M6',idata,a.max_rl_draws,s.stable_seed(a.seed,'maptrain',fi)); lte=nested.posterior_draws_mean_latents(rl,ate,'M6',idata,a.max_rl_draws,s.stable_seed(a.seed,'maptest',fi)); rtr=nested.arr_to_nested_latent_rows(rl,atr,ltr,fi,a.seed,'M6'); rte=nested.arr_to_nested_latent_rows(rl,ate,lte,fi,a.seed,'M6'); tr,_=s.prepare_frame(ddm,rtr,root/'controlled_inputs/behavioral_trial_data',700,100,700,probs,10); te,_=s.prepare_frame(ddm,rte,root/'controlled_inputs/behavioral_trial_data',700,100,700,probs,10)
  for vk in variants:
   fit=ddm.fit_one_model(tr,ddm.VARIANTS[vk],700,a.n_starts,s.stable_seed(a.seed,'mapfit',fi,vk),1e-4,.03,.03,.05); (out/'fits'/f'fold_{fi:02d}'/vk).mkdir(parents=True,exist_ok=True); (out/'fits'/f'fold_{fi:02d}'/vk/'fit.json').write_text(json.dumps(s.jsonable(fit),indent=2),encoding='utf-8'); sc=ddm.score_model(te,tr,fit,.03,.03,.05); folds.append({'fold':fi,'variant':vk,'elpd':sc['elpd'],'n_trials':len(te),**sc['ppc']})
   for sid,g in te.groupby('subj_id_ddm'):
    scs=ddm.score_model(g.reset_index(drop=True),tr,fit,.03,.03,.05); subj.append({'fold':fi,'subject_id':str(sid),'variant':vk,'elpd':scs['elpd']})
 f=pd.DataFrame(folds); u=pd.DataFrame(subj); f.to_csv(out/'mapping_fold_scores.csv',index=False); u.to_csv(out/'mapping_subject_scores.csv',index=False)
 comps=[('D14S','D14'),('D15S','D15'),('D9S','D9'),('D9X','D9')]; cr=[]; w=u.pivot_table(index=['fold','subject_id'],columns='variant',values='elpd')
 for i,(x,y) in enumerate(comps):
  z=(w[x]-w[y]).dropna().to_numpy(); m,lo,hi=boot(z,a.n_bootstrap,a.seed+i); cr.append({'alternative':x,'current':y,'n_subjects':len(z),'mean_subject_delta_elpd':m,'ci_low':lo,'ci_high':hi,'positive_subjects':int((z>0).sum()),'total_delta_elpd':float(z.sum())})
 pd.DataFrame(cr).to_csv(out/'mapping_planned_contrasts.csv',index=False); (out/'analysis_settings.json').write_text(json.dumps({'variants':variants,'planned_contrasts':comps,'deadline_ms':700,'rt_min_ms':100,'t0_ms':30,'max_rl_draws':a.max_rl_draws,'status':'claim-dependent sensitivity; no post-hoc model promotion'},indent=2),encoding='utf-8'); print('Wrote',out)
if __name__=='__main__': main()
