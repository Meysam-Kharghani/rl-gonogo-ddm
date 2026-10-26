#!/usr/bin/env python3
"""Analytic-horizon sensitivity for the behavioral-computational study.

Runs the *existing canonical right-censored DDM implementation* at fixed analytic
horizons 600/650/700/750/780 ms. 700 ms remains the historical primary; this
script produces a robustness profile and MUST NOT select a new primary horizon
based on which result looks strongest.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

HORIZONS_DEFAULT=[600,650,700,750,780]
VARIANTS=['D6','D14','D15','D9']

def need(p:Path,label:str):
    if not p.exists(): raise FileNotFoundError(f'{label} not found: {p}')
    return p

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument('--out-root',default=None)
    ap.add_argument('--horizons-ms',nargs='+',type=int,default=HORIZONS_DEFAULT)
    ap.add_argument('--max-rl-draws',type=int,default=100)
    ap.add_argument('--n-starts',type=int,default=6)
    ap.add_argument('--n-bootstrap',type=int,default=10000)
    ap.add_argument('--force',action='store_true')
    a=ap.parse_args(); root=Path(a.project_root).expanduser().resolve()
    outroot=Path(a.out_root).expanduser().resolve() if a.out_root else root/'reproduced/controlled_rerun/robustness/analytic_horizon_sensitivity'
    outroot.mkdir(parents=True,exist_ok=True)
    s54=need(root/'code/02_primary_joint_response/joint_response_time_benchmark.py','primary joint-response benchmark')
    nested=need(root/'shared/rl/nested_crossvalidated_latent_generation.py','nested support')
    rl=need(root/'shared/rl/actor_control_model_family.py','actor-control RL model family source')
    ddm=need(root/'shared/ddm/one_boundary_diffusion.py','DDM helper')
    rlout=need(root/'controlled_inputs/rl_crossvalidated_posteriors','RL posterior root')
    dataroot=need(root/'controlled_inputs/behavioral_trial_data','behavioral data root')
    cue=need(root/'configs/instructed_cue_probabilities.json','cue probabilities')
    rows=[]
    for h in a.horizons_ms:
        od=outroot/f'horizon_{h}ms'
        done=od/'crossvalidated_ddm_planned_contrasts.csv'
        if not done.exists() or a.force:
            cmd=[sys.executable,str(s54),'--nested-script',str(nested),'--rl-script',str(rl),'--ddm-script',str(ddm),
                 '--rl-out',str(rlout),'--data-root',str(dataroot),'--cue-probabilities',str(cue),'--out',str(od),
                 '--rl-variant','M6','--seed','2026','--k','5','--max-rl-draws',str(a.max_rl_draws),
                 '--variants',*VARIANTS,'--n-starts',str(a.n_starts),'--deadline-ms',str(h),'--rt-min-ms','100',
                 '--rt-max-ms',str(h),'--t0-min','0.03','--t0-max','0.03','--dist-min','0.05','--horizons','0',
                 '--n-bootstrap',str(a.n_bootstrap)]
            run(cmd)
        con=pd.read_csv(done); con['analytic_horizon_ms']=h; rows.append(con)
    allc=pd.concat(rows,ignore_index=True)
    allc.to_csv(outroot/'horizon_sensitivity_planned_contrasts.csv',index=False)
    pivot=allc.pivot_table(index=['model','reference','contrast'],columns='analytic_horizon_ms',values='mean_subject_delta_elpd')
    pivot.to_csv(outroot/'horizon_sensitivity_delta_elpd_matrix.csv')
    settings={'historical_primary_ms':700,'reported_grid_ms':a.horizons_ms,'models':VARIANTS,
              'interpretation':'Robustness profile only; do not choose a new primary horizon after seeing results.',
              'max_rl_draws':a.max_rl_draws,'n_starts':a.n_starts,'n_bootstrap':a.n_bootstrap}
    (outroot/'robustness_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    print('\nWrote',outroot)
if __name__=='__main__': main()
