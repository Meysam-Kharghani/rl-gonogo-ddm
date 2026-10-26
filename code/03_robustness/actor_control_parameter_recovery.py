#!/usr/bin/env python3
"""Parameter-recovery wrapper for the selected actor-control RL model.

This public-facing wrapper uses formal controlled-input locations and deterministic
portable source names. Existing replicate posterior files may be reused unless a
forced refit is requested by the underlying recovery implementation.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REQ = ("numpy", "pandas", "arviz", "pymc")

def py_ok(py: Path):
    q = subprocess.run(
        [str(py), "-c", "import " + ",".join(REQ) + "; print('OK')"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    return q.returncode == 0, q.stdout.strip()

def choose_py(explicit):
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    env_python = os.environ.get("RL_DDM_ANALYSIS_PYTHON")
    if env_python:
        cands.append(Path(env_python).expanduser())
    cands.append(Path(sys.executable))
    msgs=[]
    seen=set()
    for p in cands:
        p=p.resolve()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        ok,msg=py_ok(p)
        msgs.append(f"{p}: {'OK' if ok else msg[-300:]}")
        if ok:
            return p
    raise RuntimeError("No Python with PyMC/ArviZ found:\n" + "\n".join(msgs))

def readable(p):
    try:
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False

def locate_source(root: Path, explicit):
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not readable(p):
            raise FileNotFoundError(p)
        return p
    p = root / "controlled_inputs/rl_full_fit/actor_control_primary/idata.nc"
    if readable(p):
        return p.resolve()
    raise FileNotFoundError(
        "Full-fit actor-control posterior not found. Provide --source-idata or materialize "
        "controlled_inputs/rl_full_fit/actor_control_primary/idata.nc. Do not substitute a fold-specific posterior."
    )

def locate_project_out(root: Path, explicit):
    if explicit:
        p=Path(explicit).expanduser().resolve()
        if readable(p/"artifacts/arrays__actor_control_model_family.npz"):
            return p
        raise FileNotFoundError(f"Invalid --project-out: {p}")
    known=[
        root/"controlled_inputs/rl_crossvalidated_posteriors",
        root/"controlled_inputs/rl_full_fit",
    ]
    for p in known:
        if readable(p/"artifacts/arrays__actor_control_model_family.npz"):
            return p.resolve()
    hits=list(root.rglob("arrays__actor_control_model_family.npz"))
    if not hits:
        raise FileNotFoundError("arrays__actor_control_model_family.npz not found")
    return hits[0].parent.parent.resolve()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--analysis-python", default=None)
    ap.add_argument("--source-idata", default=None)
    ap.add_argument("--project-out", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=.95)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reduced-validation", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--locate-only", action="store_true")
    a=ap.parse_args()

    root=Path(a.project_root).expanduser().resolve()
    py=choose_py(a.analysis_python)
    rec=root/"shared/rl/parameter_recovery.py"
    rl=root/"shared/rl/actor_control_model_family.py"
    if not rec.exists():
        raise FileNotFoundError(
            f"Missing recovery-analysis source:\n{rec}\n"
            "Install the documented recovery dependency before running this wrapper."
        )
    if not rl.exists():
        raise FileNotFoundError(rl)
    source=locate_source(root,a.source_idata)
    project_out=locate_project_out(root,a.project_out)
    out=Path(a.out).expanduser().resolve() if a.out else root/"reproduced/controlled_rerun/robustness/actor_control_parameter_recovery"

    print("Resolved dependencies:")
    print("  analysis_python:",py)
    print("  recovery_script:",rec)
    print("  rl_script:      ",rl)
    print("  project_out:    ",project_out)
    print("  source_idata:   ",source)
    print("  output:         ",out)

    reps,draws,tune,chains,cores=(2,300,300,2,2) if a.reduced_validation else (a.reps,a.draws,a.tune,a.chains,a.cores)
    for rep in range(reps):
        ip=out/f"rep_{rep:02d}/results/M6_posterior/idata.nc"
        if ip.exists() and not a.force:
            print(f"  rep_{rep:02d}: existing idata found -> WILL REUSE ({ip})")
        else:
            print(f"  rep_{rep:02d}: no reusable idata -> WILL FIT")

    if a.locate_only:
        return 0

    cmd=[
        str(py),str(rec),
        "--rl-script",str(rl),
        "--project-out",str(project_out),
        "--source-idata",str(source),
        "--out",str(out),
        "--reps",str(reps),
        "--draws",str(draws),
        "--tune",str(tune),
        "--chains",str(chains),
        "--cores",str(cores),
        "--target-accept",str(a.target_accept),
        "--seed",str(a.seed),
    ]
    if a.force:
        cmd.append("--force")
    print("+"," ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
