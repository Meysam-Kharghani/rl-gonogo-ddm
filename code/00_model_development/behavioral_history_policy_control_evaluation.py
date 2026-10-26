#!/usr/bin/env python3
# PUBLIC HISTORICAL SOURCE SNAPSHOT
# Evidence status: HISTORICAL_OR_SUPERSEDED_DEVELOPMENT_EVIDENCE
# This file preserves a documented historical project stage in portable public form.
# The SHA-256 of the corresponding executed source snapshot is recorded in the provenance registry.

# -*- coding: utf-8 -*-
"""
Run DDM policy-control checks for nested RL->DDM latents.

This script assumes nested latent CSVs already exist under roots such as:
  results/reviewer_proofing/M6_seed2026/nested_latents/M6_seed2026_nested_trial_latents_for_ddm.csv

It runs a DDM CV suite using shared/ddm/one_boundary_diffusion.py for controls:
  D2: original RL actor-control DDM
  D5: no-p_press RL control
  D6: behavioral-summary control
  D7: residualized-policy control
  D8: within-subject policy-shuffled placebo
  D6A: train-fold behavioral-rate control

It also collects fold-wise paired deltas against D2.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cmd(cmd: List[str], dry_run: bool = False) -> None:
    print("\n" + "="*100)
    print("[cmd] " + " ".join(map(str, cmd)))
    print("="*100)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def infer_label(root: Path) -> str:
    return root.name


def infer_seed(label: str, default: int) -> int:
    m = re.search(r"seed(\d+)", label)
    return int(m.group(1)) if m else int(default)


def infer_latent_path(root: Path) -> Path:
    label = infer_label(root)
    candidates = sorted((root / "nested_latents").glob("*_nested_trial_latents_for_ddm.csv"))
    exact = root / "nested_latents" / f"{label}_nested_trial_latents_for_ddm.csv"
    if exact.exists():
        return exact
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No nested latent CSV found under {root/'nested_latents'}")


def config_grid(grid: str, deadline_ms: float, rt_min_ms: float, rt_max_ms: float,
                t0_min: float, t0_max: float) -> List[Dict]:
    configs = [
        {
            "config": "main",
            "deadline_ms": float(deadline_ms),
            "rt_min_ms": float(rt_min_ms),
            "rt_max_ms": float(rt_max_ms),
            "t0_min": float(t0_min),
            "t0_max": float(t0_max),
        }
    ]
    if grid in {"minimal", "full"}:
        configs.extend([
            {
                "config": "higher_rtmin_150",
                "deadline_ms": float(deadline_ms),
                "rt_min_ms": 150.0,
                "rt_max_ms": float(rt_max_ms),
                "t0_min": float(t0_min),
                "t0_max": float(t0_max),
            },
            {
                "config": "conservative_t0_050_150",
                "deadline_ms": float(deadline_ms),
                "rt_min_ms": float(rt_min_ms),
                "rt_max_ms": float(rt_max_ms),
                "t0_min": 0.05,
                "t0_max": 0.15,
            },
            {
                "config": "wider_t0_020_200",
                "deadline_ms": float(deadline_ms),
                "rt_min_ms": float(rt_min_ms),
                "rt_max_ms": float(rt_max_ms),
                "t0_min": 0.02,
                "t0_max": 0.20,
            },
        ])
    return configs


def run_controls(args) -> None:
    roots = [Path(r) for r in args.roots]
    out_root = Path(args.out)
    ensure_dir(out_root)
    configs = config_grid(args.grid, args.deadline_ms, args.rt_min_ms, args.rt_max_ms, args.t0_min, args.t0_max)

    manifest = {
        "time": now(),
        "roots": [str(r) for r in roots],
        "out": str(out_root),
        "ddm_script": str(args.ddm_script),
        "variants": args.variants,
        "configs": configs,
        "k": args.k,
        "n_starts": args.n_starts,
    }
    write_json(out_root / "policy_control_manifest.json", manifest)

    for root in roots:
        label = infer_label(root)
        seed = infer_seed(label, args.seed)
        latent = infer_latent_path(root)
        for cfg in configs:
            cfg_name = cfg["config"]
            run_out = out_root / label / cfg_name
            cv_csv = run_out / "cv_comparison.csv"
            if cv_csv.exists() and not args.force:
                print(f"[skip] exists: {cv_csv}")
                continue
            cmd = [
                sys.executable, str(args.ddm_script), "cv",
                "--latents", str(latent),
                "--data_root", str(args.data_root),
                "--out", str(run_out),
                "--variants", *args.variants,
                "--deadline_ms", str(cfg["deadline_ms"]),
                "--rt_min_ms", str(cfg["rt_min_ms"]),
                "--rt_max_ms", str(cfg["rt_max_ms"]),
                "--t0_min", str(cfg["t0_min"]),
                "--t0_max", str(cfg["t0_max"]),
                "--k", str(args.k),
                "--n_starts", str(args.n_starts),
                "--seed", str(seed),
                "--l2", str(args.l2),
            ]
            run_cmd(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        collect_controls(out_root)


def collect_controls(out_root: Path) -> None:
    out_root = Path(out_root)
    comp_rows = []
    fold_rows = []
    for cv_csv in sorted(out_root.glob("*/*/cv_comparison.csv")):
        label = cv_csv.parent.parent.name
        config = cv_csv.parent.name
        df = pd.read_csv(cv_csv)
        df.insert(0, "root", label)
        df.insert(1, "config", config)
        comp_rows.append(df)
        for vk in df["variant"].astype(str).tolist():
            jp = cv_csv.parent / "cv" / vk / "cv_kfold_subject.json"
            if not jp.exists():
                continue
            obj = read_json(jp)
            for fr in obj.get("folds", []):
                fold_rows.append({
                    "root": label,
                    "config": config,
                    "variant": vk,
                    "variant_name": obj.get("variant_name", ""),
                    "fold": int(fr.get("fold")),
                    "elpd": float(fr.get("elpd")),
                    "nll": float(fr.get("nll")),
                    "n_test_subjects": len(fr.get("test_subjects", [])),
                })

    if not comp_rows:
        print(f"[collect] no cv_comparison.csv under {out_root}")
        return

    comps = pd.concat(comp_rows, ignore_index=True)
    comps = comps.sort_values(["root", "config", "total_elpd"], ascending=[True, True, False])
    comps["rank"] = comps.groupby(["root", "config"])["total_elpd"].rank(ascending=False, method="first").astype(int)
    comps.to_csv(out_root / "all_policy_control_cv_ranked.csv", index=False)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_root / "all_policy_control_cv_folds.csv", index=False)

    # Paired fold-level deltas against D2.
    deltas = []
    if not folds.empty and "D2" in set(folds["variant"]):
        wide = folds.pivot_table(index=["root", "config", "fold"], columns="variant", values="elpd", aggfunc="first")
        for v in sorted([c for c in wide.columns if c != "D2"]):
            diff = wide["D2"] - wide[v]
            tmp = diff.reset_index(name="delta_elpd_D2_minus_" + v)
            for (root, config), g in tmp.groupby(["root", "config"]):
                x = pd.to_numeric(g["delta_elpd_D2_minus_" + v], errors="coerce").dropna().to_numpy(dtype=float)
                if len(x) == 0:
                    continue
                # Non-parametric bootstrap over folds (small k, so descriptive only).
                rng = np.random.default_rng(20260606)
                boot = np.array([np.mean(rng.choice(x, size=len(x), replace=True)) for _ in range(10000)]) if len(x) > 1 else np.array([x[0]])
                deltas.append({
                    "root": root,
                    "config": config,
                    "comparison": f"D2_minus_{v}",
                    "n_folds": int(len(x)),
                    "mean_delta_elpd": float(np.mean(x)),
                    "se_delta_elpd": float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
                    "ci025_boot_mean": float(np.quantile(boot, 0.025)),
                    "ci975_boot_mean": float(np.quantile(boot, 0.975)),
                    "positive_folds": int(np.sum(x > 0)),
                    "total_delta_elpd": float(np.sum(x)),
                })
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(out_root / "paired_fold_deltas_vs_D2.csv", index=False)

    # Compact summary across roots/configs for each comparison.
    if not deltas_df.empty:
        summary = (deltas_df.groupby("comparison")
                   .agg(n_configs=("comparison", "size"),
                        mean_total_delta=("total_delta_elpd", "mean"),
                        min_total_delta=("total_delta_elpd", "min"),
                        max_total_delta=("total_delta_elpd", "max"),
                        total_positive_folds=("positive_folds", "sum"))
                   .reset_index())
        summary.to_csv(out_root / "paired_delta_summary_vs_D2.csv", index=False)

    write_json(out_root / "collect_manifest.json", {
        "time": now(),
        "out": str(out_root),
        "n_cv_rows": int(len(comps)),
        "n_fold_rows": int(len(folds)),
        "n_delta_rows": int(len(deltas_df)),
        "notes": "Main outputs: all_policy_control_cv_ranked.csv and paired_fold_deltas_vs_D2.csv",
    })
    print(f"[collect] wrote {out_root}")
    print(comps.to_string(index=False))
    if not deltas_df.empty:
        print("\nPaired fold deltas vs D2:")
        print(deltas_df.to_string(index=False))


def collect_cmd(args) -> None:
    collect_controls(Path(args.out))


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--roots", nargs="+", required=True, help="Nested reviewer_proofing roots, e.g. results/reviewer_proofing/M6_seed2026")
    run.add_argument("--out", required=True, help="Output root for policy-control checks")
    run.add_argument("--ddm_script", default="shared/ddm/one_boundary_diffusion.py")
    run.add_argument("--data_root", default="data/raw/csv")
    run.add_argument("--variants", nargs="+", default=["D2", "D5", "D6", "D6A", "D7", "D8"])
    run.add_argument("--grid", choices=["main", "minimal", "full"], default="minimal")
    run.add_argument("--deadline_ms", type=float, default=700.0)
    run.add_argument("--rt_min_ms", type=float, default=100.0)
    run.add_argument("--rt_max_ms", type=float, default=1000.0)
    run.add_argument("--t0_min", type=float, default=0.03)
    run.add_argument("--t0_max", type=float, default=0.08)
    run.add_argument("--k", type=int, default=5)
    run.add_argument("--n_starts", type=int, default=8)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--l2", type=float, default=0.01)
    run.add_argument("--force", action="store_true")
    run.add_argument("--dry_run", action="store_true")
    run.set_defaults(func=run_controls)

    coll = sub.add_parser("collect")
    coll.add_argument("--out", required=True)
    coll.set_defaults(func=collect_cmd)
    return ap


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
