"""Curriculum-staging runner: the PPO and SAC curriculum chains.

Trains and evaluates the two matched four-stage curricula that are the core of
the study. Each stage fine-tunes the previous one (init_from), so the stages run
in order within a seed.

  PPO chain: phase1_ppo -> phase2_ppo_gimbal -> phase2_ppo_mass -> phase2_ppo_wind
  SAC chain: phase1_sac -> phase2_sac_gimbal -> phase2_sac_mass -> phase2_sac_wind

The sensitivity, environment, and wind-robustness sweeps live in
sensitivity_analysis.py; run this script first, because those sweeps
fine-tune from the PPO chain trained here.

Roughly 22 h from scratch on an M-series laptop CPU (the SAC chain dominates).
Resumable: a step is skipped when its completion artifact already exists.

"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

PY = ".venv/bin/python"
SEEDS = [0, 1, 2, 3, 4]

# (config, run name, trajectories saved at evaluation)
PPO_CHAIN = [
    ("configs/phase1_ppo.yaml", "phase1_ppo", 10),
    ("configs/phase2_ppo_gimbal.yaml", "phase2_ppo_gimbal", 10),
    ("configs/phase2_ppo_mass.yaml", "phase2_ppo_mass", 10),
    ("configs/phase2_ppo_wind.yaml", "phase2_ppo_wind", 10),
]
SAC_CHAIN = [
    ("configs/phase1_sac.yaml", "phase1_sac", 10),
    ("configs/phase2_sac_gimbal.yaml", "phase2_sac_gimbal", 10),
    ("configs/phase2_sac_mass.yaml", "phase2_sac_mass", 10),
    ("configs/phase2_sac_wind.yaml", "phase2_sac_wind", 10),
]


def train(config: str, name: str, seed: int, sets=()) -> dict:
    run = Path("runs") / f"{name}_s{seed}"
    cmd = [PY, "-m", "train", "--config", config,
           "--run-name", name, "--seed", str(seed), "--overwrite"]
    for s in sets:
        cmd += ["--set", s]
    return {"cmd": cmd, "done": run / "final_model.zip", "name": f"train:{name}_s{seed}"}


def evaluate(name: str, seed: int, sets=(), out: str = "eval", n: int = 100, trajs: int = 10) -> dict:
    run = Path("runs") / f"{name}_s{seed}"
    cmd = [PY, "-m", "evaluate", "--run-dir", str(run),
           "--n-episodes", str(n), "--save-trajectories", str(trajs), "--out-name", out]
    for s in sets:
        cmd += ["--set", s]
    return {"cmd": cmd, "done": run / out / "summary.json", "name": f"eval:{name}_s{seed}:{out}"}


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    # Each chain runs stage-by-stage within a seed (stage order matters because
    # each stage fine-tunes the previous one). 
    for chain in (PPO_CHAIN, SAC_CHAIN):
        for seed in SEEDS:
            for config, name, trajs in chain:
                jobs += [train(config, name, seed), evaluate(name, seed, trajs=trajs)]
    return jobs


def run_jobs(jobs: list[dict], argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="list pending jobs without running")
    p.add_argument("--only", help="run only jobs whose name contains this substring")
    args = p.parse_args(argv)

    if args.only:
        jobs = [j for j in jobs if args.only in j["name"]]
    pending = [j for j in jobs if not j["done"].exists()]
    print(f"{len(jobs)} jobs total, {len(jobs) - len(pending)} already done, {len(pending)} pending")
    if args.dry_run:
        for j in pending:
            print("  ", j["name"])
        return

    log_dir = Path("runs/matrix_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        if job["done"].exists():
            continue
        elapsed = (time.time() - t0) / 60
        print(f"[{i}/{len(jobs)} | {elapsed:6.1f} min] {job['name']}", flush=True)
        log_path = log_dir / (job["name"].replace(":", "_") + ".log")
        with open(log_path, "w") as log:
            result = subprocess.run(job["cmd"], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0 or not job["done"].exists():
            failures.append(job["name"])
            print(f"    FAILED (exit {result.returncode}), log: {log_path} — continuing", flush=True)

    hours = (time.time() - t0) / 3600
    print(f"\nFinished in {hours:.1f} h with {len(failures)} failure(s)")
    for name in failures:
        print("  failed:", name)


if __name__ == "__main__":
    run_jobs(build_jobs())
