"""Sensitivity, environment, and wind-robustness sweeps for PPO and SAC.

For each algorithm, this varies training and environment factors around its
curriculum and writes algorithm-prefixed run folders:

  1. Hyperparameter sweeps on the from-scratch phase-1 env (learning rate,
     discount factor, network size, entropy)      -> sensitivity_hp_{algo}
  2. Environment sweeps: fuel-cost weight and gimbal range (fine-tunes)
                                                    -> sensitivity_fuel_{algo}
  3. Wind robustness: fine-tune at 10 and 15 m/s, then evaluate the policies
     trained at 5/10/15 m/s across unseen wind levels
                                                    -> robustness_wind_{algo}

The two algorithms test the same FACTORS, but the hyperparameter sweep VALUES
are adapted to each one's base, because those differ:
    PPO base: lr 3e-4,  gamma 0.999, net 64^2,  entropy 0 (fixed)
    SAC base: lr 7.3e-4, gamma 0.99, net 256^2, entropy auto

DEPENDENCY: the environment and wind-robustness sweeps fine-tune from, or
evaluate, the curriculum, so run run_matrix_staged.py first (it trains the
phase-1 roots, phase2_ppo_mass / phase2_sac_mass, and phase2_ppo_wind / phase2_sac_wind).
Only the hyperparameter sweeps train from scratch and are independent.

Roughly 30h on an M-series laptop CPU (SAC dominates). 
Resumable: a step is skipped when its completion artifact exists.

"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

PY = ".venv/bin/python"
SEEDS = [0, 1, 2, 3, 4]
SWEEP_SEEDS = [0, 1]
WIND_TEST_LEVELS = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
WIND_TRAIN_LEVELS = [10.0, 15.0]    # base is 5.0
GUST_RATIO = 0.4  # gust_sigma = ratio * mean wind speed, as in training
FUEL_WEIGHTS = [0.1, 1.0, 3.0]      # base is 0.3
GIMBAL_RANGES = [5.0, 15.0]         # base is 8.0

# hyperparameter sweeps, adapted to each algorithm's base values (see docstring)
PPO_HP_SWEEPS = {
    "sweep_lr_lo": ["algo_kwargs.learning_rate=lin_1.0e-4"],
    "sweep_lr_hi": ["algo_kwargs.learning_rate=lin_1.0e-3"],
    "sweep_gamma_lo": ["algo_kwargs.gamma=0.99"],
    "sweep_gamma_hi": ["algo_kwargs.gamma=0.9999"],
    "sweep_net_sm": ["algo_kwargs.net_arch=[32,32]"],
    "sweep_net_lg": ["algo_kwargs.net_arch=[256,256]"],
    "sweep_ent": ["algo_kwargs.ent_coef=0.01"],
}
SAC_HP_SWEEPS = {
    "sweep_lr_lo": ["algo_kwargs.learning_rate=3.0e-4"],
    "sweep_lr_hi": ["algo_kwargs.learning_rate=3.0e-3"],
    "sweep_gamma_lo": ["algo_kwargs.gamma=0.9"],
    "sweep_gamma_hi": ["algo_kwargs.gamma=0.999"], 
    "sweep_net_sm": ["algo_kwargs.net_arch=[64,64]"],
    "sweep_net_lg": ["algo_kwargs.net_arch=[400,400]"],
    "sweep_ent": ["algo_kwargs.ent_coef=0.2"],       # fixed temperature vs base auto
}

# per-algorithm configs: from-scratch (hp), full-env fine-tune (fuel/wind),
# gimbal-stage fine-tune (gimbal), and the staged @5 m/s robustness baseline.
ALGOS = {
    "ppo": dict(
        hp_config="configs/phase1_ppo.yaml",
        fuel_config="configs/phase2_ppo_wind.yaml",
        gimbal_config="configs/phase2_ppo_gimbal.yaml",
        wind_config="configs/phase2_ppo_wind.yaml",
        staged_endpoint="phase2_ppo_wind",
        hp_sweeps=PPO_HP_SWEEPS,
    ),
    "sac": dict(
        hp_config="configs/phase1_sac.yaml",
        fuel_config="configs/phase2_sac_wind.yaml",
        gimbal_config="configs/phase2_sac_gimbal.yaml",
        wind_config="configs/phase2_sac_wind.yaml",
        staged_endpoint="phase2_sac_wind",
        hp_sweeps=SAC_HP_SWEEPS,
    ),
}


def wind_sets(mean: float) -> list[str]:
    return [
        "env.wind.enabled=true",
        f"env.wind.mean_speed={mean}",
        f"env.wind.gust_sigma={round(mean * GUST_RATIO, 2)}",
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


def build_algo_jobs(prefix: str, spec: dict) -> list[dict]:
    jobs: list[dict] = []

    # 1. Hyperparameter sweeps (from scratch, phase-1 env)
    for name, sets in spec["hp_sweeps"].items():
        run = f"{prefix}_{name}"
        for seed in SWEEP_SEEDS:
            jobs += [train(spec["hp_config"], run, seed, sets), evaluate(run, seed)]

    # 2. Environment sweeps (fine-tune from the curriculum)
    for w in FUEL_WEIGHTS:
        run = f"{prefix}_sweep_wfuel{w:g}"
        for seed in SWEEP_SEEDS:
            jobs += [train(spec["fuel_config"], run, seed, [f"env.reward.w_fuel={w}"]), evaluate(run, seed)]
    for g in GIMBAL_RANGES:
        run = f"{prefix}_sweep_gimbal{g:g}"
        for seed in SWEEP_SEEDS:
            jobs += [train(spec["gimbal_config"], run, seed, [f"env.gimbal_max_deg={g}"]), evaluate(run, seed)]

    # 3a. Wind-robustness training: fine-tune at higher wind levels
    for w in WIND_TRAIN_LEVELS:
        run = f"{prefix}_windtrain{w:g}"
        for seed in SWEEP_SEEDS:
            jobs += [train(spec["wind_config"], run, seed, wind_sets(w)), evaluate(run, seed)]

    # 3b. Wind-robustness evaluations (policy x unseen test level). The @5 m/s
    #     baseline is the staged endpoint (5 seeds); the higher-wind runs use 2.
    robust = [(spec["staged_endpoint"], SEEDS)] + [(f"{prefix}_windtrain{w:g}", SWEEP_SEEDS) for w in WIND_TRAIN_LEVELS]
    for run, seeds in robust:
        for seed in seeds:
            for w in WIND_TEST_LEVELS:
                jobs.append(evaluate(run, seed, wind_sets(w), out=f"eval_wind{w:g}", trajs=0))

    return jobs


def build_jobs() -> list[dict]:
    jobs: list[dict] = []
    for prefix, spec in ALGOS.items():
        jobs += build_algo_jobs(prefix, spec)
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
