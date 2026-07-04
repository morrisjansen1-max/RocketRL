"""Train PPO or SAC on the rocket-landing environment.

Each run writes a self-contained directory under runs/:
    config.yaml          resolved config (exact reproduction recipe)
    monitor/             per-env training episode logs (CSV)
    monitor_eval/        eval-env episode logs
    logs/                SB3 progress.csv + TensorBoard events
    evaluations.npz      periodic deterministic evaluations (incl. success)
    best_model.zip       best checkpoint by eval mean reward
    final_model.zip      model at the end of training
    versions.json        library versions for reproducibility
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from config import EnvConfig, RunConfig, apply_overrides, load_config, save_config
from environment.rocket_env import RocketLandingEnv
from utils import write_versions

ALGOS = {"ppo": PPO, "sac": SAC}
INFO_KEYS = ("is_success", "termination", "fuel_used")


def make_env_fn(env_cfg: EnvConfig, seed: int, monitor_path: str):
    def _init():
        env = RocketLandingEnv(env_cfg)
        env = Monitor(env, monitor_path, info_keywords=INFO_KEYS)
        env.reset(seed=seed)
        return env

    return _init


def _resolve_schedules(kwargs: dict) -> None:
    """Translate zoo-style 'lin_X' strings into linear annealing schedules."""
    for key, value in kwargs.items():
        if isinstance(value, str) and value.startswith("lin_"):
            base = float(value[4:])
            kwargs[key] = lambda progress_remaining, base=base: base * progress_remaining


def build_model(cfg: RunConfig, venv, run_dir: Path):
    kwargs = dict(cfg.algo_kwargs)
    _resolve_schedules(kwargs)
    policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
    net_arch = kwargs.pop("net_arch", None)
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)
    if cfg.init_from:
        # warm start: load weights, override hyperparameters from this config
        model = ALGOS[cfg.algo].load(
            cfg.init_from, env=venv, seed=cfg.seed, device=cfg.device, verbose=1, **kwargs
        )
    else:
        model = ALGOS[cfg.algo](
            "MlpPolicy",
            venv,
            seed=cfg.seed,
            device=cfg.device,
            policy_kwargs=policy_kwargs or None,
            verbose=1,
            **kwargs,
        )
    model.set_logger(configure(str(run_dir / "logs"), ["stdout", "csv", "tensorboard"]))
    return model


def train(cfg: RunConfig, runs_dir: str = "runs", overwrite: bool = False) -> Path:
    if cfg.init_from:
        cfg.init_from = cfg.init_from.format(seed=cfg.seed)
    run_dir = Path(runs_dir) / f"{cfg.run_name}_s{cfg.seed}"
    if run_dir.exists():
        if not overwrite:
            raise SystemExit(f"{run_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(run_dir)
    (run_dir / "monitor").mkdir(parents=True)
    (run_dir / "monitor_eval").mkdir()
    save_config(cfg, run_dir / "config.yaml")
    set_random_seed(cfg.seed)

    venv = DummyVecEnv(
        [
            make_env_fn(cfg.env, cfg.seed * 10_000 + i, str(run_dir / "monitor" / f"train_{i}"))
            for i in range(cfg.n_envs)
        ]
    )
    eval_env = DummyVecEnv(
        [make_env_fn(cfg.env, cfg.seed * 10_000 + 9_999, str(run_dir / "monitor_eval" / "eval"))]
    )

    model = build_model(cfg, venv, run_dir)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(cfg.eval.freq // cfg.n_envs, 1),
        n_eval_episodes=cfg.eval.n_episodes,
        deterministic=True,
        verbose=1,
    )

    t0 = time.time()
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_cb)
    elapsed = time.time() - t0

    model.save(run_dir / "final_model")
    write_versions(run_dir / "versions.json")
    fps = cfg.total_timesteps / elapsed
    print(f"\nTraining finished in {elapsed / 60:.1f} min ({fps:,.0f} steps/s). Artifacts: {run_dir}")
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a run config YAML")
    p.add_argument("--seed", type=int, help="override config seed")
    p.add_argument("--total-timesteps", type=int, help="override config total_timesteps")
    p.add_argument("--run-name", help="override config run_name")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="dotted-path config override, e.g. env.wind.mean_speed=10 (repeatable)")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.run_name is not None:
        cfg.run_name = args.run_name
    apply_overrides(cfg, args.overrides)

    train(cfg, runs_dir=args.runs_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
