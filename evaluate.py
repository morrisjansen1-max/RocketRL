"""Evaluate a trained run: success statistics + recorded trajectories.

Writes into <run-dir>/eval/:
    episodes.csv     one row per episode (return, outcome, touchdown stats)
    summary.json     aggregate statistics with confidence intervals
    traj_*.npz       recorded state/action time histories for plotting
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import apply_overrides, load_config
from environment.rocket_env import RocketLandingEnv
from utils import wilson_interval

STATE_FIELDS = ["t", "x", "y", "vx", "vy", "theta", "omega", "m_prop", "wind_speed"]
CMD_FIELDS = ["throttle", "gimbal", "rcs_torque"]


def rollout(env: RocketLandingEnv, model, deterministic: bool = True):
    """Run one episode; return (per-step records dict, final info, total return)."""
    obs, _ = env.reset()
    states = {k: [] for k in STATE_FIELDS}
    cmds = {k: [] for k in CMD_FIELDS}
    actions, rewards = [], []

    def snap_state():
        for k in STATE_FIELDS:
            states[k].append(getattr(env, k))

    snap_state()
    total, info = 0.0, {}
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        snap_state()
        cmds["throttle"].append(env.last_throttle)
        cmds["gimbal"].append(env.last_gimbal)
        cmds["rcs_torque"].append(env.last_rcs)
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(reward)
        total += reward
        if terminated or truncated:
            break

    record = {k: np.asarray(v) for k, v in states.items()}
    record.update({k: np.asarray(v) for k, v in cmds.items()})
    record["action"] = np.stack(actions)
    record["reward"] = np.asarray(rewards)
    return record, info, total


def evaluate(run_dir: Path, model_name: str, n_episodes: int, n_save: int, deterministic: bool,
             eval_seed: int | None, overrides=(), out_name: str = "eval"):
    cfg = load_config(run_dir / "config.yaml")
    apply_overrides(cfg, overrides)
    from train import ALGOS  # avoid importing SB3 before arg parsing

    model_path = run_dir / f"{model_name}_model.zip"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found")
    model = ALGOS[cfg.algo].load(model_path, device=cfg.device)

    env = RocketLandingEnv(cfg.env)
    env.reset(seed=eval_seed if eval_seed is not None else cfg.seed + 123_456)

    out_dir = run_dir / out_name
    out_dir.mkdir(exist_ok=True)

    rows = []
    for ep in range(n_episodes):
        record, info, total = rollout(env, model, deterministic)
        outcome = info.get("termination", "timeout")
        row = {
            "episode": ep,
            "return": total,
            "length": len(record["reward"]),
            "outcome": outcome,
            "is_success": bool(info.get("is_success", False)),
            "fuel_used": info.get("fuel_used", float("nan")),
            "touchdown_x": info.get("touchdown_x", float("nan")),
            "touchdown_vx": info.get("touchdown_vx", float("nan")),
            "touchdown_vy": info.get("touchdown_vy", float("nan")),
            "touchdown_theta_deg": math.degrees(info["touchdown_theta"]) if "touchdown_theta" in info else float("nan"),
            "touchdown_omega": info.get("touchdown_omega", float("nan")),
        }
        rows.append(row)
        if ep < n_save:
            np.savez(out_dir / f"traj_ep{ep:03d}_{outcome}.npz", outcome=np.array(outcome), **record)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "episodes.csv", index=False)

    n_success = int(df["is_success"].sum())
    lo, hi = wilson_interval(n_success, n_episodes)
    touchdown = df[df["outcome"].isin(["success", "crash"])]
    success = df[df["is_success"]]

    def stats(frame: pd.DataFrame, col: str):
        if frame.empty:
            return None
        return {"mean": float(frame[col].mean()), "std": float(frame[col].std(ddof=0))}

    summary = {
        "run_dir": str(run_dir),
        "model": model_name,
        "deterministic": deterministic,
        "overrides": list(overrides),
        "n_episodes": n_episodes,
        "success_rate": n_success / n_episodes,
        "success_rate_ci95": [lo, hi],
        "outcomes": df["outcome"].value_counts().to_dict(),
        "return": stats(df, "return"),
        "episode_length": stats(df, "length"),
        "fuel_used_kg": stats(df[df["fuel_used"].notna()], "fuel_used"),
        "touchdown_abs_x_m": stats(touchdown.assign(a=touchdown["touchdown_x"].abs()), "a"),
        "touchdown_vy_mps": stats(touchdown, "touchdown_vy"),
        "touchdown_abs_vx_mps": stats(touchdown.assign(a=touchdown["touchdown_vx"].abs()), "a"),
        "touchdown_abs_theta_deg": stats(touchdown.assign(a=touchdown["touchdown_theta_deg"].abs()), "a"),
        "success_only_touchdown_vy_mps": stats(success, "touchdown_vy"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--model", choices=["best", "final"], default=None,
                   help="which checkpoint to evaluate (default: best if present, else final)")
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--save-trajectories", type=int, default=10, dest="n_save")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of deterministic policy")
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="config override for off-nominal evaluation, e.g. env.wind.mean_speed=10")
    p.add_argument("--out-name", default="eval", help="output subdirectory name inside the run dir")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir)
    model_name = args.model
    if model_name is None:
        model_name = "best" if (run_dir / "best_model.zip").exists() else "final"
    evaluate(run_dir, model_name, args.n_episodes, args.n_save, not args.stochastic,
             args.eval_seed, args.overrides, args.out_name)


if __name__ == "__main__":
    main()
