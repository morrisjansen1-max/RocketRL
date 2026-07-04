"""Figure generation: learning curves (multi-seed mean ± std) and
trajectory/analysis plots from recorded evaluation episodes.

"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUCCESS_COLOR = "#2a9d2a"
FAIL_COLOR = "#d04040"


# --------------------------------------------------------------- learning curves


def load_training_episodes(run_dir: Path) -> pd.DataFrame:
    """Merge per-env SB3 monitor CSVs into one episode log ordered by time,
    with a cumulative env-step column."""
    files = sorted(run_dir.glob("monitor/train_*.monitor.csv"))
    if not files:
        raise FileNotFoundError(f"no monitor files in {run_dir}/monitor")
    frames = [pd.read_csv(f, skiprows=1) for f in files]
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    return df


def _band(ax, runs: list[tuple[np.ndarray, np.ndarray]], label: str, color=None):
    """Interpolate per-seed series onto a common grid and plot mean ± std."""
    max_x = min(float(x[-1]) for x, _ in runs)
    grid = np.linspace(0.0, max_x, 200)
    ys = np.stack([np.interp(grid, x, y) for x, y in runs])
    mean, std = ys.mean(axis=0), ys.std(axis=0)
    (line,) = ax.plot(grid, mean, label=f"{label} (n={len(runs)})", color=color)
    ax.fill_between(grid, mean - std, mean + std, alpha=0.25, color=line.get_color())
    return line.get_color()


def learning_curves(run_patterns: list[str], labels: list[str], out: Path, window: int = 50):
    fig, (ax_rew, ax_suc) = plt.subplots(1, 2, figsize=(11, 4))
    for pattern, label in zip(run_patterns, labels):
        run_dirs = [Path(p) for p in sorted(glob.glob(pattern))]
        if not run_dirs:
            raise FileNotFoundError(f"no runs match {pattern!r}")
        reward_series, success_series = [], []
        for rd in run_dirs:
            df = load_training_episodes(rd)
            smoothed = df["r"].rolling(window, min_periods=1).mean()
            reward_series.append((df["steps"].to_numpy(float), smoothed.to_numpy(float)))
            evals = np.load(rd / "evaluations.npz")
            if "successes" in evals:
                success_series.append(
                    (evals["timesteps"].astype(float), evals["successes"].mean(axis=1))
                )
        color = _band(ax_rew, reward_series, label)
        if success_series:
            _band(ax_suc, success_series, label, color=color)

    ax_rew.set_xlabel("environment steps")
    ax_rew.set_ylabel(f"episode return (rolling mean, {window} ep)")
    ax_rew.set_title("Training return")
    ax_rew.legend()
    ax_rew.grid(alpha=0.3)
    ax_suc.set_xlabel("environment steps")
    ax_suc.set_ylabel("success rate (deterministic eval)")
    ax_suc.set_ylim(-0.02, 1.02)
    ax_suc.set_title("Evaluation success rate")
    ax_suc.legend()
    ax_suc.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


# ------------------------------------------------------------- trajectory plots


def _load_trajs(eval_dir: Path, limit: int):
    files = sorted(eval_dir.glob("traj_*.npz"))[:limit]
    if not files:
        raise FileNotFoundError(f"no traj_*.npz in {eval_dir}")
    return [dict(np.load(f, allow_pickle=False)) for f in files]


def _outcome_color(traj) -> str:
    return SUCCESS_COLOR if str(traj["outcome"]) == "success" else FAIL_COLOR


def plot_xy(trajs, cfg_pad: float, out: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for traj in trajs:
        x, y, th = traj["x"], traj["y"], traj["theta"]
        ax.plot(x, y, color=_outcome_color(traj), alpha=0.7, lw=1.2)
        # attitude ticks: short segment along the body axis every ~1 s
        step = max(len(x) // 12, 1)
        for i in range(0, len(x), step):
            seg = 6.0
            ax.plot(
                [x[i], x[i] + seg * math.sin(th[i])],
                [y[i], y[i] + seg * math.cos(th[i])],
                color="#404a58",
                lw=0.8,
                alpha=0.8,
            )
    ax.axhline(0, color="k", lw=0.8)
    ax.plot([-cfg_pad, cfg_pad], [0, 0], color=SUCCESS_COLOR, lw=4, label="pad")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Descent trajectories (ticks show body axis)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_timehistories(trajs, out: Path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    panels = [
        ("y", "altitude y [m]", 1.0),
        ("vy", "vertical speed vy [m/s]", 1.0),
        ("vx", "horizontal speed vx [m/s]", 1.0),
        ("theta", "pitch θ [deg]", 180.0 / math.pi),
        ("omega", "pitch rate ω [deg/s]", 180.0 / math.pi),
        ("m_prop", "propellant [kg]", 1.0),
        ("throttle", "throttle [-]", 1.0),
        ("gimbal", "gimbal δ [deg]", 180.0 / math.pi),
        ("rcs_torque", "RCS torque [kN m]", 1e-3),
    ]
    cmd_keys = {"throttle", "gimbal", "rcs_torque"}
    for ax, (key, label, scale) in zip(axes.flat, panels):
        for traj in trajs:
            t = traj["t"]
            series = traj[key] * scale
            tt = t[: len(series)] if key in cmd_keys else t
            ax.plot(tt, series, color=_outcome_color(traj), alpha=0.7, lw=1.0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    fig.suptitle("State and command time histories (green = success, red = failure)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_phase_portraits(trajs, out: Path, max_landing_vy: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for traj in trajs:
        c = _outcome_color(traj)
        ax1.plot(traj["y"], traj["vy"], color=c, alpha=0.7, lw=1.0)
        ax2.plot(np.degrees(traj["theta"]), np.degrees(traj["omega"]), color=c, alpha=0.7, lw=1.0)
    ax1.axhline(-max_landing_vy, color="k", ls="--", lw=0.8, label=f"vy limit −{max_landing_vy} m/s")
    ax1.set_xlabel("altitude y [m]")
    ax1.set_ylabel("vertical speed vy [m/s]")
    ax1.set_title("Descent phase portrait")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.axvline(0, color="k", lw=0.5)
    ax2.set_xlabel("pitch θ [deg]")
    ax2.set_ylabel("pitch rate ω [deg/s]")
    ax2.set_title("Attitude phase portrait")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def trajectories(eval_dir: Path, out_dir: Path, limit: int = 10):
    from config import load_config

    cfg = load_config(eval_dir.parent / "config.yaml")
    trajs = _load_trajs(eval_dir, limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_xy(trajs, cfg.env.pad_half_width, out_dir / "traj_xy.png")
    plot_timehistories(trajs, out_dir / "traj_timehistories.png")
    plot_phase_portraits(trajs, out_dir / "traj_phase_portraits.png", cfg.env.max_landing_vy)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("curves", help="multi-seed learning curves")
    pc.add_argument("--runs", action="append", required=True, help="glob over run dirs (repeatable)")
    pc.add_argument("--label", action="append", required=True, help="label per --runs pattern")
    pc.add_argument("--out", required=True)
    pc.add_argument("--window", type=int, default=50)

    pt = sub.add_parser("trajectories", help="trajectory/analysis plots from an eval dir")
    pt.add_argument("--eval-dir", required=True)
    pt.add_argument("--out-dir", required=True)
    pt.add_argument("--limit", type=int, default=10)

    args = p.parse_args(argv)
    if args.cmd == "curves":
        if len(args.runs) != len(args.label):
            raise SystemExit("need one --label per --runs")
        learning_curves(args.runs, args.label, Path(args.out), args.window)
    else:
        trajectories(Path(args.eval_dir), Path(args.out_dir), args.limit)


if __name__ == "__main__":
    main()
