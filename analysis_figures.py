"""Analysis: tables and figures for the report.

The four curricula compared throughout:
    PPO jump    phase2_ppo              (phase1_ppo  -> full env, one fine-tune)
    PPO staged  phase2_ppo_wind         (phase1_ppo  -> gimbal -> mass -> wind)
    SAC jump    phase2_sac              (phase1_sac  -> full env, one fine-tune)
    SAC staged  phase2_sac_wind         (phase1_sac  -> gimbal -> mass -> wind)

Outputs go to results/ (CSV, comparison.md) and report/figures/ (vector PDF).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config
from plotting import SUCCESS_COLOR, _load_trajs, learning_curves

RESULTS = Path("results")
OUT = Path("figures")

# label -> (run name, colour, line style).  Colour by algorithm (PPO blue, SAC
# green); staged is dark/solid, jump is light/dashed (the weaker one).
CURRICULA = [
    ("PPO jump", "phase2_ppo", "#6fa8dc", "--"),
    ("PPO staged", "phase2_ppo_wind", "#1f4e8c", "-"),
    ("SAC jump", "phase2_sac", "#7fc97f", "--"),
    ("SAC staged", "phase2_sac_wind", "#217a21", "-"),
]
PPO_STAGES = ["phase1_ppo", "phase2_ppo_gimbal", "phase2_ppo_mass", "phase2_ppo_wind"]
SAC_STAGES = ["phase1_sac", "phase2_sac_gimbal", "phase2_sac_mass", "phase2_sac_wind"]

# Sensitivity study for both algorithms. Run folders are prefixed by algorithm
# (ppo_sweep_*, sac_windtrain*, ...); the wind-robustness "@5 m/s" baseline is the
# unprefixed staged curriculum endpoint. HP-sweep values are adapted to each
# algorithm's base (they differ), so both test the same FACTORS.
SENSITIVITY = {
    "ppo": {
        "label": "PPO", "color": "#1f4e8c", "base": "phase1_ppo",
        "robust": {"trained @ 5 m/s": "phase2_ppo_wind",
                   "trained @ 10 m/s": "ppo_windtrain10",
                   "trained @ 15 m/s": "ppo_windtrain15"},
        "hp": [("lr 1e-4", "ppo_sweep_lr_lo"), ("lr 3e-4 (base)", "phase1_ppo"), ("lr 1e-3", "ppo_sweep_lr_hi"),
               ("γ 0.99", "ppo_sweep_gamma_lo"), ("γ 0.999 (base)", "phase1_ppo"), ("γ 0.9999", "ppo_sweep_gamma_hi"),
               ("net 32²", "ppo_sweep_net_sm"), ("net 64² (base)", "phase1_ppo"), ("net 256²", "ppo_sweep_net_lg"),
               ("ent 0 (base)", "phase1_ppo"), ("ent 0.01", "ppo_sweep_ent")],
        "fuel": {0.1: "ppo_sweep_wfuel0.1", 0.3: "phase2_ppo_wind", 1.0: "ppo_sweep_wfuel1", 3.0: "ppo_sweep_wfuel3"},
    },
    "sac": {
        "label": "SAC", "color": "#217a21", "base": "phase1_sac",
        "robust": {"trained @ 5 m/s": "phase2_sac_wind",
                   "trained @ 10 m/s": "sac_windtrain10",
                   "trained @ 15 m/s": "sac_windtrain15"},
        "hp": [("lr 3e-4", "sac_sweep_lr_lo"), ("lr 7.3e-4 (base)", "phase1_sac"), ("lr 3e-3", "sac_sweep_lr_hi"),
               ("γ 0.9", "sac_sweep_gamma_lo"), ("γ 0.99 (base)", "phase1_sac"), ("γ 0.999", "sac_sweep_gamma_hi"),
               ("net 64²", "sac_sweep_net_sm"), ("net 256² (base)", "phase1_sac"), ("net 400²", "sac_sweep_net_lg"),
               ("ent auto (base)", "phase1_sac"), ("ent 0.2", "sac_sweep_ent")],
        "fuel": {0.1: "sac_sweep_wfuel0.1", 0.3: "phase2_sac_wind", 1.0: "sac_sweep_wfuel1", 3.0: "sac_sweep_wfuel3"},
    },
}

_ENV = load_config("configs/phase2_ppo_wind.yaml").env
PAD, MAX_VY, TAU_MIN = _ENV.pad_half_width, _ENV.max_landing_vy, _ENV.min_throttle
FAIL_COLOR = "#d04040"


# ============================================================ data + tables


def collect() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(Path("runs").glob("*/eval*/summary.json")):
        run_dir = summary_path.parent.parent
        m = re.match(r"(.+)_s(\d+)$", run_dir.name)
        if not m or run_dir.name.startswith("_"):
            continue
        d = json.loads(summary_path.read_text())
        eval_name = summary_path.parent.name
        wind = float(eval_name[9:]) if eval_name.startswith("eval_wind") else None
        rows.append({
            "run": m.group(1), "seed": int(m.group(2)), "eval": eval_name, "test_wind": wind,
            "success_rate": d["success_rate"],
            "vy": d["touchdown_vy_mps"]["mean"] if d.get("touchdown_vy_mps") else np.nan,
            "abs_x": d["touchdown_abs_x_m"]["mean"] if d.get("touchdown_abs_x_m") else np.nan,
            "tilt_deg": d["touchdown_abs_theta_deg"]["mean"] if d.get("touchdown_abs_theta_deg") else np.nan,
            "fuel_kg": d["fuel_used_kg"]["mean"] if d.get("fuel_used_kg") else np.nan,
        })
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "all_evals.csv", index=False)
    print(f"wrote {RESULTS / 'all_evals.csv'} ({len(df)} rows)")
    return df


def _nominal(df):
    return df[df["eval"] == "eval"]


def _per_seed_success(run, n=5):
    out = []
    for s in range(n):
        p = Path(f"runs/{run}_s{s}/eval/summary.json")
        out.append(json.loads(p.read_text())["success_rate"] if p.exists() else np.nan)
    return np.array(out)


def table(df) -> None:
    nom = _nominal(df)
    lines = ["# Results", "", "## Final policies on the full environment",
             "(5 seeds, 100 episodes/seed; mean ± std across seeds. Touchdown stats over landing seeds.)", "",
             "| curriculum | success | vy [m/s] | |x| [m] | tilt [deg] | fuel [kg] |",
             "|---|---|---|---|---|---|"]
    for label, run, _c, _s in CURRICULA:
        g = nom[nom["run"] == run].sort_values("seed")
        ok = g[g["success_rate"] > 0.5]
        def pm(col, frame=ok, prec=2):
            v = frame[col]
            return f"{v.mean():.{prec}f} ± {v.std():.{prec}f}" if len(v) else "—"
        lines.append(
            f"| {label} | {g['success_rate'].mean():.2f} ± {g['success_rate'].std():.2f} "
            f"| {pm('vy')} | {pm('abs_x')} | {pm('tilt_deg')} | {pm('fuel_kg', prec=0)} |")

    lines += ["", "## Per-seed success by curriculum stage", "",
              "| stage | s0 | s1 | s2 | s3 | s4 |", "|---|---|---|---|---|---|"]
    for run in PPO_STAGES + ["phase2_ppo"] + SAC_STAGES + ["phase2_sac"]:
        sr = _per_seed_success(run)
        lines.append(f"| {run} | " + " | ".join(f"{v:.2f}" if not np.isnan(v) else "—" for v in sr) + " |")

    out = RESULTS / "comparison.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


# ================================================================ helpers


def _representative(run, seed=0):
    """Softest-landing successful trajectory from a run's seed-0 eval dir."""
    trajs = _load_trajs(Path(f"runs/{run}_s{seed}/eval"), 20)
    succ = [t for t in trajs if str(t["outcome"]) == "success"] or trajs
    return min(succ, key=lambda d: abs(float(d["vy"][-1])))


def _ocolor(traj):
    return SUCCESS_COLOR if str(traj["outcome"]) == "success" else FAIL_COLOR


def _pulses(thr):
    return int(np.sum(np.abs(np.diff((np.asarray(thr) > 0.05).astype(int)))))


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {name}")


# ========================================================= results figures


def curriculum_necessity():
    """Two figures (PPO, SAC): staged stages plus the jump, on shared axes.
    The staged final stage locks at 100%; the jump is volatile and lower."""
    learning_curves(
        [f"runs/{s}_s*" for s in PPO_STAGES] + ["runs/phase2_ppo_s*"],
        ["1 ideal", "2a gimbal", "2b mass", "2c wind (staged)", "jump to full"],
        OUT / "curriculum_ppo.pdf",
    )
    learning_curves(
        [f"runs/{s}_s*" for s in SAC_STAGES] + ["runs/phase2_sac_s*"],
        ["1 ideal", "2a gimbal", "2b mass", "2c wind (staged)", "jump to full"],
        OUT / "curriculum_sac.pdf",
    )


def sample_efficiency():
    learning_curves(
        ["runs/phase1_ppo_s*", "runs/phase1_sac_s*"],
        ["PPO (10M steps)", "SAC (1M steps)"],
        OUT / "sample_efficiency.pdf",
    )


def staging_reliability():
    rng = np.random.default_rng(0)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (label, run, color, _s) in enumerate(CURRICULA):
        vals = _per_seed_success(run)
        axL.scatter(i + rng.uniform(-0.09, 0.09, len(vals)), vals, color=color, s=34,
                    zorder=3, edgecolor="white", linewidth=0.5)
        axL.errorbar(i, np.nanmean(vals), yerr=np.nanstd(vals), fmt="_", color=color,
                     capsize=7, markersize=26, lw=1.6, zorder=2)
    axL.set_xticks(range(4))
    axL.set_xticklabels([c[0].replace(" ", "\n") for c in CURRICULA])
    axL.set_ylabel("final success rate (dots = seeds)")
    axL.set_ylim(-0.05, 1.06)
    axL.grid(axis="y", alpha=0.3)
    axL.set_title("Reliability: jump vs staged, both algorithms")

    pairs = [("phase2_ppo", "phase2_ppo_wind", 0.0, 1.0, "#1f4e8c"),
             ("phase2_sac", "phase2_sac_wind", 2.2, 3.2, "#217a21")]
    for jr, sr, x0, x1, color in pairs:
        j, s = _per_seed_success(jr), _per_seed_success(sr)
        for k in range(len(j)):
            fail = j[k] < 0.5
            axR.plot([x0, x1], [j[k], s[k]], marker="o",
                     color="#c0392b" if fail else color,
                     lw=2.4 if fail else 1.0, alpha=0.9, zorder=3 if fail else 2)
    axR.set_xticks([0.0, 1.0, 2.2, 3.2])
    axR.set_xticklabels(["PPO\njump", "PPO\nstaged", "SAC\njump", "SAC\nstaged"])
    axR.set_ylim(-0.05, 1.06)
    axR.grid(axis="y", alpha=0.3)
    axR.set_ylabel("final success rate (per seed)")
    axR.set_title("Staging rescues the failing seeds (red)")
    fig.tight_layout()
    _save(fig, "staging_reliability.pdf")


def training_volatility():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
    for ax, (label, run, color, _s) in zip(axes.flat, CURRICULA):
        stds = []
        for seed in range(5):
            ev = np.load(f"runs/{run}_s{seed}/evaluations.npz")
            suc = ev["successes"].mean(axis=1)
            prog = ev["timesteps"] / ev["timesteps"].max()
            ax.plot(prog, suc, color=color, alpha=0.55, lw=1.0)
            stds.append(suc[len(suc) // 2:].std())
        ax.set_title(f"{label}   (2nd-half σ = {np.mean(stds):.2f})", fontsize=10)
        ax.set_ylim(-0.05, 1.06)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("training progress")
    for ax in axes[:, 0]:
        ax.set_ylabel("deterministic-eval success")
    fig.suptitle("Training-time volatility on the full environment (one line per seed)", fontsize=11)
    fig.tight_layout()
    _save(fig, "training_volatility.pdf")


def descent_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    for ax, (label, run, _c, _s) in zip(axes.flat, CURRICULA):
        for t in _load_trajs(Path(f"runs/{run}_s0/eval"), 8):
            x, y, th = t["x"], t["y"], t["theta"]
            ax.plot(x, y, color=_ocolor(t), alpha=0.7, lw=1.1)
            for i in range(0, len(x), max(len(x) // 10, 1)):
                ax.plot([x[i], x[i] + 6.0 * math.sin(th[i])], [y[i], y[i] + 6.0 * math.cos(th[i])],
                        color="#404a58", lw=0.7, alpha=0.7)
        ax.axhline(0, color="k", lw=0.7)
        ax.plot([-PAD, PAD], [0, 0], color=SUCCESS_COLOR, lw=4)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.3)
    fig.suptitle("Descent trajectories (ticks = body axis; green land, red fail)", fontsize=11)
    fig.tight_layout()
    _save(fig, "descent_trajectories.pdf")


def phase_portraits():
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for j, (label, run, _c, _s) in enumerate(CURRICULA):
        axd, axa = axes[0, j], axes[1, j]
        for t in _load_trajs(Path(f"runs/{run}_s0/eval"), 8):
            axd.plot(t["y"], t["vy"], color=_ocolor(t), alpha=0.7, lw=1.0)
            axa.plot(np.degrees(t["theta"]), np.degrees(t["omega"]), color=_ocolor(t), alpha=0.7, lw=1.0)
        axd.axhline(-MAX_VY, color="k", ls="--", lw=0.7)
        axd.set_title(label, fontsize=10)
        axd.set_xlabel("altitude y [m]")
        axa.set_xlabel("pitch θ [deg]")
        axa.axhline(0, color="k", lw=0.5)
        axa.axvline(0, color="k", lw=0.5)
        for a in (axd, axa):
            a.grid(alpha=0.3)
    axes[0, 0].set_ylabel("vertical speed vy [m/s]")
    axes[1, 0].set_ylabel("pitch rate ω [deg/s]")
    fig.suptitle("Phase portraits per curriculum: descent (top), attitude (bottom)", fontsize=11)
    fig.tight_layout()
    _save(fig, "phase_portraits.pdf")


def throttle_commands():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, (label, run, _c, _s) in zip(axes.flat, CURRICULA):
        t = _representative(run)
        thr = t["throttle"]
        tc = t["t"][: len(thr)]
        ax.fill_between(tc, 0, thr, step="post", color="#e8862e", alpha=0.75)
        ax.axhline(TAU_MIN, color="k", ls=":", lw=0.8)
        ax.set_title(f"{label}  ({_pulses(thr)} pulses)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("time t [s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("throttle τ")
    fig.suptitle(f"Throttle command, representative landing (dotted = min throttle {TAU_MIN:.2f})", fontsize=11)
    fig.tight_layout()
    _save(fig, "throttle_commands.pdf")


def representative_hoverslam():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for idx, (ax, (label, run, _c, _s)) in enumerate(zip(axes.flat, CURRICULA)):
        t = _representative(run)
        tt, y, vy, thr = t["t"], t["y"], t["vy"], t["throttle"]
        ign = int(np.argmax(np.asarray(thr) > 0.05))
        ax.axvspan(tt[0], tt[ign], color="#cfd8e6", alpha=0.5)
        ax.plot(tt, y, color="#1f4e8c")
        axb = ax.twinx()
        axb.plot(tt, vy, color="#c0392b")
        col, row = idx % 2, idx // 2
        if col == 0:
            ax.set_ylabel("altitude y [m]", color="#1f4e8c")
        if col == 1:
            axb.set_ylabel("vertical speed vy [m/s]", color="#c0392b")
        axb.annotate(f"touchdown {vy[-1]:.2f} m/s", xy=(tt[-1], vy[-1]),
                     xytext=(tt[-1] * 0.45, vy.min() * 0.55), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.7))
        ax.set_title(label, fontsize=10)
        if row == 1:
            ax.set_xlabel("time t [s]")
        ax.grid(alpha=0.3)
    fig.suptitle("Representative hoverslam per curriculum (shaded = free-fall coast)", fontsize=11)
    fig.tight_layout()
    _save(fig, "representative_hoverslam.pdf")


# ===================================================== sensitivity figures


def robustness(df, prefix):
    spec = SENSITIVITY[prefix]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    wind_df = df[df["test_wind"].notna()]
    for label, run in spec["robust"].items():
        g = wind_df[wind_df["run"] == run]
        if g.empty:
            continue
        stats = g.groupby("test_wind")["success_rate"].agg(["mean", "std", "count"])
        (line,) = ax.plot(stats.index, stats["mean"], marker="o",
                          label=f"{spec['label']} {label} (n={int(stats['count'].iloc[0])})")
        ax.fill_between(stats.index, stats["mean"] - stats["std"].fillna(0),
                        (stats["mean"] + stats["std"].fillna(0)).clip(upper=1.0),
                        alpha=0.15, color=line.get_color())
    ax.set_xlabel("test wind: mean speed [m/s] (gusts σ = 0.4 × mean)")
    ax.set_ylabel("landing success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{spec['label']} generalisation to unseen wind levels (100 episodes/seed/level)")
    fig.tight_layout()
    _save(fig, f"robustness_wind_{prefix}.pdf")


def sensitivity_hp(df, prefix):
    spec = SENSITIVITY[prefix]
    nom = _nominal(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (label, run) in enumerate(spec["hp"]):
        vals = nom[nom["run"] == run]["success_rate"].to_numpy()
        ax.bar(i, vals.mean() if len(vals) else 0.0,
               color="#888888" if "(base)" in label else spec["color"], alpha=0.6)
        ax.scatter([i] * len(vals), vals, color="k", zorder=3, s=18)
    ax.set_xticks(range(len(spec["hp"])))
    ax.set_xticklabels([lbl for lbl, _ in spec["hp"]], rotation=30, ha="right")
    ax.set_ylabel("success rate (per-seed dots)")
    ax.set_title(f"{spec['label']} from-scratch learning sensitivity (phase-1 env)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, f"sensitivity_hp_{prefix}.pdf")


def sensitivity_fuel(df, prefix):
    spec = SENSITIVITY[prefix]
    nom = _nominal(df)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ws, fuels, fstd, vys = [], [], [], []
    for w, run in sorted(spec["fuel"].items()):
        g = nom[(nom["run"] == run) & (nom["success_rate"] > 0.5)]
        ws.append(w)
        fuels.append(g["fuel_kg"].mean())
        fstd.append(g["fuel_kg"].std())
        vys.append(g["vy"].mean())
    ax1.errorbar(ws, fuels, yerr=fstd, marker="o", color=spec["color"], capsize=3)
    ax1.set_xscale("log")
    ax1.set_xlabel("fuel-cost weight $w_{fuel}$ (base 0.3)")
    ax1.set_ylabel("propellant used [kg]", color=spec["color"])
    ax2 = ax1.twinx()
    ax2.plot(ws, vys, marker="s", color="#c0392b")
    ax2.set_ylabel("touchdown $v_y$ [m/s]", color="#c0392b")
    ax1.set_title(f"{spec['label']} fuel-economy vs touchdown-softness trade-off")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, f"sensitivity_fuel_{prefix}.pdf")


# ===================================================================== main


def results(df):
    table(df)
    curriculum_necessity()
    sample_efficiency()
    staging_reliability()
    training_volatility()
    descent_trajectories()
    phase_portraits()
    throttle_commands()
    representative_hoverslam()


def sensitivity_all(df):
    for prefix in SENSITIVITY:
        robustness(df, prefix)
        sensitivity_hp(df, prefix)
        sensitivity_fuel(df, prefix)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="?", default="all", choices=["all", "results", "sensitivity", "table"])
    args = p.parse_args(argv)
    df = collect()
    if args.cmd == "table":
        table(df)
        return
    if args.cmd in ("results", "all"):
        results(df)
    if args.cmd in ("sensitivity", "all"):
        sensitivity_all(df)
    print("\nfigures written to", OUT)


if __name__ == "__main__":
    main()
