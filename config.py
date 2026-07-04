"""Config dataclasses + YAML loading.

Every run is fully described by a `RunConfig`. Defaults below describe the 
full environment: gimballed engine, weak RCS, mass depletion, drag. Wind off,
wind is the robustness knob. Phase configs override only what they simplify, 
so each YAML stays a minimal diff against the canonical environment.
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class WindConfig:
    enabled: bool = False
    mean_speed: float = 0.0 
    gust_sigma: float = 0.0
    gust_tau: float = 2.0
    randomize_direction: bool = False


@dataclass
class InitRanges:
    x: tuple[float, float] = (-30.0, 30.0)      # [m]
    y: tuple[float, float] = (90.0, 110.0)      # [m]
    vx: tuple[float, float] = (-2.0, 2.0)       # [m/s]
    vy: tuple[float, float] = (-12.0, -6.0)     # [m/s]
    theta_deg: tuple[float, float] = (-6.0, 6.0)  # [deg]
    omega: tuple[float, float] = (-0.05, 0.05)  # [rad/s]


@dataclass
class RewardConfig:
    # potential-based shaping weights (applied to normalised quantities)
    w_distance: float = 100.0
    w_x: float = 0.0         # optional extra weight on horizontal error alone
    w_velocity: float = 100.0
    w_tilt: float = 150.0
    w_omega: float = 20.0
    # per-step action costs
    w_fuel: float = 0.3          # at full throttle, per control step
    w_rcs: float = 0.03          # per unit |RCS command|, per control step
    w_gimbal: float = 0.0        # per unit |gimbal command|, per control step
    w_action_rate: float = 0.0   # per unit |action change|, per control step
    # terminal rewards
    success_bonus: float = 100.0
    crash_penalty: float = 100.0
    oob_penalty: float = 100.0
    timeout_penalty: float = 100.0  # hovering out the clock is a mission failure
    # graded touchdown penalties, applied per unit of violation of each
    # landing criterion (smooth the binary success/crash cliff)
    w_impact_vy: float = 25.0   # per m/s of |vy| beyond max_landing_vy
    w_impact_vx: float = 25.0   # per m/s of |vx| beyond max_landing_vx
    w_offpad: float = 5.0       # per m of |x| beyond pad_half_width


@dataclass
class EnvConfig:
    # vehicle (loosely Falcon-9-booster-scale, single sea-level engine)
    gravity: float = 9.81            # [m/s^2]
    dry_mass: float = 22000.0        # [kg]
    initial_propellant: float = 5000.0  # [kg] landing reserve
    max_thrust: float = 845000.0     # [N]
    min_throttle: float = 0.4        # engine cannot run below this fraction
    isp: float = 282.0               # [s] sea-level specific impulse
    length: float = 40.0             # [m] vehicle length (for inertia)
    engine_offset: float = 20.0      # [m] CoM to engine moment arm
    inertia_coef: float = 1.0 / 12.0 # I = coef * m * length^2 (uniform rod)
    # actuators
    # Merlin-class TVC range. From-scratch RL fails at any tested range
    # (8-15 deg, tilted-descent local optimum); the fine-tuning curriculum
    # masters this realistic value, so the canonical vehicle keeps it.
    # Gimbal range remains a sensitivity-study axis.
    gimbal_max_deg: float = 8.0      # [deg] thrust-vector deflection limit
    # RCS sized as a stand-in for grid-fin/aero control authority during
    # unpowered descent (not modelled): enough to trim attitude while
    # coasting, ~8x weaker than the gimbal at full thrust.
    rcs_max_torque: float = 3.0e5    # [N m]
    mass_depletion: bool = True      # if False, inertial mass stays constant
    # aerodynamics / disturbances
    cda: float = 80.0                # [m^2] drag-area product (0 disables drag)
    air_density: float = 1.225       # [kg/m^3]
    wind: WindConfig = field(default_factory=WindConfig)
    # simulation
    dt: float = 0.05                 # [s] control step (20 Hz)
    substeps: int = 5                # physics substeps per control step
    max_episode_steps: int = 600     # 30 s
    # world bounds and landing criteria
    x_max: float = 150.0             # [m]
    y_max: float = 250.0             # [m]
    pad_half_width: float = 10.0     # [m]
    tip_over_deg: float = 80.0       # [deg] abort if |theta| exceeds this
    max_landing_vy: float = 2.5      # [m/s]
    max_landing_vx: float = 1.5      # [m/s]
    max_landing_angle_deg: float = 6.0   # [deg]
    max_landing_omega: float = 0.2   # [rad/s]
    # initial-state distribution
    init: InitRanges = field(default_factory=InitRanges)
    # normalisation scales, shared by observations and shaping potential.
    # Chosen to resolve the landing zone (pad ±10 m, touchdown speeds ~m/s),
    # not the world bounds — the success window must map to O(0.1-1).
    x_scale: float = 40.0
    y_scale: float = 40.0
    v_scale: float = 15.0
    omega_scale: float = 1.0
    # reward
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class EvalConfig:
    freq: int = 25000      # evaluate every N total env steps
    n_episodes: int = 20


@dataclass
class RunConfig:
    run_name: str = "run"
    seed: int = 0
    algo: str = "ppo"                # "ppo" or "sac"
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    device: str = "cpu"
    # optional curriculum: warm-start from a previously trained model.zip
    # (same algo and network architecture required)
    init_from: str = ""
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    # passed through to the SB3 algorithm constructor (net_arch is pulled
    # out into policy_kwargs by train.build_model)
    algo_kwargs: dict = field(default_factory=dict)


def _from_dict(cls, data: dict):
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise KeyError(f"Unknown config key(s) {sorted(unknown)} for {cls.__name__}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _from_dict(hint, value)
        elif (hint is tuple or typing.get_origin(hint) is tuple) and isinstance(value, (list, tuple)):
            value = tuple(value)
        elif hint is float and isinstance(value, (int, str)):
            value = float(value)
        elif hint is int and isinstance(value, str):
            value = int(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(RunConfig, data)


def apply_overrides(cfg: RunConfig, pairs) -> None:
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        value = yaml.safe_load(raw)
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, dict):
            obj[last] = value
        else:
            if not hasattr(obj, last):
                raise SystemExit(f"unknown config key {key!r}")
            if isinstance(getattr(obj, last), tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, last, value)


def _listify(obj):
    if isinstance(obj, dict):
        return {k: _listify(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_listify(v) for v in obj]
    if isinstance(obj, list):
        return [_listify(v) for v in obj]
    return obj


def save_config(cfg: RunConfig, path: str | Path) -> None:
    data = _listify(dataclasses.asdict(cfg))
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
