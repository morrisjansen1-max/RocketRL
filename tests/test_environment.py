"""Physics, API, and reproducibility tests for RocketLandingEnv."""

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from config import EnvConfig, InitRanges, WindConfig, load_config
from environment.rocket_env import G0, RocketLandingEnv

ENGINE_OFF = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
FULL_THRUST = np.array([1.0, 0.0, 0.0], dtype=np.float32)


def quiet_cfg(**overrides) -> EnvConfig:
    """Phase-1-style config: no gimbal, ideal torque, fixed mass, no aero."""
    defaults = dict(gimbal_max_deg=0.0, rcs_max_torque=2.5e6, mass_depletion=False, cda=0.0)
    defaults.update(overrides)
    return EnvConfig(**defaults)


def place(env, x=0.0, y=100.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0):
    """White-box helper: pin the state after a reset."""
    env.reset(seed=0)
    env.x, env.y, env.vx, env.vy, env.theta, env.omega = x, y, vx, vy, theta, omega
    env.potential = env._potential()


def test_gymnasium_api():
    check_env(RocketLandingEnv(EnvConfig()), skip_render_check=True)


def test_render_returns_frame():
    env = RocketLandingEnv(quiet_cfg(), render_mode="rgb_array")
    env.reset(seed=0)
    env.step(FULL_THRUST)
    frame = env.render()
    assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3


def test_free_fall_matches_discrete_solution():
    cfg = quiet_cfg()
    env = RocketLandingEnv(cfg)
    place(env, x=5.0, y=150.0, vx=1.0, vy=0.0)
    x0, y0, vx0 = env.x, env.y, env.vx
    n_steps = 20
    for _ in range(n_steps):
        env.step(ENGINE_OFF)
    h = cfg.dt / cfg.substeps
    n = n_steps * cfg.substeps
    # semi-implicit Euler closed form under constant acceleration
    vy_expected = -cfg.gravity * h * n
    y_expected = y0 - cfg.gravity * h * h * n * (n + 1) / 2
    assert env.vy == pytest.approx(vy_expected, rel=1e-12)
    assert env.y == pytest.approx(y_expected, rel=1e-12)
    assert env.vx == pytest.approx(vx0)
    assert env.x == pytest.approx(x0 + vx0 * h * n)
    assert env.theta == 0.0 and env.omega == 0.0


def test_thrust_decelerates_descent():
    env_off = RocketLandingEnv(quiet_cfg())
    env_on = RocketLandingEnv(quiet_cfg())
    place(env_off, vy=-20.0)
    place(env_on, vy=-20.0)
    for _ in range(10):
        env_off.step(ENGINE_OFF)
        env_on.step(FULL_THRUST)
    assert env_on.vy > env_off.vy
    # T/W ~ 3 upright: net acceleration must be upward
    assert env_on.vy > -20.0


def test_rcs_torque_sign():
    env = RocketLandingEnv(quiet_cfg())
    place(env)
    env.step(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert env.omega > 0.0  # positive RCS tilts toward +x
    assert env.theta > 0.0


def test_gimbal_torque_sign_and_lateral_force():
    cfg = quiet_cfg(gimbal_max_deg=8.0)
    env = RocketLandingEnv(cfg)
    place(env)
    env.step(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    # thrust deflected toward +x pushes the vehicle to +x and torques it to -theta
    assert env.vx > 0.0
    assert env.omega < 0.0


def test_propellant_depletion_and_engine_cutout():
    cfg = quiet_cfg(mass_depletion=True, initial_propellant=50.0)  # tiny tank
    env = RocketLandingEnv(cfg)
    place(env, y=200.0)
    env.step(FULL_THRUST)
    assert env.m_prop < 50.0
    assert env.fuel_used > 0.0
    # burn the tank dry, then thrust must have no effect
    for _ in range(20):
        env.step(FULL_THRUST)
    assert env.m_prop == 0.0
    vy_before = env.vy
    env.step(FULL_THRUST)
    h_total = cfg.dt
    assert env.vy == pytest.approx(vy_before - cfg.gravity * h_total, rel=1e-9)


def test_mass_flow_rate():
    cfg = quiet_cfg(mass_depletion=True)
    env = RocketLandingEnv(cfg)
    place(env, y=200.0)
    env.step(FULL_THRUST)
    expected = cfg.max_thrust / (cfg.isp * G0) * cfg.dt
    assert env.fuel_used == pytest.approx(expected, rel=1e-9)


def test_successful_touchdown():
    env = RocketLandingEnv(quiet_cfg())
    place(env, x=1.0, y=0.03, vx=0.1, vy=-0.8)
    obs, reward, terminated, truncated, info = env.step(ENGINE_OFF)
    assert terminated and not truncated
    assert info["termination"] == "success"
    assert info["is_success"] is True
    assert reward > 50.0  # success bonus dominates
    assert abs(info["touchdown_vy"]) <= env.cfg.max_landing_vy


def test_hard_impact_is_crash():
    env = RocketLandingEnv(quiet_cfg())
    place(env, y=1.0, vy=-25.0)
    _, reward, terminated, _, info = env.step(ENGINE_OFF)
    assert terminated
    assert info["termination"] == "crash"
    assert info["is_success"] is False
    assert reward < 0.0


def test_graded_impact_penalty():
    """A harder impact must yield a lower terminal reward (smooth gradient)."""

    def crash_reward(vy):
        env = RocketLandingEnv(quiet_cfg())
        place(env, y=0.02, vy=vy)
        _, reward, terminated, _, info = env.step(ENGINE_OFF)
        assert terminated and info["termination"] == "crash"
        return reward

    assert crash_reward(-8.0) > crash_reward(-12.0) > crash_reward(-20.0)


def test_off_pad_touchdown_is_crash():
    env = RocketLandingEnv(quiet_cfg())
    place(env, x=30.0, y=0.03, vy=-0.5)  # gentle but far off the pad
    _, _, terminated, _, info = env.step(ENGINE_OFF)
    assert terminated and info["termination"] == "crash"


def test_out_of_bounds():
    env = RocketLandingEnv(quiet_cfg())
    place(env, x=149.0, y=100.0, vx=40.0)
    _, reward, terminated, _, info = env.step(ENGINE_OFF)
    assert terminated and info["termination"] == "out_of_bounds"
    assert reward < 0.0


def test_tip_over_aborts():
    env = RocketLandingEnv(quiet_cfg())
    place(env, theta=math.radians(75.0), omega=3.0)
    terminated, info = False, {}
    for _ in range(5):
        _, _, terminated, _, info = env.step(ENGINE_OFF)
        if terminated:
            break
    assert terminated and info["termination"] == "tip_over"


def test_timeout_truncates_with_penalty():
    cfg = quiet_cfg(max_episode_steps=5)
    env = RocketLandingEnv(cfg)
    env.reset(seed=0)
    terminated = truncated = False
    reward = 0.0
    for _ in range(5):
        _, reward, terminated, truncated, info = env.step(FULL_THRUST)
    assert truncated and not terminated
    assert info["termination"] == "timeout"
    assert reward < -cfg.reward.timeout_penalty / 2  # hovering the clock out is punished


def test_seed_determinism():
    cfg = EnvConfig(wind=WindConfig(enabled=True, mean_speed=5.0, gust_sigma=2.0))
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1, 1, size=(50, 3)).astype(np.float32)

    def run(seed):
        env = RocketLandingEnv(cfg)
        env.reset(seed=seed)
        states = []
        for a in actions:
            env.step(a)
            states.append((env.x, env.y, env.theta, env.wind_speed))
        return np.array(states)

    assert np.array_equal(run(42), run(42))
    assert not np.array_equal(run(42), run(43))


def test_wind_pushes_rocket():
    cfg = quiet_cfg(cda=80.0, wind=WindConfig(enabled=True, mean_speed=8.0, gust_sigma=0.0))
    env = RocketLandingEnv(cfg)
    place(env, y=200.0)
    for _ in range(20):
        env.step(ENGINE_OFF)
    assert env.vx > 0.0  # dragged downwind (+x)


def test_wind_direction_randomization():
    cfg = quiet_cfg(
        cda=80.0,
        wind=WindConfig(enabled=True, mean_speed=8.0, gust_sigma=0.0, randomize_direction=True),
    )
    env = RocketLandingEnv(cfg)
    env.reset(seed=11)
    signs = set()
    for _ in range(50):
        env.reset()
        signs.add(env.wind_sign)
    assert signs == {1.0, -1.0}


def test_potential_prefers_target_state():
    env = RocketLandingEnv(quiet_cfg())
    place(env, x=50.0, y=100.0, vx=5.0, vy=-20.0, theta=0.3)
    far = env._potential()
    place(env, x=2.0, y=10.0, vx=0.2, vy=-1.0, theta=0.01)
    near = env._potential()
    assert near > far


def test_observation_normalised_and_typed():
    env = RocketLandingEnv(EnvConfig())
    obs, _ = env.reset(seed=3)
    assert obs.shape == (7,) and obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert np.all(np.abs(obs) <= 4.0)  # initial states stay within a few scale units
    assert obs[6] == pytest.approx(1.0)  # full tank


def test_config_yaml_roundtrip(tmp_path):
    from config import RunConfig, save_config

    cfg = RunConfig()
    cfg.env.init = InitRanges(x=(-1.0, 1.0))
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.env.init.x == (-1.0, 1.0)
    assert loaded.env.gimbal_max_deg == cfg.env.gimbal_max_deg


def test_apply_overrides():
    from config import RunConfig, apply_overrides

    cfg = RunConfig()
    apply_overrides(cfg, [
        "env.wind.mean_speed=10",
        "env.wind.enabled=true",
        "algo_kwargs.learning_rate=lin_1.0e-4",
        "env.reward.w_fuel=1.5",
        "env.init.x=[-5, 5]",
        "total_timesteps=123",
    ])
    assert cfg.env.wind.mean_speed == 10
    assert cfg.env.wind.enabled is True
    assert cfg.algo_kwargs["learning_rate"] == "lin_1.0e-4"
    assert cfg.env.reward.w_fuel == 1.5
    assert cfg.env.init.x == (-5, 5)  # lists coerce to tuples
    assert cfg.total_timesteps == 123
    with pytest.raises(SystemExit):
        apply_overrides(cfg, ["env.no_such_key=1"])


def test_phase1_config_loads():
    cfg = load_config("configs/phase1_ppo.yaml")
    assert cfg.algo == "ppo"
    assert cfg.env.gimbal_max_deg == 0.0
    assert cfg.env.mass_depletion is False
    assert cfg.env.rcs_max_torque == pytest.approx(2.5e6)
    assert cfg.algo_kwargs["net_arch"] == [64, 64]
    # untouched keys fall back to canonical defaults
    assert cfg.env.max_thrust == pytest.approx(845000.0)
