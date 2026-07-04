"""Planar (2-D) rocket soft-landing environment, Gymnasium-compatible.

The rocket lives in a 2-D world with x pointing right and y pointing up.
The landing pad sits at the origin — y = 0 is where the legs touch down,
and the vehicle's centre of mass position (x, y) is tracked relative to
that point. Negative y means you've hit the ground. Pitch theta is the
angle off vertical, positive toward +x, with omega as its time derivative.

Dynamics are rigid-body with mass = dry + remaining propellant (or held
constant if mass_depletion is off). The main engine fires along the body
axis, gimbal-deflected by delta, and sits engine_offset metres below the
CoM, so it produces both thrust and a torque (-F sin(delta) * engine_offset).
A cold-gas RCS adds direct torque on top of that. Propellant burns at
F / (Isp * g0) and the engine cuts out when the tank runs dry. Aero drag
is optional: -0.5 rho CdA |v_rel| v_rel at the CoM, where v_rel is
relative to a wind field (steady component plus Ornstein-Uhlenbeck gusts).
Integration is semi-implicit Euler, with substeps physics steps per
control step.

Actions are a 3-vector in [-1, 1]:
  a[0]  throttle — anything <= 0 cuts the engine; above that it maps
        linearly onto [min_throttle, 1] * max_thrust to respect the
        minimum stable throttle a real engine needs.
  a[1]  gimbal command, scaled to [-gimbal_max, +gimbal_max].
  a[2]  RCS command, scaled to [-rcs_max_torque, +rcs_max_torque].

The observation is a 7-vector, loosely normalised to [-1, 1]:
  [x/xs, y/ys, vx/vs, vy/vs, theta/(pi/2), omega/ws, propellant fraction]
Wind is deliberately excluded — the policy should handle it blind.

Reward is potential-based shaping toward a slow, upright, centred hover,
with per-step action costs and terminal bonuses or penalties on landing.
All the weights live in RewardConfig.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rocket_rl.config import EnvConfig

G0 = 9.80665 
THETA_SCALE = math.pi / 2.0


class OUGust:
    def __init__(self, sigma: float, tau: float):
        self.sigma = sigma
        self.tau = tau
        self.value = 0.0

    def reset(self, rng: np.random.Generator) -> None:
        self.value = float(rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0

    def sample(self, rng: np.random.Generator, dt: float) -> float:
        if self.sigma <= 0:
            self.value = 0.0
        else:
            a = math.exp(-dt / self.tau)
            self.value = a * self.value + self.sigma * math.sqrt(1.0 - a * a) * float(rng.normal())
        return self.value


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        c = self.cfg
        self.gimbal_max = math.radians(c.gimbal_max_deg)
        self.tip_over = math.radians(c.tip_over_deg)
        self.m0 = c.dry_mass + c.initial_propellant
        self.gust = OUGust(c.wind.gust_sigma, c.wind.gust_tau)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-15.0, 15.0, shape=(7,), dtype=np.float32)

        self.x = self.y = self.vx = self.vy = self.theta = self.omega = 0.0
        self.m_prop = c.initial_propellant
        self.wind_speed = 0.0
        self.wind_sign = 1.0
        self.fuel_used = 0.0
        self.t = 0.0
        self.steps = 0
        self.potential = 0.0
        self.prev_action = np.zeros(3)
        self.last_throttle = 0.0
        self.last_gimbal = 0.0
        self.last_rcs = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c, rng = self.cfg, self.np_random
        i = c.init
        self.x = float(rng.uniform(*i.x))
        self.y = float(rng.uniform(*i.y))
        self.vx = float(rng.uniform(*i.vx))
        self.vy = float(rng.uniform(*i.vy))
        self.theta = math.radians(float(rng.uniform(*i.theta_deg)))
        self.omega = float(rng.uniform(*i.omega))
        self.m_prop = c.initial_propellant
        self.fuel_used = 0.0
        self.t = 0.0
        self.steps = 0
        self.gust.reset(rng)
        self.wind_sign = 1.0
        if c.wind.enabled and c.wind.randomize_direction and rng.random() < 0.5:
            self.wind_sign = -1.0
        self.wind_speed = self.wind_sign * c.wind.mean_speed + self.gust.value if c.wind.enabled else 0.0
        self.prev_action = np.zeros(3)
        self.last_throttle = self.last_gimbal = self.last_rcs = 0.0
        self.potential = self._potential()
        return self._obs(), {}

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).flatten(), -1.0, 1.0)
        a_thr, a_gim, a_rcs = float(a[0]), float(a[1]), float(a[2])

        throttle = 0.0
        if a_thr > 0.0 and self.m_prop > 0.0:
            throttle = c.min_throttle + a_thr * (1.0 - c.min_throttle)
        gimbal = a_gim * self.gimbal_max
        rcs_torque = a_rcs * c.rcs_max_torque
        self.last_throttle, self.last_gimbal, self.last_rcs = throttle, gimbal, rcs_torque

        if c.wind.enabled:
            self.wind_speed = self.wind_sign * c.wind.mean_speed + self.gust.sample(self.np_random, c.dt)

        h = c.dt / c.substeps
        outcome = None
        for _ in range(c.substeps):
            m = c.dry_mass + self.m_prop if c.mass_depletion else self.m0
            inertia = c.inertia_coef * m * c.length**2
            thrust = throttle * c.max_thrust if self.m_prop > 0.0 else 0.0

            fx = thrust * math.sin(self.theta + gimbal)
            fy = thrust * math.cos(self.theta + gimbal)
            if c.cda > 0.0:
                vrx = self.vx - self.wind_speed
                vry = self.vy
                vrel = math.hypot(vrx, vry)
                fx -= 0.5 * c.air_density * c.cda * vrel * vrx
                fy -= 0.5 * c.air_density * c.cda * vrel * vry
            torque = -thrust * math.sin(gimbal) * c.engine_offset + rcs_torque

            self.vx += (fx / m) * h
            self.vy += (fy / m - c.gravity) * h
            self.omega += (torque / inertia) * h
            self.x += self.vx * h
            self.y += self.vy * h
            self.theta += self.omega * h

            burned = min(self.m_prop, thrust / (c.isp * G0) * h)
            self.m_prop -= burned
            self.fuel_used += burned
            self.t += h

            if self.y <= 0.0:
                outcome = "touchdown"
                break
            if abs(self.x) > c.x_max or self.y > c.y_max:
                outcome = "out_of_bounds"
                break
            if abs(self.theta) > self.tip_over:
                outcome = "tip_over"
                break

        self.steps += 1
        terminated = outcome is not None
        truncated = (not terminated) and self.steps >= c.max_episode_steps

        r = c.reward
        new_potential = self._potential()
        reward = new_potential - self.potential
        self.potential = new_potential
        reward -= r.w_fuel * throttle
        reward -= r.w_rcs * abs(a_rcs)
        reward -= r.w_gimbal * abs(a_gim)
        reward -= r.w_action_rate * float(np.abs(a - self.prev_action).sum())
        self.prev_action = a

        info: dict[str, Any] = {}
        if outcome == "touchdown":
            success = (
                abs(self.vy) <= c.max_landing_vy
                and abs(self.vx) <= c.max_landing_vx
                and abs(self.theta) <= math.radians(c.max_landing_angle_deg)
                and abs(self.omega) <= c.max_landing_omega
                and abs(self.x) <= c.pad_half_width
            )
            if success:
                reward += r.success_bonus
                outcome = "success"
            else:
                reward -= r.crash_penalty
                outcome = "crash"
                reward -= r.w_impact_vy * max(0.0, abs(self.vy) - c.max_landing_vy)
                reward -= r.w_impact_vx * max(0.0, abs(self.vx) - c.max_landing_vx)
                reward -= r.w_offpad * max(0.0, abs(self.x) - c.pad_half_width)
            info.update(
                touchdown_x=self.x,
                touchdown_vx=self.vx,
                touchdown_vy=self.vy,
                touchdown_theta=self.theta,
                touchdown_omega=self.omega,
            )
        elif outcome == "out_of_bounds":
            reward -= r.oob_penalty
        elif outcome == "tip_over":
            reward -= r.crash_penalty

        if terminated or truncated:
            if outcome is None:
                outcome = "timeout"
                reward -= r.timeout_penalty
            info["termination"] = outcome
            info["is_success"] = outcome == "success"
            info["fuel_used"] = self.fuel_used

        return self._obs(), float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        c = self.cfg
        return np.array(
            [
                self.x / c.x_scale,
                self.y / c.y_scale,
                self.vx / c.v_scale,
                self.vy / c.v_scale,
                self.theta / THETA_SCALE,
                self.omega / c.omega_scale,
                self.m_prop / c.initial_propellant,
            ],
            dtype=np.float32,
        )

    def _potential(self) -> float:
        c, r = self.cfg, self.cfg.reward
        dist = math.hypot(self.x / c.x_scale, self.y / c.y_scale)
        speed = math.hypot(self.vx / c.v_scale, self.vy / c.v_scale)
        return -(
            r.w_distance * dist
            + r.w_x * abs(self.x) / c.x_scale
            + r.w_velocity * speed
            + r.w_tilt * abs(self.theta) / THETA_SCALE
            + r.w_omega * abs(self.omega) / c.omega_scale
        )

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon, Rectangle

        c = self.cfg
        fig = Figure(figsize=(5.0, 5.0), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_xlim(-c.x_max, c.x_max)
        ax.set_ylim(-15.0, c.y_max)
        ax.set_aspect("auto")
        ax.add_patch(Rectangle((-c.x_max, -15.0), 2 * c.x_max, 15.0, color="#8a7f70"))
        ax.add_patch(Rectangle((-c.pad_half_width, -2.0), 2 * c.pad_half_width, 2.0, color="#3c7a3c"))

        def body_to_world(px: float, py: float) -> tuple[float, float]:
            ct, st = math.cos(self.theta), math.sin(self.theta)
            return self.x + px * ct + py * st, self.y - px * st + py * ct

        w, length = 3.7, c.length
        body = [body_to_world(*p) for p in [(-w / 2, 0), (w / 2, 0), (w / 2, length), (0, length + 5), (-w / 2, length)]]
        ax.add_patch(Polygon(body, closed=True, color="#404a58"))
        if self.last_throttle > 0.0:
            fl = 18.0 * self.last_throttle
            ang = self.theta + self.last_gimbal
            base = body_to_world(0.0, 0.0)
            tip = (base[0] - fl * math.sin(ang), base[1] - fl * math.cos(ang))
            left = body_to_world(-w / 3, 0.0)
            right = body_to_world(w / 3, 0.0)
            ax.add_patch(Polygon([left, right, tip], closed=True, color="#e8862e"))
        ax.set_title(
            f"t={self.t:5.1f}s  v=({self.vx:+5.1f},{self.vy:+5.1f}) m/s  "
            f"θ={math.degrees(self.theta):+5.1f}°  fuel={self.m_prop / c.initial_propellant:4.0%}",
            fontsize=9,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.tight_layout()
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
        return img
