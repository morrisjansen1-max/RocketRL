# Results

## Final policies on the full environment
(5 seeds, 100 episodes/seed; mean ± std across seeds. Touchdown stats over landing seeds.)

| curriculum | success | vy [m/s] | |x| [m] | tilt [deg] | fuel [kg] |
|---|---|---|---|---|---|
| PPO jump | 0.58 ± 0.53 | -0.63 ± 0.45 | 2.56 ± 0.80 | 0.68 ± 0.33 | 1523 ± 920 |
| PPO staged | 1.00 ± 0.00 | -0.91 ± 0.80 | 2.29 ± 1.08 | 0.61 ± 0.35 | 1248 ± 731 |
| SAC jump | 0.79 ± 0.44 | -0.29 ± 0.12 | 4.49 ± 2.86 | 1.47 ± 0.84 | 1431 ± 495 |
| SAC staged | 0.99 ± 0.01 | -0.52 ± 0.30 | 2.94 ± 0.76 | 1.49 ± 1.32 | 811 ± 115 |

## Per-seed success by curriculum stage

| stage | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| phase1_ppo | 1.00 | 0.67 | 0.00 | 1.00 | 1.00 |
| phase2_ppo_gimbal | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
| phase2_ppo_mass | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 |
| phase2_ppo_wind | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| phase2_ppo | 1.00 | 0.00 | 0.00 | 0.91 | 1.00 |
| phase1_sac | 1.00 | 1.00 | 0.82 | 0.98 | 1.00 |
| phase2_sac_gimbal | 1.00 | 1.00 | 0.99 | 1.00 | 0.98 |
| phase2_sac_mass | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| phase2_sac_wind | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| phase2_sac | 1.00 | 0.96 | 1.00 | 1.00 | 0.00 |
