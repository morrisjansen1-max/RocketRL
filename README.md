# RocketRL — Landing a Rocket in 2D with Reinforcement Learning

A model-free reinforcement-learning study of autonomous powered soft-landing.
A custom continuous 2-D rocket environment is built from scratch, and two continuous-control 
algorithms PPO (on-policy) and SAC (off-policy) are trained to land a booster on a pad,
then compared head-to-head.

The vehicle has a gimballed main engine, a reaction-control system (RCS),
propellant depletion, aerodynamic drag, and unobserved wind. 
Learning is warm-started from a simple stage-1 policy and then
adapted to the full environment two ways:

- Jump: one fine-tune that adds all realism features at once.
- Staged curriculum: a matched four-stage chain that adds one realism
  feature at a time (gimbal → mass depletion → drag + wind).

The headline result is that the gradual curriculum is what buys reliability:
the single jump lands only 3/5 (PPO) and 4/5 (SAC) seeds, while the matched
four-stage curriculum lands every seed for both algorithms. Given the same
curriculum, PPO and SAC trade off cleanly, PPO is faster in wall-clock and
places the vehicle more accurately; SAC is ~6× more sample-efficient, lands more
gently, and is the most fuel-efficient policy in the study. Both algorithms
independently rediscover the minimum-fuel "hoverslam" (coast with the engine
off, then a single late burn) purely from reward.

This repository contains the environment, the training/evaluation pipeline, the
experiment runners that reproduce every policy in the report, and the analysis
scripts that regenerate the tables and figures. It accompanies the report
"Reinforcement Learning: Comparing Soft Actor-Critic and Proximal Policy
Optimization Algorithms for Landing a Rocket".

---

## Requirements & setup

- Python 3.11+, a CPU is sufficient (all configs use `device: cpu` and 8
  parallel environments; the study was run on an Apple M-series laptop).
- Everything is pinned in `requirements.txt` — the core stack is
  [Gymnasium](https://gymnasium.farama.org/) 1.3,
  [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) 2.9,
  PyTorch 2.12, NumPy 2.5, pandas 3.0, and Matplotlib 3.11.

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The experiment runners invoke the interpreter at the hard-coded path
`.venv/bin/python`, so create the virtual environment at `.venv/` in the repo
root as above (or edit the `PY` constant at the top of each runner).

Verify the environment physics and pipeline with the test suite:

```bash
python -m pytest tests/
```

---

## Quick start (single run)

Before reproducing the whole study, you can train and evaluate one policy to
confirm the pipeline works. This trains the from-scratch PPO stage-1 policy and
evaluates it:

```bash
# train -> writes runs/phase1_ppo_s0/
python -m train --config configs/phase1_ppo.yaml --run-name phase1_ppo --seed 0 --overwrite

# evaluate -> writes runs/phase1_ppo_s0/eval/
python -m evaluate --run-dir runs/phase1_ppo_s0 --n-episodes 100 --save-trajectories 10
```

---

## Reproducing the full study

The four experiment runners must be executed in this order because of data
dependencies (each stage warm-starts from a policy trained by an earlier stage).
Every runner is resumable, a step is skipped when its completion artifact
already exists — and supports `--dry-run` (list pending jobs) and
`--only <substring>` (filter jobs by name). Wall-clock estimates are for a
laptop CPU.

```bash
# 1. Curriculum chains: the PPO and SAC four-stage curricula (the core of the study).
#    Trains the from-scratch phase-1 roots AND the full staged chains
#    (phase1 -> gimbal -> mass -> wind), 5 seeds each. ~22 h (SAC dominates).
python run_matrix_staged.py

# 2. Single-jump baselines: phase1 -> full env in one fine-tune, 5 seeds.
#    Depends on step 1 (fine-tunes from the phase-1 roots). ~1.5 h.
python run_matrix_jump.py

# 3. Sensitivity + robustness sweeps: hyperparameter sweeps (from scratch),
#    fuel-weight and gimbal-range sweeps (fine-tune), and wind-robustness
#    training/evaluation across unseen wind levels, for both algorithms.
#    Depends on step 1 (fine-tunes from / evaluates the curriculum). ~30 h.
python sensitivity_analysis.py

# 4. Analysis: collect every eval, write the result tables, and render all figures.
#    Depends on steps 1-3 (reads their run folders).
python analysis_figures.py
```

After step 4 you'll have `results/all_evals.csv`, `results/comparison.md`, and
the vector-PDF figures in `figures/`.

`analysis_figures.py` also accepts a subcommand to render a subset:
`all` (default), `results`, `sensitivity`, or `table`.

---

## How the pieces fit together

The project is a small, layered pipeline. Data flows left to right:

```
config YAML ──> train.py ──> runs/<name>_s<seed>/ ──> evaluate.py ──> runs/.../eval/
   (configs/)      │            (checkpoints, logs)        │           (summary.json,
                   │                                       │            traj_*.npz)
              environment/rocket_env.py                    │
                   ▲                                       ▼
              config.py (dataclasses)          analysis_figures.py / plotting.py
                                                     │
                                                     ▼
                                          results/ (CSV, comparison.md)
                                          figures/ (PDF figures)
```

- `config.py` defines the config schema; every runnable script loads a YAML
  into it.
- `train.py` builds the environment from a config and runs SB3 PPO/SAC,
  writing a self-contained run folder.
- `evaluate.py` loads a trained run and rolls out episodes, writing success
  statistics and recorded trajectories.
- `plotting.py` / `analysis_figures.py` read the run folders back and render
  figures and tables.
- The three `run_matrix_*.py` / `sensitivity_analysis.py` scripts are thin
  orchestration layers: each builds a list of `train`/`evaluate` subprocess jobs
  and runs them, so the actual work always goes through `train.py` and
  `evaluate.py`.