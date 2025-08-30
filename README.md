# Smart Microgrid – Unified Forecasting, Sizing, and EMS (Reproducible)

This repository provides **workable reference code** for the paper:
**“Smart Microgrid Planning with Unified Probabilistic Forecasting and Deep Reinforcement Learning Control.”**

It contains three modules:
1) **Forecasting**: CNN–LSTM with **PSO** hyperparameter tuning and optional **quantile regression**.
2) **Optimization**: **NSGA-II** multi-objective sizing with **Monte-Carlo** (Gaussian-copula synthetic load profiles).
3) **EMS**: **DQN** (default) and **PPO** (discrete) agents in a minimal microgrid environment.

All scripts save the **exact config used** to `artifacts/*_config_used.json` to ensure reproducibility.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Forecasting
python forecasting/train_forecast.py

# 2) Optimization
python optimization/run_optimization.py

# 3) EMS (agent configurable via configs/ems.json: "dqn" or "ppo")
python ems/train_ems.py
```

## Configuration
Hyperparameters are stored in `configs/*.json`. You can modify them without touching code.

## Synthetic Data
- **Wind**: programmatically generated diurnal + seasonal components.
- **Load**: **Gaussian copula**–based synthetic daily profiles (residential + commercial mix).

## License
MIT (for academic reuse).
