# Appendix A — Parameter Settings

## A.1 Forecasting
- Lookback=24, Horizon=24
- PSO: particles=10, iters=5, w=0.7, c1=1.4, c2=1.4
- Bounds: conv_filters[32,128], lstm_units[64,256], dropout[0.1,0.5], lr[1e-4,1e-3]
- Training: epochs_cv=4, epochs_final=8, batch=64
- Quantiles: [0.05, 0.95] (pinball loss)

## A.2 Optimization (NSGA-II)
- Population=60, Generations=40, Crossover=0.9, Mutation sigma=0.12
- Bounds: wind_kw[10,120], battery_kwh[40,400], diesel_kw[5,80]
- Scenarios: 300 (Gaussian-copula loads)
- Constraint: LPSP ≤ 5%

## A.3 EMS
- Agent: DQN (default) or PPO
- Episodes=400, gamma=0.98
- ε (DQN): start=1.0 → 0.05 (decay=0.995)
- PPO: clip=0.2, epochs=5, batch=256, λ-GAE=0.95
