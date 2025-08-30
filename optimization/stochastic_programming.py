import numpy as np
from data.synthetic_loads import generate_hourly_profiles

def generate_scenarios(n=100, seed=0):
    rng = np.random.default_rng(seed)
    profiles = generate_hourly_profiles(days=max(n,365), corr=0.6, seed=seed)
    scenarios = []
    for i in range(n):
        wind = rng.uniform(3.0, 9.0)
        day = profiles[i % len(profiles)]
        scenarios.append((wind, day))
    return scenarios

def cost_params():
    return dict(wind_kw=900.0, batt_kwh=220.0, diesel_kw=400.0, fuel=0.6)
