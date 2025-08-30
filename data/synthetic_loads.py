import numpy as np
from scipy.stats import norm

def daily_base_curve(hours=24, peak_shift=-1.0):
    t = np.linspace(0, 2*np.pi, hours, endpoint=False)
    curve = 10 + 4*np.sin(t + peak_shift) + 1.2*np.sin(2*t)
    return np.clip(curve, 0, None)

def gaussian_copula_samples(n, corr=0.6, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, corr],[corr,1.0]])
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(n,2))
    u = norm.cdf(z @ L.T)
    return u

def generate_hourly_profiles(days=365, corr=0.6, seed=0):
    rng = np.random.default_rng(seed)
    base_res = daily_base_curve(24, peak_shift=-1.0)
    base_com = daily_base_curve(24, peak_shift=-0.3) + 2.0
    profiles = []
    for d in range(days):
        u = gaussian_copula_samples(24, corr=corr, seed=seed+d)
        res = base_res * (0.8 + 0.4*u[:,0]) + rng.normal(0, 0.5, 24)
        com = base_com * (0.8 + 0.4*u[:,1]) + rng.normal(0, 0.6, 24)
        res = np.clip(res, 0, None); com = np.clip(com, 0, None)
        profiles.append(res + com*0.6)
    return np.array(profiles)
