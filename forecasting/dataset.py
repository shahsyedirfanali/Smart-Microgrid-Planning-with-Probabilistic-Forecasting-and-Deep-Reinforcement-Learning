import numpy as np, pandas as pd

def make_synthetic_wind(n_days=365, seed=42):
    rng = np.random.default_rng(seed)
    hours = n_days*24; t = np.arange(hours)
    daily = 2.0*np.sin(2*np.pi*(t%24)/24.0 - 0.5) + 4.5
    seasonal = 0.8*np.sin(2*np.pi*t/(24*90))
    noise = rng.normal(0, 0.6, hours)
    wind = np.clip(daily + seasonal + noise, 0, None)
    df = pd.DataFrame({'wind_mps': wind})
    df.index = pd.date_range('2022-01-01', periods=hours, freq='H')
    return df

def make_sequences(series, lookback=24, horizon=24):
    X, y = [], []
    s = series.values.astype('float32').reshape(-1,1)
    for i in range(len(s)-lookback-horizon):
        X.append(s[i:i+lookback]); y.append(s[i+lookback:i+lookback+horizon].ravel())
    return np.array(X), np.array(y)
