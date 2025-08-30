import os, json, numpy as np
from sklearn.model_selection import train_test_split
from forecasting.dataset import make_synthetic_wind, make_sequences
from forecasting.model_cnn_lstm_pso import PSO, evaluate_cfg, build_cnn_lstm, quantile_intervals
from utils.seed import set_seed
from utils.metrics import mape, rmse
from tensorflow import keras
import tensorflow as tf

os.makedirs('artifacts', exist_ok=True)
set_seed(42)
cfg = json.load(open('configs/forecast.json'))
lookback = cfg['lookback']; horizon = cfg['horizon']

df = make_synthetic_wind(n_days=365)
X, y = make_sequences(df['wind_mps'], lookback=lookback, horizon=horizon)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg['train']['test_size'], shuffle=False)
Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=cfg['train']['val_size'], shuffle=False)

b = cfg['bounds']
bounds = [(b['conv_filters'][0], b['conv_filters'][1]), (b['lstm_units'][0], b['lstm_units'][1]), (b['dropout'][0], b['dropout'][1]), (b['lr'][0], b['lr'][1])]
pso = PSO(bounds, n_particles=cfg['pso']['particles'], iters=cfg['pso']['iters'], w=cfg['pso']['w'], c1=cfg['pso']['c1'], c2=cfg['pso']['c2'])
for it in range(pso.iters):
    scores = []
    for c in pso.ask():
        s, _ = evaluate_cfg(c, Xtr, ytr, Xva, yva, epochs=cfg['train']['epochs_cv'], batch_size=cfg['train']['batch_size'], verbose=0)
        scores.append(s)
    pso.tell(np.array(scores))
    print(f"[PSO] iter {it+1}/{pso.iters} best MAPE={pso.gbest_score:.3f}, cfg={pso.gbest_pos}")

mape_score, model = evaluate_cfg(pso.gbest_pos, np.vstack([Xtr,Xva]), np.vstack([ytr,yva]), Xte, yte, epochs=cfg['train']['epochs_final'], batch_size=cfg['train']['batch_size'], verbose=0)

yp = model.predict(Xte, verbose=0)
resid = (yte - yp).ravel(); resid_std = np.std(resid)
lower, upper = quantile_intervals(yp, resid_std, q=0.95)

from matplotlib import pyplot as plt
plt.figure()
plt.plot(yte[:50].ravel(), label='true'); plt.plot(yp[:50].ravel(), label='pred')
plt.fill_between(np.arange(50*horizon), lower[:50].ravel(), upper[:50].ravel(), alpha=0.2, label='95% PI')
plt.legend(); plt.title('Forecast with 95% Prediction Intervals'); plt.tight_layout(); plt.savefig('artifacts/forecast_intervals.png')

model.save('artifacts/forecast_cnn_lstm.keras')
meta = {'best_cfg': pso.gbest_pos.tolist(), 'test_mape': float(mape(yte, yp)), 'test_rmse': float(rmse(yte, yp)), 'resid_std': float(resid_std)}
json.dump(meta, open('artifacts/forecast_meta.json','w'), indent=2)
json.dump(cfg, open('artifacts/forecast_config_used.json','w'), indent=2)
print('[Done] Forecasting. Meta:', meta)
