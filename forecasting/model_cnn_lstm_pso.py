import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from utils.metrics import mape

class PSO:
    def __init__(self, bounds, n_particles=12, iters=10, w=0.7, c1=1.4, c2=1.4, seed=42):
        rng = np.random.default_rng(seed)
        self.bounds = bounds; self.np = n_particles; self.iters = iters
        self.w, self.c1, self.c2 = w, c1, c2
        self.pos = np.array([rng.uniform(l,u,size=len(bounds)) for (l,u) in bounds for _ in [0]]).reshape(self.np, len(bounds))
        self.vel = np.zeros_like(self.pos); self.best_pos = self.pos.copy()
        self.best_score = np.full(self.np, np.inf); self.gbest_pos = None; self.gbest_score = np.inf
        self.rng = rng
    def ask(self): return self.pos
    def tell(self, scores):
        imp = scores < self.best_score
        self.best_score[imp] = scores[imp]; self.best_pos[imp] = self.pos[imp]
        gidx = np.argmin(scores)
        if scores[gidx] < self.gbest_score: self.gbest_score = scores[gidx]; self.gbest_pos = self.pos[gidx].copy()
        r1 = self.rng.random(self.pos.shape); r2 = self.rng.random(self.pos.shape)
        self.vel = self.w*self.vel + self.c1*r1*(self.best_pos-self.pos) + self.c2*r2*(self.gbest_pos-self.pos)
        self.pos += self.vel
        for d,(l,u) in enumerate(self.bounds): self.pos[:,d] = np.clip(self.pos[:,d], l, u)

def build_cnn_lstm(input_shape, conv_filters=64, lstm_units=128, dropout=0.3, lr=5e-4, horizon=24):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(int(conv_filters), kernel_size=3, padding='causal', activation='relu')(inputs)
    x = layers.Conv1D(int(conv_filters), kernel_size=5, padding='causal', activation='relu')(x)
    x = layers.MaxPool1D(2)(x)
    x = layers.LSTM(int(lstm_units), return_sequences=False)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(horizon)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    return model

def evaluate_cfg(cfg, Xtr, ytr, Xva, yva, horizon=24, epochs=5, batch_size=64, verbose=0):
    conv_filters = int(cfg[0]); lstm_units = int(cfg[1]); dropout = float(cfg[2]); lr = float(cfg[3])
    model = build_cnn_lstm(Xtr.shape[1:], conv_filters, lstm_units, dropout, lr, horizon)
    model.fit(Xtr, ytr, validation_data=(Xva,yva), epochs=epochs, batch_size=int(batch_size), verbose=verbose)
    yp = model.predict(Xva, verbose=0)
    return mape(yva.ravel(), yp.ravel()), model

def quantile_intervals(pred, resid_std, q=0.95):
    from scipy.stats import norm
    z = norm.ppf((1+q)/2)
    lower = pred - z*resid_std; upper = pred + z*resid_std
    return lower, upper
