import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

class Replay:
    def __init__(self, size=8000):
        self.s = np.zeros((size,3), dtype='float32')
        self.a = np.zeros((size,), dtype='int32')
        self.r = np.zeros((size,), dtype='float32')
        self.ns = np.zeros((size,3), dtype='float32')
        self.d = np.zeros((size,), dtype='float32')
        self.i = 0; self.full=False; self.size=size
    def add(self, s,a,r,ns,d):
        self.s[self.i]=s; self.a[self.i]=a; self.r[self.i]=r; self.ns[self.i]=ns; self.d[self.i]=d
        self.i=(self.i+1)%self.size; self.full = self.full or self.i==0
    def sample(self, batch=64):
        n = self.size if self.full else self.i
        idx = np.random.randint(0, n, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.ns[idx], self.d[idx]

def build_qnet():
    inp = keras.Input((3,)); x = layers.Dense(64, activation='relu')(inp); x = layers.Dense(64, activation='relu')(x); out = layers.Dense(3)(x)
    m = keras.Model(inp, out); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse'); return m

class DQNAgent:
    def __init__(self, gamma=0.99, eps=1.0, eps_min=0.05, eps_decay=0.995, tau=0.01):
        self.q = build_qnet(); self.tq = build_qnet()
        self.gamma=gamma; self.eps=eps; self.eps_min=eps_min; self.eps_decay=eps_decay; self.tau=tau
        self.buf = Replay(8000)
    def act(self, s):
        if np.random.rand()<self.eps: return np.random.randint(0,3)
        q = self.q.predict(s[None], verbose=0)[0]; return int(np.argmax(q))
    def learn(self, batch=64):
        if (self.buf.i if not self.buf.full else self.buf.size) < 512: return
        s,a,r,ns,d = self.buf.sample(batch)
        q = self.q.predict(s, verbose=0); tq = self.tq.predict(ns, verbose=0); y = q.copy()
        idx = np.arange(batch); y[idx, a] = r + (1-d)*self.gamma*np.max(tq, axis=1)
        self.q.train_on_batch(s, y)
        w = self.q.get_weights(); tw = self.tq.get_weights()
        self.tq.set_weights([self.tau*ww + (1-self.tau)*tt for ww,tt in zip(w,tw)])
        self.eps = max(self.eps_min, self.eps*self.eps_decay)
