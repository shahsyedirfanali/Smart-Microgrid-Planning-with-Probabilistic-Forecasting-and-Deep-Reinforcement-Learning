import numpy as np
class MicrogridEnv:
    """Minimal gym-like env; discrete actions: 0=charge,1=idle,2=discharge"""
    def __init__(self, horizon=24, seed=0):
        self.rng = np.random.default_rng(seed); self.horizon=horizon
        self.batt_kwh = 80.0; self.diesel_kw = 30.0
        self.charge_rate = 8.0; self.discharge_rate = 8.0
    def reset(self):
        self.t=0; self.soc = 40.0
        self.wind_kw = self.rng.uniform(20, 60)
        self.wind_cf = np.clip(0.3 + 0.2*np.sin(np.linspace(0,2*np.pi,24)+self.rng.uniform()), 0, 1)
        self.load = 10 + 4*np.sin(np.linspace(0,2*np.pi,24)-1.0) + self.rng.normal(0,0.8,24); self.load = np.clip(self.load, 0, None)
        return self._state()
    def _state(self):
        forecast = self.wind_kw*self.wind_cf[min(self.t+1,23)]
        return np.array([self.soc/self.batt_kwh, forecast/60.0, self.load[self.t]/20.0], dtype='float32')
    def step(self, action:int):
        if action==0: self.soc = min(self.batt_kwh, self.soc + self.charge_rate)
        elif action==2: self.soc = max(0.0, self.soc - self.discharge_rate)
        wind_gen = self.wind_kw*self.wind_cf[self.t]*1.0; supply = wind_gen; demand = self.load[self.t]
        if action==2: supply += self.discharge_rate
        diesel_use = 0.0
        if supply < demand:
            gap = demand - supply
            diesel_use = min(self.diesel_kw, gap); supply += diesel_use
        unmet = max(0.0, demand - supply)
        reward = - (0.6*diesel_use + 4.0*unmet + 0.1*(self.soc<10))
        self.t += 1; done = (self.t>=self.horizon)
        return self._state(), reward, done, {'diesel': diesel_use, 'unmet': unmet}
