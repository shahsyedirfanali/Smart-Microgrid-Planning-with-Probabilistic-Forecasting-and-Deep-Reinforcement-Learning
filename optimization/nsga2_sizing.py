import numpy as np

class Individual:
    def __init__(self, x):
        self.x = x  # [wind_kw, battery_kwh, diesel_kw]
        self.objs = None; self.violation = 0.0

def evaluate(ind, scenarios, costs, lpsp_limit=5.0):
    wind_kw, batt_kwh, diesel_kw = ind.x
    unmet = 0.0; total = 0.0; fuel = 0.0
    for w, load in scenarios:
        cf = np.clip(w/12.0, 0, 1); gen_wind = wind_kw * cf
        soc = batt_kwh/2.0
        for d in load:
            total += d; supply = gen_wind
            if supply < d:
                gap = d - supply
                take = min(gap, soc); soc -= take; gap -= take
                if gap > 0:
                    dg = min(gap, diesel_kw); fuel += dg * 0.25; gap -= dg
                if gap > 0: unmet += gap
            else:
                spare = min(supply - d, batt_kwh - soc); soc += spare
    lpsp = (unmet / (total+1e-6))*100.0
    capex = costs['wind_kw']*wind_kw + costs['batt_kwh']*batt_kwh + costs['diesel_kw']*diesel_kw
    opex = costs['fuel']*fuel + 0.02*capex
    lcoe = (capex/10 + opex) / (total/365.0 + 1e-6)
    emissions = fuel * 0.7
    ind.objs = np.array([lcoe, lpsp, emissions])
    ind.violation = 0.0 if lpsp <= lpsp_limit else (lpsp - lpsp_limit)
    return ind

def tournament(pop, k=2):
    idx = np.random.randint(0, len(pop), size=k)
    return sorted([pop[i] for i in idx], key=lambda z:(z.violation>0, z.objs[0]))[0]

def crossover(a, b, rate=0.9):
    import numpy as np
    if np.random.rand() < rate:
        alpha = np.random.rand()
        return Individual(alpha*a.x + (1-alpha)*b.x)
    return Individual(a.x.copy())

def mutate(ind, lb, ub, sigma=0.1):
    import numpy as np
    x = ind.x + np.random.normal(0, sigma, size=len(ind.x))* (ub-lb)
    x = np.clip(x, lb, ub)
    return Individual(x)

def dominates(a,b):
    if a.violation==0 and b.violation>0: return True
    if a.violation>0 and b.violation==0: return False
    return np.all(a.objs<=b.objs) and np.any(a.objs<b.objs)

def fast_nondominated_sort(pop):
    F=[]; S={i:[] for i in range(len(pop))}; n = np.zeros(len(pop), dtype=int); rank = np.zeros(len(pop), dtype=int)
    for p in range(len(pop)):
        Sp=[]; np_=0
        for q in range(len(pop)):
            if dominates(pop[p], pop[q]): Sp.append(q)
            elif dominates(pop[q], pop[p]): np_ += 1
        S[p]=Sp; n[p]=np_
        if n[p]==0: rank[p]=0
    F0 = [i for i in range(len(pop)) if n[i]==0]; F.append(F0); i=0
    while F[i]:
        Q=[]
        for p in F[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: rank[q]=i+1; Q.append(q)
        i+=1; F.append(Q)
    return F[:-1], rank

def crowding_distance(front, pop):
    import numpy as np
    m = len(pop[0].objs); dist = np.zeros(len(front))
    if len(front)==0: return dist
    for k in range(m):
        f = np.array([pop[i].objs[k] for i in front]); order = np.argsort(f)
        dist[order[0]] = dist[order[-1]] = np.inf
        fmin, fmax = f[order[0]], f[order[-1]]
        if fmax-fmin==0: continue
        for idx in range(1,len(front)-1):
            dist[order[idx]] += (f[order[idx+1]] - f[order[idx-1]])/(fmax-fmin)
    return dist
