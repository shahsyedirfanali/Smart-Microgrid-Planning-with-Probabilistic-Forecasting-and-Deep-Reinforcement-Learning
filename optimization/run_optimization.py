import os, json, numpy as np
from optimization.stochastic_programming import generate_scenarios, cost_params
from optimization.nsga2_sizing import Individual, evaluate, tournament, crossover, mutate, fast_nondominated_sort, crowding_distance

os.makedirs('artifacts', exist_ok=True)
cfg = json.load(open('configs/optimization.json'))
lb = np.array([cfg['bounds']['wind_kw'][0], cfg['bounds']['battery_kwh'][0], cfg['bounds']['diesel_kw'][0]])
ub = np.array([cfg['bounds']['wind_kw'][1], cfg['bounds']['battery_kwh'][1], cfg['bounds']['diesel_kw'][1]])
scenarios = generate_scenarios(n=cfg['num_scenarios'], seed=1); costs = cost_params()

pop_size = cfg['population']; gens = cfg['generations']; cx_rate = cfg['crossover_rate']; mut_sigma = cfg['mutation_sigma']

pop = [Individual(np.random.uniform(lb, ub)) for _ in range(pop_size)]
for ind in pop: evaluate(ind, scenarios, costs, lpsp_limit=cfg['reliability_constraint_lpsp_percent'])

for g in range(gens):
    offspring = []
    while len(offspring) < pop_size:
        a = tournament(pop); b = tournament(pop)
        child = crossover(a,b, rate=cx_rate); child = mutate(child, lb, ub, sigma=mut_sigma)
        evaluate(child, scenarios, costs, lpsp_limit=cfg['reliability_constraint_lpsp_percent']); offspring.append(child)
    combined = pop + offspring
    fronts, rank = fast_nondominated_sort(combined); new_pop = []
    for F in fronts:
        if len(new_pop)+len(F) <= pop_size:
            new_pop.extend([combined[i] for i in F])
        else:
            dist = crowding_distance(F, combined); order = np.argsort(-dist)
            new_pop.extend([combined[F[i]] for i in order[:pop_size-len(new_pop)]]); break
    pop = new_pop
    best_lcoe = min(ind.objs[0] for ind in pop)
    print(f"[Gen {g+1}] best LCOE={best_lcoe:.4f} (pop {len(pop)})")

pareto = [{'x': ind.x.tolist(), 'objs': ind.objs.tolist()} for ind in pop]
with open('artifacts/pareto.json','w') as f: json.dump(pareto, f, indent=2)
json.dump(cfg, open('artifacts/optimization_config_used.json','w'), indent=2)
print('[Done] Optimization. Saved artifacts/pareto.json')
