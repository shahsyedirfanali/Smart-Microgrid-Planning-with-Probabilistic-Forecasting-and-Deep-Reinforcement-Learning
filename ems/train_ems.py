import os, json
from ems.env_microgrid import MicrogridEnv
from ems.dqn_agent import DQNAgent
from ems.ppo_agent import PPOAgent

os.makedirs('artifacts', exist_ok=True)
cfg = json.load(open('configs/ems.json'))
env = MicrogridEnv(horizon=cfg['horizon'], seed=cfg['seed'])

agent_name = cfg.get('agent','dqn').lower()
if agent_name == 'ppo':
    agent = PPOAgent(gamma=cfg['gamma'], lam=cfg['ppo']['gae_lambda'], clip=cfg['ppo']['clip'], lr=cfg['ppo']['lr'], epochs=cfg['ppo']['epochs'], batch=cfg['ppo']['batch'])
else:
    agent = DQNAgent(gamma=cfg['gamma'], eps=cfg['epsilon_start'], eps_min=cfg['epsilon_min'], eps_decay=cfg['epsilon_decay'], tau=cfg['target_tau'])

episodes = cfg['episodes']
stats = {'reward': []}

if agent_name == 'ppo':
    rollout = 32  # episodes per update
    buf = {'s':[], 'a':[], 'r':[], 'd':[], 'logp':[]}
    for ep in range(episodes):
        s = env.reset(); done=False; total=0.0
        while not done:
            a, logp, v = agent.act(s)
            ns, r, done, info = env.step(a)
            buf['s'].append(s); buf['a'].append(a); buf['r'].append(r); buf['d'].append(float(done)); buf['logp'].append(logp)
            s = ns; total += r
        stats['reward'].append(total)
        if (ep+1) % rollout == 0:
            agent.learn(buf); buf = {'s':[], 'a':[], 'r':[], 'd':[], 'logp':[]}
        if (ep+1)%50==0: print(f"[PPO Ep {ep+1}] return={total:.2f}")
else:
    for ep in range(episodes):
        s = env.reset(); done=False; total=0.0
        while not done:
            a = agent.act(s)
            ns, r, done, info = env.step(a)
            agent.buf.add(s,a,r,ns,float(done)); agent.learn(batch=cfg['batch'])
            s = ns; total += r
        stats['reward'].append(total)
        if (ep+1)%50==0: print(f"[DQN Ep {ep+1}] return={total:.2f}, eps={agent.eps:.2f}")

json.dump(stats, open('artifacts/ems_training_stats.json','w'), indent=2)
json.dump(cfg, open('artifacts/ems_config_used.json','w'), indent=2)
print('[Done] EMS training. Saved artifacts/ems_training_stats.json')
