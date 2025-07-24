import asyncio
import numpy as np
from env import MinesweeperEnv

async def run_ep(env, agent):
    s, total, done = env.reset(), 0, False
    while not done:
        a = agent.select(s)
        s2, r, done = env.step(a)
        agent.store(s, a, r, s2, done)
        s, total = s2, total + r
    return total

async def train(agent, episodes=200, n_envs=8):
    from tqdm.notebook import tqdm
    history = []
    for ep in tqdm(range(episodes)):
        envs = [MinesweeperEnv() for _ in range(n_envs)]
        results = await asyncio.gather(*[run_ep(env, agent) for env in envs])
        agent.update()
        avg = np.mean(results)
        history.append(avg)
        tqdm.write(f"Episode {ep+1}/{episodes} — Avg: {avg:.2f} — Eps: {agent.eps:.2f}")
    return history
