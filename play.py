from gym_sts.envs.base import SlayTheSpireGymEnv
from gym_sts.spaces.observations import ObservationError

from sts_playground.agents.random import RandomAgent


env = SlayTheSpireGymEnv("../gym-sts/lib", "../gym-sts/mods", "../gym-sts/out", headless=False)
agent = RandomAgent(env)
SEED = 17
agent.seed(SEED)

_, info = env.reset(options={"sts_seed": "SLAY"})
print(SEED, info["sts_seed"], info["seed"])

while True:
    action = agent.predict()[0]
    try:
        obs, _, done, _, _ = env.step(action)
    except ObservationError as e:
        breakpoint()
        pass

    if done:
        SEED += 1
        agent.seed(SEED)
        _, info = env.reset()
        print(SEED, info["sts_seed"], info["seed"])
