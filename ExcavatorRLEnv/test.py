import sys
sys.path.append(r'C:\CM Labs\Vortex Studio 2021a\bin')
sys.path.append(r'C:\Users\Prana\OneDrive\Desktop\CMLabs\InvRL_Auto-Evaluate')

import warnings
warnings.filterwarnings('ignore')

from agent import Agent
from environment import env
from policy_arguments import get_args

if __name__ == "__main__":

    args = get_args()

    env = env(args)
    agent = Agent(args)

    env.render(active=True)
    print(f"Test Started")
    agent = Agent(args)
    agent.load_weights()
    
    state, _ = env.reset()

    for _ in range(int(args.steps_per_episode)):
        action = agent.act(state)
        state_, reward, dyna_penalty, safe_penalty, done, _ = env.step(list(action))

        state = state_

        if done:
            break

del env