import sys
sys.path.append(r'C:\CM Labs\Vortex Studio 2021a\bin')
sys.path.append(r'C:\Users\Prana\Desktop\CM_Labs\InvRL_Auto-Evaluate')

import warnings
warnings.filterwarnings('ignore')

from agent import Agent
from environment import env
from arguments import get_args

if __name__ == "__main__":

    args = get_args()
    is_training = False

    env = env(args)
    agent = Agent(args)

    env.render(active=True)
    print(f"Test Started")
    agent = Agent(args)
    agent.load_weights()
    
    state, _ = env.reset()

    for _ in range(int(args.steps_per_episode)):
        action = agent.act(state, is_training)
        state_, reward, penalty, done, _ = env.step(list(action))

        state = state_

        if done:
            break

del env
