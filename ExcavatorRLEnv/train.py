import os
import sys
sys.path.append(r'C:\CM Labs\Vortex Studio 2022.1\bin')

import warnings
warnings.filterwarnings('ignore')

import math
import time
import wandb
import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
import torch.optim as optim
from environment import env
from arguments import get_args
import matplotlib.pyplot as plt
import torch.nn.functional as F

args = get_args()
is_training = args.is_training

if __name__ == "__main__":

    env = env()
    wandb.init(name="Excavator Learning Policy", config=args)

    agent = Agent(args)


    agent.lets_init_weights()

    i_ep = 0
    agent.save_weights()

    while True:
        env.render()
        if 'saved_buffer.pkl' not in os.listdir():
            print(f"Collecting Episode: {i_ep}")
            mean_score = []
            mean_sand = []
            agent = Agent(args)
            time.sleep(5)
            agent.load_weights()
            state, _ = env.reset()
            env.interface.getInputContainer()['Episode'].value = str(i_ep)

            for t in range(int(args.steps_per_episode)):
                action = agent.act(state, is_training)
                state_, reward, done, _ = env.step(list(action))
                mean_score.append(sum(reward))
                mean_sand.append(reward[-1])
                env.interface.getInputContainer()['Reward'].value = str(sum(reward))

                agent.save_eps(state, sum(reward), action, done, state_)
                state = state_

                if done:
                    break

            agent.memory.saveBuffer()

            wandb.log({"Average Total Reward":np.mean(mean_score)})
            wandb.log({"Average Sand":np.mean(mean_sand)})
            i_ep += 1
    del env
