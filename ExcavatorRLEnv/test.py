import os
import sys
sys.path.append(r'C:\CM Labs\Vortex Studio 2021a\bin')
sys.path.append(r'C:\Users\Prana\Desktop\CM_Labs\InvRL_Auto-Evaluate')

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
import torch.nn.functional as F

if __name__ == "__main__":

    args = get_args()
    is_training = args.is_training

    env = env(args)

    agent = Agent(args)

    i_ep = 0

    env.render(active=True)
    print(f"Test Started")
    agent = Agent(args)
    agent.load_weights()
    state, _ = env.reset()

    for t in range(int(args.steps_per_episode)):
        action = agent.act(state, is_training)
        state_, reward, penalty, done, _ = env.step(list(action))

        state = state_

        if done:
            break

del env
