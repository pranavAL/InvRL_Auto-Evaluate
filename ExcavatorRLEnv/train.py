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
    wandb.init(name=f"{args.test_id}_{args.ppo_episodes}", config=args)

    agent = Agent(args)

    i_ep = 0
    complexity = 0
    thres_dist = 2.0
    agent.save_weights()

    while True:
        env.render(active=False)
        if 'saved_buffer.pkl' not in os.listdir():
            not_ready = True
            print(f"Collecting Episode: {i_ep}")
            agent = Agent(args)

            while not_ready:
                try:
                    agent.load_weights()
                except Exception as e:
                    not_ready = True
                else:
                    not_ready = False

            state, _ = env.reset(complexity, thres_dist)
            mean_reward = []
            mean_penalty = []
            total_reward = []

            for t in range(int(args.steps_per_episode)):
                action = agent.act(state, is_training)
                state_, reward, penalty, done, _ = env.step(list(action))

                mean_reward.append(reward)
                mean_penalty.append(penalty)

                if args.test_id == "Dynamic_Dense":
                    agent.save_eps(state, reward*penalty, action, done, state_)
                    total_reward.append(reward*penalty)
                elif args.test_id == "Dense":
                    agent.save_eps(state, reward, action, done, state_)
                    total_reward.append(reward)
                elif args.test_id == "Dynamic":
                    agent.save_eps(state, penalty, action, done, state_)
                    total_reward.append(penalty)
                else:
                    print("Error: Please choose a reward type: Dynamic_Dense or Dense or Dynamic")
                state = state_

                if done:
                    break

            if env.initial_complexity > env.final_complexity:
                complexity += 1
                args.steps_per_episode += 50
                thres_dist = max(thres_dist - 1, 1.0)

            print(f"Complexity: {complexity} --- Total distance: {env.max_dist}")
            print(f"Distance Left: {env.goal_distance}")

            agent.memory.saveBuffer()

            wandb.log({'Avg. Penalty per Episode':np.mean(mean_penalty)})
            wandb.log({'Avg. Goal Reward per Episode':np.mean(mean_reward)})
            wandb.log({'Avg. Total Reward per Episode':np.mean(total_reward)})
            wandb.log({'Complexity':complexity})
            wandb.log({'Exercise Number of goals met':env.num_goal})
            wandb.log({'Collisions with environment':env.coll_env})
            wandb.log({'Number of times machine was left idling':env.num_idle})
            wandb.log({'Number of times user had to restart an arc':env.arc_restart})
            wandb.log({'Number of tennis balls knocked over by operator':env.ball_knock})
            wandb.log({'Number of poles touched':env.pole_touch})
            wandb.log({'Number of poles that fell over':env.pole_fell})
            wandb.log({'Number of barrels touches':env.barr_touch})
            wandb.log({'Number of barrels knocked over':env.barr_knock})
            wandb.log({'Number of equipment collisions':env.equip_coll})
            wandb.log({'Exercise Number of goals met':env.num_goal})
            wandb.log({'Exercise Time':env.ex_time})

            i_ep += 1
    del env
