import os
import sys
sys.path.append(r'C:\CM Labs\Vortex Studio 2021a\bin')
sys.path.append(r'C:\Users\Prana\Desktop\CM_Labs\InvRL_Auto-Evaluate')

import warnings
warnings.filterwarnings('ignore')

import wandb
import numpy as np
from agent import Agent
from environment import env
from arguments import get_args

if __name__ == "__main__":

    args = get_args()
    is_training = args.is_training

    env = env(args)
    agent = Agent(args)

    if args.wandb_id:
        wandb.init(id=args.wandb_id, resume="must")
        agent.load_weights()
    else:
        wandb.init(name=f"{args.test_id}_{args.complexity}", config=args, settings=wandb.Settings(start_method="thread"))

    agent.save_weights()
    eps_count = 0
    mean_embedd_loss = []
    mean_reward = []

    while not env.is_complete:
        env.render(active=False)
        state, _ = env.reset()
        print("New Episode Started")
        done = False
        total_reward = []
        total_embedd_loss = []

        while not done:

            action = agent.act(state, is_training)
            state_, reward, penalty, done, _ = env.step(list(action))

            total_reward.append(reward)
            total_embedd_loss.append(penalty)

            if args.test_id == "Dynamic":
                agent.save_eps(state, reward + 10 * penalty, action, done, state_)
            elif args.test_id == "Dense":
                agent.save_eps(state, reward, action, done, state_)
            else:
                print("Error: Please choose a reward type: Dense or Dynamic")

            state = state_

            if done:
                break

        print(f"Complexity: {env.initial_complexity}  Distance Left: {env.goal_distance}")
        print("Updating policy")
        
        agent.memory.saveBuffer()
        
        mean_reward.append(np.mean(total_reward))
        mean_embedd_loss.append(np.mean(total_embedd_loss))

        wandb.log({'Avg. Total Reward last 100 episodes':np.mean(mean_reward[-100:])})
        wandb.log({'Avg. Embedding Reward last 100 episodes':np.mean(mean_embedd_loss[-100:])})
        wandb.log({'Avg. Torque (in %) last 100 Episode':np.mean(env.tor_avg[-100:])})
        wandb.log({'Avg. Power (in %) last 100 Episode':np.mean(env.pow_avg[-100:])})
        wandb.log({'Avg. Fuel Consumption (in %) last 100 Episode':np.mean(env.avg_fuel[-100:])})
        wandb.log({'Average Collisions with environment last 100 episodes':np.mean(env.env_col[-100:])})
        wandb.log({'Average Number of tennis balls knocked last 100 episodes':np.mean(env.knock_ball[-100:])})
        wandb.log({'Average Number of poles touched last 100 episodes':np.mean(env.touch_pole[-100:])})
        wandb.log({'Average Number of poles fell last 100 episodes':np.mean(env.fell_pole[-100:])})
        wandb.log({'Average Number of equipment collisions last 100 episodes':np.mean(env.coll_equip[-100:])})

        while 'saved_buffer.pkl' in os.listdir():
            continue

        not_ready = True
        agent = Agent(args)

        while not_ready:
            try:
                agent.load_weights()
            except Exception as e:
                not_ready = True
            else:
                not_ready = False

    del env