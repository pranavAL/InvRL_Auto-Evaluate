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
    is_training = True

    wandb.init(name=f"{args.test_id}_{args.complexity}_{args.expert}", config=args)
    
    agent = Agent(args)
    env = env(args)

    agent.save_weights()
    mean_embedd_loss = []
    mean_reward = []

    for ep in range(1):
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
                agent.save_eps(state, reward * penalty, action, done, state_)
            elif args.test_id == "Dense":
                agent.save_eps(state, reward, action, done, state_)
            else:
                print("Error: Please choose a reward type: Dense or Dynamic")

            state = state_

            if done:
                break

        print(f"Episode: {ep} Distance Left: {env.goal_distance}")
        print("Updating policy")
        
        agent.memory.saveBuffer()
        
        mean_reward.append(np.mean(total_reward))
        mean_embedd_loss.append(np.mean(total_embedd_loss))

        wandb.log({'Avg. Total Reward last 10 episodes':np.mean(mean_reward[-10:])})
        wandb.log({'Avg. Embedding Reward last 10 episodes':np.mean(mean_embedd_loss[-10:])})
        wandb.log({'Avg. Torque (in %) last 10 Episode':np.mean(env.tor_avg[-10:])})
        wandb.log({'Avg. Power (in %) last 10 Episode':np.mean(env.pow_avg[-10:])})
        wandb.log({'Avg. Fuel Consumption (in %) last 10 Episode':np.mean(env.avg_fuel[-10:])})
        wandb.log({'Collisions with environment last 10 episodes':sum(env.env_col[-10:])})
        wandb.log({'Number of tennis balls knocked last 10 episodes':sum(env.knock_ball[-10:])})
        wandb.log({'Number of poles touched last 10 episodes':sum(env.touch_pole[-10:])})
        wandb.log({'Number of poles fell last 10 episodes':sum(env.fell_pole[-10:])})
        wandb.log({'Number of equipment collisions last 10 episodes':sum(env.coll_equip[-10:])})

        # while 'saved_buffer.pkl' in os.listdir():
        #     continue

        # not_ready = True
        # agent = Agent(args)

        # while not_ready:
        #     try:
        #         agent.load_weights()
        #     except Exception as e:
        #         not_ready = True
        #     else:
        #         not_ready = False

    del env