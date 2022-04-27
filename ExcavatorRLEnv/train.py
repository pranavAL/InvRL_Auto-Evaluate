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
from policy_arguments import get_args

if __name__ == "__main__":

    args = get_args()
    is_training = True

    wandb.init(name=f"{args.test_id}_{args.complexity}", config=args)

    agent = Agent(args)
    env = env(args)

    agent.save_weights()
    mean_reward = []
    mean_dyna_loss = []
    mean_penalty_loss = []
    
    for ep in range(500):
        env.render(active=False)
        state, _ = env.reset()
        print("New Episode Started")
        done = False
        total_dyna_loss = []
        total_penalty_loss = []

        while not done:

            action = agent.act(state, is_training)
            state_, reward, dyna_penalty, safe_penalty, done, _ = env.step(list(action))

            total_dyna_loss.append(dyna_penalty)
            total_penalty_loss.append(safe_penalty)

            if args.test_id == "Dynamic":
                agent.save_eps(state, reward + dyna_penalty, action, done, state_)
            elif args.test_id == "Safety":
                agent.save_eps(state, reward + safe_penalty, action, done, state_)    
            elif args.test_id == "DynamicSafety":
                agent.save_eps(state, reward + 0.5 * safe_penalty + dyna_penalty, action, done, state_)        
            elif args.test_id == "Task":
                agent.save_eps(state, reward, action, done, state_)
            else:
                print("Error: Please choose correct reward")
                break

            state = state_

            if done:
                mean_reward.append(env.goal_distance)
                break

        print(f"Episode: {ep} Distance Left: {env.goal_distance}")
        print("Updating policy")

        agent.memory.saveBuffer()

        mean_dyna_loss.append(np.mean(total_dyna_loss))
        mean_penalty_loss.append(np.mean(total_penalty_loss))

        wandb.log({'Avg. Final Distance from Goal last 10 episodes': np.mean(mean_reward[-10:])})
        wandb.log({'Avg. Dynamic Reward last 10 episodes':np.mean(mean_dyna_loss[-10:])})
        wandb.log({'Avg. Penalty Reward last 10 episodes':np.mean(mean_penalty_loss[-10:])})
        wandb.log({'Avg. Torque (in %) last 10 Episode':np.mean(env.tor_avg[-10:])})
        wandb.log({'Avg. Power (in %) last 10 Episode':np.mean(env.pow_avg[-10:])})
        wandb.log({'Avg. Fuel Consumption (in %) last 10 Episode':np.mean(env.avg_fuel[-10:])})
        wandb.log({'Collisions with environment last 10 episodes':sum(env.env_col[-10:])})
        wandb.log({'Number of tennis balls knocked last 10 episodes':sum(env.knock_ball[-10:])})
        wandb.log({'Number of poles touched last 10 episodes':sum(env.touch_pole[-10:])})
        wandb.log({'Number of poles fell last 10 episodes':sum(env.fell_pole[-10:])})
        wandb.log({'Number of equipment collisions last 10 episodes':sum(env.coll_equip[-10:])})

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
