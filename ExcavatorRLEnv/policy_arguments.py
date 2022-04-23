import os
import argparse
import torch
import random
import numpy as np

def get_args():

    parser = argparse.ArgumentParser(description='Train a PPO agent')

    parser.add_argument('--policy_clip', type=float, default=0.2, metavar='G', help='Value to Clip Policy')
    parser.add_argument('--value_clip', type=float, default=0.2, metavar='G', help='Threshold to Clip Value')
    parser.add_argument('--weight_entropy', type=float, default=0.001, metavar='G', help='Importance of Entropy')
    parser.add_argument('--weight_for_value', type=float, default=1.0, metavar='G', help='Importance of Value')
    parser.add_argument('--ppo_epochs', type=int, default=15, metavar='G', help='Number of epochs for PPO')
    parser.add_argument('--lr_act', type=float, default=3e-4, metavar='G', help='Learning Rate Actor')
    parser.add_argument('--lr_crit', type=float, default=1e-3, metavar='G', help='Learning Rate Critic')
    parser.add_argument('--lam', type=float, default=0.95, metavar='G', help='GAE Factor')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='Discount Factor')
    parser.add_argument('--state_dim', type=int, default=19, metavar='G', help='Dimension of State Space')
    parser.add_argument('--action_dim', type=int, default=4, metavar='G', help='Dimension of Action Space')
    parser.add_argument('--std', type=float, default=0.6, metavar='G', help='Standard Deviation for Policy Exploration')

    parser.add_argument('--seed', type=int, metavar='N', default=0, help='random seed (default: 0)')
    parser.add_argument('--save-dir', type=str, default='saved_models', help='the path to save the models')
    parser.add_argument('--run_id', type=str, default='train', help="name of the run")
    parser.add_argument('--test_id', type=str, required=True, help="Experiment ID")
    parser.add_argument('--wandb_id', type=str, default=None, help="Wandb ID")

    parser.add_argument('--is_training', default=1, type=int, help='1 for training and 0 for testing')
    parser.add_argument('--steps_per_episode', default=300, type=int, help='Steps per Episode')
    parser.add_argument('--complexity', required=True, type=int, help='State the required complexity')
    parser.add_argument('--expert', required=True, type=int, help='Select Your Expert')

    args = parser.parse_args()
    args.is_training = bool(args.is_training)

    args.save_dir = os.path.join(args.save_dir, 'train', f"{args.test_id}_{args.complexity}_{args.expert}")
    os.makedirs(args.save_dir, exist_ok=True)

    if 'saved_buffer.pkl' in os.listdir():
        os.remove('saved_buffer.pkl')

    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    if use_cuda:
        torch.cuda.manual_seed(1)

    return args
