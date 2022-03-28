import os
import argparse
import torch
import shutil
import random
import numpy as np

def get_args():

    parser = argparse.ArgumentParser(description='Train a PPO agent')

    parser.add_argument('--policy_clip', type=float, default=0.2, metavar='G', help='Value to Clip Policy')
    parser.add_argument('--value_clip', type=float, default=0.2, metavar='G', help='Threshold to Clip Value')
    parser.add_argument('--weight_entropy', type=float, default=0.001, metavar='G', help='Importance of Entropy')
    parser.add_argument('--weight_for_value', type=float, default=1.0, metavar='G', help='Importance of Value')
    parser.add_argument('--ppo_epochs', type=int, default=15, metavar='G', help='Number of epochs for PPO')
    parser.add_argument('--ppo_episodes', type=int, default=1000, metavar='G', help='Number of episodes for PPO')
    parser.add_argument('--lr_act', type=float, default=3e-4, metavar='G', help='Learning Rate Actor')
    parser.add_argument('--lr_crit', type=float, default=1e-3, metavar='G', help='Learning Rate Critic')
    parser.add_argument('--lam', type=float, default=0.95, metavar='G', help='GAE Factor')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='Discount Factor')

    parser.add_argument('--seed', type=int, metavar='N', default=0, help='random seed (default: 0)')
    parser.add_argument('--save-dir', type=str, default='saved_models', help='the path to save the models')
    parser.add_argument('--run_id', type=str, default='train', help="name of the run")
    parser.add_argument('--test_id', type=str, required=True, help="Experiment ID")
    parser.add_argument('--wandb_id', type=str, default=None, help="Wandb ID")

    parser.add_argument('--is_training', default=1, type=int, help='1 for training and 0 for testing')
    parser.add_argument('--steps_per_episode', default=1000, type=int, help='Steps per Episode')

    parser.add_argument('-sq','--seq_len', type=int, default=32, help="Sequence Length for input to LSTM")
    parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0003, help="Neural Network Learning Rate")
    parser.add_argument('-mp', '--model_path', type=str, default='lstm_vae.pth', help="Saved model path")
    parser.add_argument('-ls','--latent_spc', type=int,default=8, help='Size of Latent Space')
    parser.add_argument('-fcd','--fc_dim', type=int, default=64, help="Number of FC Nodes")
    parser.add_argument('-nf','--n_features', type=int, default=7, help="Length of feature for each sample")
    parser.add_argument('-me','--max_epochs', type=int, default=1000, help="Number of epchs to train")
    parser.add_argument('-kldc','--beta', type=float, default=0.001, help='weighting factor of KLD')

    args = parser.parse_args()
    args.is_training = bool(args.is_training)

    args.save_dir = os.path.join(args.save_dir, 'train', f"{args.test_id}")
    os.makedirs(args.save_dir, exist_ok=True)

    if 'saved_buffer.pkl' in os.listdir():
        os.remove('saved_buffer.pkl')

    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if use_cuda else "cpu")
    dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    if use_cuda:
        torch.cuda.manual_seed(1)

    return args
