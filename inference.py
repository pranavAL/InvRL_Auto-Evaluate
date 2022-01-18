import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor
import matplotlib.patches as mpatches

color_cycle = [(1,0,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(0,1,0),
             (0.5,0,0),(0,0,0.5),(0.5,0.5,0),(0,0.5,0.5),
             (0,0.5,0),(0.5,0,0.5),(0.25,0,0.25),(0,0,0.25),
             (0.25,0,0)]

parser = argparse.ArgumentParser(description="Save models")
parser.add_argument('-mp', '--model_path', type=str, help="Saved model path")
parser.add_argument('-sess', '--sessions', type=int, help="Number of Sessions to compare")
parser.add_argument('-ts', '--to_save', type=bool, help="To save image or not")
args = parser.parse_args()

model_name = f"{args.model_path.split('.')[0]}"
model_type = model_name.split('_')[3]
model_path = os.path.join('save_model', args.model_path)
df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))

train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

p = dict(
    seq_len = 8,
    batch_size = 8,
    max_epochs = 100,
    n_features = 9,
    embedding_dim = 4,
    num_layers = 1,
    learning_rate = 0.001
)

model = LSTMPredictor(
    n_features = p['n_features'],
    embedding_dim = p['embedding_dim'],
    seq_len = p['seq_len'],
    batch_size = p['batch_size'],
    num_layers = p['num_layers'],
    learning_rate = p['learning_rate'],
    model_type = model_type
)

model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

def sort_sessions(data):
    sorted_sess = {}

    for sess in data['Session id'].unique():
        sorted_sess[sess] = data.loc[data['Session id']==sess,"Current trainee score at that time"].sum()
    sorted_sess = dict(sorted(sorted_sess.items(), key = lambda x:x[1], reverse=True))

    return sorted_sess

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def plot_sample(data, ax, color):
    _, mu, _ = model(data.unsqueeze(0).to(model.device))
    mu = get_numpy(mu)
    ax.scatter(*mu, color=color)
    return ax

def plot(ax, patches, save):
    ax.legend(handles=patches, loc='best',
          ncol=3, fancybox=True, shadow=True)
    if save:
        plt.savefig(os.path.join('outputs',f"{model_name}.png"))
    plt.title(model_name)
    plt.show()

def plot_by_session(sess_feat, sorted_sess, ax, patches, i):

    for j in range(0,len(sess_feat) - p['seq_len']):
        score = sum(sess_feat.iloc[j:j+p['seq_len'],:]["Current trainee score at that time"].values)
        ax = plot_sample(torch.tensor(sess_feat.iloc[j:j+p['seq_len'],:][train_feats].values).float(), ax, color_cycle[i])
    patches.append(mpatches.Patch(color=color_cycle[i], label=f"Score {sorted_sess[sess]}"))

    return ax, patches

sorted_sess = sort_sessions(df)

with torch.no_grad():
    fig, ax = plt.subplots()
    patches = []
    uniq_scores = []

    for i, sess in enumerate(list(sorted_sess.keys())[:args.sessions]):
        sess_feat = df.loc[df["Session id"]==sess,:]

        ax, patches = plot_by_session(sess_feat, sorted_sess, ax, patches, i)

    plot(ax, patches, args.to_save)
