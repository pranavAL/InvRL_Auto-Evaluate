import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor
import matplotlib.patches as mpatches

color_cycle = {0.0: (1,0,0),-10.0: (0,0,1),-2.0: (1,1,0),-5.0:
              (0,1,1),-9.0: (1,0,1),-18.0: (0,1,0),
            -11.0: (0.5,0,0),-7.0: (0,0,0.5),-32.0: (0.5,0.5,0),
            -15.0: (0,0.5,0.5),-16.0: (0,0.5,0),-20:(0.5,0,0.5)}

parser = argparse.ArgumentParser(description="Save models")
parser.add_argument('-mp', '--model_path', type=str, help="Saved model path")
parser.add_argument('-sess', '--sessions', type=int, help="Number of Sessions to compare")
args = parser.parse_args()

model_name = args.model_path.split('.')[0]
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
    learning_rate = p['learning_rate']
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

sorted_sess = sort_sessions(df)

with torch.no_grad():
    fig, ax = plt.subplots()
    patches = []
    uniq_scores = []
    for i, sess in enumerate(list(sorted_sess.keys())[:args.sessions]):
        sess_feat = df.loc[df["Session id"]==sess,:]
        iter  = (1 if i < 2 else p['seq_len'])
        for j in range(0,len(sess_feat) - p['seq_len'], iter):
            features = sess_feat.iloc[j:j+p['seq_len'],:][train_feats].values
            score = sum(sess_feat.iloc[j:j+p['seq_len'],:]["Current trainee score at that time"].values)
            if score not in uniq_scores:
                uniq_scores.append(score)

            if score < 0 or i < 2:
                ax = plot_sample(torch.tensor(features).float(), ax, color_cycle[score])

    patches = [mpatches.Patch(color=color_cycle[score], label=f"Score {score}") for score in sorted(uniq_scores)]
    ax.legend(handles=set(patches), loc='upper center', bbox_to_anchor=(0.4, 1.17),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(os.path.join('outputs',f"penaltyplot{model_name}.png"))
    plt.show()
