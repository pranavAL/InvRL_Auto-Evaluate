import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor
import matplotlib.patches as mpatches
from celluloid import Camera

color_cycle = [(1,0,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(0,1,0),
             (0.5,0,0),(0,0,0.5),(0.5,0.5,0),(0,0.5,0.5),
             (0,0.5,0),(0.5,0,0.5),(0.25,0,0.25),(0,0,0.25),
             (0.25,0,0),(0,0,0)]

parser = argparse.ArgumentParser(description="Save models")
parser.add_argument('-mp', '--model_path', type=str, help="Saved model path")
parser.add_argument('-sess', '--sessions', type=int, help="Number of Sessions to compare")
parser.add_argument('-ts', '--to_save', type=bool, default=False, help="To save image or not")
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
    recon, mu, _ = model(data.unsqueeze(0).to(model.device))
    mu = get_numpy(mu)
    ax.scatter(*mu, color=color)
    return ax, mu

def plot(ax, patches, save, sess_name, meta_data, is_animate, fig):
    ax.legend(handles=patches, loc='best',
          ncol=3, fancybox=True, shadow=True)

    plt.title(f"{model_name}")
    plt.xlim([-0.075,0.05])
    plt.ylim([-0.075,0.015])

    if is_animate:
        animate(meta_data, fig, ax)

    if save:
        plt.savefig(os.path.join('outputs',f"{sess_name}_{model_name}.png"))

def plot_by_session(sess_feat, sorted_sess, ax, patches, i, sess, meta_data):

    for j in range(0,len(sess_feat) - p['seq_len']):
        score = sum(sess_feat.iloc[j:j+p['seq_len'],:]["Current trainee score at that time"].values)
        ax, mu = plot_sample(torch.tensor(sess_feat.iloc[j:j+p['seq_len'],:][train_feats].values).float(), ax, color_cycle[i])
        meta_data.loc[len(meta_data)] = [i, mu[0], mu[1], score]
    patches.append(mpatches.Patch(color=color_cycle[i], label=f"Score {sorted_sess[sess]}"))

    return ax, patches, meta_data

def plot_datas(num_sess, df, sorted_sess, type_data, save_meta_data, is_animate):

    fig, ax = plt.subplots()
    patches = []
    uniq_scores = []
    meta_data = pd.DataFrame(columns=['rank', 'coordX', 'coordY', 'penalty'])

    for i, sess in enumerate(list(sorted_sess.keys())[:num_sess]):
        sess_feat = df.loc[df["Session id"]==sess,:]
        ax, patches, meta_data = plot_by_session(sess_feat, sorted_sess, ax, patches, i, sess, meta_data)

    plot(ax, patches, args.to_save, type_data, meta_data, is_animate, fig)
    if save_meta_data:
        meta_data.to_csv(f"outputs/meta_data.csv")

def getXYZ(meta_data, rank):
    X = list(meta_data.loc[meta_data['rank']==rank,'coordX'].values)
    Y = list(meta_data.loc[meta_data['rank']==rank,'coordY'].values)
    Z = list(meta_data.loc[meta_data['rank']==rank,'penalty'].values)
    score = sum(Z)
    return X, Y, Z, score

def animate(meta_data, fig, ax):
    camera = Camera(fig)
    meta_data = pd.read_csv(f"outputs/meta_data.csv")
    all_sess = meta_data['rank'].unique()
    expertsX1, expertsY1, _, _ = getXYZ(meta_data, 0)
    expertsX2, expertsY2, _, _ = getXYZ(meta_data, 1)

    for sess in all_sess[2:]:
        X,Y,Z,score = getXYZ(meta_data, sess)
        for (x,y,z) in zip(X, Y, Z):
            plt.title(f"Rank: {sess}, Penalty: {z}, Total Score: {score}")
            ax.scatter(expertsX1, expertsY1, color=(1,0,0))
            ax.scatter(expertsX2, expertsY2, color=(0,0,1))
            ax.scatter(x, y, color=(0,0,0))
            camera.snap()

    animate = camera.animate()
    animate.save('output.mp4',fps=1)

sorted_sess = sort_sessions(df)

with torch.no_grad():

    plot_datas(args.sessions, df,sorted_sess, type_data="all", save_meta_data=True, is_animate=False)
    plot_datas(2, df,sorted_sess, type_data="best", save_meta_data=False, is_animate=True)
