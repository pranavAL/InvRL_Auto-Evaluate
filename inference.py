import os
import glob
import torch
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from random import uniform as u
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.patches as mpatches
from train_LSTM import LSTMPredictor, args

model_name = f"{args.model_path.split('.')[0]}"
cmd = f'ffmpeg -framerate 10 -y -i temp/%d.png -c:v libx264 -pix_fmt yuv420p outputs/{model_name}.mp4'
model_path = os.path.join('save_model', args.model_path)
os.makedirs(f"temp", exist_ok = True)
df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))
df_train = pd.read_csv(os.path.join("datasets","train_df.csv"))
color_cycle = {0:(0,0,1), 1:(0,1,0), 2:(1,0,0), 3:(1,0,1), 4:(1,1,0)}

train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
        'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

def normalize(df):
    df.loc[:,train_feats] = ((df.loc[:,train_feats] - df.loc[:,train_feats].min())
                            /(df.loc[:,train_feats].max() - df.loc[:,train_feats].min()))
    return df

df = normalize(df)

model = LSTMPredictor(
    n_features = args.n_features,
    fc_dim = args.fc_dim,
    seq_len = args.seq_len,
    batch_size = args.batch_size,
    latent_spc = args.latent_spc,
    learning_rate = args.learning_rate
)

model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_embeddings(data, label):
    data = data.unsqueeze(0).to(model.device)
    label = label.view(1,1,-1).to(model.device)
    recon, mu, _, _ = model(data, label, is_train=False)
    mu = get_numpy(mu)
    return mu

def plot_sample(data, ax, color):
    ax.scatter(*data, color=color)
    return ax

def plot_by_session(ax, sess, meta_data):
    sess_feat = meta_data.loc[meta_data['sess']==sess,:]
    final_score = sess_feat.iloc[0,:]['FinalScore']
    bucket = abs(final_score) // 25
    color = color_cycle[bucket]
    X = sess_feat.iloc[:,:]['X']
    Y = sess_feat.iloc[:,:]['Y']
    ax.scatter(X, Y, color=color, label=f"Rank {bucket+1}")
    return ax

def save_meta_data(df):
    meta_data = pd.DataFrame(columns=['sess', 'embeddings', 'penalty', 'FinalScore'])
    for sess in df["Session id"].unique():
        sess_feat = df.loc[df["Session id"]==sess,:]
        final_score = sum(sess_feat["Current trainee score at that time"].values)
        terminate = args.seq_len
        for i in range(0,len(sess_feat)-terminate):
            score = sum(sess_feat.iloc[i:i+args.seq_len,:]["Current trainee score at that time"].values)
            train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
            init_label = list(sess_feat.iloc[i+args.seq_len-1,:][train_feats].values)
            mu = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
            meta_data.loc[len(meta_data)] = [sess, list(mu.astype(float)), score, final_score]
    meta_data = get_tsne(meta_data)
    meta_data.to_csv(f"outputs/{model_name}")
    return meta_data

def get_tsne(meta_data):
    embeddings = list(meta_data['embeddings'].values)
    tsne = TSNE(n_components=2, verbose=1, perplexity=500, n_iter=5000, n_jobs=-1)
    Z_tsne = tsne.fit_transform(embeddings)
    meta_data['X'] = Z_tsne[:,0]
    meta_data['Y'] = Z_tsne[:,1]
    return meta_data

def plot_datas(meta_data, ax):

    for sess in df_train["Session id"].unique():
        ax = plot_by_session(ax, sess, meta_data)

    return ax

def animate(meta_data):
    count = 0
    for index, row in meta_data.iterrows():
        if row['sess'] in df_train["Session id"].unique():
            continue
        fig, ax = plt.subplots()
        penalty = row['penalty']
        final_score = row['FinalScore']
        rank = (abs(final_score) // 25) + 1
        x = row['X']
        y = row['Y']
        ax = plot_datas(meta_data, ax)
        ax = plot_sample([x,y], ax, (0,0,0))

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, lines))
        plt.legend(by_label.values(), by_label.keys(), loc = 'best')
        plt.title(f"Rank: {rank}, Penalty: {penalty}, Final Score: {final_score}")
        fig.savefig(f"temp/{count}.png")
        plt.close()
        count+=1

    os.system(cmd)
    shutil.rmtree(f"temp")

with torch.no_grad():

    meta_data = save_meta_data(df)
    #meta_data = pd.read_csv(f"outputs/{model_name}")
    animate(meta_data)
