import os
import cv2
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
import matplotlib.patches as mpatches
from train_LSTM import LSTMPredictor, args

model_path = os.path.join('save_model', args.model_path)
df = pd.read_csv(os.path.join("datasets","train_df.csv"))

train_feats = ['Bucket Angle','Bucket Height']

df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())
train_feats += ['Current trainee score at that time']

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

def get_embeddings(input, label):
    input = input.unsqueeze(0).to(model.device)
    label = label.view(1,1,-1).to(model.device)
    recon, mu, _ = model(input, label, is_train=False)
    recon = get_numpy(recon)
    return recon

y_pred = []
x_input = []
n_sessions = len(df["Session id"].unique())

for sess in df["Session id"].unique():
    sess_feat = df.loc[df["Session id"]==sess,:]
    sess_feat.loc[:,"Current trainee score at that time"] = abs((sess_feat.loc[:,"Current trainee score at that time"].sum()//25)/4)
    terminate = args.seq_len
    for i in range(0,len(sess_feat)-terminate,terminate):
        train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
        init_label = list(sess_feat.iloc[i+args.seq_len-1,:][train_feats].values)
        recon = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
        y_pred.append(recon)
        x_input.append(train)

fig, ax = plt.subplots(1,3)
fig.set_size_inches(15, 10)
train = np.reshape(np.array(x_input), (n_sessions, -1, args.n_features))
label = np.reshape(np.array(y_pred), (n_sessions, -1, args.n_features))
for i in range(args.n_features):
    ax[i].plot(train[5,:,i], color='r', label="Original")
    ax[i].plot(label[5,:,i], color='b', label="Predicted")
    ax[i].set_title(f"Feature {i}")

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
plt.show()
#plt.savefig(f"outputs/{args.seq_len}_steps_lookahead.png", dpi=100)
