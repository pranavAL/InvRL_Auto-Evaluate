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
df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))

train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

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
    label = label.unsqueeze(0).to(model.device)
    recon, mu, _ = model(input, label, is_train=False)
    recon = get_numpy(recon)
    return recon

for sess in df["Session id"].unique()[36:38]:
    sess_feat = df.loc[df["Session id"]==sess,:]
    terminate = args.seq_len
    y_pred = []
    x_input = []
    for i in range(0,len(sess_feat)-terminate,terminate):
        train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
        init_label = list(sess_feat.iloc[i+args.seq_len-1,:][train_feats].values)
        recon = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
        y_pred.append(recon)
        x_input.append(train)

fig, ax = plt.subplots(3,3)
fig.set_size_inches(15, 10)
train = np.reshape(np.array(x_input), (len(x_input)*args.seq_len, args.n_features))
label = np.reshape(np.array(y_pred), (len(y_pred)*args.seq_len, args.n_features))
for i in range(args.n_features):
    ax[i//3,i%3].plot(train[:,i], color='r', label="Original")
    ax[i//3,i%3].plot(label[:,i], color='b', label="Predicted")
    ax[i//3,i%3].set_title(f"Feature {i}")

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
plt.savefig(f"outputs/{args.seq_len}_steps_lookahead.png", dpi=100)
