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
from train_LSTM import LSTMPredictor

parser = argparse.ArgumentParser(description="Hyperparameter Values")
parser.add_argument('-sq','--seq_len', default=32, type=int, help="Sequence Length for input to LSTM")
parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
parser.add_argument('-me','--max_epochs', type=int, default=1000, help="Number of epchs to train")
parser.add_argument('-nf','--n_features', type=int, default=3, help="Length of feature for each sample")
parser.add_argument('-ls','--latent_spc', type=int,default=8, help='Size of Latent Space')
parser.add_argument('-kldc','--beta', type=float, default=0.001, help='weighting factor of KLD')
parser.add_argument('-gam','--gamma', type=float, default=0.1, help='weighting factor of MSE')
parser.add_argument('-fcd','--fc_dim', type=int, default=64, help="Number of FC Nodes")
parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help="Neural Network Learning Rate")
parser.add_argument('-mp', '--model_path', type=str, default='lstm_vae.pth', help="Saved model path")
parser.add_argument('-istr','--is_train', type=bool, help="Train or Testing")
args = parser.parse_args()

model_path = os.path.join('save_model', args.model_path)
df = pd.read_csv(os.path.join("datasets",'features_to_train.csv'))

train_feats = ['Engine Average Power', 'Engine Torque Average']
print(f"Min: {df.loc[:,train_feats].min()}")
print(f"Max: {df.loc[:,train_feats].max()}")

df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

model = LSTMPredictor(
    n_features = args.n_features,
    fc_dim = args.fc_dim,
    seq_len = args.seq_len,
    batch_size = args.batch_size,
    latent_spc = args.latent_spc,
    learning_rate = args.learning_rate,
    epochs = args.max_epochs,
    beta = args.beta
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
    ax[i].plot(train[0,:,i], color='r', label="Original")
    ax[i].plot(label[0,:,i], color='b', label="Predicted")
    ax[i].set_title(f"Feature {i}")

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
#plt.savefig(f"outputs/{args.seq_len}_steps_lookahead.png", dpi=100)
plt.show()
