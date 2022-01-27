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
from train_LSTM import LSTMPredictor
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser(description="Save models")
parser.add_argument('-sq','--seq_len', type=int, default=16, help="Sequence Length for input to LSTM")
parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
parser.add_argument('-me','--max_epochs', type=int, default=100, help="Number of epchs to train")
parser.add_argument('-nf','--n_features', type=int, default=9, help="Length of feature for each sample")
parser.add_argument('-mp', '--model_path', type=str, help="Saved model path")
parser.add_argument('-fcd','--fc_dim', type=int, default=256, help="Number of FC Nodes")
parser.add_argument('-ts', '--to_save', type=bool, default=False, help="To save image or not")
parser.add_argument('-nc','--num_classes', type=int, default=40, help="Number of users")
parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help="Neural Network Learning Rate")
parser.add_argument('-ls','--latent_spc', type=int,default=64, help='Size of Latent Space')

args = parser.parse_args()

color_sample = [(u(0,1),u(0,1),u(0,1)) for i in range(40)]

model_name = f"{args.model_path.split('.')[0]}"
cmd = f'ffmpeg -framerate 10 -y -i temp/%d.png -c:v libx264 -pix_fmt yuv420p outputs/{model_name}.mp4'
model_path = os.path.join('save_model', args.model_path)
os.makedirs(f"temp", exist_ok = True)
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
    learning_rate = args.learning_rate,
    n_class = args.num_classes
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

def get_embeddings(data):
    recon, mu, _, _ = model(data.unsqueeze(0).to(model.device))
    mu = get_numpy(mu)
    return mu

def plot_sample(data, ax, color):
    ax.scatter(*data, color=color)
    return ax

def plot_by_session(ax, patches, sess, meta_data):
    sess_feat = meta_data.loc[meta_data['rank']==sess,:]
    final_score = sess_feat.iloc[0,:]['FinalScore']
    color = color_sample[sess]
    X = sess_feat.iloc[:,:]['X']
    Y = sess_feat.iloc[:,:]['Y']
    ax.scatter(X, Y, color=color)
    patches.append(mpatches.Patch(color=color, label=f"Rank {sess}"))
    return ax, patches

def save_meta_data(df, sorted_sess):
    meta_data = pd.DataFrame(columns=['rank', 'embeddings', 'penalty', 'FinalScore'])
    for i, sess in enumerate(list(sorted_sess.keys())):
        sess_feat = df.loc[df["Session id"]==sess,:]
        final_score = sum(sess_feat["Current trainee score at that time"].values)
        for j in range(0,len(sess_feat) - args.seq_len):
            score = sum(sess_feat.iloc[j:j+args.seq_len,:]["Current trainee score at that time"].values)
            mu = get_embeddings(torch.tensor(sess_feat.iloc[j:j+args.seq_len,:][train_feats].values).float())
            meta_data.loc[len(meta_data)] = [i, list(mu.astype(float)), score, final_score]
    meta_data = get_tsne(meta_data)
    meta_data.to_csv(f"outputs/{model_name}")
    return meta_data

def get_tsne(meta_data):
    embeddings = list(meta_data['embeddings'].values)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=6000, n_jobs=-1)
    Z_tsne = tsne.fit_transform(embeddings)
    meta_data['X'] = Z_tsne[:,0]
    meta_data['Y'] = Z_tsne[:,1]
    return meta_data

def plot_datas(meta_data, ax):

    patches = []

    for i, sess in enumerate(meta_data['rank'].unique()):
        ax, patches = plot_by_session(ax, patches, sess, meta_data)

    return ax, patches


def animate(meta_data):

    count = 0

    for index, row in meta_data.iterrows():
        print(count)
        fig, ax = plt.subplots()
        rank = row['rank']
        penalty = row['penalty']
        final_score = row['FinalScore']
        x = row['X']
        y = row['Y']
        ax, patches = plot_datas(meta_data, ax)
        ax = plot_sample([x,y], ax, (0,0,0))
        #ax.legend(handles=patches, loc='best', ncol=3, fancybox=True, shadow=True)
        plt.title(f"Rank: {rank}, Penalty: {penalty}, Final Score: {final_score}")
        fig.savefig(f"temp/{count}.png")
        plt.close()
        count+=1
        if index > 5000:
            break

    os.system(cmd)
    shutil.rmtree(f"temp")

sorted_sess = sort_sessions(df)

with torch.no_grad():

    #meta_data = save_meta_data(df, sorted_sess)
    meta_data = pd.read_csv(f"outputs/{model_name}")
    animate(meta_data)
