import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vae_arguments import get_args
from model_infractions import SafetyPredictor

def normalize(df):
    df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(
                            df.loc[:,train_feats].max() - df.loc[:,train_feats].min())
    return df

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_embeddings(data):
    data = data.unsqueeze(0).to(model.device)
    z, mu, _ = model.encoder(data)
    embedd = get_numpy(z)
    return list(embedd)

def save_meta_data(df):
    meta_data = pd.DataFrame(columns=['sess', 'X', 'Y','infractions'])

    for i, sess in enumerate(df["Session id"].unique()):
        print(f"Session: {i}")
        sess_feat = df.loc[df["Session id"]==sess,:]
        for i in range(0,len(sess_feat)):
            train = list(sess_feat.iloc[i,:][train_feats].values)
            penlt_feat = sum(list(map(lambda x: x * 20, list(sess_feat.iloc[i,:][train_feats].values))))
            embedd = get_embeddings(torch.tensor(train).float())
            meta_data.loc[len(meta_data)] = [sess, embedd[0], embedd[1], penlt_feat]

    return meta_data

def animate(meta_data):

    plt.axis('off')
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off',
                    labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.scatter(meta_data.X, meta_data.Y, s=50, c=meta_data['infractions'], cmap="jet")
    plt.colorbar(label="Total Infractions")
    plt.savefig(f"outputs/vae_safety.png", dpi=100)
    plt.show()

args = get_args()

model_path = os.path.join('save_model', 'vae_recon_safety.pth')

df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))

train_feats = ['Number of tennis balls knocked over by operator','Number of equipment collisions',
               'Number of poles that fell over', 'Number of poles touched', 'Collisions with environment']

for f in train_feats:
   df[f] = [np.random.randint(0,20) for _ in range(len(df))]

df = normalize(df)

model = SafetyPredictor(
    n_features = args.n_features_safety,
    fc_dim = args.fc_dim_safety,
    batch_size = args.batch_size_safety,
    latent_spc = args.latent_spc_safety,
    learning_rate = args.learning_rate,
    epochs = args.max_epochs,
    beta = args.beta
)

model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

with torch.no_grad():

    meta_data = save_meta_data(df)
    animate(meta_data)
