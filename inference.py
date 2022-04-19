import os
import torch
import pandas as pd
from arguments import get_args
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor

def normalize(df):
    df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(
                            df.loc[:,train_feats].max() - df.loc[:,train_feats].min())
    return df

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_embeddings(data, label):
    data = data.unsqueeze(0).to(model.device)
    label = label.unsqueeze(0).to(model.device)
    z, _, _, _ = model.encoder(data)
    embedd = get_numpy(z)
    return list(embedd)

def save_meta_data(df):
    meta_data = pd.DataFrame(columns=['sess', 'X', 'Y','Number of tennis balls knocked over by operator',
               'Number of equipment collisions', 'Number of poles that fell over',
               'Number of poles touched', 'Collisions with environment'])          
    
    for sess in df["Session id"].unique():
        sess_feat = df.loc[df["Session id"]==sess,:]
        terminate = args.seq_len
        for i in range(0,len(sess_feat)-terminate,terminate):
            train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
            init_label = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
            penlt_feat = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats[3:]].max().values)
            embedd = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
            meta_data.loc[len(meta_data)] = [sess, embedd[0], embedd[1], *penlt_feat]

    meta_data.to_csv(f"outputs/{model_name}")
    return meta_data

def animate(meta_data):

    meta_data['Total Infractions'] = meta_data.loc[:,train_feats[3:]].sum(axis=1)

    plt.scatter(meta_data.X, meta_data.Y, s=50, c=meta_data['Total Infractions'], cmap="RdBu")
    plt.colorbar()
    plt.show()

args = get_args()
model_name = f"{args.model_path.split('.')[0]}"
model_path = os.path.join('save_model', args.model_path)

df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))

train_feats = ['Engine Average Power', 'Engine Torque Average', 'Fuel Consumption Rate Average', 
               'Number of tennis balls knocked over by operator','Number of equipment collisions', 
               'Number of poles that fell over', 'Number of poles touched', 'Collisions with environment']

df = normalize(df)

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

with torch.no_grad():

    meta_data = save_meta_data(df)
    animate(meta_data)
