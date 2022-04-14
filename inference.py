import os
import torch
import pandas as pd
from arguments import get_args
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train_LSTM import LSTMPredictor

args = get_args()
model_name = f"{args.model_path.split('.')[0]}"
cmd = f'ffmpeg -framerate 10 -y -i temp/%d.png -c:v libx264 -pix_fmt yuv420p outputs/{model_name}.mp4'
model_path = os.path.join('save_model', args.model_path)
os.makedirs(f"temp", exist_ok = True)

df = pd.read_csv(os.path.join("datasets","features_to_train.csv"))

train_feats = ['Engine Average Power', 'Engine Torque Average', 'Fuel Consumption Rate Average', 
               'Number of tennis balls knocked over by operator','Number of equipment collisions', 'Number of poles that fell over',
               'Number of poles touched', 'Collisions with environment']

full_data_path = os.path.join("datasets", "features_to_train.csv")
fd = pd.read_csv(full_data_path)

def normalize(df):
    df.loc[:,train_feats] = (df.loc[:,train_feats] - fd.loc[:,train_feats].min())/(fd.loc[:,train_feats].max() - fd.loc[:,train_feats].min())
    return df

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

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_embeddings(data, label):
    data = data.unsqueeze(0).to(model.device)
    label = label.unsqueeze(0).to(model.device)
    _, mu, _ = model(data, label, is_train=False)
    mu = get_numpy(mu)
    return mu

def save_meta_data(df):
    meta_data = pd.DataFrame(columns=['sess', 'embeddings','Number of tennis balls knocked over by operator',
               'Number of equipment collisions', 'Number of poles that fell over',
               'Number of poles touched', 'Collisions with environment'])          
    
    for sess in df["Session id"].unique():
        sess_feat = df.loc[df["Session id"]==sess,:]
        terminate = args.seq_len
        for i in range(0,len(sess_feat)-terminate):
            train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
            init_label = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
            penlt_feat = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats[3:]].max().values)
            mu = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
            meta_data.loc[len(meta_data)] = [sess, list(mu.astype(float)), *penlt_feat]

    meta_data = get_tsne(meta_data)
    meta_data.to_csv(f"outputs/{model_name}")
    return meta_data

def get_tsne(meta_data):
    embeddings = list(meta_data['embeddings'].values)
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000, n_jobs=-1,
                early_exaggeration=12, learning_rate=200.0)
    Z_tsne = tsne.fit_transform(embeddings)
    meta_data['X'] = Z_tsne[:,0]
    meta_data['Y'] = Z_tsne[:,1]
    return meta_data

def animate(meta_data):

    meta_data.loc[:,train_feats[3:]] = meta_data.loc[:,train_feats[3:]] * (fd.loc[:,train_feats[3:]].max() - fd.loc[:,train_feats[3:]].min()) + fd.loc[:,train_feats[3:]].min()
    meta_data['Total Infractions'] = meta_data.loc[:,train_feats[3:]].sum(axis=1)

    plt.scatter(meta_data.X, meta_data.Y, s=50, c=meta_data['Total Infractions'], cmap="RdBu")
    
    plt.colorbar()
    plt.show()
   
with torch.no_grad():

    meta_data = save_meta_data(df)
    animate(meta_data)
