import os
import torch
import numpy as np
import pandas as pd
from vae_arguments import get_args
import matplotlib.pyplot as plt
from model_dynamics import DynamicsPredictor

args = get_args()

model_path = os.path.join('save_model', 'lstm_vae_dynamic.pth')
df = pd.read_csv(os.path.join("datasets",'features_to_train.csv'))

train_feats = ['Engine Average Power', 'Engine Torque Average', 'Fuel Consumption Rate Average']

full_data_path = os.path.join("datasets", "features_to_train.csv")
fd = pd.read_csv(full_data_path)

df.loc[:,train_feats] = (df.loc[:,train_feats] - fd.loc[:,train_feats].min())/(
                        fd.loc[:,train_feats].max() - fd.loc[:,train_feats].min())

model = DynamicsPredictor(
    n_features = args.n_features_dynamics,
    fc_dim = args.fc_dim_dynamics,
    seq_len = args.seq_len_dynamics,
    batch_size = args.batch_size_dynamics,
    latent_spc = args.latent_spc_dynamics,
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
    label = label.unsqueeze(0).to(model.device)
    recon, _, _ = model(input, label, is_train=False)
    recon = get_numpy(recon)
    return recon

y_pred = []
x_input = []
sessions = df["Session id"].unique()
feats = ['Avg. Power', 'Avg. Torque', 'Avg. Fuel Consmp.']

for sess in sessions[4:5]: # Select Sessions to view
    sess_feat = df.loc[df["Session id"]==sess,:]
    terminate = args.seq_len_dynamics
    for i in range(0,len(sess_feat)-terminate,terminate):
        train = sess_feat.iloc[i:i+args.seq_len_dynamics,:][train_feats].values.tolist()
        init_label = list(sess_feat.iloc[i:i+args.seq_len_dynamics,:][train_feats].values)
        recon = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
        y_pred.append(recon[0])
        x_input.append(train[0])

fig, ax = plt.subplots(1,3)
fig.set_size_inches(15, 10)
x_input = np.reshape(np.array(x_input), (-1, 3))
y_pred = np.reshape(np.array(y_pred), (-1, 3))

for i in range(3):
    ax[i].plot(x_input[:,i], color='r', label="Original")
    ax[i].plot(y_pred[:,i], color='b', label="Predicted")
    ax[i].set_title(f"{feats[i]}")

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
plt.savefig(f"outputs/lstm_vae_dynamic.png", dpi=100)
plt.show()
