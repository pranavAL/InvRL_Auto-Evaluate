import os
import torch
import numpy as np
import pandas as pd
from arguments import get_args
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor

args = get_args()

model_path = os.path.join('save_model', args.model_path)
df = pd.read_csv(os.path.join("datasets",'features_to_train.csv'))

train_feats = ['Engine Average Power', 'Engine Torque Average', 'Fuel Consumption Rate Average',
               'Number of tennis balls knocked over by operator','Number of equipment collisions', 'Number of poles that fell over',
               'Number of poles touched', 'Collisions with environment']

full_data_path = os.path.join("datasets", "features_to_train.csv")
fd = pd.read_csv(full_data_path)

df.loc[:,train_feats] = (df.loc[:,train_feats] - fd.loc[:,train_feats].min())/(
                        fd.loc[:,train_feats].max() - fd.loc[:,train_feats].min())

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
    label = label.unsqueeze(0).to(model.device)
    recon, _, _ = model(input, label, is_train=False)
    recon = get_numpy(recon)
    return recon

y_pred = []
x_input = []
sessions = df["Session id"].unique()
feats = ['Avg. Power', 'Avg. Torque', 'Avg. Fuel Consmp.',
        'Balls Knocked','Eqip. Collison', 'Poles fell',
        'Poles touched', 'Env. Collison']

for sess in sessions[4:5]: # Select Sessions to view
    sess_feat = df.loc[df["Session id"]==sess,:]
    terminate = args.seq_len
    for i in range(0,len(sess_feat)-terminate,terminate):
        train = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
        init_label = list(sess_feat.iloc[i:i+args.seq_len,:][train_feats].values)
        recon = get_embeddings(torch.tensor(train).float(), torch.tensor(init_label).float())
        y_pred.append(recon)
        x_input.append(train)

fig, ax = plt.subplots(2,4)
fig.set_size_inches(15, 10)
train = np.reshape(np.array(x_input), (-1, args.n_features))
label = np.reshape(np.array(y_pred), (-1, args.n_features))

for i in range(args.n_features):
    ax[i//4][i%4].plot(train[:,i], color='r', label="Original")
    ax[i//4][i%4].plot(label[:,i], color='b', label="Predicted")
    ax[i//4][i%4].set_title(f"{feats[i]}")

lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper center')
#plt.savefig(f"outputs/{args.seq_len}_steps_lookahead.png", dpi=100)
plt.show()
