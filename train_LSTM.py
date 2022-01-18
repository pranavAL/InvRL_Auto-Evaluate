import os
import torch
import wandb
import argparse
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

warnings.filterwarnings("ignore")

class CraneDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

class CraneDatasetModule():
    def __init__(self, seq_len, batch_size, num_workers=2):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

        data_path = os.path.join("datasets","features_to_train.csv")

        df = pd.read_csv(data_path)

        train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

        df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

        best_sess = [sess for sess in df["Session id"].unique() if df.loc[df["Session id"]==sess,"Current trainee score at that time"].min()==0]

        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for sess in df["Session id"].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            for i in range(0,len(sess_feat) - (2*self.seq_len)):
                if sess in best_sess:
                    X_train.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                    Y_train.append(sess_feat.iloc[(i+self.seq_len)-1:(i+(2*self.seq_len))-1,:][train_feats].values)
                else:
                    X_test.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                    Y_test.append(sess_feat.iloc[(i+self.seq_len)-1:(i+(2*self.seq_len))-1,:][train_feats].values)

        self.X_train = torch.tensor(X_train).float()
        self.Y_train = torch.tensor(Y_train).float()
        self.X_test = torch.tensor(X_test).float()
        self.Y_test = torch.tensor(Y_test).float()

    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train, self.Y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return train_loader

    def test_dataloader(self):
        test_dataset = CraneDataset(self.X_test, self.Y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return test_loader

class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim, seq_len, num_layers):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.latent_spc = int(self.embedding_dim / 2)

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.hidden_dim,
                             num_layers=num_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, self.latent_spc)
        self.fc2 = nn.Linear(self.embedding_dim, self.latent_spc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        _, (h_out, _) = self.lstm(x)
        out = self.fc(h_out)
        mu, logvar = self.fc1(out), self.fc2(out)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, input_dim, seq_len, num_layers, n_features):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.seq_len, self.input_dim = seq_len, input_dim

        self.initial_layer = nn.Linear(self.input_dim, self.n_features)

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.n_features,
                             num_layers=num_layers)

        self.output_layer = nn.Linear(self.n_features, n_features)

    def forward(self, x, last_feat):

        out = last_feat.unsqueeze(0)
        batch_size = out.size()[1]

        hidden = self.initial_layer(x)
        cell = self.initial_layer(x)
        outputs = torch.zeros(self.seq_len, batch_size, self.n_features).cuda()

        for i in range(self.seq_len):
            out, (hidden, cell) = self.lstm(out, (hidden, cell))
            self.output_layer(out)
            outputs[i] = out

        outputs = outputs.reshape((batch_size, self.seq_len, self.n_features))
        return outputs

class LSTMPredictor(pl.LightningModule):
    def __init__(self, n_features, embedding_dim, seq_len, batch_size, num_layers, learning_rate):
        super(LSTMPredictor,self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.latent_spc = int(embedding_dim / 2)

        self.encoder = Encoder(n_features, embedding_dim, seq_len, num_layers)
        self.decoder = Decoder(self.latent_spc, seq_len, self.num_layers, n_features)

        self.save_hyperparameters()

    def forward(self, x):
        last_feat = x[:,-1,:]
        x, mu, logvar = self.encoder(x)
        x = self.decoder(x, last_feat)
        return x, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)
        rloss = F.mse_loss(y_hat, y)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=1)
        loss = rloss + kld
        self.log('train/recon_loss', rloss, on_epoch=True)
        self.log('train/kld', kld, on_epoch=True)
        self.log('train/total_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)
        rloss = F.mse_loss(y_hat, y)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = rloss + kld
        self.log('val/recon_loss', rloss, on_epoch=True)
        self.log('val/kld', kld, on_epoch=True)
        self.log('val/total_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mu, logvar = self(x)
        rloss = F.mse_loss(y_hat, y)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=0)
        loss = rloss + kld
        self.log('test/recon_loss', rloss, on_epoch=True)
        self.log('test/kld', kld, on_epoch=True)
        self.log('test/total_loss', loss, on_epoch=True)
        return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameter Values")
    parser.add_argument('-mt', '--model_type', type=str, help="Teacher Forcing or reconstruct same output")
    parser.add_argument('-sq','--seq_len', type=int, help="Sequence Length for input to LSTM")
    parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
    parser.add_argument('-me','--max_epochs', type=int, default=100, help="Number of epchs to train")
    parser.add_argument('-nf','--n_features', type=int, default=9, help="Length of feature for each sample")
    parser.add_argument('-ed','--embedding_dim', type=int, default=4, help="Dimension of Sample Space")
    parser.add_argument('-nl','--num_layers', type=int, default=1, help="Number of LSTM layers")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help="Neural Network Learning Rate")
    args = parser.parse_args()


    p = dict(
    seq_len = args.seq_len,
    batch_size = args.batch_size,
    max_epochs = args.max_epochs,
    n_features = args.n_features,
    embedding_dim = args.embedding_dim,
    num_layers = args.num_layers,
    learning_rate = args.learning_rate
)
    dm = CraneDatasetModule(
        seq_len = p['seq_len'],
        batch_size = p['batch_size']
    )

    model_path = os.path.join('save_model',f"{p['seq_len']}seq_lstm_vae({args.mt}).pth")
    wandb.init(name = f"{p['seq_len']}seq_lstm_vae(Recon_New_seq)")

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    seed_everything(1)

    wandb_logger = WandbLogger(project="lit-wandb")
    trainer = Trainer(max_epochs=p['max_epochs'],
                    gpus = 1,
                    logger=wandb_logger,
                    log_every_n_steps=5)

    model = LSTMPredictor(
        n_features = p['n_features'],
        embedding_dim = p['embedding_dim'],
        seq_len = p['seq_len'],
        batch_size = p['batch_size'],
        num_layers = p['num_layers'],
        learning_rate = p['learning_rate']
    )

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_dataloaders=test_loader)

    torch.save(model.state_dict(), model_path)

    wandb.finish()
