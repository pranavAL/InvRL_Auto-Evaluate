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
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

warnings.filterwarnings("ignore")

class CraneDataset(Dataset):
    def __init__(self, X: np.ndarray, Y_recon: np.ndarray, Y_class: np.ndarray):
        self.X = X
        self.Y_recon = Y_recon
        self.Y_class = Y_class

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y_recon[index], self.Y_class[index]

class CraneDatasetModule():
    def __init__(self, seq_len, batch_size, num_workers=2):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        data_path = os.path.join("datasets","features_to_train.csv")

        df = pd.read_csv(data_path)

        train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

        df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

        sorted_sess = {}
        for sess in df['Session id'].unique():
            sorted_sess[sess] = df.loc[df['Session id']==sess,"Current trainee score at that time"].sum()
        sorted_sess = dict(sorted(sorted_sess.items(), key = lambda x:x[1], reverse=True))
        sess2rank = {sess:i for i,sess in enumerate(sorted_sess)}

        X = []
        Y_recon = []
        Y_rank = []

        for sess in df["Session id"].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            terminate = 2*self.seq_len
            for i in range(0,len(sess_feat) - terminate):
                X.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                Y_recon.append(sess_feat.iloc[(i+self.seq_len)-1:(i+(2*self.seq_len))-1,:][train_feats].values)
                Y_rank.append(sess2rank[sess])

        X = np.array(X)
        Y_recon = np.array(Y_recon)
        Y_rank = np.array(Y_rank)

        tot_indx = np.random.randint(low=0,high=len(X),size=len(X))
        train_size = int(0.8 * len(X))

        X_train, X_test = X[tot_indx[:train_size]], X[tot_indx[train_size:]]
        Y_train_recon, Y_test_recon = Y_recon[tot_indx[:train_size]], Y_recon[tot_indx[train_size:]]
        Y_train_class, Y_test_class = Y_rank[tot_indx[:train_size]], Y_rank[tot_indx[train_size:]]

        self.X_train, self.X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
        self.Y_train_recon, self.Y_test_recon = torch.tensor(Y_train_recon).float(), torch.tensor(Y_test_recon).float()
        self.Y_train_class, self.Y_test_class = torch.tensor(Y_train_class), torch.tensor(Y_test_class)

    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train, self.Y_train_recon, self.Y_train_class)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        return train_loader

    def test_dataloader(self):
        test_dataset = CraneDataset(self.X_test, self.Y_test_recon, self.Y_test_class)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return test_loader

class Encoder(nn.Module):
    def __init__(self, n_features, seq_len, latent_spc, fc_dim):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.latent_spc = latent_spc
        self.fc_dim =  fc_dim

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.n_features,
                             batch_first=True)

        self.elu = nn.ELU()
        self.fc = nn.Linear(self.n_features, self.fc_dim)
        self.fc1 = nn.Linear(self.fc_dim, self.fc_dim)
        self.ls1 = nn.Linear(self.fc_dim, self.latent_spc)
        self.ls2 = nn.Linear(self.fc_dim, self.latent_spc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        _, (h_out, _) = self.lstm(x)
        out = self.elu(self.fc(h_out))
        out = self.elu(self.fc1(out))
        mu, logvar = self.ls1(out), self.ls2(out)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, seq_len, latent_spc, n_features):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_spc = latent_spc

        self.initial_layer = nn.Linear(latent_spc, n_features)
        self.elu = nn.ELU()

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.n_features)

        self.output_layer = nn.Linear(self.n_features, n_features)

    def forward(self, x, last_feat):

        out = last_feat.unsqueeze(0)
        batch_size = out.size()[1]
        hidden = self.elu(self.initial_layer(x))
        cell = self.elu(self.initial_layer(x))
        outputs = torch.zeros(self.seq_len, batch_size, self.n_features).cuda()

        for i in range(self.seq_len):
            out, (hidden, cell) = self.lstm(out, (hidden, cell))
            self.output_layer(out)
            outputs[i] = out

        outputs = outputs.reshape((batch_size, self.seq_len, self.n_features))

        return outputs

class Classifier(nn.Module):
    def __init__(self, latent_spc, fc_dim, n_class):
        super(Classifier, self).__init__()
        self.latent_spc = latent_spc
        self.fc_dim = fc_dim
        self.n_class = n_class

        self.elu = nn.ELU()
        self.fc1 = nn.Linear(self.latent_spc, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.out = nn.Linear(self.fc_dim, self.n_class)

    def forward(self, x):
        out = self.elu(self.fc1(x))
        out = self.elu(self.fc2(out))
        out = self.out(out)

        return out

class LSTMPredictor(pl.LightningModule):
    def __init__(self, n_features, fc_dim, seq_len, batch_size, latent_spc, learning_rate, n_class):
        super(LSTMPredictor,self).__init__()
        self.n_features = n_features
        self.fc_dim = fc_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent_spc = latent_spc
        self.n_class = n_class

        self.encoder = Encoder(n_features, seq_len, latent_spc, fc_dim)
        self.decoder = Decoder(seq_len, latent_spc, n_features)
        self.classifier = Classifier(latent_spc, fc_dim, n_class)

        self.save_hyperparameters()

    def forward(self, x):
        last_feat = x[:,-1,:]
        x, mu, logvar = self.encoder(x)
        rank = self.classifier(x)
        x = self.decoder(x, last_feat)
        return x, mu, logvar, rank

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y_decod, y_rank = batch
        y_hat, mu, logvar, y_hat_rank = self(x)
        y_hat_rank = y_hat_rank.squeeze(0)
        rloss = F.mse_loss(y_hat, y_decod)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=1)
        bce = nn.CrossEntropyLoss()(y_hat_rank, y_rank)
        loss = rloss + kld + bce
        self.log('train/recon_loss', rloss, on_epoch=True)
        self.log('train/kld', kld, on_epoch=True)
        self.log('train/total_loss', loss, on_epoch=True)
        self.log('train/Classification_loss', bce, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_decod, y_rank = batch
        y_hat, mu, logvar, y_hat_rank = self(x)
        y_hat_rank = y_hat_rank.squeeze(0)
        rloss = F.mse_loss(y_hat, y_decod)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=0)
        bce = nn.CrossEntropyLoss()(y_hat_rank, y_rank)
        loss = rloss + kld + bce
        self.log('val/recon_loss', rloss, on_epoch=True)
        self.log('val/kld', kld, on_epoch=True)
        self.log('val/Classification_loss', bce, on_epoch=True)
        self.log('val/total_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y_decod, y_rank = batch
        y_hat, mu, logvar, y_hat_rank = self(x)
        y_hat_rank = y_hat_rank.squeeze(0)
        rloss = F.mse_loss(y_hat, y_decod)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp(), dim=1), dim=0)
        bce = nn.CrossEntropyLoss()(y_hat_rank, y_rank)
        loss = rloss + kld + bce
        self.log('test/recon_loss', rloss, on_epoch=True)
        self.log('test/kld', kld, on_epoch=True)
        self.log('test/Classification_loss', bce, on_epoch=True)
        self.log('test/total_loss', loss, on_epoch=True)
        return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameter Values")
    parser.add_argument('-sq','--seq_len', type=int, help="Sequence Length for input to LSTM")
    parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
    parser.add_argument('-me','--max_epochs', type=int, default=100, help="Number of epchs to train")
    parser.add_argument('-nf','--n_features', type=int, default=9, help="Length of feature for each sample")
    parser.add_argument('-ls','--latent_spc', type=int,default=64, help='Size of Latent Space')
    parser.add_argument('-fcd','--fc_dim', type=int, default=256, help="Number of FC Nodes")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001, help="Neural Network Learning Rate")
    parser.add_argument('-nc','--num_classes', type=int, default=40, help="Number of users")
    args = parser.parse_args()

    dm = CraneDatasetModule(
        seq_len = args.seq_len,
        batch_size = args.batch_size
    )

    model_path = os.path.join('save_model',f"{args.seq_len}seq_lstm_vae_.pth")
    wandb.init(name = f"{args.seq_len}seq_lstm_vae_Recon_New_seq_")

    train_loader = dm.train_dataloader()
    test_loader = dm.test_dataloader()

    seed_everything(1)

    wandb_logger = WandbLogger(project="lit-wandb")
    trainer = Trainer(max_epochs=args.max_epochs,
                    gpus = 1,
                    logger=wandb_logger,
                    log_every_n_steps=5)

    model = LSTMPredictor(
        n_features = args.n_features,
        fc_dim = args.fc_dim,
        seq_len = args.seq_len,
        batch_size = args.batch_size,
        latent_spc = args.latent_spc,
        learning_rate = args.learning_rate,
        n_class = args.num_classes
    )

    trainer.fit(model, train_loader, test_loader)
    trainer.test(model, test_dataloaders=test_loader)

    torch.save(model.state_dict(), model_path)

    wandb.finish()
