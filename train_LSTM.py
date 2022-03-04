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
    def __init__(self, X: np.ndarray, Y_recon: np.ndarray):
        self.X = X
        self.Y_recon = Y_recon

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y_recon[index]

class CraneDatasetModule():
    def __init__(self, seq_len, batch_size, num_workers=2):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.X_train, self.Y_train_recon = self.get_data('features_to_train.csv')
        self.X_val, self.Y_val_recon = self.get_data('val_df.csv')
        #self.X_test, self.Y_test_recon = self.get_data('test_df.csv')

    def get_data(self, file_type):

        train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)']

        train_data_path = os.path.join("datasets",file_type)

        df = pd.read_csv(train_data_path)

        df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())

        input = []
        pred = []
        for sess in df['Session id'].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            terminate = 2*self.seq_len
            score = abs(sess_feat.loc[:,"Current trainee score at that time"].sum())
            if score <= 25:
                for i in range(0,len(sess_feat)-terminate):
                    input.append(list(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values))
                    pred.append(list(sess_feat.iloc[i+self.seq_len-1:i+terminate-1,:][train_feats].values))

        return torch.tensor(input).float(), torch.tensor(pred).float()

    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train, self.Y_train_recon)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

        return train_loader

    def val_dataloader(self):
        val_dataset = CraneDataset(self.X_val, self.Y_val_recon)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

        return val_loader

    # def test_dataloader(self):
    #     test_dataset = CraneDataset(self.X_test, self.Y_test_recon)
    #     test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
    #
    #     return test_loader

class Encoder(nn.Module):
    def __init__(self, n_features, latent_spc, fc_dim):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.latent_spc = latent_spc
        self.fc_dim =  fc_dim

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.n_features,
                             batch_first=True,
                             dropout=0.2)

        self.elu = nn.ELU()
        self.fc = nn.Linear(self.n_features, self.fc_dim)
        self.fc1 = nn.Linear(self.fc_dim, self.fc_dim)
        self.ls1 = nn.Linear(self.fc_dim, self.latent_spc)
        self.ls2 = nn.Linear(self.fc_dim, self.latent_spc)
        self.final = nn.Linear(self.latent_spc, self.n_features)

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
        z = self.elu(self.final(z))
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self, n_features, fc_dim):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.fc_dim =  fc_dim

        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.n_features,
                             batch_first=True,
                             dropout=0.2)

    def forward(self, inp, hidden):
        out, hidden = self.lstm(inp, hidden)
        return out, hidden

class LSTMPredictor(pl.LightningModule):
    def __init__(self, n_features, fc_dim, seq_len, batch_size, latent_spc, learning_rate):
        super(LSTMPredictor,self).__init__()
        self.n_features = n_features
        self.fc_dim = fc_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent_spc = latent_spc

        self.encoder = Encoder(n_features, latent_spc, fc_dim)
        self.decoder = Decoder(n_features, fc_dim)

        self.save_hyperparameters()

    def forward(self, x, y_decod, is_train):
        inp_batch_size = x.size(0)
        x, mu, logvar = self.encoder(x)
        hidden = (x, x)
        output = []

        if is_train:
            out, _ = self.decoder(y_decod, hidden)
            output = out
        else:
            batch_size = y_decod.size()[0]
            out = y_decod[:,0,:].unsqueeze(1)
            for i in range(args.seq_len):
                out, hidden = self.decoder(out, hidden)
                output.append(out)
            output = torch.stack(output, dim=0)
            output = torch.reshape(output, (batch_size, args.seq_len, self.n_features))

        return output, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def final_process(self, batch, p_type, is_train):
        x, y_decod = batch
        y_hat, mu, logvar = self(x, y_decod, is_train)

        rloss = F.mse_loss(y_hat, y_decod)
        kld = -0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp())
        beta = ((self.current_epoch//50)/args.max_epochs) * args.beta

        loss = rloss + kld * beta

        self.log(f'{p_type}/recon_loss', rloss, on_epoch=True)
        self.log(f'{p_type}/kld', kld, on_epoch=True)
        self.log(f'{p_type}/total_loss', loss, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'train', is_train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'val', is_train=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'test', is_train=False)
        return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameter Values")
    parser.add_argument('-sq','--seq_len', default=32, type=int, help="Sequence Length for input to LSTM")
    parser.add_argument('-bs','--batch_size', type=int, default=8, help="Batch Size")
    parser.add_argument('-me','--max_epochs', type=int, default=1000, help="Number of epchs to train")
    parser.add_argument('-nf','--n_features', type=int, default=7, help="Length of feature for each sample")
    parser.add_argument('-ls','--latent_spc', type=int,default=8, help='Size of Latent Space')
    parser.add_argument('-kldc','--beta', type=float, default=0.001, help='weighting factor of KLD')
    parser.add_argument('-gam','--gamma', type=float, default=0.1, help='weighting factor of MSE')
    parser.add_argument('-fcd','--fc_dim', type=int, default=64, help="Number of FC Nodes")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help="Neural Network Learning Rate")
    parser.add_argument('-mp', '--model_path', type=str, default='lstm_vae.pth', help="Saved model path")
    parser.add_argument('-istr','--is_train', type=bool, help="Train or Testing")
    args = parser.parse_args()

    dm = CraneDatasetModule(
        seq_len = args.seq_len,
        batch_size = args.batch_size
    )

    model_path = os.path.join('save_model',f"lstm_vae.pth")
    wandb.init(name = f"{args.seq_len}seq_lstm_vae_Recon_w_class")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # test_loader = dm.test_dataloader()

    seed_everything(1)

    model = LSTMPredictor(
        n_features = args.n_features,
        fc_dim = args.fc_dim,
        seq_len = args.seq_len,
        batch_size = args.batch_size,
        latent_spc = args.latent_spc,
        learning_rate = args.learning_rate
    )

    wandb_logger = WandbLogger(project="lit-wandb")

    trainer = Trainer(max_epochs=args.max_epochs,
                    gpus = 1,
                    logger=wandb_logger,
                    log_every_n_steps=500)

    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_loader)
    # trainer.test(model, test_dataloaders=test_loader)

    torch.save(model.state_dict(), model_path)

    wandb.finish()
