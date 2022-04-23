import os
import torch
import wandb
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from vae_arguments import get_args
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
    def __init__(self, batch_size, num_workers=2):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.X_train, self.Y_train_recon = self.get_data('train_df.csv')
        self.X_val, self.Y_val_recon = self.get_data('val_df.csv')
        self.X_test, self.Y_test_recon = self.get_data('test_df.csv')

    def get_data(self, file_type):

        train_feats = ['Number of tennis balls knocked over by operator','Number of equipment collisions',
                       'Number of poles that fell over', 'Number of poles touched', 'Collisions with environment']

        train_data_path = os.path.join("datasets",file_type)
        full_data_path = os.path.join("datasets", "features_to_train.csv")
        fd = pd.read_csv(full_data_path)

        df = pd.read_csv(train_data_path)

        for f in train_feats:
           df[f] = [np.random.randint(0,20) for _ in range(len(df))]

        df.loc[:,train_feats] = (df.loc[:,train_feats] - fd.loc[:,train_feats].min())/(
                                 fd.loc[:,train_feats].max() - fd.loc[:,train_feats].min())
                                
        input = []
        pred = []
        for sess in df['Session id'].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            for i in range(0,len(sess_feat)):
                input.append(list(sess_feat.iloc[i,:][train_feats].values))
                pred.append(sum(list(sess_feat.iloc[i,:][train_feats].values)))
        pred = pred / max(pred)
        return torch.tensor(input).float(), torch.tensor(pred).float()

    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train, self.Y_train_recon)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

        return train_loader

    def val_dataloader(self):
        val_dataset = CraneDataset(self.X_val, self.Y_val_recon)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

        return val_loader

    def test_dataloader(self):
        test_dataset = CraneDataset(self.X_test, self.Y_test_recon)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)

        return test_loader

class Encoder(nn.Module):
    def __init__(self, n_features, latent_spc, fc_dim):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.latent_spc = latent_spc
        self.fc_dim =  fc_dim

        self.elu = nn.ELU()
        self.fc = nn.Linear(self.n_features, self.fc_dim)
        self.ls1 = nn.Linear(self.fc_dim, self.latent_spc)
        self.ls2 = nn.Linear(self.fc_dim, self.latent_spc)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        out = self.elu(self.fc(x))
        mu, logvar = self.ls1(out), self.ls2(out)
        z_latent = self.reparameterize(mu, logvar)
        return z_latent, mu, logvar

class Decoder(nn.Module):
    def __init__(self, n_features, latent_spc, fc_dim):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.fc_dim =  fc_dim
        self.latent_spc = latent_spc

        self.elu = nn.ELU()
        self.fc = nn.Linear(self.latent_spc, self.fc_dim)
        self.final = nn.Linear(self.fc_dim, 1)

    def forward(self, inp):
        out = self.elu(self.fc(inp))
        out = self.final(out)
        return out

class SafetyPredictor(pl.LightningModule):
    def __init__(self, n_features, fc_dim, batch_size, latent_spc, learning_rate, epochs, beta):
        super(SafetyPredictor,self).__init__()
        self.n_features = n_features
        self.fc_dim = fc_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent_spc = latent_spc
        self.max_epochs = epochs
        self.beta = beta

        self.encoder = Encoder(n_features, latent_spc, fc_dim)
        self.decoder = Decoder(n_features, latent_spc, fc_dim)

        self.save_hyperparameters()

    def forward(self, x):
        x, mu, logvar = self.encoder(x)
        out = self.decoder(x)

        return out, mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def final_process(self, batch, p_type):
        x, y_decod = batch
        y_hat, mu, logvar = self(x)

        rloss = F.mse_loss(y_hat, y_decod)
        kld = -0.5 * torch.sum(1 + logvar -mu.pow(2) - logvar.exp())
        loss = 10 * rloss + kld * self.beta * 0.1

        self.log(f'{p_type}/recon_loss', rloss, on_epoch=True)
        self.log(f'{p_type}/kld', kld, on_epoch=True)
        self.log(f'{p_type}/total_loss', loss, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.final_process(batch, 'test')
        return loss

if __name__ == "__main__":

    args = get_args()

    dm = CraneDatasetModule(
        batch_size = args.batch_size_safety
    )

    model_path = os.path.join('save_model',f"vae_recon_safety.pth")
    wandb.init(name = f"vae_recon_safety")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    seed_everything(1)

    model = SafetyPredictor(
        n_features = args.n_features_safety,
        fc_dim = args.fc_dim_safety,
        batch_size = args.batch_size_safety,
        latent_spc = args.latent_spc_safety,
        learning_rate = args.learning_rate,
        epochs = args.max_epochs,
        beta = args.beta
    )

    wandb_logger = WandbLogger(project="lit-wandb")

    trainer = Trainer(max_epochs=args.max_epochs,
                    gpus = 1,
                    logger=wandb_logger,
                    log_every_n_steps=500,
                    )

    wandb_logger.watch(model, log="all")

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

    torch.save(model.state_dict(), model_path)
    print("Saved")

    wandb.finish()
