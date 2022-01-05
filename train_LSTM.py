import os
import torch
import wandb
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

wandb.login()

class CraneDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.X[index]
    
class CraneDatasetModule():
    def __init__(self, seq_len, batch_size, num_workers=2):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        self.Y_penalty = None
        
        data_path = os.path.join("datasets","features_to_train.csv")
        
        df = pd.read_csv(data_path)
        
        train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

        df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())
        
        best_sess = [sess for sess in df["Session id"].unique() if df.loc[df["Session id"]==sess,"Current trainee score at that time"].min()==0]
        X_train = []
        X_test = []
        Y_penalty = []
        for sess in df["Session id"].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            for i in range(0,len(sess_feat) - self.seq_len):
                if sess in best_sess:
                    X_train.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                else:
                    X_test.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                    Y_penalty.append(sess_feat.iloc[i:i+self.seq_len,:]["Current trainee score at that time"].sum())
                    
        self.X_train = torch.tensor(X_train).float()
        self.X_test = torch.tensor(X_test).float()  
        print(len(self.X_train))
        print(len(self.X_test))
        self.Y_penalty = Y_penalty                      
            
    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        return train_loader
    
    def test_dataloader(self):
        test_dataset = CraneDataset(self.X_test)
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
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).cuda())
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).cuda())
        
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = self.fc(h_out)
        mu, logvar = self.fc1(out), self.fc2(out)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
            
class Decoder(nn.Module):
    def __init__(self, input_dim, seq_len, num_layers, n_features):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.seq_len, self.input_dim = seq_len, input_dim
        self.embedding_dim = 2 * input_dim
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                             hidden_size=self.embedding_dim,
                             num_layers=num_layers,
                             batch_first=True)
        
        self.output_layer = nn.Linear(self.embedding_dim, n_features)
        
    def forward(self, x):
        x = x.repeat(self.seq_len, 1, 1)
        x = x.reshape((-1, self.seq_len, self.input_dim))
        
        out, _ = self.lstm(x)
        out = out.reshape((-1, self.seq_len, self.embedding_dim))
        
        return self.output_layer(out)
             
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
        x, mu, logvar = self.encoder(x)
        x = self.decoder(x)
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
    
    def predict_step(self, batch, batch_idx):
        return self(batch)
    
p = dict(
    seq_len = 8,
    batch_size = 8,
    max_epochs = 100,
    n_features = 9,
    embedding_dim = 4,
    num_layers = 1,
    learning_rate = 0.001
)       
dm = CraneDatasetModule(
    seq_len = p['seq_len'],
    batch_size = p['batch_size']
) 

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
    
model.cuda()
model.eval()

def get_sample(data, penalty=None):
    samples = []
    for i,d in enumerate(data):
        _, mu, _ = model(d.unsqueeze(0).to(model.device))
        if penalty is None:
            samples.append(mu)
        else:
            samples.append((mu,penalty[i]))   
    return samples    

with torch.no_grad():
    train_pred = get_sample(dm.X_train)
    test_pred = get_sample(dm.X_test, dm.Y_penalty)

np.save(os.path.join("outputs","train.npy"), np.array(train_pred))
np.save(os.path.join("outputs","test.npy"), np.array(test_pred))
    
wandb.finish()                       
