import torch
import wandb
import numpy as np
import torchmetrics
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer, seed_everything

wandb.login()

class CraneDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index]).float(), torch.tensor(self.X[index]).float()
    
class CraneDatasetModule(pl.LightningDataModule):
    def __init__(self, seq_len, batch_size, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        
    def prepare_data(self):
        pass
    
    def setup(self, stage=None):
             
        data_path = "features_to_train.csv"
        
        df = pd.read_csv(data_path)
        
        train_feats = ['Bucket Angle','Bucket Height','Engine Average Power','Current Engine Power','Engine Torque', 'Engine Torque Average',
                'Engine RPM (%)', 'Tracks Ground Pressure Front Left', 'Tracks Ground Pressure Front Right']

        df.loc[:,train_feats] = (df.loc[:,train_feats] - df.loc[:,train_feats].min())/(df.loc[:,train_feats].max() - df.loc[:,train_feats].min())
        
        best_sess = [sess for sess in df["Session id"].unique() if df.loc[df["Session id"]==sess,"Current trainee score at that time"].min()==100.0]
        X_train = []
        X_test = []
        for sess in df["Session id"].unique():
            sess_feat = df.loc[df["Session id"]==sess,:]
            for i in range(0,len(sess_feat) - self.seq_len):
                if sess in best_sess:
                    X_train.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)
                else:
                    X_test.append(sess_feat.iloc[i:i+self.seq_len,:][train_feats].values)    
                  
        X_val, X_test = train_test_split(X_test, test_size=0.2, shuffle=False)
        
        if stage == 'fit' or stage is None:
            self.X_train = X_train
            self.X_val = X_val
        
        if stage == 'test' or stage is None:
            self.X_test = X_test  
            
    def train_dataloader(self):
        train_dataset = CraneDataset(self.X_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        return train_loader
    
    def val_dataloader(self):
        val_dataset = CraneDataset(self.X_val)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return val_loader
    
    def test_dataloader(self):
        test_dataset = CraneDataset(self.X_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
        return test_loader
    
class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim, seq_len, latent_spc, num_layers, dropout):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.latent_spc = latent_spc
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim 
        
        self.lstm = nn.LSTM(input_size=self.n_features,
                             hidden_size=self.hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        
        self.fc = nn.Linear(self.hidden_dim, self.latent_spc)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).cuda())
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim).cuda())
        
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        return self.fc(h_out)
            
class Decoder(nn.Module):
    def __init__(self, input_dim, seq_len, num_layers, dropout, n_features):
        super(Decoder, self).__init__()
        self.n_features = n_features
        self.seq_len, self.input_dim = seq_len, input_dim
        self.embedding_dim, self.hidden_dim = 2 * input_dim, input_dim
        
        self.lstm = nn.LSTM(input_size=input_dim, 
                             hidden_size=input_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):
        x = x.repeat(self.seq_len, 1, 1)
        x = x.reshape((-1, self.seq_len, self.input_dim))
        
        out, _ = self.lstm(x)
        out = out.reshape((-1, self.seq_len, self.hidden_dim))
        
        return self.output_layer(out)
             
class LSTMPredictor(pl.LightningModule):
    def __init__(self, n_features, embedding_dim, seq_len, latent_spc, batch_size, num_layers, dropout, learning_rate, criterion):
        super(LSTMPredictor,self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.latent_spc = latent_spc
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        
        self.encoder = Encoder(n_features, embedding_dim, seq_len, latent_spc, num_layers, dropout)
        self.decoder = Decoder(latent_spc, seq_len, self.num_layers, self.dropout, n_features)
        
        self.save_hyperparameters()
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log('train/loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log("val/loss_epoch", loss, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        return loss
    
p = dict(
    seq_len = 24,
    batch_size = 70,
    latent_spc = 128, 
    criterion = nn.L1Loss(reduction='sum'),
    max_epochs = 20,
    n_features = 9,
    embedding_dim = 100,
    num_layers = 1,
    dropout = 0,
    learning_rate = 0.001
)       

seed_everything(1)

wandb_logger = WandbLogger(project="lit-wandb")
trainer = Trainer(max_epochs=p['max_epochs'],
                  gpus = 1,
                  logger=wandb_logger,
                  log_every_n_steps=50)    

model = LSTMPredictor(
    n_features = p['n_features'],
    embedding_dim = p['embedding_dim'],
    seq_len = p['seq_len'],
    latent_spc = p['latent_spc'],
    batch_size = p['batch_size'],
    criterion = p['criterion'],
    num_layers = p['num_layers'],
    dropout = p['dropout'],
    learning_rate = p['learning_rate']
)

dm = CraneDatasetModule(
    seq_len = p['seq_len'],
    batch_size = p['batch_size']
)        

trainer.fit(model, dm)
trainer.test(datamodule=dm)        
                                  
wandb.finish()                       


                                 

