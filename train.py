import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


warnings.filterwarnings("ignore")

EPOCHS = 100
SEQ_LEN = 10
BATCH_SIZE = 1
learning_rate = 0.01
input_size = 28
hidden_size = 64
num_layers = 1

file_path = "features_to_train.csv"

class CraneData(Dataset):
    
    def __init__(self, path_to_data):
    
        self.data = pd.read_csv(path_to_data)
        self.seq_length = SEQ_LEN
        self.x = []
        self.y = []
        
        self.normalize_columns()
        self.create_dataset()
        
    def normalize_columns(self):
        
        for cols in self.data.columns[3:-1]:
            self.data[cols] = self.data[cols] / self.data[cols].abs().max()
            
    def create_dataset(self):
        
        for sess in self.data["Session id"].unique():
            filter = (self.data["Session id"]==sess)
            num_of_elements = len(self.data[filter])
            features = self.data[filter].drop(columns=["reward", "Quality of Action"])
            labels = self.data[filter]["reward"]
        
            for i in range(num_of_elements-self.seq_length-1):
                x_ = [features.iloc[i,3:].to_list() for i in range(i,i+self.seq_length)]
                y_ = labels[i:i+self.seq_length].sum()
                self.x.append(x_)
                self.y.append(y_)
                
    def __len__(self):
    
        return len(self.x)
    
    def __getitem__(self, index):
    
        inp_data = self.x[index]
        labels = self.y[index]
        return torch.tensor(inp_data), torch.tensor(labels)
    
data = CraneData(file_path)
train_size = int(len(data)*0.8)
test_size = len(data) - train_size

train_dataset, test_dataset = random_split(data, [train_size, test_size])
print(f"Train Size: {len(train_dataset)}")
print(f"Test Size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = SEQ_LEN
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1))
        
    def forward(self, x):
        
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        output, (_, _) = self.lstm(x, (h_0, c_0))
        output = output[:,-1,:]
        out = self.fc(output)
        return out
    
net = LSTMNet(input_size,hidden_size,num_layers)
criterion = nn.MSELoss()               
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
tr_loss = []
te_loss = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    test_loss = 0.0
    count = 0
    for indx, (inp, lab) in enumerate(train_loader):
        inp, lab = inp.float(), lab.float()
        testinp, testlab = next(iter(test_loader))
        optimizer.zero_grad()
        outputs = net(inp)
        loss = criterion(outputs, lab)
        
        toutputs = net(testinp.float())
        tloss = criterion(toutputs, testlab.float())
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        test_loss += tloss.item()
                
        count += 1

    print(f"Epoch:{epoch} - Train Loss:{train_loss/count},---------,Val Loss:{test_loss/count}")
    tr_loss.append(train_loss/count)
    te_loss.append(test_loss/count)

print("Training Complete") 
 
plt.xlabel("Epochs")
plt.ylabel("Loss") 
plt.plot(range(len(tr_loss)),tr_loss, color='red')        
plt.plot(range(len(te_loss)),te_loss, color='blue') 
plt.legend(["Train", "Test"])       
plt.savefig("output.png")
plt.show()


