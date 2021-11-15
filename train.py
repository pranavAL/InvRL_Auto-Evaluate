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

EPOCHS = 50

file_path = "features_to_train.csv"

class CraneData(Dataset):
    
    def __init__(self, path_to_data):
        self.data = pd.read_csv(path_to_data)
        for column in self.data.columns[1:]:
            self.data[column] = (self.data[column] - self.data[column].min()) / (self.data[column].max() - self.data[column].min())
            
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        inp_data = self.data.iloc[index, 3:].to_list()    
        labels = self.data["Current trainee score at that time"][index]
        return torch.tensor(inp_data), torch.tensor(labels)
    
data = CraneData(file_path)
train_size = int(len(data)*0.8)
test_size = len(data) - train_size

train_dataset, test_dataset = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(28,128),
                                nn.ReLU(),
                                nn.Linear(128,256),
                                nn.ReLU(),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,1))
        
    def forward(self, x):
        out = self.fc(x)
        return out
    
net = Net()
criterion = nn.MSELoss()               
optimizer = optim.Adam(net.parameters(), lr=0.005)
tr_loss = []
te_loss = []

for epoch in range(EPOCHS):
    train_loss = 0.0
    test_loss = 0.0
    count = 0
    for indx, (inp, lab) in enumerate(train_loader, 0):
        inp, lab = inp.float(), lab.float()
        testinp, testlab = next(iter(test_loader))
        optimizer.zero_grad()
        outputs = net(inp)
        loss = criterion(outputs, lab)
        
        toutputs = net(testinp.float())
        tloss = criterion(toutputs, testlab.float())
        
        loss.backward()
        train_loss += loss.detach().item()
        test_loss += tloss.detach().item()
        
        optimizer.step()
        
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
plt.show()
plt.savefig("output.png")


