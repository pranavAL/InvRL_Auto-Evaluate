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
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

EPOCHS = 500
BATCH_SIZE = 1
learning_rate = 0.01
input_size = 6
hidden_size = [32,64,128,64,32]
outsize = 10

file_path = "features_to_train.csv"

class CraneData(Dataset):
    
    def __init__(self, path_to_data):
    
        self.data = pd.read_csv(path_to_data)
        
        self.label_encoder()
        self.normalize_columns()
    
    def label_encoder(self):
        lb = LabelEncoder()
        self.data["reward"] = lb.fit_transform(self.data["reward"])
             
    def normalize_columns(self):
        for cols in self.data.columns[3:-1]:
            self.data[cols] = self.data[cols] / self.data[cols].abs().max()
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
    
        inp_data = self.data.iloc[index][self.data.columns[3:-1]]
        labels = self.data.iloc[index,-1]
        return torch.tensor(inp_data), torch.tensor(labels)
    
data = CraneData(file_path)
train_size = int(len(data)*0.8)
test_size = len(data) - train_size

train_dataset, test_dataset = random_split(data, [train_size, test_size])
print(f"Train Size: {len(train_dataset)}")
print(f"Test Size: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = outsize
                
        self.fc1 = nn.Sequential(nn.Linear(self.input_size, self.hidden_size[0]),
                                nn.ReLU())
        self.droput = nn.Dropout(0.25)
        self.fc2 = nn.Sequential(nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                                nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(self.hidden_size[1], self.hidden_size[2]),
                                nn.ReLU())
        self.fc4 = nn.Sequential(nn.Linear(self.hidden_size[2], self.hidden_size[3]),
                                nn.ReLU())
        self.fc5 = nn.Sequential(nn.Linear(self.hidden_size[3], self.hidden_size[4]),
                                nn.ReLU())
        self.fc6 = nn.Linear(self.hidden_size[4], self.out_size)
        
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.droput(out)
        out = self.fc5(out)
        out = self.droput(out)
        out = self.fc6(out)
        return out
    
net = Net(input_size,hidden_size)
criterion = nn.CrossEntropyLoss()               
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
tr_loss = []
te_loss = []
tr_acc = []
te_acc = []


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags==y_test).float()
    acc = correct_pred.sum() /len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

for epoch in range(EPOCHS):
    train_loss = 0.0
    test_loss = 0.0
    count = 0
    train_acc = 0.0
    test_acc = 0.0
    
    for indx, (inp, lab) in enumerate(train_loader):
        net.train()
        inp, lab = inp.float(), lab
        optimizer.zero_grad()
        outputs = net(inp)
        loss = criterion(outputs, lab)
        train_acc += multi_acc(outputs, lab)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        with torch.no_grad():
            net.eval()
            testinp, testlab = next(iter(test_loader))
            toutputs = net(testinp.float())
            tloss = criterion(toutputs, testlab)
            tacc = multi_acc(toutputs, testlab)
        
            test_loss += tloss.item()
            test_acc += tacc
                
        count += 1

    print(f"Epoch:{epoch} - Train Loss:{train_loss/count},---------,Val Loss:{test_loss/count}")
    print(f"Epoch:{epoch} - Train Acc:{train_acc/count},---------,Val Loss:{test_acc/count}")
    tr_loss.append(train_loss/count)
    te_loss.append(test_loss/count)
    tr_acc.append(train_acc/count)
    te_acc.append(test_acc/count)
    

print("Training Complete") 

plt.subplot(121) 
plt.xlabel("Epochs")
plt.ylabel("Loss") 
plt.plot(range(len(tr_loss)),tr_loss, color='red')        
plt.plot(range(len(te_loss)),te_loss, color='blue')
plt.legend(["TrainLoss", "TestLoss"])       

plt.subplot(122)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(range(len(tr_acc)),tr_acc, color='green')        
plt.plot(range(len(te_acc)),te_acc, color='yellow')  
plt.legend(["Train Acc","TestAcc"])

plt.savefig("output.png")
plt.show()


