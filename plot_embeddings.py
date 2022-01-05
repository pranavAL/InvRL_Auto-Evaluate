import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train_LSTM import LSTMPredictor

checkpoint_path = os.path.join('save_model','checkpoints')
model = LSTMPredictor.load_from_checkpoint()
model.cuda()
model.eval()
print(model.learning_rate)
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

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_xy(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    return x,y

def plot_embeddings(data):
    
    for keys in data:
        plot_data = data[keys][0]
        reward = data[keys][1][:250]
        x = list(map(get_numpy, plot_data[:250]))
        x, y = get_xy(x)
        plt.scatter(x, y, c=reward, cmap='bwr')
    plt.title('Latent Space of Expert vs Novice')
    plt.savefig(os.path.join("outputs","Latent Space of Expert and Novice.png"))
    plt.show()    

train_embedd = np.load(os.path.join("outputs","train.npy"), allow_pickle=True)
test_embedd = np.load(os.path.join("outputs","test.npy"), allow_pickle=True)
filtered = list(filter(lambda x: x[1] < 0, test_embedd))
penalty_values = list(map(lambda x: x[1], filtered))
filtered = list(map(lambda x: x[0], filtered))
reward_values = np.zeros_like(train_embedd)

plot_data = dict(train=[train_embedd, reward_values],
                 test=[filtered, penalty_values])
        
plot_embeddings(plot_data)        