import os
import numpy as np
import matplotlib.pyplot as plt

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