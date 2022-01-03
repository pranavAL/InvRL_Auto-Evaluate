import numpy as np
import matplotlib.pyplot as plt

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_xy(data):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    return x,y

def plot_embeddings(data):
    fig, ax = plt.subplots()
    for keys in data:
        plot_data = data[keys][0]
        color = data[keys][1]
        user = data[keys][2]
        x = list(map(get_numpy, plot_data[:250]))
        x, y = get_xy(x)
        ax.scatter(x, y, c=color, label=user)
    fig.suptitle('Latent Space of Expert vs Novice')
    ax.legend()
    fig.savefig("Latent Space of Expert and Novice.png")
    plt.show()    

train_embedd = np.load("train.npy", allow_pickle=True)
test_embedd = np.load("test.npy", allow_pickle=True)

plot_data = dict(train=[train_embedd,'red','expert'],
                 test=[test_embedd,'blue','novice'])
        
plot_embeddings(plot_data)        