import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Values")
    parser.add_argument('-sq','--seq_len', default=32, type=int, help="Sequence Length for input to LSTM")
    parser.add_argument('-bs','--batch_size', type=int, default=2, help="Batch Size")
    parser.add_argument('-me','--max_epochs', type=int, default=20, help="Number of epchs to train")
    parser.add_argument('-nf','--n_features', type=int, default=8, help="Length of feature for each sample")
    parser.add_argument('-ls','--latent_spc', type=int,default=8, help='Size of Latent Space')
    parser.add_argument('-kldc','--beta', type=float, default=0.001, help='weighting factor of KLD')
    parser.add_argument('-gam','--gamma', type=float, default=0.1, help='weighting factor of MSE')
    parser.add_argument('-fcd','--fc_dim', type=int, default=64, help="Number of FC Nodes")
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help="Neural Network Learning Rate")
    parser.add_argument('-mp', '--model_path', type=str, default='lstm_vae.pth', help="Saved model path")
    parser.add_argument('-istr','--is_train', type=bool, help="Train or Testing")
    args = parser.parse_args()

    return args