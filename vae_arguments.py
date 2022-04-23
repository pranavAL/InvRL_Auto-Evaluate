import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Values")

    parser.add_argument('-me','--max_epochs', type=int, default=100, help="Number of epchs to train")
    parser.add_argument('-kldc','--beta', type=float, default=0.001, help='weighting factor of KLD')
    parser.add_argument('-gam','--gamma', type=float, default=0.1, help='weighting factor of MSE')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0001, help="Neural Network Learning Rate")
    parser.add_argument('-istr','--is_train', type=bool, help="Train or Testing")

    parser.add_argument('-sqdy','--seq_len_dynamics', default=32, type=int, help="Sequence Length for input to LSTM")
    parser.add_argument('-bsdy','--batch_size_dynamics', type=int, default=8, help="Batch Size for dynamics")
    parser.add_argument('-nfdy','--n_features_dynamics', type=int, default=3, help="Length of feature for dynamics")
    parser.add_argument('-lsdy','--latent_spc_dynamics', type=int,default=8, help='Size of Latent Space for dynamics')
    parser.add_argument('-fcdy','--fc_dim_dynamics', type=int, default=64, help="Number of FC Nodes for dynamics")

    parser.add_argument('-bssy','--batch_size_safety', type=int, default=2, help="Batch Size for safety")
    parser.add_argument('-nfsy','--n_features_safety', type=int, default=5, help="Length of feature for safety")
    parser.add_argument('-lssy','--latent_spc_safety', type=int,default=2, help='Size of Latent Space for safety')
    parser.add_argument('-fcsy','--fc_dim_safety', type=int, default=8, help="Number of FC Nodes for safety")

    args = parser.parse_args()

    return args
