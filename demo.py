import os
import torch
import numpy as np
import pandas as pd
from vae_arguments import get_args
from model_dynamics import DynamicsPredictor
from model_infractions import SafetyPredictor

def get_numpy(x):
    return x.squeeze().to('cpu').detach().numpy()

def get_penalty(expert, novice, model, type):
    expert = expert.unsqueeze(0).to(model.device)
    novice = novice.unsqueeze(0).to(model.device)
    _, mu1, _ = model.encoder(expert)
    _, mu2, logvar = model.encoder(novice)
    if type=="dynamic":
        penalty = -torch.sum(1 + logvar - mu2.pow(2)-logvar.exp()) * 10000
    else:
        penalty = torch.dist(mu1.squeeze(), mu2.squeeze(), 2)
    penalty = get_numpy(penalty)
    return penalty            

args = get_args()

full_data_path = os.path.join("datasets","features_to_train.csv")
fd = pd.read_csv(full_data_path)

dyna_feats = ["Engine Average Power", "Engine Torque Average",
              "Fuel Consumption Rate Average"]

safe_feats = ["Number of tennis balls knocked over by operator",
              "Number of equipment collisions",
              "Number of poles that fell over", "Number of poles touched",
              "Collisions with environment"]

fd['Dynamic Reward'] = 0
fd['Safety Reward'] = 0

max_val = fd.loc[:,dyna_feats+safe_feats].max()
min_val = fd.loc[:,dyna_feats+safe_feats].min()

fd.loc[:,dyna_feats+safe_feats] = (fd.loc[:,dyna_feats+safe_feats]
                                   -min_val)/(max_val - min_val)

dynamic_model_path = os.path.join("save_model", "lstm_vae_dynamic.pth")
safe_model_path = os.path.join("save_model", "vae_recon_safety.pth")

dynamicsmodel = DynamicsPredictor(
    n_features = args.n_features_dynamics,
    fc_dim = args.fc_dim_dynamics,
    seq_len = args.seq_len_dynamics,
    batch_size = args.batch_size_dynamics,
    latent_spc = args.latent_spc_dynamics,
    learning_rate = args.learning_rate,
    epochs = args.max_epochs,
    beta = args.beta
)

safetymodel = SafetyPredictor(
    n_features = args.n_features_safety,
    fc_dim = args.fc_dim_safety,
    batch_size = args.batch_size_safety,
    latent_spc = args.latent_spc_safety,
    learning_rate = args.learning_rate,
    epochs = args.max_epochs,
    beta = args.beta
)

dynamicsmodel.load_state_dict(torch.load(dynamic_model_path))
dynamicsmodel.cuda()
dynamicsmodel.eval()

safetymodel.load_state_dict(torch.load(safe_model_path))
safetymodel.cuda()
safetymodel.eval()

safety_step = 0
dynfeat = []
saffeat = []

for indx, sess in enumerate(fd["Session id"].unique()):
    sess_feat = fd.loc[fd["Session id"]==sess,:]
    curr_safety = [0.0,0.0,0.0,0.0,0.0]
    
    dynamic_step = safety_step

    for i in range(0,len(sess_feat)):
        EngAvgPow = sess_feat.iloc[i,:]['Engine Average Power']
        EngTorAvg = sess_feat.iloc[i,:]['Engine Torque Average']
        fuelCons = sess_feat.iloc[i,:]['Fuel Consumption Rate Average']
        ball_knock = sess_feat.iloc[i,:]['Number of tennis balls knocked over by operator']
        equip_coll = sess_feat.iloc[i,:]['Number of equipment collisions']
        pole_fell = sess_feat.iloc[i,:]['Number of poles that fell over']
        pole_touch = sess_feat.iloc[i,:]['Number of poles touched']
        coll_env = sess_feat.iloc[i,:]['Collisions with environment']
        
        RewardVal = [EngAvgPow, EngTorAvg, fuelCons, ball_knock,
                    equip_coll, pole_fell, pole_touch, coll_env]

        RewardVal = list(np.divide(np.subtract(np.array(RewardVal), np.array(min_val)),
                                np.subtract(np.array(max_val), np.array(min_val))))

        new_safety = RewardVal[3:]
        if curr_safety != new_safety:
            infractions = new_safety
            curr_safety = new_safety
        else:
            infractions = [0.0,0.0,0.0,0.0,0.0]
            
        dynfeat.append(RewardVal[:3])
        saffeat.append(infractions)        

        if len(dynfeat) >= args.seq_len_dynamics:
            pol_dyn = torch.tensor(dynfeat[dynamic_step:dynamic_step+args.seq_len_dynamics]).float()
            exp_dyn = pol_dyn
            dyna_penalty = get_penalty(exp_dyn, pol_dyn, dynamicsmodel, type="dynamic")
            dyna_penalty = (dyna_penalty - 1.6844) / (1.90 - 1.6844)
            fd.at[dynamic_step,"Dynamic Reward"] = 1 - dyna_penalty
            dynamic_step += 1
            
        exp_saf = torch.tensor([0,0,0,0,0]).float()
        pol_saf = torch.tensor(list(saffeat[safety_step])).float()
        safe_penalty = get_penalty(exp_saf, pol_saf, safetymodel, type="safety")
        safe_penalty = (safe_penalty) / (0.1382)
        fd.at[safety_step, "Safety Reward"] = 1 - safe_penalty
        safety_step += 1
        
fd.to_csv('demo_result.csv')        
