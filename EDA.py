import os
import warnings
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = [9.50, 3.50]
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

warnings.filterwarnings("ignore")

file = os.path.join('datasets','ExtractedFeatures.csv')
file1 = os.path.join("datasets","Ex_ArcSwipe_Scoring_Rules.csv")
scoring_rule = os.path.join("datasets","Ex_ArcSwipe_Scoring_Rules.csv")
metric_name = os.path.join("datasets","ArcSwipe List of Metrics - Sheet1.csv")

score_data = pd.read_csv(scoring_rule, header=None)
metric_data = pd.read_csv(metric_name, header=None)

metric_name = {key: value for (key, value) in zip(metric_data[0],metric_data[1])}

score_data.replace(to_replace=[keys for keys in metric_name.keys()], 
             value=[values for values in metric_name.values()],
             inplace=True)


# ### Feature Space

feature_space = score_data[1].unique()
print(feature_space)

data = pd.read_csv(file)
data.drop(columns=['Unnamed: 0'], inplace=True)

data1 = pd.read_csv(file1, header=None)

# ### Session with the best performance

def duration_of_session(session : str) -> int:
    duration = data.loc[data["Session id"]==session, "Time Step"]
    return len(duration)

def total_score(session: str):
    score = data.loc[data['Session id']==session,"Current trainee score at that time"].min()
    return score    

sessions = data["Session id"].unique()

print("All Features")
print(data.columns[3:])

features_to_skip = ['Safety violation unsafe parking position', 'Safety violation Flipped Vehicle', 'Safety violation human contact',
                    'Safety violation load over human', 'Safety violation electrical lines', 'Safety violation dump truck contact', 
                    'Safety violation bucket over truck cab', 'Bucket Self Contact', 'Wind speed', 'Number of barrels touches', 
                    'Number of barrels knocked over','Average score per path', 'Current path score']

print(f"Total Original Features: {len(data.columns[3:])}")
print(f"Total Features Removed: {len(features_to_skip)}")

data.drop(columns=features_to_skip, inplace=True)


# #### Remove Correlated features
# 
# Using correlated features are redundant information as they add the same information

data_corr = data.loc[:,data.columns[3:]].corr()

def reward_normalizel():
    data["Current trainee score at that time"] = data["Current trainee score at that time"][:] - 100
    for sess in data["Session id"].unique():
        test_cond = (data["Session id"]==sess)
        indx = data.index[test_cond].tolist()
        succes = data["Current trainee score at that time"][indx[1:]].tolist()
        predec = data["Current trainee score at that time"][indx[:-1]].tolist()
        output = list(np.array(succes)-np.array(predec))
        data["Current trainee score at that time"][indx[1:]] = output

reward_normalizel()      

corr_columns = []
for i in range(len(data_corr)):
    for j in range(i+1, len(data_corr)):
        if abs(data_corr.iloc[i,j]) > 0.97 and data_corr.columns[j] not in corr_columns:
            corr_columns.append(data_corr.columns[j])

print(f"Total Correlated features: {len(corr_columns)}") 
print(f"Correlated features: {corr_columns}")  

print(f"Final Features for training: {data.columns[3:]}")


for feature in data.columns[3:]:
    print(f"Feature: {feature}, Max Value: {data[feature].max()}, Min Value: {data[feature].min()}")

def create_dataset(sessions, dataframe):
    for sess in sessions:
        sess_feat = data.loc[data["Session id"]==sess,:]
        for i in range(len(sess_feat)):
            dataframe.loc[len(dataframe)] = sess_feat.iloc[i,:]
    return dataframe        

test_sess = ['5f29aee75503690dc400e6ed', '5efb87a555036917a005aedb', '5f0f52175503691a38012a4f', '5f0f5715bcf56303ec02201e']
valid_sess = ['5f298dab5503691770019164', '5f16fba5bcf563077800defe', '5efcedc85503691934046c9f', '5f0f357e55036922bc0394c1', 
         '5efceb355503691934044c21', '5f29ae6bb1a0e0078400bca5', '5f0f19c6bcf5631cc40054d5', '5f0f4991bcf56303ec009f44']
train_sess = [] 
for sess in data['Session id'].unique():
    if sess not in (test_sess + valid_sess):
        train_sess.append(sess)

train_df = pd.DataFrame(columns=data.columns)
val_df = pd.DataFrame(columns=data.columns)
test_df = pd.DataFrame(columns=data.columns)

train_df = create_dataset(train_sess, train_df)
val_df = create_dataset(valid_sess, val_df)
test_df = create_dataset(test_sess, test_df)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct

data.to_csv(os.path.join("datasets","features_to_train.csv"), index=False)
train_df.to_csv(os.path.join("datasets","train_df.csv"), index=False)
val_df.to_csv(os.path.join("datasets","val_df.csv"), index=False)
test_df.to_csv(os.path.join("datasets","test_df.csv"), index=False)

