import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# ### Import Data
# * **Ex_ArcSwipe_Data.csv** contains value for each features for a given time step
# * **ArcSwipe List of Metrics - Sheet1.csv** information about each metric name

file = pd.read_csv(os.path.join("datasets","Ex_ArcSwipe_Data.csv"), header=None)
file2 = pd.read_csv(os.path.join("datasets","ArcSwipe List of Metrics - Sheet1.csv"), header=None)

col_name = ['Session id', 'Start Time', 'Equipment Name', 'Excercise Name', 'Pass/Fail', 'Score', 'Metric Name',
            'Time Step', 'Value', 'Unit']

file.set_axis(col_name, axis=1, inplace=True)

# Irrelevant features are dropped
file.drop(["Start Time", "Equipment Name", "Excercise Name"], axis=1, inplace=True)


metric_name = {key: value for (key, value) in zip(file2[0],file2[1])}

file.replace(to_replace=[keys for keys in metric_name.keys()], 
             value=[values for values in metric_name.values()],
             inplace=True)

new_columns = ["Session id", "Time Step", *metric_name.values()]

all_sessions = file["Session id"].unique()

def Duration_of_Single_Sessions(sess_id):
    return len(list(file[file["Metric Name"]=="Exercise Time"][file["Session id"]==sess_id]["Time Step"]))
    
def Duration_of_All_Sessions():
    count = 0
    for ids in all_sessions:
         count += Duration_of_Single_Sessions(ids)
    return count

def value_at_trigger(sess_id, parameter):
    time_step = file[file["Metric Name"]==parameter][file["Session id"]==sess_id]["Time Step"].to_list()
    value = list(file[file["Metric Name"]==parameter][file["Session id"]==sess_id]["Value"])
    return time_step, value

def check_len(sess_id, parameter):
    total_time = file[file["Metric Name"]==parameter][file["Session id"]==sess_id]["Time Step"].to_list()
    return len(total_time)

total_rows = Duration_of_All_Sessions()
new_df = pd.DataFrame(columns = new_columns, index = range(total_rows))

old_indx = new_indx = 0

for ids in all_sessions:
    time_steps = Duration_of_Single_Sessions(ids)
    new_indx += time_steps
    new_df["Session id"][old_indx:new_indx] = ids
    x, _ = value_at_trigger(ids, "Average time out of path range")
    new_df["Time Step"][old_indx:new_indx] = range(time_steps)
    time_of_reward, value_of_reward = value_at_trigger(ids, "Current trainee score at that time")
    full_columns = [m for m in metric_name.values() if check_len(ids, m) == time_steps]
    one_value_columns = [m for m in metric_name.values() if check_len(ids, m) == 1]
    left_columns = [m for m in metric_name.values() if m not in (full_columns + one_value_columns)]

    for param in one_value_columns:
        time_step, values = value_at_trigger(ids, param)
        new_df[param][old_indx:new_indx] = values * time_steps 
        
    for param in full_columns:
        time_step, values = value_at_trigger(ids, param)
        new_df[param][old_indx:new_indx] = values  
    
    for param in left_columns:
        t_step, values = value_at_trigger(ids, param)
        init_indx = old_indx
        
        for i in range(len(t_step)-1):
            trigr_indx = list(map(lambda k: k > t_step[i+1], time_step)).index(True)
            new_init = old_indx + trigr_indx
            new_df[param][init_indx:new_init] = values[i]
            init_indx = new_init
        
        new_df[param][init_indx:old_indx+len(time_step)] = values[-1]
                     
    old_indx = new_indx


new_df.to_csv(os.path.join("datasets","ExtractedFeatures.csv"))
