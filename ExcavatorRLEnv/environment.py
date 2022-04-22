import warnings
warnings.filterwarnings('ignore')

import Vortex
import vxatp3
import numpy as np
import torch
import os
import pandas as pd
from train_LSTM import LSTMPredictor
from math import dist

SUB_STEPS = 5
class env():

    def __init__(self, args):
        # VxMechanism variable for the mechanism to be loaded.
        self.vxscene = None
        self.scene = None
        self.interface = None
        self.args = args
        
        full_data_path = os.path.join("..","datasets","features_to_train.csv")
        fd = pd.read_csv(full_data_path)
        self.train_feats = ['Engine Average Power', 'Engine Torque Average', 'Fuel Consumption Rate Average', 
                            'Number of tennis balls knocked over by operator', 'Number of equipment collisions', 
                            'Number of poles that fell over', 'Number of poles touched', 
                            'Collisions with environment']
        
        self.max_val = fd.loc[:,self.train_feats].max()
        self.min_val = fd.loc[:,self.train_feats].min()
        fd.loc[:,self.train_feats] = (fd.loc[:,self.train_feats] - self.min_val)/(self.max_val - self.min_val)               
        self.experts = ['5efb9aacbcf5631c14097d5d', '5efcee755503691934047938']
        self.exp_values = fd.loc[fd["Session id"]==self.experts[self.args.expert], self.train_feats]
        
        self.args.steps_per_episode = len(self.exp_values) - self.args.seq_len

        # Define the setup and scene file paths
        self.setup_file = 'Setup.vxc'
        self.content_file = f'C:\CM Labs\Vortex Construction Assets 21.1\\assets\Excavator\Scenes\ArcSwipe\EX_Arc_Swipe{args.complexity}.vxscene'

        # Create the Vortex Application
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)

        # Create a display window
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(500, 500, 1280, 720))

        model_path = os.path.join('..','save_model', self.args.model_path)

        self.model = LSTMPredictor(
            n_features = self.args.n_features,
            fc_dim = self.args.fc_dim,
            seq_len = self.args.seq_len,
            batch_size = self.args.batch_size,
            latent_spc = self.args.latent_spc,
            learning_rate = self.args.learning_rate,
            epochs = self.args.max_epochs,
            beta = self.args.beta
        )

        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()

        self.env_col = []
        self.knock_ball = []
        self.touch_pole = []
        self.fell_pole = []
        self.coll_equip = []
        self.tor_avg = []
        self.pow_avg = []
        self.avg_fuel = []

    def reset(self):
        # Initialize Reward and Step Count
        self.reward = 0
        self.current_steps = 0
        self.rewfeatures = []
        self.per_step_fuel = []
        self.per_step_power = []
        self.per_step_torque = []

        self.load_scene()
        self.get_goals()
        
        while len(self.rewfeatures) < self.args.seq_len:
            state, reward, _ = self._get_obs()

        return state, reward

    def step(self, action):  # takes a numpy array as input

        # Apply actions
        self.ControlInterface.getInputContainer()['Swing [3] | Control Input'].value = action[0].item()
        self.ControlInterface.getInputContainer()['Bucket [2] | Control Input'].value = action[1].item()
        self.ControlInterface.getInputContainer()['Stick [1] | Control Input'].value = action[2].item()
        self.ControlInterface.getInputContainer()['Boom [0] | Control Input'].value = action[3].item()

        # Step the simulation
        for _ in range(SUB_STEPS):
            self.application.update()

        # Observations
        obs, reward, penalty = self._get_obs()
        
        # Done flag
        if self.current_steps + 1 > self.args.steps_per_episode:
            print("Episode over")
            done = True
            self.store_logs()
        else:
            done = False

        self.current_steps += 1

        return obs, reward, penalty, done, {}

    def _get_obs(self):
        reward = 0
        penalty = 0

        self.SwingLinPos = self.ControlInterface.getOutputContainer()['State | Actuator Swing LinPosition'].value
        self.BoomLinPos = self.ControlInterface.getOutputContainer()['State | Actuator Boom LinPosition'].value
        self.BuckLinPos = self.ControlInterface.getOutputContainer()['State | Actuator Bucket LinPosition'].value
        self.StickLinPos = self.ControlInterface.getOutputContainer()['State | Actuator Arm LinPosition'].value

        self.SwingAngVel = self.ControlInterface.getOutputContainer()['State | Actuator Swing AngVelocity'].value
        self.BoomAngvel = self.ControlInterface.getOutputContainer()['State | Actuator  Boom AngVelocity'].value
        self.BuckAngvel = self.ControlInterface.getOutputContainer()['State | Actuator Bucket AngVelocity'].value
        self.StickAngvel = self.ControlInterface.getOutputContainer()['State | Actuator Arm AngVelocity'].value

        self.BoomLinvel = self.ControlInterface.getOutputContainer()['State | Actuator Boom LinVelocity'].value
        self.BuckLinvel = self.ControlInterface.getOutputContainer()['State | Actuator Bucket LinVelocity'].value
        self.StickLinvel = self.ControlInterface.getOutputContainer()['State | Actuator Arm LinVelocity'].value

        self.SwingAngPos = self.ControlInterface.getOutputContainer()['State | Actuator Swing AngPosition'].value
        self.BoomAngPos = self.ControlInterface.getOutputContainer()['State | Actuator  Boom AngPosition'].value
        self.BuckAngPos = self.ControlInterface.getOutputContainer()['State | Actuator Bucket AngPosition'].value
        self.StickAngPos = self.ControlInterface.getOutputContainer()['State | Actuator Arm AngPosition'].value

        self.goal = self.goals[self.args.complexity]
        self.get_heuristics()
        
        RewardVal = [self.EngAvgPow, self.EngTorAvg, self.fuelCons, self.ball_knock,
                     self.equip_coll, self.pole_fell, self.pole_touch, self.coll_env]

        RewardVal = self.normalize(RewardVal)

        self.rewfeatures.append(RewardVal)

        if len(self.rewfeatures) >= self.args.seq_len:
            exp_input = torch.tensor(list(self.exp_values.iloc[self.current_steps:self.current_steps+self.args.seq_len,:][self.train_feats].values)).float()
            pol_input = torch.tensor(self.rewfeatures[self.current_steps:self.current_steps+self.args.seq_len]).float()
            penalty = self.get_penalty(exp_input, pol_input)                           

        states = np.array([*self.SwingLinPos, *self.BoomLinPos, *self.BuckLinPos, *self.StickLinPos,
                           self.SwingAngVel, self.BoomAngvel, self.BuckAngvel, self.StickAngvel,
                           self.BoomLinvel, self.BuckLinvel, self.StickLinvel])

        states = (states - np.mean(states))/(np.std(states))
        self.goal_distance = dist(self.goal,self.BuckLinPos)
        reward =  1 - self.goal_distance/10.0

        return states, reward, penalty

    def get_heuristics(self):
        self.ball_knock = self.MetricsInterface.getOutputContainer()['Number of tennis balls knocked over by operator'].value
        self.pole_touch = self.MetricsInterface.getOutputContainer()['Number of poles touched'].value
        self.pole_fell = self.MetricsInterface.getOutputContainer()['Number of poles that fell over'].value
        self.equip_coll = self.MetricsInterface.getOutputContainer()['Number of equipment collisions'].value
        self.coll_env = self.MetricsInterface.getOutputContainer()['Collisions with environment'].value
        self.EngAvgPow = self.MetricsInterface.getOutputContainer()['Engine Average Power'].value
        self.EngTorAvg = self.MetricsInterface.getOutputContainer()['Engine Torque Average'].value
        self.fuelCons = self.MetricsInterface.getOutputContainer()['Fuel Consumption Rate Average'].value

        self.per_step_torque.append(self.EngTorAvg)
        self.per_step_power.append(self.EngAvgPow)
        self.per_step_fuel.append(self.fuelCons)

    def get_goals(self):
        self.goal1 = self.MetricsInterface.getOutputContainer()['Path3 Easy Transform'].value
        self.goal2 = self.MetricsInterface.getOutputContainer()['Path8 Easy Transform'].value
        self.goal3 = self.MetricsInterface.getOutputContainer()['Path13 Hard Transform'].value

        self.goals = [self.goal1, self.goal2, self.goal3]

    def store_logs(self):
        self.knock_ball.append(self.ball_knock)
        self.touch_pole.append(self.pole_touch)
        self.fell_pole.append(self.pole_fell)
        self.coll_equip.append(self.equip_coll)
        self.env_col.append(self.coll_env)
        self.tor_avg.append(np.mean(self.per_step_torque))
        self.pow_avg.append(np.mean(self.per_step_power))
        self.avg_fuel.append(np.mean(self.per_step_fuel))    

    def normalize(self, features):
        features = list(np.divide(np.subtract(np.array(features), np.array(self.min_val)),
                                  np.subtract(np.array(self.max_val), np.array(self.min_val))))
        return features                            
    
    def get_numpy(self, x):
        return x.squeeze().to('cpu').detach().numpy()

    def get_penalty(self, expert, novice):
        expert = expert.unsqueeze(0).to(self.model.device)
        novice = novice.unsqueeze(0).to(self.model.device)
        _, mu1, _ = self.model.encoder(expert)
        _, mu2, _ = self.model.encoder(novice)
        penalty = torch.dist(mu1.squeeze(), mu2.squeeze(), 2)
        penalty = self.get_numpy(penalty) * 1000
        return 10.0 - penalty

    def load_scene(self):
        # The first time we load the scene
        if self.vxscene is None:
            # Switch to Editing Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

            # Load scene file and get the mechanism interface
            self.vxscene = self.application.getSimulationFileManager().loadObject(self.content_file)
            self.scene = Vortex.SceneInterface(self.vxscene)

            # Get the RL Interface VHL
            self.ControlInterface = self.scene.findExtensionByName('ICD Controls - VHL')
            self.MetricsInterface = self.scene.findExtensionByName('States and Metrics')

            # Switch to Simulation Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

            # Initialize first key frame
            self.application.update()
            self.keyFrameList = self.application.getContext().getKeyFrameManager().createKeyFrameList("KeyFrameList",
                                                                                                      False)
            self.application.update()

            self.keyFrameList.saveKeyFrame()
            self.waitForNbKeyFrames(1, self.application, self.keyFrameList)
            self.key_frames_array = self.keyFrameList.getKeyFrames()

            # Other times we reset the environment
        else:
            # Switch to Simulation Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

            # Load first key frame
            self.keyFrameList.restore(self.key_frames_array[0])
            self.application.update()

        self.ControlInterface.getInputContainer()['Control | Engine Start Switch'].value = True
    
    def waitForNbKeyFrames(self, expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter
    
    def render(self, active=True):

        # Find current list of displays
        current_displays = self.application.findExtensionsByName('3D Display')

        # If active, add a display and activate Vsync
        if active and len(current_displays) == 0:
            self.application.add(self.display)
            self.application.setSyncMode(Vortex.kSyncSoftwareAndVSync)

        # If not, remove the current display and deactivate Vsync
        elif not active:
            if len(current_displays) == 1:
                self.application.remove(current_displays[0])
            self.application.setSyncMode(Vortex.kSyncNone)

    def __del__(self):
        # It is always a good idea to destroy the VxApplication when we are done with it.
        self.application = None        