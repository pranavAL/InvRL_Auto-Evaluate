import warnings
warnings.filterwarnings('ignore')

import os
import torch
import Vortex
import vxatp3
import numpy as np
import pandas as pd
from math import dist

from model_dynamics import DynamicsPredictor
from model_infractions import SafetyPredictor

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
        self.dyna_feats = ['Engine Average Power', 'Engine Torque Average',
                           'Fuel Consumption Rate Average']

        self.safe_feats = ['Number of tennis balls knocked over by operator',
                           'Number of equipment collisions',
                           'Number of poles that fell over', 'Number of poles touched',
                           'Collisions with environment']

        self.max_val = fd.loc[:,self.dyna_feats+self.safe_feats].max()
        self.min_val = fd.loc[:,self.dyna_feats+self.safe_feats].min()

        fd.loc[:,self.dyna_feats+self.safe_feats] = (fd.loc[:,self.dyna_feats+self.safe_feats]
                                                    - self.min_val)/(self.max_val - self.min_val)

        self.experts = '5efb9aacbcf5631c14097d5d'
        self.exp_values = fd.loc[fd["Session id"]==self.experts,
                                self.dyna_feats+self.safe_feats]

        self.args.steps_per_episode = len(self.exp_values) - self.args.seq_len_dynamics

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

        dynamics_model_path = os.path.join('..','save_model', 'lstm_vae_dynamic.pth')
        safe_model_path = os.path.join('..','save_model', 'vae_recon_safety.pth')

        self.dynamicsmodel = DynamicsPredictor(
            n_features = self.args.n_features_dynamics,
            fc_dim = self.args.fc_dim_dynamics,
            seq_len = self.args.seq_len_dynamics,
            batch_size = self.args.batch_size_dynamics,
            latent_spc = self.args.latent_spc_dynamics,
            learning_rate = self.args.learning_rate,
            epochs = self.args.max_epochs,
            beta = self.args.beta
        )

        self.safetymodel = SafetyPredictor(
            n_features = self.args.n_features_safety,
            fc_dim = self.args.fc_dim_safety,
            batch_size = self.args.batch_size_safety,
            latent_spc = self.args.latent_spc_safety,
            learning_rate = self.args.learning_rate,
            epochs = self.args.max_epochs,
            beta = self.args.beta
        )

        self.dynamicsmodel.load_state_dict(torch.load(dynamics_model_path))
        self.dynamicsmodel.cuda()
        self.dynamicsmodel.eval()

        self.safetymodel.load_state_dict(torch.load(safe_model_path))
        self.safetymodel.cuda()
        self.safetymodel.eval()

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
        self.dynfeat = []
        self.saffeat = []
        self.curr_safety = [0.0,0.0,0.0,0.0,0.0]
        self.current_steps = 0
        self.per_step_fuel = []
        self.per_step_power = []
        self.per_step_torque = []

        self.load_scene()
        self.get_goals()

        while len(self.dynfeat) < self.args.seq_len_dynamics:
            state, reward, _, _ = self._get_obs()

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
        obs, reward, dyna_penalty, safe_penalty = self._get_obs()

        # Done flag
        if self.current_steps + 1 > self.args.steps_per_episode or self.goal_distance < 1.0:
            print("Episode over")
            done = True
            self.store_logs()
        else:
            done = False

        self.current_steps += 1

        return obs, reward, dyna_penalty, safe_penalty, done, {}

    def _get_obs(self):
        reward = 0
        dyna_penalty = 0
        safe_penalty = 0

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
        self.new_safety = RewardVal[3:]
        if self.curr_safety != self.new_safety:
            infractions = self.new_safety
            self.curr_safety = self.new_safety
        else:
            infractions = [0.0,0.0,0.0,0.0,0.0] 

        self.dynfeat.append(RewardVal[:3])
        self.saffeat.append(infractions)

        if len(self.dynfeat) >= self.args.seq_len_dynamics:
            exp_dyn = torch.tensor(list(self.exp_values.iloc[self.current_steps:self.current_steps+self.args.seq_len_dynamics,:][self.dyna_feats].values)).float()
            pol_dyn = torch.tensor(self.dynfeat[self.current_steps:self.current_steps+self.args.seq_len_dynamics]).float()
            dyna_penalty = self.get_penalty(exp_dyn, pol_dyn, self.dynamicsmodel, type="dynamic")

        exp_saf = torch.tensor([0,0,0,0,0]).float()
        pol_saf = torch.tensor(list(self.saffeat[self.current_steps])).float()
        safe_penalty = self.get_penalty(exp_saf, pol_saf, self.safetymodel, type="safety")

        states = np.array([*self.SwingLinPos, *self.BoomLinPos, *self.BuckLinPos, *self.StickLinPos,
                           self.SwingAngVel, self.BoomAngvel, self.BuckAngvel, self.StickAngvel,
                           self.BoomLinvel, self.BuckLinvel, self.StickLinvel])

        states = (states - np.mean(states))/(np.std(states))
        self.goal_distance = dist(self.goal,self.BuckLinPos)
        reward =  1 - self.goal_distance/10.0
        dyna_penalty = (1 - dyna_penalty - 0.970)/(1-0.970)
        safe_penalty = (1 - safe_penalty - 0.998)/(1-0.998)
       
        return states, reward, dyna_penalty, safe_penalty

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
        self.goal1 = self.MetricsInterface.getOutputContainer()['Path6 Easy Transform'].value
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

    def get_penalty(self, expert, novice, model, type):
        expert = expert.unsqueeze(0).to(model.device)
        novice = novice.unsqueeze(0).to(model.device)
        _, mu1, _ = model.encoder(expert)
        _, mu2, logvar = model.encoder(novice)
        if type=="dynamic":
            penalty =  - torch.sum(1 + logvar -mu2.pow(2) - logvar.exp()) * 10000
        else:
            penalty = torch.dist(mu1.squeeze(), mu2.squeeze(), 2)   
        penalty = self.get_numpy(penalty)
        return penalty

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
