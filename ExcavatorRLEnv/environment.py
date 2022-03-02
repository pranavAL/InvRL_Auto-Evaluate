import warnings
warnings.filterwarnings('ignore')

import Vortex
import vxatp3
import random
import numpy as np
import math
import torch
import os
from train_LSTM import LSTMPredictor
from numpy.linalg import norm
from math import dist

SUB_STEPS = 5
MAX_STEPS = 200

class env():

    def __init__(self, args):
        # VxMechanism variable for the mechanism to be loaded.
        self.vxscene = None
        self.scene = None
        self.interface = None
        self.args = args

        # Define the setup and scene file paths
        self.setup_file = 'Setup.vxc'
        self.content_file = r'C:\CM Labs\Vortex Construction Assets 21.1\assets\Excavator\Scenes\ArcSwipe\EX_Arc_Swipe.vxscene'

        # Create the Vortex Application
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)

        # Create a display window
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(500, 500, 1280, 720))

        #Standardisation
        self.min = [-2.390703, -0.786492, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.max = [2.202948, 5.192437, 38.898521, 146173.979222, 889.100139,
                    73.347609, 1866.730765]

        model_name = f"{self.args.model_path.split('.')[0]}"
        model_path = os.path.join('..','save_model', self.args.model_path)

        self.model = LSTMPredictor(
            n_features = self.args.n_features,
            fc_dim = self.args.fc_dim,
            seq_len = self.args.seq_len,
            batch_size = self.args.batch_size,
            latent_spc = self.args.latent_spc,
            learning_rate = self.args.learning_rate
        )

        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()

    def __del__(self):
        # It is always a good idea to destroy the VxApplication when we are done with it.
        self.application = None

    def get_numpy(self, x):
        return x.squeeze().to('cpu').detach().numpy()

    def get_penalty(self, data, label):
        data = data.unsqueeze(0).to(self.model.device)
        label = label.view(1,1,-1).to(self.model.device)
        _, mu, logvar = self.model(data, label, is_train=False)
        penalty = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.get_numpy(penalty)

    def reset(self):
        # Initialize Reward and Step Count
        self.current_step = 0
        self.reward = 0
        self.rewfeatures = []
        self.last_step = np.zeros(8)

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
        self.goals = self.get_goals()

        while len(self.rewfeatures) < self.args.seq_len:
            state, reward = self._get_obs()

        return state, reward

    def waitForNbKeyFrames(self, expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

    def step(self, action):  # takes a numpy array as input

        # Apply actions
        self.ControlInterface.getInputContainer()['Swing [3] | Control Input'].value = action[0].item()
        self.ControlInterface.getInputContainer()['Bucket [2] | Control Input'].value = action[1].item()
        self.ControlInterface.getInputContainer()['Stick [1] | Control Input'].value = action[2].item()
        self.ControlInterface.getInputContainer()['Boom [0] | Control Input'].value = action[3].item()

        # Step the simulation
        for i in range(SUB_STEPS):
            self.application.update()

        # Observations
        obs, reward = self._get_obs()

        # Done flag
        if self.current_step >= MAX_STEPS:
            done = True
        else:
            done = False

        self.current_step += 1

        return obs, reward, done, {}


    def _get_obs(self):
        reward = 0
        swingpos = self.ControlInterface.getOutputContainer()['State | Actuator Swing Position'].value
        BoomLinPos = self.ControlInterface.getOutputContainer()['Actuator Boom Position'].value
        BuckLinPos = self.ControlInterface.getOutputContainer()['Actuator Bucket Position'].value
        StickLinPos = self.ControlInterface.getOutputContainer()['Actuator Arm Position'].value
        states = np.array([swingpos, *BoomLinPos, *BuckLinPos, *StickLinPos])
        states = (states - np.mean(states))/ (np.std(states))

        BuckAng = self.MetricsInterface.getOutputContainer()['Bucket Angle'].value
        BuckHeight = self.MetricsInterface.getOutputContainer()['Bucket Height'].value
        EngAvgPow = self.MetricsInterface.getOutputContainer()['Engine Average Power'].value
        CurrEngPow = self.MetricsInterface.getOutputContainer()['Current Engine Power'].value
        EngTor = self.MetricsInterface.getOutputContainer()['Engine Torque'].value
        EngTorAvg = self.MetricsInterface.getOutputContainer()['Engine Torque Average'].value
        Engrpm = self.MetricsInterface.getOutputContainer()['Engine RPM (%)'].value

        RewardVal = [BuckAng, BuckHeight, EngAvgPow, CurrEngPow, EngTor, EngTorAvg,
                    Engrpm]

        RewardVal = list(np.divide(np.subtract(np.array(RewardVal), np.array(self.min)),
                                   np.subtract(np.array(self.max), np.array(self.min))))

        self.rewfeatures.append(RewardVal)

        if len(self.rewfeatures) >= self.args.seq_len:
            trainfeatures = np.array(self.rewfeatures)
            penalty = self.get_penalty(torch.tensor(trainfeatures[-self.args.seq_len:,:]).float(), torch.tensor(trainfeatures[-1,:]).float())
            reward = -penalty - dist(self.goal2,BoomLinPos) - dist(self.goal2,BuckLinPos) - dist(self.goal2,StickLinPos)

        self.get_heuristics()

        return states, reward

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

    def get_heuristics(self):
        self.arc_restart = self.MetricsInterface.getOutputContainer()['Number of times user had to restart an arc'].value
        self.cur_score = self.MetricsInterface.getOutputContainer()['Current path score'].value
        self.tot_out_path = self.MetricsInterface.getOutputContainer()['Total time out of path'].value
        self.cur_pat_tim = self.MetricsInterface.getOutputContainer()['Current path time'].value
        self.avg_tim_pat = self.MetricsInterface.getOutputContainer()['Average time per path'].value
        self.avg_sco_pat = self.MetricsInterface.getOutputContainer()['Average score per path'].value
        self.cur_pat_out_time = self.MetricsInterface.getOutputContainer()['Current path time out of range'].value
        self.avg_pat_out_time = self.MetricsInterface.getOutputContainer()['Average time out of path range'].value
        self.ball_knock = self.MetricsInterface.getOutputContainer()['Number of tennis balls knocked over by operator'].value
        self.pole_touch = self.MetricsInterface.getOutputContainer()['Number of poles touched'].value
        self.pole_fell = self.MetricsInterface.getOutputContainer()['Number of poles that fell over'].value
        self.barr_touch = self.MetricsInterface.getOutputContainer()['Number of barrels touches'].value
        self.barr_knock = self.MetricsInterface.getOutputContainer()['Number of barrels knocked over'].value
        self.equip_coll = self.MetricsInterface.getOutputContainer()['Number of equipment collisions'].value
        self.num_idle = self.MetricsInterface.getOutputContainer()['Number of times machine was left idling'].value
        self.buck_self = self.MetricsInterface.getOutputContainer()['Bucket Self Contact'].value
        self.rat_idle = self.MetricsInterface.getOutputContainer()['Ratio of time that operator runs equipment vs idle time'].value
        self.coll_env = self.MetricsInterface.getOutputContainer()['Collisions with environment'].value
        self.num_goal = self.MetricsInterface.getOutputContainer()['Exercise Number of goals met'].value
        self.ex_time = self.MetricsInterface.getOutputContainer()['Exercise Time'].value

    def get_goals(self):
        self.goal2 = self.MetricsInterface.getOutputContainer()['Path2 Easy Transform'].value
        self.goal3 = self.MetricsInterface.getOutputContainer()['Path3 Easy Transform'].value
        self.goal4 = self.MetricsInterface.getOutputContainer()['Path4 Easy Transform'].value
        self.goal5 = self.MetricsInterface.getOutputContainer()['Path5 Easy Transform'].value
        self.goal6 = self.MetricsInterface.getOutputContainer()['Path6 Easy Transform'].value
        self.goal7 = self.MetricsInterface.getOutputContainer()['Path7 Easy Transform'].value
        self.goal8 = self.MetricsInterface.getOutputContainer()['Path8 Easy Transform'].value
