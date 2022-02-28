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
        self.content_file = r'C:\CM Labs\Vortex Construction Assets 21.1\assets\Excavator\Scenes\BasicControls\EX_Basic_Controls.vxscene'

        # Create the Vortex Application
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)

        # Create a display window
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(500, 500, 1280, 720))

        #Standardisation
        self.min = [-2.390703, -0.786492, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.max = [2.202948, 5.192437, 38.898521, 146173.979222, 889.100139,
                    73.347609, 1866.730765, 226386.504786, 120065.185196]

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

    def get_distribution(self, data, label):
        data = data.unsqueeze(0).to(self.model.device)
        label = label.view(1,1,-1).to(self.model.device)
        _, mu, logvar = model(data, label, is_train=False)
        return self.get_numpy(mu), self.get_numpy(logvar)

    def reset(self):
        # Initialize Reward and Step Count
        self.current_step = 0
        self.reward = 0
        self.rewfeatures = []

        # The first time we load the scene
        if self.vxscene is None:
            # Switch to Editing Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

            # Load scene file and get the mechanism interface
            self.vxscene = self.application.getSimulationFileManager().loadObject(self.content_file)
            self.scene = Vortex.SceneInterface(self.vxscene)

            # Get the RL Interface VHL
            self.interface = self.scene.findExtensionByName('ICD Controls - VHL')

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

        self.interface.getInputContainer()['Control | Engine Start Switch'].value = True

        while len(self.rewfeatures) < 32:
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
        self.interface.getInputContainer()['Swing [3] | Control Input'].value = action[0].item()
        self.interface.getInputContainer()['Bucket [2] | Control Input'].value = action[1].item()
        self.interface.getInputContainer()['Stick [1] | Control Input'].value = action[2].item()
        self.interface.getInputContainer()['Boom [0] | Control Input'].value = action[3].item()

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
        penalty = 0
        swingpos = self.interface.getOutputContainer()['Actuator Swing Position'].value
        BoomLinPos = self.interface.getOutputContainer()['Actuator Boom Position'].value
        BuckLinPos = self.interface.getOutputContainer()['Actuator Bucket Position'].value
        StickLinPos = self.interface.getOutputContainer()['Actuator Arm Position'].value
        states = np.array([swingpos, BoomLinPos, BuckLinPos, StickLinPos])

        states = (states - np.mean(states)) / np.std(states)

        BuckAng = self.interface.getOutputContainer()['Reward | Bucket Angle'].value
        BuckHeight = self.interface.getOutputContainer()['Reward | Bucket Height'].value
        EngAvgPow = self.interface.getOutputContainer()['Reward | Engine Average Power'].value
        CurrEngPow = self.interface.getOutputContainer()['Reward | Current Engine Power'].value
        EngTor = self.interface.getOutputContainer()['Reward | Engine Torque'].value
        EngTorAvg = self.interface.getOutputContainer()['Reward | Engine Torque Average'].value
        Engrpm = self.interface.getOutputContainer()['Reward | Engine RPM'].value
        PressLeft = self.interface.getOutputContainer()['Reward | Front Left Pressure'].value
        PressRight = self.interface.getOutputContainer()['Reward | Front Right Pressure'].value

        RewardVal = [BuckAng, BuckHeight, EngAvgPow, CurrEngPow, EngTor, EngTorAvg,
                    Engrpm, PressLeft, PressRight]

        RewardVal = list(np.divide(np.subtract(np.array(RewardVal), np.array(self.min)),
                                   np.subtract(np.array(self.max), np.array(self.min))))

        self.rewfeatures.append(RewardVal)

        if len(self.rewfeatures) >= 32:
            _, mu, logvar = self.get_distribution(torch.tensor(self.rewfeatures[:-32]).float(), torch.tensor(self.rewfeatures[-1]).float())
            penalty = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return states, -penalty

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
