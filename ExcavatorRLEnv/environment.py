import warnings
warnings.filterwarnings('ignore')

import Vortex
import vxatp3
import random
import numpy as np
import math

SUB_STEPS = 5
MAX_STEPS = 200

class env():

    def __init__(self):
        # VxMechanism variable for the mechanism to be loaded.
        self.vxscene = None
        self.scene = None
        self.interface = None

        # Define the setup and scene file paths
        self.setup_file = 'C:\CM Labs\Vortex Studio Content 2022.1\Demo Scenes\Equipment\Excavator\Dynamic\Design\Setup.vxc'
        self.content_file = 'C:\CM Labs\Vortex Studio Content 2022.1\Demo Scenes\Scenario\Excavator Scene\ExcavatorWorkshop.vxscene'

        # Create the Vortex Application
        self.application = vxatp3.VxATPConfig.createApplication(self, 'Excavator App', self.setup_file)

        # Create a display window
        self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue("Windowed")
        self.display.setName('3D Display')
        self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(500, 500, 1280, 720))

        # Initialize Action and Observation Spaces for the NN
        self.max_torque = 5.0

        self.action_space = 8
        self.observation_space = 12

    def __del__(self):
        # It is always a good idea to destroy the VxApplication when we are done with it.
        self.application = None

    def reset(self):
        # Initialize Reward and Step Count
        self.current_step = 0
        self.reward = 0

        # The first time we load the scene
        if self.vxscene is None:
            # Switch to Editing Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

            # Load scene file and get the mechanism interface
            self.vxscene = self.application.getSimulationFileManager().loadObject(self.content_file)
            self.scene = Vortex.SceneInterface(self.vxscene)

            # Get the RL Interface VHL
            self.interface = self.scene.findExtensionByName('RL Interface')

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

        self.interface.getInputContainer()['SwingVelocity'].value = 2 * random.random() -1
        self.interface.getInputContainer()['BucketVelocity'].value = 2 * random.random() -1
        self.interface.getInputContainer()['StickVelocity'].value = 2 * random.random() -1
        self.interface.getInputContainer()['BoomVelocity'].value = 2 * random.random() -1

        self.interface.getInputContainer()['SwingState'].value = 2 * random.random() -1
        self.interface.getInputContainer()['BucketState'].value = 2 * random.random() -1
        self.interface.getInputContainer()['BoomState'].value = 2 * random.random() -1
        self.interface.getInputContainer()['StickState'].value = 2 * random.random() -1

        return self._get_obs()

    def waitForNbKeyFrames(self, expectedNbKeyFrames, application, keyFrameList):
        maxNbIter = 100
        nbIter = 0
        while len(keyFrameList.getKeyFrames()) != expectedNbKeyFrames and nbIter < maxNbIter:
            if not application.update():
                break
            ++nbIter

    def step(self, action):  # takes a numpy array as input

        # Apply actions
        self.interface.getInputContainer()['SwingVelocity'].value = action[0].item() * self.max_torque
        self.interface.getInputContainer()['BucketVelocity'].value = ((action[1].item() + 1) / 2.0) * self.max_torque
        self.interface.getInputContainer()['StickVelocity'].value = action[2].item() * self.max_torque
        self.interface.getInputContainer()['BoomVelocity'].value = action[3].item() * self.max_torque

        self.interface.getInputContainer()['SwingState'].value = action[4].item() * self.max_torque
        self.interface.getInputContainer()['BucketState'].value = action[5].item() * self.max_torque
        self.interface.getInputContainer()['BoomState'].value = action[6].item() * self.max_torque
        self.interface.getInputContainer()['StickState'].value = action[7].item() * self.max_torque

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

        # Reward Function

        self.current_step += 1

        return obs, reward, done, {}


    def _get_obs(self):
        swingpos = self.interface.getOutputContainer()['swingPos'].value
        BoomLinPos = self.interface.getOutputContainer()['boomLinPos'].value
        BuckLinPos = self.interface.getOutputContainer()['BuckLinPos'].value
        StickLinPos = self.interface.getOutputContainer()['stickLinPos'].value
        bucket_sand = self.interface.getOutputContainer()['Reward_Sand'].value / 500.0
        Reward = (math.dist(BuckLinPos, swingpos) - 5.0) / (8.0 - 5.0) + bucket_sand
        features = np.array([*swingpos, *BoomLinPos, *BuckLinPos, *StickLinPos])
        features = (features - np.mean(features)) / (np.std(features))

        return features, [Reward,bucket_sand]

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
