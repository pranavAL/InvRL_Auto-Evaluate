import math
import Vortex


def on_simulation_start(extension):
    extension.outputs.RPM.value = 0
    extension.outputs.Torque.value = 0
    extension.outputs.Power.value = 0

def post_step(extension):

    # Calculate the desired speeds for the actuators
    rpm = max(extension.inputs.RPM.value, 0)  # engine speed in rpm
    torque = max(extension.inputs.Torque.value, 0) # engine torque in N.m
    power = (math.pi * rpm / 30.0) * torque # power in W    

    extension.outputs.RPM.value = rpm
    extension.outputs.Torque.value = torque
    extension.outputs.Power.value = power
    
    bucketTm = extension.inputs.Bucket_Tip.value
    baseTm = extension.inputs.Excavator_Base.value
    extension.outputs.Bucket_Height.value = Vortex.getTranslation(bucketTm).z - Vortex.getTranslation(baseTm).z


