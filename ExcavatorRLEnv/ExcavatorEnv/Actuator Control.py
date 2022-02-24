import random

def on_simulation_start(extension):
    # Define time constants to use to filter the signals to the actuators
    extension.SwingTC = 0.0
    extension.BoomTC = 0.0
    extension.StickTC = 0.0
    extension.BucketTC = 0.0

def post_step(extension):

    extension.outputs.Throttle.value = 0
    extension.outputs.Swing.value = 0
    extension.outputs.Bucket.value = random.random()
    extension.outputs.Stick.value =  random.random()
    extension.outputs.Boom.value = 0
    extension.outputs.Left_Track.value = 0
    extension.outputs.Right_Track.value = 0

    print(extension.inputs.Reward)
