
def on_simulation_start(extension):
    # Define time constants to use to filter the signals to the actuators
    extension.SwingTC = 1.0
    extension.BoomTC = 0.05
    extension.StickTC = 0.05
    extension.BucketTC = 0.05
    extension.BladeTC = 0.1

    # Define max speeds of actuators
    extension.SwingMaxSpeed = 1.0
    extension.BoomMaxSpeed = 0.25
    extension.StickMaxSpeed = 0.5
    extension.BucketMaxSpeed = 0.75
    extension.BladeMaxSpeed = 0.1


def post_step(extension):

    # Calculate the desired speeds for the actuators
    throttleFactor = max(extension.inputs.Throttle.value, 0.2)
    swingSignal = extension.inputs.Swing_Signal.value * extension.SwingMaxSpeed * throttleFactor
    boomSignal = extension.inputs.Boom_Signal.value * extension.BoomMaxSpeed * throttleFactor
    stickSignal = extension.inputs.Stick_Signal.value * extension.StickMaxSpeed * throttleFactor
    bucketSignal = extension.inputs.Bucket_Signal.value * extension.BucketMaxSpeed * throttleFactor
    bladeSignal = extension.inputs.Blade_Signal.value * extension.BladeMaxSpeed * throttleFactor

    # Use a low pass filter on the actuator speeds so they are not instantly responsive
    extension.outputs.Swing_Speed.value = lowPassFilter(extension, swingSignal, extension.outputs.Swing_Speed.value, extension.SwingTC )
    extension.outputs.Boom_Speed.value = lowPassFilter(extension, boomSignal, extension.outputs.Boom_Speed.value, extension.BoomTC )
    extension.outputs.Stick_Speed.value = lowPassFilter(extension, stickSignal, extension.outputs.Stick_Speed.value, extension.StickTC )
    extension.outputs.Bucket_Speed.value = lowPassFilter(extension, bucketSignal, extension.outputs.Bucket_Speed.value, extension.BucketTC )
    extension.outputs.Blade_Speed.value = lowPassFilter(extension, bladeSignal, extension.outputs.Blade_Speed.value, extension.BladeTC )

# Basic low pass filter function that smooths out signal based on timeConstant.
def lowPassFilter(extension, inputSignal, outputSignal, timeConstant ):
    timeStep = extension.getApplicationContext().getSimulationTimeStep()
    value = ( (timeStep * inputSignal) + (timeConstant * outputSignal) ) / (timeStep + timeConstant )
    return value