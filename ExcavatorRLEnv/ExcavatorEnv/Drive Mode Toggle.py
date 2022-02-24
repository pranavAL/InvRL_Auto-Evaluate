# This script is a simple toggle, selecting between drive mode (control the tracks)
# and control mode (actuate the bucket)


def on_simulation_start(extension):
    # Save a variable of the previous input state so we can tell if it changed.
    extension.prevMode_Toggle = False
    # Start with the help turned on
    extension.outputs.Drive_Mode.value = False

def pre_step(extension):
    # Rising edge detection: the output should be toggled only once when the button is pressed.
    if extension.inputs.Drive_Mode_Toggle.value and not extension.prevMode_Toggle:
        extension.outputs.Drive_Mode.value = not extension.outputs.Drive_Mode.value

    extension.prevMode_Toggle = extension.inputs.Drive_Mode_Toggle.value

