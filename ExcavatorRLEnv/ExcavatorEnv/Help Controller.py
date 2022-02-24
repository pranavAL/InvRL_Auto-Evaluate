# This script is a simple toggle turning on or off the display of help
# If help is on, it will select either Drive or Actuate

def on_simulation_start(extension):
    # Save a variable of the previous input state so we can tell if it changed.
    extension.prevHelp_Toggle = False
    # Start with the help turned on
    extension.help_on = True

def pre_step(extension):
    # Rising edge detection: the output should be toggled only once when the button is pressed.
    if extension.inputs.Help_Toggle.value and not extension.prevHelp_Toggle:
        extension.help_on = not extension.help_on

    extension.outputs.Help_Drive.value = extension.inputs.Drive_Mode.value and extension.help_on
    extension.outputs.Help_Actuate.value = not extension.inputs.Drive_Mode.value and extension.help_on

    extension.prevHelp_Toggle = extension.inputs.Help_Toggle.value

