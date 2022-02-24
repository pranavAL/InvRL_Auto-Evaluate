'''
>===========================================================================<
> Signal interpreter for Excavator                                          <
>===========================================================================<
> This signal interpreter:                                                  <
> Takes the inputs from controllers and prepares them for output            <
>===========================================================================<
'''

def on_simulation_start(extension):
    # Internal Variables:
    # initialize some variables
    extension.mEngineTogglePrev          = False
    extension.mEngineStatus              = False
    extension.mEngineThrottle            = 0.0

def pre_step(extension):
    # toggle the engine switch between on and off
    extension.mEngineTogglePrev, extension.mEngineStatus = toggleButton(extension.inputs.Engine_Start_Toggle_Switch.value, extension.mEngineTogglePrev, extension.mEngineStatus)

    # Throttle
    # Speed Dial has 10 positions (0-9). Set Throttle between 0 and 0.7 depending on dial position.
    if extension.mEngineStatus:
        extension.mEngineThrottle = extension.inputs.Engine_Speed_Dial.value * 0.7/ 9.0
    # Take alternate input (for example a gamepad) if HMI is not available
    if extension.inputs.Throttle_set_from_gamepad.value:
        extension.mEngineThrottle = 0.7

    # assigning outputs
    extension.outputs.Engine_Status.value               = extension.mEngineStatus
    extension.outputs.Engine_Throttle.value             = extension.mEngineThrottle
    extension.outputs.Engine_Auto_Idle_Switch.value     = extension.inputs.Engine_Auto_Idle_Switch.value
    extension.outputs.Travel_Speed_Switch.value         = extension.inputs.Travel_Speed_Switch.value
    extension.outputs.SAE_ISO_Toggle_Switch.value       = extension.inputs.SAE_ISO_Toggle_Switch.value  # 0 is SAE and 1 is ISO
    extension.outputs.Work_Light_Switch.value           = extension.inputs.Work_Light.value

# FUNCTIONS:
# toggle function
def toggleButton(buttonState, buttonStatePrev, currentState):
    if buttonState and not buttonStatePrev:
        currentState = not currentState
    buttonStatePrev = buttonState
    return buttonStatePrev, currentState

