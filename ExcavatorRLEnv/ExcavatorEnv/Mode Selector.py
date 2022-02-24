# This script routes the signals to switch between three different control modes:
# - drive
# - control bucket
# - control from external signals

def post_step(extension):
    if not extension.inputs.Engine_Status.value:
        # Freeze actuators and tracks
        extension.outputs.Swing_Signal.value = 0.0
        extension.outputs.Boom_Signal.value = 0.0
        extension.outputs.Stick_Signal.value = 0.0
        extension.outputs.Bucket_Signal.value = 0.0
        extension.outputs.Left_Track_Signal.value = 0.0
        extension.outputs.Right_Track_Signal.value = 0.0
    else:
        if not extension.inputs.External_Override.value:
            if extension.inputs.Drive_Mode.value:
                # Freeze actuators and drive tracks
                extension.outputs.Swing_Signal.value = 0.0
                extension.outputs.Boom_Signal.value = 0.0
                extension.outputs.Stick_Signal.value = 0.0
                extension.outputs.Bucket_Signal.value = 0.0
                extension.outputs.Left_Track_Signal.value = extension.inputs.Gamepad_Left_Track.value
                extension.outputs.Right_Track_Signal.value = extension.inputs.Gamepad_Right_Track.value
            else:
                # Freeze tracks and control actuators
                extension.outputs.Swing_Signal.value = extension.inputs.Gamepad_Swing.value
                extension.outputs.Boom_Signal.value = extension.inputs.Gamepad_Boom.value
                extension.outputs.Stick_Signal.value = extension.inputs.Gamepad_Stick.value
                # Note that the gamepad signal for the bucket is inverted, so a - is added here
                extension.outputs.Bucket_Signal.value = - extension.inputs.Gamepad_Bucket.value
                extension.outputs.Left_Track_Signal.value = 0.0
                extension.outputs.Right_Track_Signal.value = 0.0
        else:
            # Use external signals
            extension.outputs.Swing_Signal.value = extension.inputs.External_Swing.value
            extension.outputs.Boom_Signal.value = extension.inputs.External_Boom.value
            extension.outputs.Stick_Signal.value = extension.inputs.External_Stick.value
            extension.outputs.Bucket_Signal.value = extension.inputs.External_Bucket.value
            extension.outputs.Left_Track_Signal.value = extension.inputs.External_Left_Track.value
            extension.outputs.Right_Track_Signal.value = extension.inputs.External_Right_Track.value
