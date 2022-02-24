def pre_step(extension):
    # Pitch of hydraulic pump:
    minPitchHydraulicPump = 0.9
    maxPitchHydraulicPump = 1.1
    throttle = max(extension.inputs.Throttle.value, 0.2)
    pitchHydraulicPump = minPitchHydraulicPump + ((maxPitchHydraulicPump - minPitchHydraulicPump) * throttle)
    extension.outputs.Pitch_Hydraulic_Pump.value = pitchHydraulicPump

    #Gain of hydraulic pump:   
    minGainHydraulicPump = 0.15
    maxGainHydraulicPump = 0.35
    gainHydraulicPump = minGainHydraulicPump + ((maxGainHydraulicPump - minGainHydraulicPump) * throttle)
    extension.outputs.Gain_Hydraulic_Pump.value = gainHydraulicPump

    # Beep when moving backwards
    trackMotionThreshold = -0.05
    if (extension.inputs.Right_Track_Signal.value < trackMotionThreshold and extension.inputs.Left_Track_Signal.value <  trackMotionThreshold):
        extension.outputs.Reversing.value = True
    else:
        extension.outputs.Reversing.value = False
