def timeToString(seconds):
    hours = seconds // 3600
    seconds %= (3600)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def pre_step(extension):
    elapsedTime = extension.getApplicationContext().getSimulationTime()
    extension.outputs.Session_Time_Text.value = timeToString(elapsedTime)
    pass
