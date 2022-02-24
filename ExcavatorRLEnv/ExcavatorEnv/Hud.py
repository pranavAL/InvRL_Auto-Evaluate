def post_step(extension):
    unitSystem = extension.getApplicationContext().getUnitSystem();
    
    excavatorMass = unitSystem.convertToPreferred(extension.inputs.Payload_Mass)
    extension.outputs.Mass_Text.value = str(round(excavatorMass, 1)) + ' ' + unitSystem.getPreferredUnitSymbol(extension.inputs.Payload_Mass)

    dumptruckMass = unitSystem.convertToPreferred(extension.inputs.DumpTruck_Mass)
    extension.outputs.DumpTruck_Text.value = str(round(dumptruckMass, 1)) + ' ' + unitSystem.getPreferredUnitSymbol(extension.inputs.DumpTruck_Mass)

    bucketHeight = unitSystem.convertToPreferred(extension.inputs.Bucket_Height)
    extension.outputs.Bucket_Height_Text.value = str(round(bucketHeight, 2)) + ' ' + unitSystem.getPreferredUnitSymbol(extension.inputs.Bucket_Height)

    bucketAngle = int(round(unitSystem.convertToPreferred(extension.inputs.Bucket_Angle), 0))
    extension.outputs.Bucket_Angle_Text.value = str(bucketAngle) + ' ' + unitSystem.getPreferredUnitSymbol(extension.inputs.Bucket_Angle)
