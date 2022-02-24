import Vortex


def post_step(extension):
    extension.outputs.Left_Track_UV_Offset.value = (
        1/extension.parameters.Texture_Length.value * Vortex.VxVector2(extension.inputs.Left_Track_Displacement.value, 0.0))
    extension.outputs.Right_Track_UV_Offset.value = (
        1/extension.parameters.Texture_Length.value * Vortex.VxVector2(extension.inputs.Right_Track_Displacement.value, 0.0))