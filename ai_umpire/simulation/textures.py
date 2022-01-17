import pychrono as chrono
import pychrono.postprocess as postprocess

__all__ = [
    "CUBE_TEXTURE_CHRONO",
    "CUBE_TEXTURE_RED_CHRONO",
    "WOOD_TEXTURE_POVRAY",
    "GLASS_TEXTURE_POVRAY",
    "BALL_TEXTURE_POVRAY",
    "RED_TEXTURE_POVRAY",
    "BLUE_TEXTURE_POVRAY",
    "WALL_DECAL_A_POVRAY",
]


CUBE_TEXTURE_CHRONO = chrono.ChTexture()
CUBE_TEXTURE_CHRONO.SetTextureFilename("\\assets\\textures\\cubetexture.png")

CUBE_TEXTURE_RED_CHRONO = chrono.ChTexture()
CUBE_TEXTURE_RED_CHRONO.SetTextureFilename("\\assets\\textures\\cubetexture_red.png")

WOOD_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
WOOD_TEXTURE_POVRAY.SetCommands(
    "texture { pigment { DMFDarkOak } finish { diffuse 0.3 } }"
)

GLASS_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
GLASS_TEXTURE_POVRAY.SetCommands(
    "pigment { color rgbf <1, 1, 1, 0.8> }\nfinish { reflection 0.001 refraction 0.7 ior 1.5 phong 1.0 }"
)

BALL_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
BALL_TEXTURE_POVRAY.SetCommands(
    "pigment { color rgb <250, 249, 246> } finish { diffuse 0.2 }"
)

RED_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
RED_TEXTURE_POVRAY.SetCommands("texture { pigment { Red } }")

BLUE_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
BLUE_TEXTURE_POVRAY.SetCommands("texture { pigment { Blue } }")

WALL_DECAL_A_POVRAY = postprocess.ChPovRayAssetCustom()
WALL_DECAL_A_POVRAY.SetCommands("pigment { image_map { gif \"uos.gif\" once} scale <-1.5, 1.5, 1> } translate <0.65, -.6, 0>")


