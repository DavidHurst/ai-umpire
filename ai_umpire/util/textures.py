import pychrono as chrono
import pychrono.postprocess as postprocess

__all__ = [
    "CUBE_TEXTURE_CHRONO",
    "CUBE_TEXTURE_RED_CHRONO",
    "WOOD_TEXTURE_POVRAY",
    "GLASS_TEXTURE_POVRAY",
    "BALL_TEXTURE_POVRAY",
    "RED_TEXTURE_POVRAY",
    "PURPLE_TEXTURE_POVRAY",
    "ORANGE_TEXTURE_POVRAY",
    "WALL_DECAL_A_POVRAY",
    "WALL_DECAL_B_POVRAY",
    "WALL_DECAL_C_POVRAY",
]


CUBE_TEXTURE_CHRONO = chrono.ChTexture()
CUBE_TEXTURE_CHRONO.SetTextureFilename("\\assets\\textures\\cubetexture.png")

CUBE_TEXTURE_RED_CHRONO = chrono.ChTexture()
CUBE_TEXTURE_RED_CHRONO.SetTextureFilename("\\assets\\textures\\cubetexture_red.png")

WOOD_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
WOOD_TEXTURE_POVRAY.SetCommands(
    "texture { pigment { DMFDarkOak } finish { diffuse 0.1 } }"
)

GLASS_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
GLASS_TEXTURE_POVRAY.SetCommands(
    "texture {Glass} finish { reflection {0.035} diffuse 0.2 }"
)

BALL_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
BALL_TEXTURE_POVRAY.SetCommands(
    """
    texture
    {
       pigment {color rgb <1,1,1>}
       normal  {granite 0.05 scale 0.1}
       finish
       {
         specular 0.2 roughness 0.08 metallic 0.5
         subsurface
         {
           translucency rgb 20*<2,0.2,0.02>
         }
       }
    }
    
    interior
    {
       ior 1.52
    }
    """
)

RED_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
RED_TEXTURE_POVRAY.SetCommands(
    """
    pigment{ rgb<0.980, 0, 0.070> } 
    finish{ brilliance .6 }
    """
)

PURPLE_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
PURPLE_TEXTURE_POVRAY.SetCommands(
    "texture { pigment { DarkPurple } finish { diffuse 0.9 }}"
)

ORANGE_TEXTURE_POVRAY = postprocess.ChPovRayAssetCustom()
ORANGE_TEXTURE_POVRAY.SetCommands(
    "texture { pigment { OrangeRed } finish { diffuse 0.9 }}"
)

WALL_DECAL_A_POVRAY = postprocess.ChPovRayAssetCustom()
WALL_DECAL_A_POVRAY.SetCommands(
    'pigment { image_map { png "../UOS-logo.png" once transmit all 0.7} translate <-0.5, -0.5, 0> scale <-1.3, 1.3, 1> }'
)

WALL_DECAL_B_POVRAY = postprocess.ChPovRayAssetCustom()
WALL_DECAL_B_POVRAY.SetCommands(
    'pigment { image_map { png "david.png" once transmit all 0.7} translate <-0.5, -0.5, 0> scale <-1.3, 1.3, 1> }'
)

WALL_DECAL_C_POVRAY = postprocess.ChPovRayAssetCustom()
WALL_DECAL_C_POVRAY.SetCommands(
    'pigment { image_map { png "hurst.png" once transmit all 0.7} translate <-0.5, -0.5, 0> scale <-1.3, 1.3, 1> }'
)
