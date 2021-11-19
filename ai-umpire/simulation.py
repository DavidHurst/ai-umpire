import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pathlib import Path
import pychrono.postprocess as postprocess

mm = 0.001
WALL_THICKNESS = 5 * mm
COURT_LENGTH = 9750 * mm
WALL_HEIGHT = 5250 * mm
COURT_WIDTH = 6400 * mm
BACK_WALL_OUT_LINE_HEIGHT = 2130 * mm
TIN_HEIGHT = 430 * mm
LINE_MARKING_WIDTH = 50 * mm
PAINT_THICKNESS = 1 * mm
FRONT_WALL_OUT_LINE_HEIGHT = 4570 * mm

# Set location of data file
data_path = str(Path("C:\\Users\\david\\miniconda3\\pkgs\\pychrono-6.0.0-py37_0\\Library\\data\\").resolve())
chrono.SetChronoDataPath(data_path)

# System
system = chrono.ChSystemNSC()

# Floor material
surface_mat = chrono.ChMaterialSurfaceNSC()

# Ball material
sph_mat = chrono.ChMaterialSurfaceNSC()
sph_mat.SetFriction(0.25)

# Ball
ball = chrono.ChBodyEasySphere(20 * mm, 0.0001, True, True, sph_mat)
ball.SetPos(chrono.ChVectorD(0, 100 * mm, COURT_LENGTH * 0.6))
ball.SetName('Ball')
ball.SetPos_dt(chrono.ChVectorD(8, 20, 55))  # Speed
ball.SetPos_dtdt(chrono.ChVectorD(5, 5, 5))  # Acceleration
ball.SetRot_dt(chrono.ChQuaternionD(0, 0, 0.5, 0))  # Rotation
ball.SetRot_dtdt(chrono.ChQuaternionD(0, 0, 0.5, 0))  # Rotation acceleration

# Floor
floor = chrono.ChBodyEasyBox(COURT_WIDTH, WALL_THICKNESS, COURT_LENGTH, 1, True, True, surface_mat)
floor.SetName('Floor')
floor.SetPos(chrono.ChVectorD(0, 0, 0))
floor.SetBodyFixed(True)

# Tin
tin = chrono.ChBodyEasyBox(COURT_WIDTH, TIN_HEIGHT, LINE_MARKING_WIDTH, 1, True, True, surface_mat)
tin.SetName('Tin')
tin.SetPos(chrono.ChVectorD(0, TIN_HEIGHT / 2, (COURT_LENGTH / 2) - LINE_MARKING_WIDTH / 2))
tin.SetBodyFixed(True)

# Left wall
left_wall = chrono.ChBodyEasyBox(WALL_THICKNESS, WALL_HEIGHT, COURT_LENGTH, 1, True, True, surface_mat)
left_wall.SetName('Left Wall')
left_wall.SetPos(chrono.ChVectorD(-(COURT_WIDTH / 2), WALL_HEIGHT / 2, 0))
left_wall.SetBodyFixed(True)

# Right wall
right_wall = chrono.ChBodyEasyBox(WALL_THICKNESS, WALL_HEIGHT, COURT_LENGTH, 1, True, True, surface_mat)
right_wall.SetName('Right Wall')
right_wall.SetPos(chrono.ChVectorD(COURT_WIDTH / 2, WALL_HEIGHT / 2, 0))
right_wall.SetBodyFixed(True)

# Front wall
front_wall = chrono.ChBodyEasyBox(COURT_WIDTH, WALL_HEIGHT, WALL_THICKNESS, 1000, True, True, surface_mat)
front_wall.SetName('Front Wall')
front_wall.SetPos(chrono.ChVectorD(0, WALL_HEIGHT / 2, COURT_LENGTH / 2))
front_wall.SetBodyFixed(True)

# Back wall
back_wall = chrono.ChBodyEasyBox(COURT_WIDTH, BACK_WALL_OUT_LINE_HEIGHT, WALL_THICKNESS, 1000, True, True, surface_mat)
back_wall.SetName('Back Wall')
back_wall.SetPos(chrono.ChVectorD(0, BACK_WALL_OUT_LINE_HEIGHT / 2, -COURT_LENGTH / 2))
back_wall.SetBodyFixed(True)

# Front wall line
front_wall_out_line = chrono.ChBodyEasyBox(COURT_WIDTH, LINE_MARKING_WIDTH, PAINT_THICKNESS, 1000, True, True,
                                           surface_mat)
front_wall_out_line.SetName('Front Wall Out-Line')
front_wall_out_line.SetPos(chrono.ChVectorD(0, FRONT_WALL_OUT_LINE_HEIGHT + (LINE_MARKING_WIDTH / 2),
                                            (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS))
front_wall_out_line.SetBodyFixed(True)

# Service line
service_line = chrono.ChBodyEasyBox(COURT_WIDTH, LINE_MARKING_WIDTH, PAINT_THICKNESS, 1000, True, True, surface_mat)
service_line.SetName('Service Line')
service_line.SetPos(chrono.ChVectorD(0, (1780 * mm) + (LINE_MARKING_WIDTH / 2),
                                     (COURT_LENGTH / 2) - (WALL_THICKNESS / 2) - PAINT_THICKNESS))
service_line.SetBodyFixed(True)

# Left-wall out-line line
# left_wall_out_line = chrono.ChBodyEasyBox(2, 2, 4, 1000, True, True,
#                                           surface_mat)
# left_wall_out_line.SetName('Left Wall Out Line')
# left_wall_out_line.SetPos(chrono.ChVectorD(0, 3, 0))
# left_wall_out_line >> chrono.ChQuaternionD(3, 3, 4, 1)
# left_wall_out_line.SetBodyFixed(True)
#
# # Right-wall out-line line
# right_wall_out_line = chrono.ChBodyEasyBox(2, 2, 4, 1000, True, True,
#                                            surface_mat)
# right_wall_out_line.SetName('Left Wall Out Line')
# right_wall_out_line.SetPos(chrono.ChVectorD(0, 3, 0))
# right_wall_out_line >> chrono.ChQuaternionD(3, 3, 4, 1)
# right_wall_out_line.SetBodyFixed(True)

# Add objects to system
system.Add(ball)
system.Add(floor)
system.Add(left_wall)
system.Add(right_wall)
system.Add(front_wall)
system.Add(back_wall)
system.Add(tin)
system.Add(front_wall_out_line)
system.Add(service_line)
# system.Add(left_wall_out_line)
# system.Add(right_wall_out_line)

# Box texture
default_box_texture = chrono.ChTexture()
default_box_texture.SetTextureFilename(chrono.GetChronoDataFile('\\textures\\cubetexture_borders_ref.png'))

# Red box texture
red_texture = chrono.ChTexture()
red_texture.SetTextureFilename(chrono.GetChronoDataFile('\\textures\\red_box.png'))

# Give the body surfaces some texture's which PyChrono will defer to POV-Ray
floor_texture_povray = postprocess.ChPovRayAssetCustom()
floor_texture_povray.SetCommands('texture { pigment { DMFDarkOak } finish { diffuse 0.3 } }')

wall_texture_povray = postprocess.ChPovRayAssetCustom()
wall_texture_povray.SetCommands('pigment { color rgbf <1, 1, 1, 0.8> }\nfinish { reflection 0.001 refraction 0.7 ior '
                                '1.5 phong 1.0 }')

ball_texture_povray = postprocess.ChPovRayAssetCustom()
ball_texture_povray.SetCommands('pigment { color rgb <250, 249, 246> } finish { diffuse 0.2 }')

red_texture_povray = postprocess.ChPovRayAssetCustom()
red_texture_povray.SetCommands('texture { pigment { Red } }')

# left_wall.AddAsset(default_box_texture)
# right_wall.AddAsset(default_box_texture)
# front_wall.AddAsset(default_box_texture)
# back_wall.AddAsset(default_box_texture)
# floor.AddAsset(default_box_texture)
# ball.AddAsset(default_box_texture)
# tin.AddAsset(red_texture)
#
# front_wall_out_line.AddAsset(red_texture)
# service_line.AddAsset(red_texture)
# left_wall_out_line.AddAsset(red_texture)
# right_wall_out_line.AddAsset(red_texture)

left_wall.AddAsset(wall_texture_povray)
right_wall.AddAsset(wall_texture_povray)
front_wall.AddAsset(wall_texture_povray)
back_wall.AddAsset(wall_texture_povray)
floor.AddAsset(floor_texture_povray)
ball.AddAsset(ball_texture_povray)
tin.AddAsset(red_texture_povray)

front_wall_out_line.AddAsset(red_texture_povray)
service_line.AddAsset(red_texture_povray)


# Class that reports contact and allows user defined actions upon contact.
class MyReportContactCallback(chrono.ReportContactCallback):
    def __init__(self):
        super(MyReportContactCallback, self).__init__()

    def OnReportContact(self, contact_point_A, contact_point_B, plane_coord, distance, eff_radius, react_forces,
                        react_torques, contactobjA, contactobjB):
        bodyUpA = chrono.CastContactableToChBody(contactobjA)
        nameA = bodyUpA.GetName()
        bodyUpB = chrono.CastContactableToChBody(contactobjB)
        nameB = bodyUpB.GetName()
        if nameB != 'Floor' and nameA != 'Floor':
            print(f'Contact between {nameA} & {nameB} @ {contact_point_A}, dist={distance}')
        return True  # return False to stop reporting contacts


reporter = MyReportContactCallback()

# Visualise system with Irrlicht app
# vis_app = chronoirr.ChIrrApp(system, 'Ball Visualisation', chronoirr.dimension2du(800, 800))
#
# vis_app.AddTypicalCamera(chronoirr.vector3df(0, BACK_WALL_OUT_LINE_HEIGHT + 1500 * mm, -COURT_LENGTH))
# vis_app.AddTypicalSky(chrono.GetChronoDataFile('\\skybox\\'))
# # vis_app.AddTypicalLights()
# vis_app.AddLight(chronoirr.vector3df(0, 30, 0), 200)
# vis_app.AddShadowAll()
# vis_app.AssetBindAll()
# vis_app.AssetUpdateAll()
#
# # -- Run simulation -- #
# vis_app.SetTimestep(0.01)
# vis_app.SetTryRealtime(True)
#
# while vis_app.GetDevice().run():
#     vis_app.BeginScene()
#     vis_app.DrawAll()
#     vis_app.DoStep()
#     vis_app.EndScene()
#
#     system.GetContactContainer().ReportAllContacts(reporter)

# Export simulation data so that it can be visualised in POV-Ray
pov_exporter = postprocess.ChPovRay(system)
pov_exporter.SetTemplateFile(f'{data_path}\\_template_POV.pov')
pov_exporter.SetBasePath(str(Path('C:\\Users\\david\\Downloads\\generated_povray').resolve()))
pov_exporter.SetCamera(chrono.ChVectorD(0, BACK_WALL_OUT_LINE_HEIGHT + 1500 * mm, -COURT_LENGTH * 1.1),
                       chrono.ChVectorD(0, WALL_HEIGHT / 2, COURT_LENGTH / 2), 0)
pov_exporter.SetLight(chrono.ChVectorD(0, 30, 0), chrono.ChColor(1, 1, 1, 1), True)
pov_exporter.SetBackground(chrono.ChColor(0.2, 0.2, 0.2, 1))
pov_exporter.SetPictureSize(852, 480)  # 1280, 720
pov_exporter.SetAntialiasing(True, 6, 0.3)

# Add bodies to export
pov_exporter.AddAll()

pov_exporter.ExportScript()

# Perform a short simulation
while system.GetChTime() <= 1:
    system.DoStepDynamics(0.01)
    system.GetContactContainer().ReportAllContacts(reporter)
    print(f'Time Step = {system.GetChTime():.2f}')
    pov_exporter.ExportData()
