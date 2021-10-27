import pychrono as chrono
import pychrono.irrlicht as chronoirr
from pathlib import Path
import pychrono.postprocess as postprocess

SZ = 50

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
ball = chrono.ChBodyEasySphere(0.5, 5000, True, True, sph_mat)
ball.SetPos(chrono.ChVectorD(22, 3, 20))
ball.SetName('Ball')
ball.SetPos_dt(chrono.ChVectorD(-20, 25, 40))  # Speed
ball.SetPos_dtdt(chrono.ChVectorD(-20, 25, 40))  # Acceleration
ball.SetRot_dt(chrono.ChQuaternionD(0, 0, 0.5, 0))  # Rotation
ball.SetRot_dtdt(chrono.ChQuaternionD(0, 0, 0.5, 0))  # Rotation acceleration

# Floor
floor = chrono.ChBodyEasyBox(SZ, 0.2, SZ, 1000, True, True, surface_mat)
floor.SetName('Floor')
floor.SetPos(chrono.ChVectorD(0, 0, 0))
floor.SetBodyFixed(True)

# Left wall
left_wall = chrono.ChBodyEasyBox(0.2, SZ, SZ, 1000, True, True, surface_mat)
left_wall.SetName('Left Wall')
left_wall.SetPos(chrono.ChVectorD(25, 25, 0))
left_wall.SetBodyFixed(True)

# Right wall
right_wall = chrono.ChBodyEasyBox(0.2, SZ, SZ, 1000, True, True, surface_mat)
right_wall.SetName('Right Wall')
right_wall.SetPos(chrono.ChVectorD(-25, 25, 0))
right_wall.SetBodyFixed(True)

# Front wall
front_wall = chrono.ChBodyEasyBox(SZ, SZ, 0.2, 1000, True, True, surface_mat)
front_wall.SetName('Front Wall')
front_wall.SetPos(chrono.ChVectorD(0, 25, 25))
front_wall.SetBodyFixed(True)

# Back wall
# @To-Do: add back wall once you figure out how to change opacity / use wall_texture as texture,
# might have to be done in VTK/POV-Ray.

# Add objects to system
system.Add(ball)
system.Add(floor)
system.Add(left_wall)
system.Add(right_wall)
system.Add(front_wall)

# Give the body surfaces some texture's which PyChrono will defer to POV-Ray
floor_texture_povray = postprocess.ChPovRayAssetCustom()
floor_texture_povray.SetCommands('texture { pigment { DMFDarkOak } finish { diffuse 0.3 } }')

wall_texture_povray = postprocess.ChPovRayAssetCustom()
wall_texture_povray.SetCommands('pigment { color rgbf <1, 1, 1, 0.8> }\nfinish { reflection 0.1 refraction 1.0 ior '
                                '1.5 phong 1.0 }')

ball_texture_povray = postprocess.ChPovRayAssetCustom()
ball_texture_povray.SetCommands('pigment { color rgb <250, 249, 246> } finish { diffuse 0.2 }')

left_wall.AddAsset(wall_texture_povray)
right_wall.AddAsset(wall_texture_povray)
front_wall.AddAsset(wall_texture_povray)
floor.AddAsset(floor_texture_povray)
ball.AddAsset(ball_texture_povray)


# Class that reports contact and allows user defined actions upon contact.
class MyReportContactCallback(chrono.ReportContactCallback):
    def __init__(self):
        super(MyReportContactCallback, self).__init__()

    def OnReportContact(self, contact_point_A, contact_point_B, plane_coord, distance, eff_radius, react_forces, react_torques, contactobjA, contactobjB):
        bodyUpA = chrono.CastContactableToChBody(contactobjA)
        nameA = bodyUpA.GetName()
        bodyUpB = chrono.CastContactableToChBody(contactobjB)
        nameB = bodyUpB.GetName()
        print(f'Contact between {nameA} & {nameB} @ {contact_point_A}, dist={distance}')
        return True  # return False to stop reporting contacts


reporter = MyReportContactCallback()

# Visualise system with Irrlicht app
# vis_app = chronoirr.ChIrrApp(system, 'Ball Visualisation', chronoirr.dimension2du(800, 800))
#
# vis_app.AddTypicalCamera(chronoirr.vector3df(0, 25, -70))
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
pov_exporter.SetCamera(chrono.ChVectorD(0, 40, -75), chrono.ChVectorD(0, 15, 25), 0)
pov_exporter.SetLight(chrono.ChVectorD(0, 30, 0), chrono.ChColor(1, 1, 1, 1), True)
pov_exporter.SetBackground(chrono.ChColor(0.2, 0.2, 0.2, 1))
pov_exporter.SetPictureSize(1280, 720)
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
