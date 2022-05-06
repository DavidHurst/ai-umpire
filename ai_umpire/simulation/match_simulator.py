import logging
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pychrono as chrono
import pychrono.irrlicht as chronoirr
import pychrono.postprocess as postprocess
from tqdm import tqdm

__all__ = ["MatchSimulator"]

from ai_umpire.simulation.fixed_sim_objs import *
from ai_umpire.util import (
    PLAYER_HEIGHT,
    BALL_TEXTURE_POVRAY,
    ORANGE_TEXTURE_POVRAY,
    PURPLE_TEXTURE_POVRAY,
    BACK_WALL_OUT_LINE_HEIGHT,
    COURT_LENGTH,
)

# Ball material, parameters define collision characteristics
BALL_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
BALL_MAT.SetRestitution(1)
BALL_MAT.SetDampingF(1)
BALL_MAT.SetCompliance(0.001)

# Player material
PLAYER_MAT: chrono.ChMaterialSurfaceNSC = chrono.ChMaterialSurfaceNSC()
PLAYER_MAT.SetSfriction(0.2)

# ToDo list:
""" 
     * Automatically include the screen.inc file in rendered povray folders on export.
     * Edit .ini file to change final frame to match number of frames produced, defaults to 1000
     * Add continue if cancel/fail to .ini file
     * Change output file type to JPEG
"""

"""
Once data is exported, the following must be appended to the render_sim_# file:

        #include "screen.inc"     
        #local cam_loc = <0,3,-16>;         
        #local cam_look_at = <0,3,16>;       
        #local cam_transform = transform
            {
                matrix
                <
                    1, 0, 0,
                    0, 1, 0,
                    0, 0, 1,
                    0, 0, 0
                >                       
            }  

        Set_Camera_Location(cam_loc)  
        Set_Camera_Look_At(cam_look_at)     
        Set_Camera_Transform(cam_transform)

And the screen.inc file must be place in the same directory as the file.
The setting Final_Frame in the .ini file must have its value change to the actual number of states produced, 399 in the
case of the script, append the following to the render_sim_#.ini file
        Output_File_Type=J
        Quality=8
        Continue_Trace=on       
        Work_Threads=2048
        Final_Frame=0399
"""


class MatchSimulator:
    def __init__(
        self,
        sim_id: int,
        root: Path,
        sim_step_sz: float,
        ball_init_pos: chrono.ChVectorD,
        ball_vel: chrono.ChVectorD,
        ball_acc: chrono.ChVectorD,
        ball_rot_dt: chrono.ChQuaternionD,
        p1_init_x: float,
        p1_init_z: float,
        p1_vel: chrono.ChVectorD,
        p2_init_x: float,
        p2_init_z: float,
        p2_vel: chrono.ChVectorD,
        output_res: Tuple,
    ) -> None:
        self._id: int = sim_id
        self._root: Path = root
        self._out_res = output_res
        self._povray_out_file: str = f"sim_{sim_id}_povray"
        self._povray_out_dir_path: Path = (
            self._root / "generated_povray" / self._povray_out_file
        )
        self._ball_pos_out_path = root / "ball_pos"
        self._sys: chrono.ChSystemNSC = chrono.ChSystemNSC()
        self._time_step: float = sim_step_sz

        self._sys.SetStep(self._time_step)

        # Initialise ball body
        self._ball: chrono.ChBodyEasySphere = chrono.ChBodyEasySphere(
            0.04, 0.5, True, True, BALL_MAT
        )
        self._ball.SetPos(ball_init_pos)
        self._ball.SetName("Ball")
        self._ball.SetPos_dt(ball_vel)
        self._ball.SetPos_dtdt(ball_acc)
        self._ball.SetRot_dt(ball_rot_dt)
        self._ball.AddAsset(BALL_TEXTURE_POVRAY)

        # Initialise player 1 body
        self._player1: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
            0.5, PLAYER_HEIGHT, 0.4, 50, True, True, PLAYER_MAT
        )
        self._player1.SetName("Player 1")
        p1_pos: chrono.ChVectorD = chrono.ChVectorD(
            p1_init_x, (PLAYER_HEIGHT / 2) + 0.01, p1_init_z
        )
        self._player1.SetPos(p1_pos)
        self._player1.SetPos_dt(p1_vel)
        self._player1.AddAsset(ORANGE_TEXTURE_POVRAY)

        # Initialise player 2 body
        self._player2: chrono.ChBodyEasyBox = chrono.ChBodyEasyBox(
            0.4, PLAYER_HEIGHT - 0.3, 0.4, 50, True, True, PLAYER_MAT
        )
        self._player2.SetName("Player 2")
        p2_pos: chrono.ChVectorD = chrono.ChVectorD(
            p2_init_x, ((PLAYER_HEIGHT - 0.3) / 2) + 0.01, p2_init_z
        )
        self._player2.SetPos(p2_pos)
        self._player2.SetPos_dt(p2_vel)
        self._player2.AddAsset(PURPLE_TEXTURE_POVRAY)

        # Add bodies to physics system
        self._sys.Add(self._ball)
        self._sys.Add(self._player1)
        self._sys.Add(self._player2)
        self._add_fixed_objects()

    def _add_fixed_objects(self):
        self._sys.Add(FLOOR)
        self._sys.Add(LEFT_WALL)
        self._sys.Add(RIGHT_WALL)
        self._sys.Add(FRONT_WALL)
        self._sys.Add(BACK_WALL)
        self._sys.Add(TIN)
        self._sys.Add(FRONT_WALL_OUT_LINE)
        self._sys.Add(LEFT_WALL_OUT_LINE)
        self._sys.Add(RIGHT_WALL_OUT_LINE)
        self._sys.Add(SERVICE_LINE)
        self._sys.Add(HALF_COURT_LINE)
        self._sys.Add(SHORT_LINE)
        self._sys.Add(LSB_VERTICAL)
        self._sys.Add(LSB_HORIZONTAL)
        self._sys.Add(RSB_VERTICAL)
        self._sys.Add(RSB_HORIZONTAL)
        self._sys.Add(FRONT_WALL_DECAL_CENTERED)
        # self._sys.Add(FRONT_WALL_DECAL_LEFT)
        # self._sys.Add(FRONT_WALL_DECAL_RIGHT)

    def run_sim(
        self, duration: float, export: bool = True, visualise: bool = False
    ) -> List[List]:
        if not visualise and not export:
            e: ValueError = ValueError(
                "Simulation did not run, visualise and export both disabled."
            )
            logging.exception(e)
            raise e
        logging.info(f"Export enabled: {export}.")
        logging.info(f"Visualise enabled: {visualise}.")

        random_moves: List[chrono.ChVectorD] = [
            chrono.ChVectorD(-1, 0, 1),
            chrono.ChVectorD(1, 0, -1),
            chrono.ChVectorD(1, 0, 1),
            chrono.ChVectorD(-1, 0, -1),
        ]
        ball_pos: List[List[float, float, float]]

        if export:
            ball_pos: List[List[float, float, float]] = [[], [], []]
            logging.info("Simulating, rendering and exporting.")
            # Set up object that exports the simulation data to a format that POV-Ray can render
            pov_exporter: postprocess.ChPovRay = postprocess.ChPovRay(self._sys)
            pov_exporter.SetTemplateFile(
                str(Path(".\\assets\\_template_POV.pov").resolve())
            )
            pov_exporter.SetBasePath(str(self._povray_out_dir_path))
            pov_exporter.SetOutputScriptFile(f"render_sim_{self._id}")
            pov_exporter.SetCamera(
                chrono.ChVectorD(0, 3, -11),
                chrono.ChVectorD(0, 3, 11),
                0,
            )
            pov_exporter.SetLight(
                chrono.ChVectorD(0, 7, 0), chrono.ChColor(1.2, 1.2, 1.2, 1), True
            )
            pov_exporter.SetBackground(chrono.ChColor(0.2, 0.2, 0.2, 1))
            pov_exporter.SetPictureSize(
                self._out_res[0], self._out_res[1]
            )  # Output resolution
            pov_exporter.SetAntialiasing(True, 6, 0.3)
            pov_exporter.AddAll()
            pov_exporter.ExportScript()

            # Run simulation one time step at a time exporting data for rendering at each time step
            pbar: tqdm = tqdm(
                total=int(duration / self._time_step), desc="Running simulation"
            )
            while self._sys.GetChTime() < duration - self._time_step:
                ball_pos[0].append(self._ball.GetPos().x)
                ball_pos[1].append(self._ball.GetPos().y)
                ball_pos[2].append(self._ball.GetPos().z)
                pov_exporter.ExportData()
                self._sys.DoStepDynamics(self._time_step)
                pbar.update(1)
                # # Emulate random player movement
                # if self._player1.GetPos_dt() <= chrono.ChVectorD(0, 0, 0):
                #     self._player1.SetPos_dt(random.choice(random_moves))
                #
                # if self._player2.GetPos_dt() <= chrono.ChVectorD(0, 0, 0):
                #     self._player2.SetPos_dt(random.choice(random_moves))
        if visualise:
            logging.info("Visualising simulation.")
            # contact_reporter.reset()
            # Visualise system with Irrlicht app
            vis_app = chronoirr.ChIrrApp(
                self._sys, "Ball Visualisation", chronoirr.dimension2du(1200, 800)
            )
            vis_app.AddTypicalCamera(
                chronoirr.vector3df(0, BACK_WALL_OUT_LINE_HEIGHT + 1.5, -COURT_LENGTH)
            )
            vis_app.AddLight(chronoirr.vector3df(-5, 90, 5), 100)
            vis_app.AddLight(chronoirr.vector3df(5, 90, 5), 100)
            vis_app.AddLight(chronoirr.vector3df(0, 90, 0), 100)
            vis_app.AddLight(chronoirr.vector3df(-5, 90, -5), 100)
            vis_app.AddLight(chronoirr.vector3df(5, 90, -5), 100)
            vis_app.AddShadowAll()
            vis_app.AssetBindAll()
            vis_app.AssetUpdateAll()
            vis_app.SetTimestep(self._time_step)
            vis_app.SetTryRealtime(True)
            vis_app.SetShowInfos(True)
            vis_app.SetPaused(True)

            while vis_app.GetDevice().run():
                vis_app.DoStep()
                vis_app.BeginScene()
                vis_app.DrawAll()
                vis_app.DoStep()
                vis_app.EndScene()

                # self._sys.GetContactContainer().ReportAllContacts(contact_reporter)

                # # Emulate random player movement
                # if self._player1.GetPos_dt() <= chrono.ChVectorD(0, 0, 0):
                #     self._player1.SetPos_dt(random.choice(random_moves))
                #
                # if self._player2.GetPos_dt() <= chrono.ChVectorD(0, 0, 0):
                #     self._player2.SetPos_dt(random.choice(random_moves))
                #
                # if self.get_sim_time() >= duration:
                #     vis_app.SetPaused(True)

        # Save ball pos to file
        if export:
            df: pd.DataFrame = pd.DataFrame(
                {"x": ball_pos[0], "y": ball_pos[1], "z": ball_pos[2]}
            )
            df.to_csv(str(self._ball_pos_out_path / f"sim_{self._id}.csv"))

        if not export:
            return None
        else:
            return ball_pos

    def get_sim_time(self) -> float:
        return self._sys.GetChTime()

    def get_step_sz(self) -> float:
        return self._time_step
