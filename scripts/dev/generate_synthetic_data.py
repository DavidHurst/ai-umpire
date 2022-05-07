from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
import pandas as pd
import pychrono as chrono
from matplotlib import pyplot as plt

from ai_umpire.util import wc_to_ic

ROOT_DIR_PATH: Path = Path() / "data"
SIM_LENGTH: float = 2.0
SIM_STEP_SIZE: float = 0.005
N_RENDERED_IMAGES: int = int(SIM_LENGTH / SIM_STEP_SIZE)
DESIRED_FPS: int = 50
N_FRAMES_TO_AVERAGE: int = int(N_RENDERED_IMAGES / DESIRED_FPS)
START_X_POS: List[int] = [-2, -1, 0, 1]
START_Z_POS: List[int] = [-2, -1, 0, 1]

if __name__ == "__main__":
    # Define starting positions and velocities of players and ball
    ball_start_positions = [
        chrono.ChVectorD(-1, 0.5, -4.5),
        chrono.ChVectorD(2.8, 0.2, -4),
    ]
    ball_start_velocities = [chrono.ChVectorD(1, 9, 14), chrono.ChVectorD(-5, 8, 20)]
    ball_start_accelerations = [chrono.ChVectorD(-1, 2, 3), chrono.ChVectorD(-2, 4, 5)]

    p1_start_positions = [(-2.5, 1.0), (2.0, -4.5)]
    p1_start_velocities = [chrono.ChVectorD(1, 0, -1.5), chrono.ChVectorD(-2.5, 0, 2)]

    p2_start_positions = [(2.5, -4.0), (0, -1.5)]
    p2_start_velocities = [
        chrono.ChVectorD(-1, 0, 1.5),
        chrono.ChVectorD(-2.0, 0.0, -2),
    ]

    # Run simulations
    # for i in range(1, 2):
    #     # Check no simulation data has be stored for this sim id
    #     pov_data_file = ROOT_DIR_PATH / "generated_povray" / f"sim_{i}_povray"
    #     if pov_data_file.exists():
    #         raise FileExistsError(
    #             f"Simulation ID {i} already has simulation data generated."
    #         )
    #
    #     # Run simulation
    #     sim = MatchSimulator(
    #         sim_id=i,
    #         root=ROOT_DIR_PATH,
    #         sim_step_sz=SIM_STEP_SIZE,
    #         ball_init_pos=ball_start_positions[i],
    #         ball_vel=ball_start_velocities[i],
    #         ball_acc=ball_start_accelerations[i],
    #         ball_rot_dt=chrono.ChQuaternionD(0, 0, 0.0436194, 0.9990482),
    #         p1_init_x=p1_start_positions[i][0],
    #         p1_init_z=p1_start_positions[i][1],
    #         p1_vel=p1_start_velocities[i],
    #         p2_init_x=p2_start_positions[i][0],
    #         p2_init_z=p2_start_positions[i][1],
    #         p2_vel=p2_start_velocities[i],
    #         output_res=(1280, 720),
    #     )
    #     sim.run_sim(SIM_LENGTH) #, export=False, visualise=True)

    # Render videos using simulation data
    # for i in range(1):
    #     video_fname = f"sim_{i}.mp4"
    #     video_file = ROOT_DIR_PATH / "videos" / video_fname
    #     # Check no video has been generated for this sim id
    #     if video_file.exists():
    #         raise FileExistsError(f"Video already encoded for Simulation ID {i}.")
    #
    #     # Generate video from images rendered by POV Ray of
    #     vid_gen = VideoGenerator(root_dir=ROOT_DIR_PATH)
    #     vid_gen.convert_frames_to_vid(i, DESIRED_FPS)

    # Visually verify correct generation, plotted ball position should align with ball in frame
    for i in range(1):
        # Check that ball data file exists
        data_file_path = ROOT_DIR_PATH / "ball_pos" / f"sim_{i}.csv"
        if not data_file_path.exists():
            raise IOError("Data file not found")

        # Load true ball positions at each state
        ball_pos_WC = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])

        # Convert state positions to blurred positions, taking the leading position in the sliding window i.e. the
        # leading point of the streak
        ball_pos_blurred_WC = ball_pos_WC.iloc[
            N_FRAMES_TO_AVERAGE::N_FRAMES_TO_AVERAGE, :
        ].reset_index(drop=True)

        for j in range(len(ball_pos_blurred_WC["x"])):
            frame_ball_pos = np.array(
                [
                    ball_pos_blurred_WC["x"][j],
                    ball_pos_blurred_WC["y"][j],
                    ball_pos_blurred_WC["z"][j],
                ]
            )

            frame = cv.imread(
                str(
                    ROOT_DIR_PATH
                    / "frames"
                    / f"sim_{i}"
                    / f"frame{str(j).zfill(5)}.jpg"
                )
            )
            frame_dims = list(frame.shape[:2])
            ball_x_ic, ball_y_ic = wc_to_ic(
                frame_ball_pos,
                frame_dims,
            )

            plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            plt.scatter(ball_x_ic, ball_y_ic, label="Ball", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.axis("off")
            plt.show()
