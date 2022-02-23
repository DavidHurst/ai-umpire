from pathlib import Path
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_umpire import SimVideoGen, Localiser
from ai_umpire.util import sim_to_pixel_coord

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
sim_frames_path: Path = root_dir_path / "sim_frames" / f"sim_{sim_id}_frames"
sim_blurred_frames_path: Path = (
    root_dir_path / "blurred_frames" / f"sim_{sim_id}_blurred"
)
vid_dir: Path = root_dir_path / "videos"
sim_length = 2.0
sim_step_sz = 0.005
n_rendered_frames = int(sim_length / sim_step_sz)
desired_fps = 50
n_frames_to_avg = int(n_rendered_frames / desired_fps)
img_dims = np.reshape(np.array([1024, 768]), (2, 1))
cam_transform_mat_homog = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.75, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, -13.0, 1.0],
        ]
    )

if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    if not (vid_dir / f"sim_{sim_id}.mp4").exists():
        print(f"Generating video for sim id {sim_id}")
        vid_gen = SimVideoGen(root_dir=root_dir_path)
        vid_gen.convert_frames_to_vid(sim_id, 50)

    # Check that ball data file exisits
    data_file_path = root_dir_path / "ball_pos" / f"sim_{sim_id}.csv"
    if not data_file_path.exists():
        raise IOError("Data file not found")

    ball_pos = pd.DataFrame(pd.read_csv(data_file_path), columns=["x", "y", "z"])

    print(f"Homogenous transform mat: \n{cam_transform_mat_homog}")

    for i in range(len(ball_pos)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        im1 = cv2.imread(
            str(root_dir_path / f"test_frames/picture{str(i).zfill(3)}.png"),
            1,
        )
        im2 = cv2.imread(
            str(root_dir_path / f"test_frames/picture{str(i).zfill(3)}.png"),
            1,
        )

        ball_wc_homog = np.reshape(
            np.array([ball_pos["x"][i], ball_pos["y"][i], ball_pos["z"][i], 1]), (1, 4)
        )
        transformed_ball_wc_homog = np.dot(
            ball_wc_homog, np.linalg.inv(cam_transform_mat_homog)
        )
        transformed_ball_wc = transformed_ball_wc_homog[:, :-1]
        ball_ic_coefs = (
            np.array(
                [
                    0.5 + (transformed_ball_wc[:, 0] / transformed_ball_wc[:, -1]),
                    0.5 - (transformed_ball_wc[:, 1] / transformed_ball_wc[:, -1]),
                ]
            )
        )
        ball_ic = np.multiply(img_dims, ball_ic_coefs)

        print(f'Frame #{i}{"-" * 15}')
        print(f'Ball WC homogenous: {ball_wc_homog}')
        print(f'TFormed ball WC homogenous: {transformed_ball_wc_homog}')
        print(f'TFormed ball WC: {transformed_ball_wc}')
        print(f'Ball IC coefs:\n{ball_ic_coefs}')
        print(f'Ball IC:\n{ball_ic}')


        cv2.circle(
            im2,
            (int(ball_ic[0]), int(ball_ic[1])),
            5,
            (0, 255, 0),
            1,
        )

        axes[0].imshow(im1)
        axes[1].imshow(im2)
        axes[0].set_title(f"Frame #{i}")
        axes[1].set_title(f"Frame #{i} Ball Circled")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Generate ball candidates per frame in video
    # loc = Localiser()
    # candidates = loc.get_ball_candidates(vid_dir / f"sim_{sim_id}.mp4")
