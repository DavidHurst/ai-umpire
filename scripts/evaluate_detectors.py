from pathlib import Path
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ai_umpire import SimVideoGen
from ai_umpire.util import wc_to_ic

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
    ball_x_ic, ball_y_ic = wc_to_ic(
            ball_pos["x"][i], ball_pos["y"][i], ball_pos["z"][i], [1024, 768]
        )

    # fig, axes = plt.subplots(figsize=(15, 7))
    # for i in range(len(ball_pos)):
    #     im1 = cv2.cvtColor(
    #         cv2.imread(
    #             str(root_dir_path / f"test_frames/picture{str(i).zfill(3)}.png"),
    #             cv2.IMREAD_COLOR,
    #         ),
    #         cv2.COLOR_BGR2RGB,
    #     )
    #
    #     ball_x_ic, ball_y_ic = wc_to_ic(
    #         ball_pos["x"][i], ball_pos["y"][i], ball_pos["z"][i], [1024, 768]
    #     )
    #
    #     axes.annotate(
    #         f'(x_IC, y_IC, z_WC) = ({ball_x_ic:.2f}, {ball_y_ic:.2f}, {ball_pos["z"][i]:.2f})',
    #         (ball_x_ic, ball_y_ic),
    #         xytext=(-300, 0),
    #         textcoords="offset points",
    #         bbox=dict(boxstyle="round", fc="0.7"),
    #         arrowprops=dict(arrowstyle="->", color="green", linewidth=2),
    #     )
    #
    #     axes.imshow(im1)
    #     axes.set_title(f"Frame #{i}")
    #     axes.axis("off")
    #     plt.tight_layout()
    #     plt.savefig(f"ball_true_{i}.png")
    #     plt.show()

    # Generate ball candidates per frame in video
    # loc = Localiser()
    # candidates = loc.get_ball_candidates(vid_dir / f"sim_{sim_id}.mp4")
