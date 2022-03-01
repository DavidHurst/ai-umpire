from pathlib import Path
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
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

    # avg_indices: List[int] = [
    #     _
    #     for _ in range(
    #         0, (n_frames_to_avg * desired_fps) + n_frames_to_avg, n_frames_to_avg
    #     )
    # ]
    # averaged_positions: List[List] = []
    # for i in range(len(avg_indices) - 1):
    #     start = avg_indices[i]
    #     end = avg_indices[i + 1] - 1
    #     averaged_positions.append(list(ball_pos[start:end].mean(axis=0)))
    #
    # averaged_positions_df = pd.DataFrame(averaged_positions, columns=list("xyz"))

    # for i in range(len(averaged_positions_df)):
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #     im1 = cv2.imread(
    #         str(sim_blurred_frames_path / f"frame{str(i).zfill(5)}.png"),
    #         cv2.COLOR_BGR2RGB,
    #     )
    #     im2 = cv2.imread(
    #         str(sim_blurred_frames_path / f"frame{str(i).zfill(5)}.png"),
    #         cv2.COLOR_BGR2RGB,
    #     )
    #
    #     cv2.circle(
    #         im2,
    #         sim_to_pixel_coord(
    #             averaged_positions_df["x"][i], averaged_positions_df["y"][i]
    #         ),
    #         4,
    #         (0, 255, 0),
    #         1,
    #     )
    #
    #     axes[0].imshow(im1)
    #     axes[1].imshow(im2)
    #     axes[0].set_title("Frame")
    #     axes[1].set_title("Ball True")
    #     for ax in axes:
    #         ax.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    for i in range(len(ball_pos)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        im1 = cv2.imread(
            str(sim_frames_path / f"picture{str(i).zfill(3)}.png"),
            cv2.COLOR_BGR2RGB,
        )
        im2 = cv2.imread(
            str(sim_frames_path / f"picture{str(i).zfill(3)}.png"),
            cv2.COLOR_BGR2RGB,
        )

        ball_sim_x: int = ball_pos["x"][i] / ball_pos["z"][i]
        ball_sim_y: int = ball_pos["y"][i] / ball_pos["z"][i]

        print(f'Converted coord: {sim_to_pixel_coord(ball_sim_x, ball_sim_y)}')

        cv2.circle(
            im2,
            sim_to_pixel_coord(ball_sim_x, ball_sim_y),
            5,
            (0, 255, 0),
            1,
        )

        axes[0].imshow(im1)
        axes[1].imshow(im2)
        axes[0].set_title(f"Frame #{i}")
        axes[1].set_title("Frame #{i} Ball Circled")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Generate ball candidates per frame in video
    # loc = Localiser()
    # candidates = loc.get_ball_candidates(vid_dir / f"sim_{sim_id}.mp4")
