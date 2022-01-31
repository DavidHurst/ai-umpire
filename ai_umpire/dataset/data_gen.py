__all__ = ["SimVideoGen"]

import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class SimVideoGen:
    def __init__(self, root_dir: Path):
        self._root: Path = root_dir
        self._vid_dir: Path = root_dir / "videos"
        self._blurred_dir: Path = root_dir / "blurred_frames"
        self._frames_dir: Path = root_dir / "sim_frames"

    def _apply_motion_blur(self, n_frames_avg: int, sim_id: int) -> None:
        logging.info("Blurring frames...")
        blurred_out_dir: Path = self._blurred_dir / f"sim_{sim_id}_blurred"
        try:
            blurred_out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logging.error(f"Directory {blurred_out_dir} already exists. {e}")
        else:
            logging.info(f"Directory {blurred_out_dir} created.")

        sim_frames_path: Path = self._root / "sim_frames" / f"sim_{sim_id}_frames"
        frame_paths: list = glob.glob(f"{str(sim_frames_path)}{os.path.sep}*.png")
        template_img: np.ndarray = cv2.imread(frame_paths[0], 1)
        averaged_frames: np.ndarray = np.zeros_like(template_img, float)
        i: int = 0
        blurred_frame_count: int = 0

        for f_path in tqdm(frame_paths, desc="Applying motion blur"):
            frame = cv2.imread(f_path, 1)
            frame_rbg = frame[..., ::-1].copy()
            frame_arr = np.array(frame_rbg, dtype=float)
            averaged_frames += frame_arr / n_frames_avg
            i += 1

            if i == n_frames_avg:
                blurred = np.array(np.round(averaged_frames), dtype=np.uint8)
                blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

                fname = (
                    blurred_out_dir / f"frame{str(blurred_frame_count).zfill(5)}.png"
                )

                cv2.imwrite(str(fname), blurred)

                averaged_frames = np.zeros_like(template_img, float)
                logging.info(
                    f"Saved blurred frame #{str(blurred_frame_count).zfill(5)} to {fname}."
                )
                i = 0
                blurred_frame_count += 1

        logging.info(
            f"{blurred_frame_count} blurred frames  and saved to {blurred_out_dir}."
        )

    def convert_frames_to_vid(
        self,
        sim_id: int,
        desired_fps: int = 50,
    ) -> None:
        logging.info("Converting blurred frames to video...")
        sim_frames_path: Path = self._frames_dir / f"sim_{sim_id}_frames"
        if not sim_frames_path.exists():
            raise FileNotFoundError(
                f"Rendered frames from simulation {sim_id} not found."
            )

        # Apply motion blur to frames
        num_frames: int = len(glob.glob(f"{sim_frames_path}{os.path.sep}*.png"))
        self._apply_motion_blur(int(num_frames / desired_fps), sim_id)

        # Encode blurred with a .mp4 encoder
        sim_blurred_frames_path: Path = self._blurred_dir / f"sim_{sim_id}_blurred"
        frame_paths: list = glob.glob(f"{sim_blurred_frames_path}{os.path.sep}*.png")
        img_array: list = [cv2.imread(f_path) for f_path in frame_paths]
        f_height: int
        f_width: int
        f_height, f_width, _ = cv2.imread(frame_paths[0]).shape
        f_size: tuple = (f_width, f_height)
        vid_fname: str = f"sim_{sim_id}.mp4"

        writer = cv2.VideoWriter(
            filename=vid_fname,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=desired_fps,
            frameSize=f_size,
        )

        for i in tqdm(range(len(img_array)), desc="Processing blurred frames"):
            logging.info(f"Processing blurred frame #{i}")
            writer.write(img_array[i])
        writer.release()

        # Move video to data directory
        vid_path: Path = Path(".") / vid_fname
        vid_path.rename(self._vid_dir / vid_fname)

        logging.info("Converted blurred frames to video.")
