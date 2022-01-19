__all__ = ["DataGenerator"]

import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class DataGenerator:
    def __init__(self, root_dir_path: Path):
        self.root_path: Path = root_dir_path

    def apply_motion_blur(
        self, n_frames_avg: int, sim_frames_path: Path, blurred_out_dir: Path
    ) -> None:
        logging.info("Blurring frames...")
        try:
            blurred_out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as e:
            logging.error(f"Directory {blurred_out_dir} already exists. {e}")
        else:
            logging.info(f"Directory {blurred_out_dir} created.")

        frame_paths: list = glob.glob(f"{sim_frames_path}{os.path.sep}*.png")
        template_img: np.ndarray = cv2.imread(frame_paths[0], 1)
        averaged_frames: np.ndarray = np.zeros_like(template_img, float)
        i = 0
        blurred_frame_count = 0

        for f_path in tqdm(frame_paths, desc="Processing frames"):
            frame = cv2.imread(f_path, 1)
            frame_rbg = frame[..., ::-1].copy()
            frame_arr = np.array(frame_rbg, dtype=float)
            averaged_frames += frame_arr / n_frames_avg
            i += 1

            if i == n_frames_avg:
                blurred = np.array(np.round(averaged_frames), dtype=np.uint8)
                blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

                fname = (
                    blurred_out_dir / f"frame{str(blurred_frame_count).zfill(5)}.jpg"
                )

                cv2.imwrite(str(fname), blurred)

                averaged_frames = np.zeros_like(template_img, float)
                logging.info(
                    f"Saved blurred frame #{str(blurred_frame_count).zfill(5)} to {fname}."
                )
                i = 0
                blurred_frame_count += 1

        logging.info(
            f"{blurred_frame_count} frames blurred and saved to {blurred_out_dir}."
        )

    def convert_frames_to_vid(
        self,
        vid_out_dir_path: Path,
        blurred_frames_dir_path: Path,
        sim_id: int,
        fps: int = 50,
    ) -> None:
        logging.info("Converting blurred frames to video...")
        if not blurred_frames_dir_path.exists():
            e = FileNotFoundError("Simulation frames not blurred.")
            logging.exception(e)
            raise e
        frame_paths: list = glob.glob(f"{blurred_frames_dir_path}{os.path.sep}*.jpg")
        img_array: list = [cv2.imread(f_path) for f_path in frame_paths]
        f_height: int
        f_width: int
        f_height, f_width, _ = cv2.imread(frame_paths[0]).shape
        f_size: tuple = (f_width, f_height)
        vid_fname: str = f"sim_{sim_id}.mp4"

        writer = cv2.VideoWriter(
            filename=vid_fname,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=f_size,
        )

        for i in tqdm(range(len(img_array)), desc="Processing blurred frames"):
            logging.info(f"Processing blurred frame #{i}")
            writer.write(img_array[i])
        writer.release()

        # Move video to data directory
        vid_path: Path = Path(".") / vid_fname
        vid_path.rename(vid_out_dir_path / vid_fname)

        logging.info("Converted blurred frames to video.")
