from pathlib import Path
import cv2 as cv
from ai_umpire import Detector

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 5
vid_dir: Path = root_dir_path / "videos"

if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    video_fname: str = f"sim_{sim_id}.mp4"
    if not (vid_dir / video_fname).exists():
        raise FileNotFoundError(f"Video file for sim ID {sim_id}not found.")

    detector = Detector(root_dir_path)
    frame_detections = detector.get_ball_detections(
        vid_fname=video_fname,
        sim_id=sim_id,
        morph_op="close",
        morph_op_iters=11,
        morph_op_se_shape=(2, 2),
        blur_kernel_size=(31, 31),
        blur_sigma=3,
        binary_thresh=130,
        struc_el_shape=cv.MORPH_RECT,
    )
