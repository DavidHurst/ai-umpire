from pathlib import Path
import cv2 as cv
from ai_umpire import BallDetector

root_dir_path = Path("C:\\Users\\david\\Data\\AI Umpire DS")
sim_id = 0
vid_dir: Path = root_dir_path / "videos"

if __name__ == "__main__":
    # Generate video from simulation frames if it does not already exist
    video_fname: str = f"sim_{sim_id}.mp4"
    if not (vid_dir / video_fname).exists():
        raise FileNotFoundError(f"Video file for sim ID {sim_id}not found.")

    detector = BallDetector(root_dir_path)
    # all_detections = detector.get_ball_detections(
    #     vid_fname=video_fname,
    #     sim_id=sim_id,
    #     morph_op="close",
    #     morph_op_iters=5,
    #     morph_op_se_shape=(14, 14),
    #     blur_kernel_size=(41, 41),
    #     blur_sigma=3,
    #     binary_thresh=120,
    #     struc_el_shape=cv.MORPH_RECT,
    # )
    filtered_dets = detector.get_filtered_ball_detections(
        vid_fname=video_fname,
        sim_id=sim_id,
        morph_op="close",
        morph_op_iters=1,
        morph_op_se_shape=(4, 4),
        blur_kernel_size=(11, 11),
        blur_sigma=4,
        binary_thresh=130,
        struc_el_shape=cv.MORPH_RECT,
        min_ball_travel_dist=1,
        max_ball_travel_dist=70,
        min_det_area=1,
        max_det_area=30,
        init_ball_pos=(310, 400)
    )
    print(filtered_dets)




