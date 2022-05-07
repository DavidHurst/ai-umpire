import glob
import os
from pathlib import Path

import pytest

from ai_umpire import VideoGenerator

SIM_ID = 5
ROOT_DIR = Path("C:\\Users\\david\\Data\\AI Umpire DS")
VID_DIR = ROOT_DIR / "videos"
SIM_FRAMES_PATH = ROOT_DIR / "sim_frames" / f"sim_{SIM_ID}_frames"
SIM_BLURRED_FRAMES_PATH = ROOT_DIR / "blurred_frames" / f"sim_{SIM_ID}_blurred"


@pytest.fixture
def data_gen_instance() -> VideoGenerator:
    data_gen = VideoGenerator(root_dir=ROOT_DIR)
    yield data_gen

    # Teardown
    # if SIM_BLURRED_FRAMES_PATH.exists():
    #     shutil.rmtree(SIM_BLURRED_FRAMES_PATH)
    # vid_path: Path = VID_DIR / f"sim_{SIM_ID}.mp4"
    # if vid_path.exists():
    #     vid_path.unlink()


def test_init(data_gen_instance) -> None:
    assert data_gen_instance is not None


def test_apply_motion_blur(data_gen_instance) -> None:
    assert not SIM_BLURRED_FRAMES_PATH.exists()
    n_frames_to_avg: int = 8
    data_gen_instance._apply_motion_blur(
        n_frames_avg=n_frames_to_avg,
        sim_id=SIM_ID,
    )

    n_frames: int = len(glob.glob(f"{SIM_FRAMES_PATH}{os.path.sep}*.png"))
    expected_n_blurred_frames: int = int(n_frames / n_frames_to_avg)
    assert (
        len(glob.glob(f"{SIM_BLURRED_FRAMES_PATH}{os.path.sep}*.png"))
        == expected_n_blurred_frames
    )


def test_vid_conversion(data_gen_instance) -> None:
    vid_count_prior: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))
    data_gen_instance.convert_frames_to_vid(sim_id=SIM_ID, desired_fps=50)
    vid_count_post: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))

    assert vid_count_post == vid_count_prior + 1
