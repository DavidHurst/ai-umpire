import glob
import shutil
import os
from pathlib import Path

import pytest

from ai_umpire.dataset.data_gen import DataGenerator

ROOT_DIR = Path("C:\\Users\\david\\Data\\AI Umpire DS")
VID_DIR = ROOT_DIR / "videos"
SIM_ID = 0
SIM_FRAMES_PATH = ROOT_DIR / "sim_frames" / f"sim_{SIM_ID}_frames"
SIM_BLURRED_FRAMES_PATH = ROOT_DIR / "blurred_frames" / f"sim_{SIM_ID}_blurred"


@pytest.fixture
def data_gen_instance() -> DataGenerator:
    data_gen = DataGenerator(root_dir_path=ROOT_DIR)
    yield data_gen

    # Teardown
    if SIM_BLURRED_FRAMES_PATH.exists():
        shutil.rmtree(SIM_BLURRED_FRAMES_PATH)
    vid_path: Path = VID_DIR / f'sim{SIM_ID}.mp4'
    if vid_path.exists():
        vid_path.unlink()


def test_init(data_gen_instance) -> None:
    assert data_gen_instance is not None


def test_blurring(data_gen_instance) -> None:
    n_frames_to_avg: int = 50
    data_gen_instance.apply_motion_blur(
        n_frames_avg=n_frames_to_avg,
        sim_frames_path=SIM_FRAMES_PATH,
        blurred_out_dir=SIM_BLURRED_FRAMES_PATH,
    )

    n_frames: int = len(glob.glob(f"{SIM_FRAMES_PATH}{os.path.sep}*.png"))
    expected_n_blurred_frames: int = int(n_frames / n_frames_to_avg)
    assert len(glob.glob(f"{SIM_BLURRED_FRAMES_PATH}{os.path.sep}*.jpg")) == expected_n_blurred_frames


def test_vid_conversion(data_gen_instance) -> None:
    data_gen_instance.apply_motion_blur(
        n_frames_avg=50,
        sim_frames_path=SIM_FRAMES_PATH,
        blurred_out_dir=SIM_BLURRED_FRAMES_PATH,
    )

    vid_count_prior: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))
    data_gen_instance.convert_frames_to_vid(
        vid_out_dir_path=VID_DIR,
        blurred_frames_dir_path=SIM_BLURRED_FRAMES_PATH,
        sim_id=0
    )
    vid_count_post: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))

    assert vid_count_post == vid_count_prior + 1
