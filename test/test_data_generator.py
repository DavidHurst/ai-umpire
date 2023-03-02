import pytest
from pathlib import Path

from ai_umpire import VideoGenerator

DATA_DIR_PATH = str(Path.cwd() / "data")


@pytest.fixture
def data_gen_instance() -> VideoGenerator:
    assert any(Path(DATA_DIR_PATH).iterdir()) is False  # Empty dir
    data_gen = VideoGenerator(DATA_DIR_PATH)
    yield data_gen


def test_gen_single_video(data_gen_instance) -> None:
    data_gen_instance.generate_video(0)

    with pytest.raises(IndexError) as excinfo:
        # Test idx OOB
        for i in range(4):
            data_gen_instance.generate_video(i)


def test_gen_all_videos(data_gen_instance) -> None:
    data_gen_instance.generate_all_videos()


# def test_init(data_gen_instance) -> None:
#     assert data_gen_instance is not None
#     with pytest.raises(FileNotFoundError):
#         read_yaml(file_path="source/data/non_existing_file.yaml")
#
#     with pytest.raises(yaml.scanner.ScannerError):
#         # only show the first error
#         read_yaml(file_path="source/data/sample_invalid.yaml")
#
#     with pytest.raises(yaml.parser.ParserError):
#         # only show the first error
#         read_yaml(file_path="source/data/sample_invalid.json")
#
# def test_apply_motion_blur(data_gen_instance) -> None:
#     assert not SIM_BLURRED_FRAMES_PATH.exists()
#     n_frames_to_avg: int = 8
#     data_gen_instance._apply_motion_blur(
#         n_frames_avg=n_frames_to_avg,
#         sim_id=SIM_ID,
#     )
#
#     n_frames: int = len(glob.glob(f"{SIM_FRAMES_PATH}{os.path.sep}*.png"))
#     expected_n_blurred_frames: int = int(n_frames / n_frames_to_avg)
#     assert (
#         len(glob.glob(f"{SIM_BLURRED_FRAMES_PATH}{os.path.sep}*.png"))
#         == expected_n_blurred_frames
#     )
#
#
# def test_vid_conversion(data_gen_instance) -> None:
#     vid_count_prior: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))
#     data_gen_instance.convert_frames_to_vid(sim_id=SIM_ID, desired_fps=50)
#     vid_count_post: int = len(glob.glob(f"{VID_DIR}{os.path.sep}*.mp4"))
#
#     assert vid_count_post == vid_count_prior + 1
