import glob
from pathlib import Path

import pytest

from ai_umpire.simulation.sim_gen import SimGenerator


@pytest.fixture
def sim_gen_instance() -> SimGenerator:
    sim_gen = SimGenerator(frames_file=Path("C:\\Users\\david\\Data\\sim_0_frames"))
    return sim_gen


def test_init(sim_gen_instance) -> None:
    assert sim_gen_instance is not None


def test_conversion(sim_gen_instance) -> None:
    out_file: Path = Path("C:\\Users\\david\\Data") / 'Videos' / 'sim_0.mp4'

    vid_count_prior: int = len(glob.glob(f'{out_file}/*.mp4'))
    sim_gen_instance.convert_frames_to_vid(out_file)
    vid_count_post: int = len(glob.glob(f'{out_file}/*.mp4'))

    assert vid_count_post == vid_count_prior + 1
