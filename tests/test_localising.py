from pathlib import Path

import numpy as np
import pytest

from ai_umpire import Localiser

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID = 5
VID_PATH = ROOT / "videos" / f"sim_{SIM_ID}.mp4"


@pytest.fixture
def localiser_instance() -> Localiser:
    loc = Localiser()
    return loc


def test_init(localiser_instance) -> None:
    assert localiser_instance is not None


@pytest.mark.parametrize("method", ["log"]) #["log", "dog", "blob_filter", "hough_circle"])
def test_get_candidates(localiser_instance, method):
    candidates: np.ndarray = localiser_instance.get_ball_candidates(
        vid_path=VID_PATH,
        morph_op="open",
        detection_method=method,
        morph_op_iters=2,
        morph_op_SE_shape=(15, 15),
        blur_kernel_size=(51, 51),
        blur_sigma_x=4,
        binary_thresh_low=245
    )

    assert candidates is not None

