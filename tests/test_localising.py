import glob
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ai_umpire.localisation.localiser import Localiser

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
BLURRED_DIR_PATH = ROOT / "blurred_frames"
SIM_ID = 0
VID_PATH = ROOT / "videos" / f"sim_{SIM_ID}.mp4"


@pytest.fixture
def localiser_instance() -> Localiser:
    loc = Localiser(root=ROOT)
    return loc


def test_init(localiser_instance) -> None:
    assert localiser_instance is not None


def test_extract_frames(localiser_instance) -> None:
    assert VID_PATH.exists()
    extracted_frames: np.ndarray = localiser_instance.extract_frames(VID_PATH)

    assert extracted_frames is not None
    assert extracted_frames.shape[0] > 0
    assert (
            len(glob.glob(str(BLURRED_DIR_PATH / f"sim_{SIM_ID}_blurred" / "*.jpg)")))
            == extracted_frames.shape[0]
    )


def test_segment_foreground(localiser_instance) -> None:
    extracted_frames: np.ndarray = localiser_instance.extract_frames(VID_PATH)
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(
        extracted_frames
    )

    assert extracted_frames is not None
    assert extracted_frames is not None
    assert foreground_segmented_frames.shape[0] == extracted_frames.shape[0] - 1


def test_detection_blob_filter(localiser_instance) -> None:
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(
        localiser_instance.extract_frames(VID_PATH)
    )

    localiser_instance.localise_ball_blob_filter(foreground_segmented_frames)


def test_detection_hough_circle(localiser_instance) -> None:
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(
        localiser_instance.extract_frames(VID_PATH)
    )

    localiser_instance.localise_ball_hough_circle(foreground_segmented_frames)


def test_detection_hough(localiser_instance) -> None:
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(
        localiser_instance.extract_frames(VID_PATH)
    )

    localiser_instance.localise_ball_hough(foreground_segmented_frames)


@pytest.mark.parametrize("detection_method", ['log'])
def test_detection_blob(localiser_instance, detection_method) -> None:
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(
        localiser_instance.extract_frames(VID_PATH)
    )

    localiser_instance.localise_ball_blob(foreground_segmented_frames, detection_method)

