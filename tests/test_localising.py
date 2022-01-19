from pathlib import Path

import numpy as np
import pytest

from ai_umpire.localisation.localiser import Localiser

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
VID_PATH = ROOT / 'videos' / 'sim_0.mp4'


@pytest.fixture
def localiser_instance() -> Localiser:
    loc = Localiser(root=ROOT)
    return loc


def test_init(localiser_instance) -> None:
    assert localiser_instance is not None


def test_extract_frames(localiser_instance) -> None:
    assert VID_PATH.exists()
    extracted_frames: np.ndarray = localiser_instance.extract_frames(VID_PATH)
    # Should test that the number of frames extracted matches the number of frames for the matching sim id's blurred
    # frames file
    assert extracted_frames is not None
    assert extracted_frames.shape[0] > 0


def test_segment_foreground(localiser_instance) -> None:
    extracted_frames: np.ndarray = localiser_instance.extract_frames(VID_PATH)
    foreground_segmented_frames: np.ndarray = localiser_instance.segment_foreground(extracted_frames)

    assert extracted_frames is not None
    assert extracted_frames is not None
    assert foreground_segmented_frames.shape[0] == extracted_frames.shape[0] - 1


