from pathlib import Path

import numpy as np
import pytest

from ai_umpire.util import (
    extract_frames_from_vid,
    blur_frames,
    binarize_frames,
    difference_frames,
    apply_morph_op,
)

ROOT_DIR = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID = 1
VID_PATH = ROOT_DIR / "videos" / f"sim_{SIM_ID}.mp4"
BLURRED_FRAMES_PATH = ROOT_DIR / "blurred_frames" / f"sim_{SIM_ID}_blurred"


def test_extract_frames() -> None:
    assert VID_PATH.exists()
    extracted_frames: np.ndarray = extract_frames_from_vid(VID_PATH)

    assert extracted_frames is not None
    assert extracted_frames.shape[0] > 0
    assert extracted_frames.shape[-1] == 3
    assert extracted_frames.dtype == np.uint8
    assert len(list(Path(BLURRED_FRAMES_PATH).iterdir())) == extracted_frames.shape[0]


def test_frame_differencing() -> None:
    extracted_frames: np.ndarray = extract_frames_from_vid(VID_PATH)
    differenced_frames: np.ndarray = difference_frames(extracted_frames)

    assert differenced_frames.shape[0] == extracted_frames.shape[0] - 2
    assert len(differenced_frames.shape) == 4  # Colour image


def test_blur_frames() -> None:
    extracted_frames: np.ndarray = extract_frames_from_vid(VID_PATH)
    blurred_frames: np.ndarray = blur_frames(extracted_frames)

    assert blurred_frames is not None
    assert blurred_frames.shape[-1] == 3
    assert blurred_frames.dtype == np.uint8
    assert extracted_frames.shape == blurred_frames.shape


def test_binarize_frames() -> None:
    extracted_frames: np.ndarray = extract_frames_from_vid(VID_PATH)
    blurred_frames: np.ndarray = blur_frames(extracted_frames)
    binary_frames: np.ndarray = binarize_frames(blurred_frames)

    assert binary_frames is not None
    assert len(binary_frames.shape) == 3  # Greyscale images
    assert blurred_frames.dtype == np.uint8


def test_norm_pdf() -> None:
    # x = np.array([[0], [0]])
    # mu = np.array([[0], [0]])
    # cov = np.eye(2)
    #
    # print(multivariate_norm_pdf(x, mu, cov) == 0.15915494309189535)
    #
    # exit()
    pass


@pytest.mark.parametrize("morph_op", ["erode", "open"])
def test_morph_op(morph_op) -> None:
    extracted_frames: np.ndarray = extract_frames_from_vid(VID_PATH)
    binary_frames: np.ndarray = binarize_frames(extracted_frames)
    morph_op_frames: np.ndarray = apply_morph_op(binary_frames, morph_op, 1)

    assert morph_op_frames.dtype == binary_frames.dtype
    assert morph_op_frames.shape == binary_frames.shape
    assert morph_op_frames.max() == binary_frames.max()
    assert morph_op_frames.min() == binary_frames.min()
