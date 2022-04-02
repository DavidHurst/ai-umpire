from pathlib import Path

import pytest

from ai_umpire.tracking.kalman import Tracker

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID = 1


@pytest.fixture
def tracker_instance():
    tracker: Tracker = Tracker()
    return tracker


def test_init(tracker_instance):
    assert tracker_instance is not None
