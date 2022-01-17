import pytest

from ai_umpire.simulation import SimGenerator


@pytest.fixture
def sim_instance():
    sim_gen = SimGenerator()
    return sim_gen


def test_init(sim_instance):
    assert sim_instance is not None
