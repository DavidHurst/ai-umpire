from pathlib import Path

import pytest
import pychrono as chrono

from ai_umpire.simulation import Simulation


@pytest.fixture
def sim_instance():
    sim_gen = Simulation(
        sim_id=0,
        root=Path("C:\\Users\\david\\Downloads").resolve(),
        out_file=Path("C:\\Users\\david\\Data").resolve(),
        step_sz=0.001,
        ball_origin=chrono.ChVectorD(-2, 1, -2),
        ball_speed=chrono.ChVectorD(5.5, 6, 70),
        ball_acc=chrono.ChVectorD(0, 2, 3),
        p1_pos_x=0.0,
        p1_pos_z=-2.5,
        p1_speed=chrono.ChVectorD(-2, 0, 4),
        p2_pos_x=1.0,
        p2_pos_z=-1.0,
        p2_speed=chrono.ChVectorD(1.5, 0, 4),
    )
    return sim_gen


def test_init(sim_instance):
    assert sim_instance is not None
    assert sim_instance.get_sim_time() == 0.0
    assert len(sim_instance._sys.Get_bodylist()) > 0


def test_run_sim(sim_instance):
    sim_duration = 1.0
    sim_instance.run_sim(sim_duration, export=True, visualise=False)
    assert sim_duration / sim_instance.get_step_sz() == 1000

