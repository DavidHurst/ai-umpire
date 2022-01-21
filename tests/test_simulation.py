from pathlib import Path

import pytest
import pychrono as chrono

from ai_umpire.simulation.sim import Simulation

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID = 1


@pytest.fixture
def sim_instance():
    sim = Simulation(
        sim_id=SIM_ID,
        root=ROOT,
        step_sz=0.001,
        ball_origin=chrono.ChVectorD(3, 0.25, -4),
        ball_speed=chrono.ChVectorD(0, 7, 55),
        ball_acc=chrono.ChVectorD(0, 0, 0),
        ball_rot_dt=chrono.ChQuaternionD(0, 0, 0.0436194, 0.9990482),
        p1_pos_x=2,
        p1_pos_z=-3,
        p1_speed=chrono.ChVectorD(-2, 0, 3),
        p2_pos_x=0,
        p2_pos_z=-1.5,
        p2_speed=chrono.ChVectorD(2, 0, -3),
    )
    return sim


def test_init(sim_instance):
    assert sim_instance is not None
    assert sim_instance.get_sim_time() == 0.0
    assert len(sim_instance._sys.Get_bodylist()) > 0


def test_run_sim(sim_instance):
    sim_duration = 5.0
    sim_instance.run_sim(sim_duration, export=True, visualise=False)
    assert sim_duration / sim_instance.get_step_sz() == 5000
