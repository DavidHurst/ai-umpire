from pathlib import Path
from random import randint, uniform, choice, sample
from typing import List

import pytest
import pychrono as chrono

from ai_umpire.simulation.sim import Simulation

ROOT = Path("C:\\Users\\david\\Data\\AI Umpire DS")
SIM_ID = 5
START_X_POS: List[int] = [-2, -1, 0, 1]
START_Z_POS: List[int] = [-2, -1, 0, 1]


@pytest.fixture
def sim_instance():
    players_x: List[int] = sample(START_X_POS, 2)
    players_z: List[int] = sample(START_Z_POS, 2)
    sim = Simulation(
        sim_id=SIM_ID,
        root=ROOT,
        step_sz=0.005,
        ball_origin=chrono.ChVectorD(
            choice([-3, 3, -2.5, 2.5]), uniform(0.2, 0.8), randint(-4, -2)
        ),
        ball_speed=chrono.ChVectorD(randint(-5, 5), randint(6, 15), randint(7, 25)),
        ball_acc=chrono.ChVectorD(-1, 2, 10),
        ball_rot_dt=chrono.ChQuaternionD(0, 0, 0.0436194, 0.9990482),
        p1_pos_x=players_x[0],
        p1_pos_z=players_z[0],
        p1_speed=chrono.ChVectorD(-1, 0, 1),
        p2_pos_x=players_x[1],
        p2_pos_z=players_z[1],
        p2_speed=chrono.ChVectorD(1, 0, -1),
    )
    return sim


def test_init(sim_instance):
    assert sim_instance is not None
    assert sim_instance.get_sim_time() == 0.0
    assert len(sim_instance._sys.Get_bodylist()) > 0


def test_run_sim(sim_instance):
    sim_duration: float = 2.0
    export: bool = True
    ball_pos: List[List] = sim_instance.run_sim(
        sim_duration, export=export, visualise=False
    )
    if export:
        assert (ROOT / "generated_povray" / f"sim_{SIM_ID}_povray").exists()
    assert len(ball_pos[0]) == sim_duration / sim_instance.get_step_sz()
    assert len(ball_pos[1]) == sim_duration / sim_instance.get_step_sz()
    assert len(ball_pos[2]) == sim_duration / sim_instance.get_step_sz()
