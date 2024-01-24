from control.jobs import Job
from simulation.sim import Sim
from scheduling.scheduling_split_tasks import Schedule
import numpy as np


def test_randon_number_gen():
    task_case = '1'
    seed = 7
    agent = 'Human'
    fail_prob = [0.1, 0.9]
    second_mode = [3, 3]
    scale = 2

    job = Job(task_case)

    h_agent_1 = Sim(agent, job, seed=seed, fail_prob=fail_prob, second_mode=second_mode, scale=scale)
    h_agent_2 = Sim(agent, job, seed=seed, fail_prob=fail_prob, second_mode=second_mode, scale=scale)
    array1 = np.array(list(h_agent_1.task_duration.values()), dtype=float)
    array2 = np.array(list(h_agent_2.task_duration.values()), dtype=float)

    assert np.allclose(array1, array2)

    schedule_model = Schedule(job)
    schedule = schedule_model.set_schedule(seed=seed, fail_prob=fail_prob, second_mode=second_mode, scale=scale)

    assert np.allclose(array1, schedule_model.task_duration[agent])

    schedule_model = Schedule(job)
    schedule = schedule_model.set_schedule(seed=seed+1, fail_prob=fail_prob, second_mode=second_mode, scale=scale)

    assert not np.allclose(array1, schedule_model.task_duration[agent])



