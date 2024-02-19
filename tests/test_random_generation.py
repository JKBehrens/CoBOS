from control.jobs import Job
from simulation.sim import Sim
from typing import Optional
from methods.scheduling_split_tasks import Schedule
import numpy as np


def test_randon_number_gen():
    task_case = '1'
    seed = 7

    job = Job(task_case, seed=seed)

    h_agent_1 = Sim("Human", job, seed=seed)
    h_agent_2 = Sim("Robot", job, seed=seed)

    schedule_model = Schedule(job, seed=seed)

    array1 = np.array(list(schedule_model.task_duration['Human'].values()), dtype=float)
    array2 = np.array(list(h_agent_1.task_duration['Human'].values()), dtype=float)
    assert np.allclose(array1, array2)

    array3 = np.array(list(schedule_model.task_duration['Robot'].values()), dtype=float)
    array4 = np.array(list(h_agent_2.task_duration['Robot'].values()), dtype=float)
    assert np.allclose(array3, array4)


    # schedule = schedule_model.set_schedule(seed=seed, fail_prob=fail_prob, second_mode=second_mode, scale=scale)
    #
    # assert np.allclose(array1, schedule_model.task_duration[agent])
    #
    # schedule_model = Schedule(job, seed=seed+1)
    # schedule = schedule_model.set_schedule(seed=seed+1, fail_prob=fail_prob, second_mode=second_mode, scale=scale)
    #
    # assert not np.allclose(array1, schedule_model.task_duration[agent])


def get_random_number(rand_gen:Optional[np.random.Generator]=None, seed=None):
    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)

    return [rand_gen.random(size=1), rand_gen.random(size=1)]


def some_function_with_using_random(rand_gen:Optional[np.random.Generator], seed):
    rand_gen.random(size=1)
    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)
    return get_random_number(rand_gen)


def some_function_without_using_random(rand_gen: Optional[np.random.Generator]):
    return get_random_number(rand_gen)


def test_random_number_in_function():
    seed = 8
    rand_gen = np.random.default_rng(seed)
    assert not np.allclose(get_random_number(rand_gen), get_random_number(rand_gen))
    assert np.allclose(get_random_number(seed=8), get_random_number(seed=8))

    rand_gen = np.random.default_rng(seed)
    a = get_random_number(rand_gen)
    assert not np.allclose(a[0], a[1])
    b = get_random_number(rand_gen)
    assert not np.allclose(b[0], b[1])
    assert not np.allclose(a[0], b[0])
    assert not np.allclose(a[1], b[1])

    rand_gen = np.random.default_rng(seed)
    a = get_random_number(rand_gen)
    rand_gen = np.random.default_rng(seed)
    b = some_function_without_using_random(rand_gen)
    assert np.allclose(a, b)

    rand_gen = np.random.default_rng(seed)
    a = get_random_number(rand_gen)
    rand_gen = np.random.default_rng(seed)
    b = some_function_with_using_random(rand_gen, seed)
    assert not np.allclose(a, b)

    values_1 = []
    rand_gen = np.random.default_rng(seed)
    for i in range(5):
        values_1.append(get_random_number(rand_gen))

    values_2 = []
    rand_gen = np.random.default_rng(seed)
    for i in range(5):
        values_2.append(get_random_number(rand_gen))

    assert np.allclose(values_1[0], values_2[0])
    assert np.allclose(values_1[4], values_2[4])


def test_schedule():
    task_case = '5'
    seed = 7

    job = Job(task_case, seed=seed)

    schedule_model1 = Schedule(job, seed=seed)
    output1 = schedule_model1.set_schedule()

    schedule_model2 = Schedule(job, seed=seed)
    output2 = schedule_model2.set_schedule()

    assert output1["Human"][0].id == output2["Human"][0].id
