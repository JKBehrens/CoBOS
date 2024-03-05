from typing import Optional
from control.jobs import Task
from inputs.config import (
    n,
    mean_min,
    mean_max,
    deviation_min,
    deviation_max,
    allocation_weights,
)
import numpy as np


TASK_NUM = 8
CASES_LENGTH = 16
HUMAN_TASKS = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]
ROBOT_TASKS = ["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"]
ALLOCATABLE_TASKS = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8"]
TEST_CASE_DATA = ["r4", "h7", "a1", "r5"]
TEST_CASE_CONDITIONS = [[], [], [], []]
# TEST_CASE_CONDITIONS = [[], [0], [], [0, 1]]

CONDITIONS = {
    2: [
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [],
        [],
        [],
        [],
    ],
    3: [
        [4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7],
        [8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11],
        [12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15],
        [],
        [],
        [],
        [],
        [],
    ],
    5: [
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
        [10],
        [11],
        [12],
        [13],
        [14],
        [15],
        [],
        [],
        [],
        [],
    ],
    6: [
        [4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7],
        [8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [10, 11],
        [12, 13],
        [12, 13, 14],
        [13, 14, 15],
        [14, 15],
        [],
        [],
        [],
        [],
        [],
    ],
    7: [[], [], [], [0, 1], [0, 2], [3], [4], [3, 4]]
}
X = ["A", "B", "C", "D"]
Y = ["1", "2", "3", "4"]


def set_random_sequence(
    case: int, rand_gen: np.random.Generator, length: int = CASES_LENGTH
):
    if case in [1, 2, 3]:
        weights = (0.5, 0.5, 0.0)
    else:
        weights = allocation_weights
    if case == 0:
        return TEST_CASE_DATA
    weights_for_each_task: list[float] = []
    for weight in weights:
        if weight != 0:
            for _ in range(TASK_NUM):
                weights_for_each_task.append(weight / TASK_NUM)
    if (weights[1] == 0) & (weights[2] == 0):
        cubes = HUMAN_TASKS
    elif (weights[0] == 0) & (weights[2] == 0):
        cubes = ROBOT_TASKS
    elif weights[2] == 0:
        cubes = HUMAN_TASKS + ROBOT_TASKS
    else:
        cubes = HUMAN_TASKS + ROBOT_TASKS + ALLOCATABLE_TASKS
    sequence = rand_gen.choice(
        cubes, size=length, replace=False, p=weights_for_each_task
    )
    return sequence.astype(dtype=str)


def set_distribution_parameters(
    rand_gen: np.random.Generator | None = None, seed: int | None = None
) -> tuple[tuple[int, int], tuple[int, int], tuple[float, float]]:

    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)

    mean = np.sort(rand_gen.integers(low=mean_min, high=mean_max, size=n))
    deviation = np.linspace(deviation_min, deviation_max, num=n) / 100
    scale = np.round(deviation * mean).astype(int)
    fail_prob = np.flip(np.sort(rand_gen.dirichlet(np.ones(n), size=1)))[0]

    # return tuple([tuple(mean), tuple(scale), tuple(fail_prob)])
    return tuple(mean), tuple(scale), tuple(fail_prob)


def set_rejection_prob(
    rand_gen: Optional[np.random.Generator] = None, seed: int = None
):
    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)

    return float(rand_gen.dirichlet(np.ones(2), size=1)[0][0])


def set_input(case: int, seed: int) -> list[Task]:
    rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)
    job_description = []
    ID_counter = 0
    cubes_sequence = set_random_sequence(case, rand_gen)

    for x in X:
        for y in Y:
            if case == 7 and ID_counter == 8:
                break
            task_description = {}
            task_description["id"] = ID_counter
            try:
                task_description["action"] = {"object": cubes_sequence[ID_counter]}
            except IndexError as e:
                return job_description

            task_description["universal"] = False
            if "h" in cubes_sequence[ID_counter]:
                task_description["agent"] = "Human"
            elif "r" in cubes_sequence[ID_counter]:
                task_description["agent"] = "Robot"
            else:
                task_description["agent"] = ["Human", "Robot"]
                task_description["universal"] = True
            task_description["action"]["place"] = x + y
            if case in (1, 4):
                task_description["conditions"] = []
            elif case == 0:
                task_description["conditions"] = TEST_CASE_CONDITIONS[ID_counter]
            else:
                task_description["conditions"] = CONDITIONS[case][ID_counter]

            task_description["distribution"] = [
                set_distribution_parameters(rand_gen=rand_gen) for _ in range(3)
            ]
            task_description["rejection_prob"] = set_rejection_prob(rand_gen=rand_gen)
            ID_counter += 1
            task = Task(**task_description)
            job_description.append(task)

    return job_description
