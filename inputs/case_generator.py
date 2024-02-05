from simulation import sim_param_path
from typing import Optional
from inputs.config import n, mean_min, mean_max, deviation_min, deviation_max, allocation_weights
import numpy as np
import json


TASK_NUM = 8
CASES_LENGTH = 16
HUMAN_TASKS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
ROBOT_TASKS = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8']
ALLOCABLE_TASKS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8']

CONDITIONS = {'2': [[4], [5], [6], [7],
                    [8], [9], [10], [11],
                    [12], [13], [14], [15],
                    [], [], [], []],
              '3': [[4, 5], [4, 5, 6], [5, 6, 7], [6, 7],
                    [8, 9], [8, 9, 10], [9, 10, 11], [10, 11],
                    [12, 13], [12, 13, 14], [13, 14, 15], [14, 15],
                    [], [], [], [], []],
              '5':  [[4], [5], [6], [7],
                     [8], [9], [10], [11],
                     [12], [13], [14], [15],
                     [], [], [], []],
              '6': [[4, 5], [4, 5, 6], [5, 6, 7], [6, 7],
                    [8, 9], [8, 9, 10], [9, 10, 11], [10, 11],
                    [12, 13], [12, 13, 14], [13, 14, 15], [14, 15],
                    [], [], [], [], []]}
X = ['A', 'B', 'C', 'D']
Y = ['1', '2', '3', '4']


def set_random_sequence(case, rand_gen:Optional[np.random.Generator], length=CASES_LENGTH):
    if case in ['1', '2', '3']:
        weights = (0.5, 0.5, 0)
    else:
        weights = allocation_weights
    weights_for_each_task = []
    for weight in weights:
        if weight != 0:
            for i in range(TASK_NUM):
                weights_for_each_task.append(weight / TASK_NUM)
    if (weights[1] == 0) & (weights[2] == 0):
        cubes = HUMAN_TASKS
    elif (weights[0] == 0) & (weights[2] == 0):
        cubes = ROBOT_TASKS
    elif weights[2] == 0:
        cubes = HUMAN_TASKS + ROBOT_TASKS
    else:
        cubes = HUMAN_TASKS + ROBOT_TASKS + ALLOCABLE_TASKS
    sequence = rand_gen.choice(cubes, size=length, replace=False, p=weights_for_each_task)
    return sequence.astype(dtype=str)


def set_distribution_parameters(rand_gen:Optional[np.random.Generator]=None):

    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)

    mean = np.sort(rand_gen.integers(mean_min, mean_max + 1, size=n))
    deviation = np.linspace(deviation_min, deviation_max, num=n)/100
    scale = np.round(deviation*mean)
    fail_prob = np.flip(np.sort(rand_gen.dirichlet(np.ones(n), size=1)))[0]

    return [mean, scale, fail_prob]


def set_rejection_prob(rand_gen:Optional[np.random.Generator]=None):
    if rand_gen is None:
        rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)

    return rand_gen.dirichlet(np.ones(2), size=1)[0][0]


def set_input(case, seed):
    rand_gen = np.random.default_rng(seed)
    assert isinstance(rand_gen, np.random.Generator)
    job_description = []
    ID_counter = 0
    cubes_sequence = set_random_sequence(case, rand_gen)

    for x in X:
        for y in Y:
            task_description = {}
            task_description['ID'] = ID_counter
            task_description['Object'] = cubes_sequence[ID_counter]
            if 'h' in cubes_sequence[ID_counter]:
                task_description['Agent'] = 'Human'
            elif 'r' in cubes_sequence[ID_counter]:
                task_description['Agent'] = 'Robot'
            else:
                task_description['Agent'] = 'Both'
            task_description['Place'] = x + y
            if case in ('1', '4'):
                task_description['Conditions'] = []
            else:
                task_description['Conditions'] = CONDITIONS[case][ID_counter]

            task_description['Distribution'] = [set_distribution_parameters(np.random.default_rng(seed+ID_counter+i))
                                                for i in range(3)]
            task_description['Rejection_prob'] = set_rejection_prob(np.random.default_rng(seed+ID_counter))
            ID_counter += 1
            job_description.append(task_description)
    return job_description



