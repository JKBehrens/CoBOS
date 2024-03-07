# This module defines task execution time constants

CUBE_ARRAYS = {'Robot': {0: ['r1', 'r2', 'r3', 'r4'],
                         1: ['r5', 'r6', 'r7', 'r8'],
                         2: ['a1', 'a2', 'a3', 'a4'],
                         3: ['a5', 'a6', 'a7', 'a8']},
               'Human': {0: ['h1', 'h2', 'h3', 'h4'],
                         1: ['h5', 'h6', 'h7', 'h8'],
                         2: ['a5', 'a6', 'a7', 'a8'],
                         3: ['a1', 'a2', 'a3', 'a4']}}

GRID_ARRAYS = {0: ['A1', 'A2', 'B1', 'B2'],
               1: ['C1', 'C2', 'D1', 'D2'],
               2: ['A3', 'A4', 'B3', 'B4'],
               3: ['C3', 'C4', 'D3', 'D4'],
               4: ['A5', 'B5', 'C5', 'D5']}

GO_DOWN = 1
GO_UP = 1
CLOSE_GRIPPER = 1
OPEN_GRIPPER = 1
GRASPING = 4
RELEASE = 3
GO_OVER_CUBE = {'Robot': [3, 4, 3, 4], 'Human': [2, 3, 3, 4]}
GO_OVER_TARGET_POSITION = {'Robot': [[2, 3, 3, 4, 5],
                                     [2, 2, 2, 3, 4],
                                     [2, 2, 2, 3, 4],
                                     [2, 2, 3, 2, 4]],
                           'Human': [[3, 2, 3, 2, 4],
                                     [4, 3, 4, 3, 5],
                                     [2, 2, 3, 2, 4],
                                     [1, 2, 2, 3, 4]]}
GO_HOME = {'Robot': [2, 3, 3, 4, 5], 'Human': [3, 2, 3, 2, 4]}


def get_array_index(dictionary, position):
    for value in dictionary.values():
        if position in value:
            array_index = list(dictionary.keys())[list(dictionary.values()).index(value)]
            return array_index
    return None


def get_approximated_task_duration(agent, action):
    if ('r' in action['Object'] and agent == 'Robot') or ('h' in action['Object'] and agent == 'Human') \
            or ('a' in action['Object']):
        cube_array = get_array_index(CUBE_ARRAYS[agent], action['Object'])
        position_array = get_array_index(GRID_ARRAYS, action['Place'])
        if agent == 'Robot':
            grasping = GO_DOWN + CLOSE_GRIPPER + GO_UP
            release = GO_DOWN + OPEN_GRIPPER + GO_UP
        else:
            grasping = GRASPING
            release = RELEASE
        preparation = GO_OVER_CUBE[agent][cube_array] + grasping
        execution = GO_OVER_TARGET_POSITION[agent][cube_array][position_array] + release
        completion = GO_HOME[agent][position_array]

        return [preparation, execution, completion]
    return [0, 0, 0]

