


from enum import IntEnum


class TaskState(IntEnum):
    AVAILABLE = 0

def test_enum():
    state = TaskState.AVAILABLE

    assert state == 0
