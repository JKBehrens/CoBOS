from enum import Enum, auto, IntEnum


class AgentState(Enum):
    IDLE = auto()
    REJECTION = auto()
    ACCEPTANCE = auto()
    PREPARATION = auto()
    WAITING = auto()
    EXECUTION = auto()
    COMPLETION = auto()
    DONE = auto()
    InPROGRESS = auto()
    ANSWERING = auto()


class TaskState(IntEnum):
    UNAVAILABLE = -1
    AVAILABLE = 0
    InProgress = 1
    COMPLETED = 2
