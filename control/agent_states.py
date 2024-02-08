from enum import Enum, auto


class AgentState(Enum):
    IDLE = auto()
    REJECT = auto()
    PREPARATION = auto()
    WAITING = auto()
    EXECUTION = auto()
    COMPLETION = auto()
    DONE = auto()
    InPROGRESS = auto()
    ANSWERING = auto()
