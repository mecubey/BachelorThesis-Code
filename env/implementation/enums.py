from enum import IntEnum

class States(IntEnum):
    ORIGIN = -1
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    EXECUTE_TASK = 4
    ALL_OUTGOING = 5