"""
Contains implementations of low level nodes and high level nodes
for LaCAM.
"""

from dataclasses import dataclass, field
from collections import deque
from ..header import Position, Config

@dataclass
class LowLevelNode:
    """Low level node in LaCAM. Contains a constraint tree."""
    who: list[int] = field(default_factory=lambda: [])
    where: list[Position] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Position) -> LowLevelNode:
        """Gets child of this low level node."""
        return LowLevelNode(who=self.who + [who],
                            where=self.where + [where],
                            depth=self.depth + 1)


@dataclass
class HighLevelNode:
    """Represents high level node in LaCAM"""
    q: Config
    order: list[int]
    parent: HighLevelNode | None = None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: float = 0
    h: float = 0
    f: float = field(init=False)
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __post_init__(self) -> None:
        self.f = self.g + self.h

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HighLevelNode):
            return self.q == other.q
        return False

    def __hash__(self) -> int:
        return self.q.__hash__()
