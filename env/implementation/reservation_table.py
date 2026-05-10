"""
Contains ReservationTable class.
"""

from . import header as h
from collections import defaultdict

class ReservationTable:
    """
    Represents a reservation table. 
    Used to reserve positions at specific timesteps.
    """
    def __init__(self, max_timestep: int) -> None:
        self.table: dict[h.Position, bytearray] = defaultdict(lambda:
                                                            bytearray(max_timestep + 1))

    def reservations_at(self, pos: h.Position) -> bytearray:
        """
        Returns the reservations at a specified position.

        Args:
            pos (h.Position): Specified 

        Returns:
            bytearray: Reservations, in the form a byte array.
        """
        return self.table[pos]

    def reserve_position(self, pos: h.Position, t: int):
        """Reserve a position at a specified timestep.

        Args:
            pos (h.Position): h.Position in grid.
            t (int): Timestep at which `pos` is being reserved.
        """
        assert self.table[pos][t] == 0, \
        f"{pos} is already reserved at t={t}"

        self.table[pos][t] = 1

    def clear_reserved_position(self, pos: h.Position, t: int):
        """Clear a reserved position at a specified timestep.

        Args:
            pos (h.Position): h.Position in grid.
            t (int): Timestep at which the reservation is being cleared.
        """
        assert self.table[pos][t] == 1, \
        f"{pos} is not reserved at t={t}"

        self.table[pos][t] = 0

    def reserve_interval(self, pos: h.Position, interval: h.Interval):
        """Reserve a position for a specified interval

        Args:
            pos (h.Position): h.Position in grid.
            interval (h.Interval): h.Interval to be reserved.
        """
        for t in range(interval.start_time, interval.end_time+1):
            self.reserve_position(pos, t)

    def clear_reserved_interval(self, pos: h.Position, interval: h.Interval):
        """Clear reservations of a position for a specified interval

        Args:
            pos (h.Position): h.Position in grid.
            interval (h.Interval): h.Interval to be cleared.
        """
        for t in range(interval.start_time, interval.end_time+1):
            self.clear_reserved_position(pos, t)

    def is_reserved(self, pos: h.Position, t: int) -> bool:
        """Returns true if the position is reserved for the specified timestep, false otherwise.

        Args:
            pos (h.Position): h.Position to be checked.

        Returns:
            bool: `True` if `pos` is reserved at t, `False` otherwise.
        """
        return self.table[pos][t] == 1
