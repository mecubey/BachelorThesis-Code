"""
Contains ReservationTable class.
"""

from . import header as h
from .reservation_table import ReservationTable

class SafeIntervalTable:
    """
    Represents a safe interval table. 
    Used to calculate safe intervals given reservations.
    """
    def __init__(self, max_timestep: int, reservations: ReservationTable) -> None:
        self.max_timestep: int = max_timestep
        self.reservations: ReservationTable = reservations
        self.table: dict[h.Position, list[h.Interval]] = {}

    def get_safe_intervals_at(self, pos: h.Position) -> list[h.Interval]:
        """
        Gets the list of safe intervals at a specified position.

        Args:
            pos (h.Position): Specified position.

        Returns:
            list[h.Interval]: Safe intervals at that position.
        """
        return self.table.get(pos, [h.Interval(0, self.max_timestep)])

    def build_safe_intervals_at(self, pos: h.Position) -> None:
        """
        Builds list of safe intervals at the specified position.

        Args:
            pos (h.Position): h.Position for which to build intervals.
            timeline (bytearray): Reservation timeline at that position.
        """
        intervals: list[h.Interval] = []

        start = None

        for t, occupied in enumerate(self.reservations.reservations_at(pos)):

            # start of safe interval
            if not occupied and start is None:
                start = t

            # end of safe interval
            elif occupied and start is not None:
                intervals.append(h.Interval(start, t-1))
                start = None

        # timeline ended while still inside safe interval
        if start is not None:
            intervals.append(h.Interval(start, self.max_timestep))

        self.table[pos] = intervals

    def build_entire_table(self) -> None:
        """
        Build all safe intervals for each position in the given
        reservation table.

        Args:
            reservation_table (ReservationTable): Reservation table used to build safe intervals.
        """
        for pos in self.reservations.table.keys():
            self.build_safe_intervals_at(pos)
