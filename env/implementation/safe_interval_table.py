"""
Contains ReservationTable class.
"""

from . import header as h
from collections import defaultdict

class SafeIntervalTable:
    """
    Represents a safe interval table. 
    Used to calculate safe intervals given reservations.
    """
    def __init__(self, max_timestep: int) -> None:
        self.table: dict[h.Position, list[h.Interval]] = defaultdict(lambda:
                                                                     [h.Interval(0, max_timestep)])

    def get_safe_intervals_at(self, pos: h.Position) -> list[h.Interval]:
        """
        Gets the list of safe intervals at a specified position.

        Args:
            pos (h.Position): Specified position.

        Returns:
            list[h.Interval]: Safe intervals at that position.
        """
        return self.table[pos]

    def set_safe_intervals_cleared_timestep(self, pos: h.Position, t: int):
        """
        Sets the list of safe interval at the specified position according
        to the newly cleared timestep.

        Args:
            pos (h.Position): Specified position.
            t (int): Timestep which was cleared.
        """
        # TODO: filter out [a, a] intervals
        all_intervals = self.get_safe_intervals_at(pos)

        # intervals list is empty
        if not all_intervals:
            all_intervals.append(h.Interval(t, t))
            return

        # before first interval
        if t < all_intervals[0].start_time-1:
            all_intervals.insert(0, h.Interval(t, t))
            return

        # extend left side of last interval
        if t == all_intervals[-1].start_time-1:
            all_intervals[-1] = h.Interval(all_intervals[-1].start_time-1,
                                           all_intervals[-1].end_time)
            return

        # after last interval
        if t > all_intervals[-1].end_time+1:
            all_intervals.append(h.Interval(t, t))
            return

        for i in range(len(all_intervals)-1):
            # between this and next interval
            if t > all_intervals[i].end_time+1 and \
               t < all_intervals[i+1].start_time-1:
                all_intervals.insert(i+1, h.Interval(t, t))
                break

            # extend left side
            elif t == all_intervals[i].start_time-1:
                all_intervals[i] = h.Interval(all_intervals[i].start_time-1,
                                              all_intervals[i].end_time)
                break

            # extend right side
            elif t == all_intervals[i].end_time+1:
                all_intervals[i] = h.Interval(all_intervals[i].start_time,
                                              all_intervals[i].end_time+1)
                break

    def set_safe_intervals_cleared_interval(self, pos: h.Position, cleared_interval: h.Interval):
        """
        Sets the list of safe interval at the specified position according
        to the newly cleared interval.

        Args:
            pos (h.Position): Specified position.
            cleared_interval (h.Interval): Interval which was cleared.
        """
        # TODO: filter out [a, a] intervals
        all_intervals = self.get_safe_intervals_at(pos)

        # intervals list is empty
        if not all_intervals:
            all_intervals.append(cleared_interval)
            return

        # before first interval
        if cleared_interval.start_time < all_intervals[0].start_time-1 and \
           cleared_interval.end_time < all_intervals[0].start_time-1:
            all_intervals.insert(0, cleared_interval)
            return

        # extend left side of last interval
        if cleared_interval.end_time == all_intervals[-1].start_time-1 and \
           cleared_interval.start_time > all_intervals[-2].end_time+1:
            all_intervals[-1] = h.Interval(cleared_interval.start_time,
                                           all_intervals[-1].end_time)

        # after last interval
        if cleared_interval.start_time > all_intervals[-1].end_time+1 and \
           cleared_interval.end_time > all_intervals[-1].end_time+1:
            all_intervals.append(cleared_interval)
            return

        result: list[h.Interval] = []
        i: int = 0
        n = len(all_intervals)

        while i < n:
            # bridge two intervals
            if all_intervals[i].end_time+1 == cleared_interval.start_time and i < n-1 and \
               all_intervals[i+1].start_time-1 == cleared_interval.end_time:
                result.append(h.Interval(all_intervals[i].start_time, all_intervals[i+1].end_time))
                i += 2

            # extend to the right
            elif all_intervals[i].end_time+1 == cleared_interval.start_time:
                result.append(h.Interval(all_intervals[i].start_time, cleared_interval.end_time))
                i += 1

            # extend to the left
            elif all_intervals[i].start_time-1 == cleared_interval.end_time:
                result.append(h.Interval(cleared_interval.start_time, all_intervals[i].end_time))
                i += 1

            else:
                result.append(all_intervals[i])

                # between this and the next interval
                if cleared_interval.start_time > all_intervals[i].end_time+1 and i < n-1 and \
                   cleared_interval.end_time < all_intervals[i+1].start_time-1:
                    result.append(cleared_interval)
                i += 1

        self.table[pos] = result

    def set_safe_intervals_reserved_timestep(self, pos: h.Position, t: int):
        """
        Given a reserved timestep t and specified position,
        splits the safe interval list accordingly.

        Args:
            pos (h.Position): Specified position.
            t (int): Reserved timestep.
        """
        result: list[h.Interval] = []

        # TODO: filter out [a, a] intervals
        for interval in self.get_safe_intervals_at(pos):
            if interval.contains_timestep(t):
                result.append(h.Interval(interval.start_time, t-1))
                result.append(h.Interval(t+1, interval.end_time))
                break
            else:
                result.append(interval)

        self.table[pos] = result

    def set_safe_intervals_reserved_interval(self, pos: h.Position, reserved_interval: h.Interval):
        """
        Given a reserved interval and specified position,
        splits the safe interval list accordingly.

        Args:
            pos (h.Position): Specified position.
            interval (h.Interval): Reserved interval.
        """
        result: list[h.Interval] = []

        # TODO: filter out [a, a] intervals
        for interval in self.get_safe_intervals_at(pos):
            if interval.start_time <= reserved_interval.start_time and \
               reserved_interval.end_time <= interval.end_time:
                if interval.start_time < reserved_interval.start_time:
                    result.append(h.Interval(interval.start_time, reserved_interval.start_time-1))

                if reserved_interval.end_time < interval.end_time:
                    result.append(h.Interval(reserved_interval.end_time+1, interval.end_time))
            else:
                result.append(interval)

        self.table[pos] = result
