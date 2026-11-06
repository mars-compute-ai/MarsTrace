from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

RecordT = TypeVar("RecordT")
GroupedT = TypeVar("GroupedT")


@dataclass
class BoundaryPair(Generic[RecordT]):
    """A start boundary on one row and the end boundary that closed it."""

    start: RecordT
    end: RecordT
    depth: int  # the starts still open on the row once this pair closed


@dataclass
class BoundaryPairing(Generic[RecordT]):
    """Boundary records paired row by row into closed spans, and the boundaries left unpaired.

    A row is whatever owns one run of boundaries: a thread, or a warp. On each row an end
    closes the latest still-open start of its own key, so a lost boundary costs a single span
    rather than shifting every pair after it.
    """

    list_pair: list[BoundaryPair[RecordT]] = field(default_factory=list)
    # the ends that found no open start, in the order they arrived
    list_unpaired_end: list[RecordT] = field(default_factory=list)
    # the starts no end closed, row by row in the order rows first appeared, oldest first
    list_unpaired_start: list[RecordT] = field(default_factory=list)

    @staticmethod
    def group_by_row(
        list_record: list[GroupedT],
        row_of: Callable[[GroupedT], int],
        sort_key: Callable[[GroupedT], Any] | None = None,
    ) -> dict[int, list[GroupedT]]:
        """The records of each row, rows in the order they first appear.

        Args:
            list_record: the records to group
            row_of: the row a record belongs to
            sort_key: what each row's records are ordered by, or None to keep arrival order;
                records that tie keep their arrival order

        Returns:
            row -> its records
        """
        map_row: dict[int, list[GroupedT]] = {}
        for record in list_record:
            map_row.setdefault(
                row_of(record),  # key
                [],              # default
            ).append(record)
        if sort_key is not None:
            for list_row_record in map_row.values():
                list_row_record.sort(key=sort_key)
        return map_row

    @classmethod
    def pair(
        cls,
        list_record: list[RecordT],
        row_of: Callable[[RecordT], int],
        is_start: Callable[[RecordT], bool],
        is_end: Callable[[RecordT], bool],
        key_of: Callable[[RecordT], Hashable] | None = None,
    ) -> "BoundaryPairing[RecordT]":
        """Pair the records, in the order given, into the spans each row closes.

        A record that is a start opens a span on its row; otherwise a record that is an end
        closes the latest still-open start of its row with the same key. A record that is
        neither is passed over.

        Args:
            list_record: the boundary records, each row's in the order they happened
            row_of: the row a record belongs to
            is_start: whether a record opens a span
            is_end: whether a record closes a span
            key_of: what a start and the end that closes it share; None lets an end close
                the latest open start whatever it is

        Returns:
            the closed spans in the order their ends arrived, and the boundaries left unpaired
        """
        pairing = cls()
        map_open: dict[int, list[RecordT]] = {}
        for record in list_record:
            list_open = map_open.setdefault(
                row_of(record),  # key
                [],              # default
            )
            if is_start(record):
                list_open.append(record)
                continue
            if not is_end(record):
                continue
            match_index = None
            for open_index in range(
                len(list_open) - 1,
                -1,
                -1,
            ):
                if (
                    key_of is None
                    or key_of(list_open[open_index]) == key_of(record)
                ):
                    match_index = open_index
                    break
            if match_index is None:
                pairing.list_unpaired_end.append(record)
                continue
            start = list_open.pop(match_index)
            pairing.list_pair.append(
                BoundaryPair(
                    start=start,
                    end=record,
                    depth=len(list_open),
                )
            )
        for list_open in map_open.values():
            pairing.list_unpaired_start.extend(list_open)
        return pairing


__all__ = [
    "BoundaryPair",
    "BoundaryPairing",
]
