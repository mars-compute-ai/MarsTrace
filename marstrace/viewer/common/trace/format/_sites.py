import statistics
from dataclasses import dataclass
from typing import Any
from marstrace.viewer.common.pairing import BoundaryPairing
from marstrace.viewer.common.trace._schema import TraceSchema
from marstrace.viewer.common.trace._result import ScopeTraceRecord, SiteTraceRecord, TraceRecord


@dataclass
class SiteRegion:
    """A region the analyst reads off the kernel: the site entering it and the one leaving it."""

    enter_site: int
    leave_site: int
    label: str

    def to_json(self) -> list[Any]:
        """The region as the JSON archive keeps it.

        Returns:
            ``[enter_site, leave_site, label]``
        """
        return [
            self.enter_site,
            self.leave_site,
            self.label,
        ]

    @staticmethod
    def normalise_all(list_site_region: Any | None) -> list["SiteRegion"]:
        """The caller's region list as :class:`SiteRegion` entries.

        A caller may name a region as a triple, or as a pair whose label defaults to the
        two site ids. A region naming the same site twice is rejected: an instruction
        cannot both open and close the same interval.

        Args:
            list_site_region: the regions as the caller or a saved report named them, each a
                ``(enter_site, leave_site)`` pair or an ``(enter_site, leave_site, label)``
                triple

        Returns:
            the regions

        Raises:
            ValueError: a region names the same site as its enter and its leave
        """
        list_region: list[SiteRegion] = []
        for entry in list_site_region or []:
            if len(entry) == 3:
                enter, leave, label = entry
            else:
                enter, leave = entry
                label = f"site {enter}-{leave}"
            enter, leave = int(enter), int(leave)
            if enter == leave:
                raise ValueError(
                    f"a region's enter and leave are the same site ({enter}); "
                    f"one instruction cannot both open and close an interval"
                )
            list_region.append(
                SiteRegion(
                    enter_site=enter,
                    leave_site=leave,
                    label=str(label),
                )
            )
        return list_region


@dataclass
class SiteStat:
    """How often one site fired, over how many rows, and its repeat interval."""

    site_id: int
    count: int
    rows: int
    records_per_row: float
    t_first: int
    t_last: int
    repeat_min: int
    repeat_mean: float
    repeat_max: int

    def to_json(self) -> dict[str, Any]:
        """The statistics as the JSON archive keeps them.

        Returns:
            the statistics, keyed by field name
        """
        return {
            "site_id": self.site_id,
            "count": self.count,
            "rows": self.rows,
            "records_per_row": self.records_per_row,
            "t_first": self.t_first,
            "t_last": self.t_last,
            "repeat_min": self.repeat_min,
            "repeat_mean": self.repeat_mean,
            "repeat_max": self.repeat_max,
        }

    @classmethod
    def collect(cls, list_record: list[SiteTraceRecord]) -> list["SiteStat"]:
        """Per site: how often it fired, over how many rows, and its repeat interval.

        The repeat interval is the gap between one row's consecutive firings of that
        site, which on a loop-body site is the loop's per-iteration time.

        Args:
            list_record: the run's site records

        Returns:
            one entry per site that fired, with its count, rows and repeat interval
        """
        map_row = SiteRecords.by_row(list_record)
        map_count: dict[int, int] = {}
        map_rows: dict[int, set] = {}
        map_gap: dict[int, list[int]] = {}
        map_first: dict[int, int] = {}
        map_last: dict[int, int] = {}

        for tid, list_row_record in map_row.items():
            map_last_stamp: dict[int, int] = {}
            for record in list_row_record:
                site_id = record.site_id
                stamp = record.timestamp
                map_count[site_id] = (
                    map_count.get(
                        site_id,  # key
                        0,        # default
                    )
                    + 1
                )
                map_rows.setdefault(
                    site_id,  # key
                    set(),    # default
                ).add(tid)
                if site_id in map_last_stamp:
                    map_gap.setdefault(
                        site_id,  # key
                        [],       # default
                    ).append(stamp - map_last_stamp[site_id])
                map_last_stamp[site_id] = stamp
                if (
                    site_id not in map_first
                    or stamp < map_first[site_id]
                ):
                    map_first[site_id] = stamp
                if (
                    site_id not in map_last
                    or stamp > map_last[site_id]
                ):
                    map_last[site_id] = stamp

        list_stat: list[SiteStat] = []
        for site_id in sorted(map_count):
            list_gap = map_gap.get(
                site_id,  # key
                [],       # default
            )
            list_stat.append(
                cls(
                    site_id=site_id,
                    count=map_count[site_id],
                    rows=len(
                        map_rows.get(
                            site_id,  # key
                            (),       # default
                        )
                    ),
                    records_per_row=map_count[site_id]
                    / max(
                        1,
                        len(
                            map_rows.get(
                                site_id,  # key
                                (),       # default
                            )
                        ),
                    ),
                    t_first=map_first.get(
                        site_id,  # key
                        0,        # default
                    ),
                    t_last=map_last.get(
                        site_id,  # key
                        0,        # default
                    ),
                    repeat_min=min(list_gap) if list_gap else 0,
                    repeat_mean=statistics.fmean(list_gap) if list_gap else 0.0,
                    repeat_max=max(list_gap) if list_gap else 0,
                )
            )
        return list_stat


@dataclass
class RegionStat:
    """What pairing one region's boundaries managed, and what it had to drop."""

    label: str
    enter_site: int
    leave_site: int
    intervals: int  # the intervals the region paired
    rows: int  # the rows those intervals lie on
    dropped: int  # the boundaries left without a partner
    pair_rate: float  # twice the intervals over the boundaries the region saw

    def to_json(self) -> dict[str, Any]:
        """The statistics as the JSON archive keeps them.

        Returns:
            the statistics, keyed by field name
        """
        return {
            "label": self.label,
            "enter_site": self.enter_site,
            "leave_site": self.leave_site,
            "intervals": self.intervals,
            "rows": self.rows,
            "dropped": self.dropped,
            "pair_rate": self.pair_rate,
        }


@dataclass
class SitePairing:
    """Site records paired into regions: the scope records they make, and how each region fared."""

    list_record: list[ScopeTraceRecord]
    list_region_stat: list[RegionStat]

    @staticmethod
    def _untangle_ties(
        list_row_record: list[SiteTraceRecord],
        map_enter: dict[int, tuple[int, str]],
        map_leave: dict[int, tuple[int, str]],
    ) -> list[SiteTraceRecord]:
        """One row's records, with boundaries sharing a timestamp put back in order.

        A clock coarser than a region gives both boundaries of one pass the same
        timestamp, and sorting by timestamp then leaves them in whatever order the
        records arrived -- for a region that runs back to back, two enters followed
        by two leaves, which pairs as one long interval and drops the rest. On one
        row a region's boundaries alternate, so within a run of equal timestamps each
        region's boundaries are ordered to continue from whether it is open, which is
        the order they were written in whatever order the sort left them.
        """
        map_region_open: dict[int, bool] = {}
        list_out: list[SiteTraceRecord] = []
        start = 0
        while start < len(list_row_record):
            end = start
            stamp = list_row_record[start].timestamp
            while (
                end < len(list_row_record)
                and list_row_record[end].timestamp == stamp
            ):
                end += 1
            map_run: dict[int | None, list[SiteTraceRecord]] = {}
            for record in list_row_record[start:end]:
                site_id = record.site_id
                region = map_enter.get(
                    site_id,  # key
                    map_leave.get(
                        site_id,  # key
                        (None,),  # default
                    ),            # default
                )[0]
                map_run.setdefault(
                    region,  # key
                    [],      # default
                ).append(record)
            for region, list_run in map_run.items():
                if region is None:
                    list_out.extend(list_run)
                    continue
                list_enter = [record for record in list_run if record.site_id in map_enter]
                list_leave = [record for record in list_run if record.site_id in map_leave]
                is_open = map_region_open.get(
                    region,  # key
                    False,   # default
                )
                while (
                    list_enter
                    or list_leave
                ):
                    if (
                        is_open
                        and list_leave
                    ):
                        list_out.append(list_leave.pop(0))
                        is_open = False
                    elif (
                        not is_open
                        and list_enter
                    ):
                        list_out.append(list_enter.pop(0))
                        is_open = True
                    else:
                        # an unpartnered boundary: the pairing counts it as dropped
                        list_out.extend(list_leave if is_open else list_enter)
                        break
                map_region_open[region] = is_open
            start = end
        return list_out

    @classmethod
    def pair(
        cls,
        list_record: list[SiteTraceRecord],
        list_region: list[SiteRegion],
    ) -> "SitePairing":
        """Turn regions into scope-shaped records.

        A leave closes the latest still-open enter of its own region, walking each row
        in time order, so a truncated head or one corrupted record costs a single
        interval rather than cascading mispairs down the run. Sites no region names are
        dropped.

        Each region also gets one statistics entry: how many intervals it paired, over
        how many rows, how many boundaries it had to drop, and the pair rate -- twice
        the intervals over the boundaries the region saw. A run is gated on that rate,
        so the pairing is what measures it rather than the caller pairing again.

        Args:
            list_record: the run's site records
            list_region: the regions to pair

        Returns:
            the paired intervals as scope-start and scope-end records, and one statistics
            entry per region
        """
        map_enter = {
            region.enter_site: (
                index,
                region.label,
            )
            for index, region in enumerate(list_region)
        }
        map_leave = {
            region.leave_site: (
                index,
                region.label,
            )
            for index, region in enumerate(list_region)
        }
        list_row_ordered = [
            record
            for list_row_record in SiteRecords.by_row(list_record).values()
            for record in cls._untangle_ties(
                list_row_record=list_row_record,
                map_enter=map_enter,
                map_leave=map_leave,
            )
        ]
        boundary_pairing = BoundaryPairing.pair(
            list_record=list_row_ordered,
            row_of=lambda record: record.global_tid,
            is_start=lambda record: record.site_id in map_enter,
            is_end=lambda record: record.site_id in map_leave,
            key_of=lambda record: (
                map_enter[record.site_id]
                if record.site_id in map_enter
                else map_leave[record.site_id]
            )[0],
        )

        # every enter opens a scope; a leave closes one only where it found its enter
        set_paired_end = {id(boundary_pair.end) for boundary_pair in boundary_pairing.list_pair}
        list_out: list[ScopeTraceRecord] = []
        for record in list_row_ordered:
            if record.site_id in map_enter:
                type_str = "SCOPE_START"
                scope_id, scope_label = map_enter[record.site_id]
            elif id(record) in set_paired_end:
                type_str = "SCOPE_END"
                scope_id, scope_label = map_leave[record.site_id]
            else:
                continue
            list_out.append(
                ScopeTraceRecord(
                    global_tid=record.global_tid,
                    type_id=TraceSchema.MAP_TYPE_ID_RESERVED[type_str],
                    type_str=type_str,
                    scope_id=scope_id,
                    scope_str=str(scope_id),
                    scope_label=scope_label,
                    payload=record.timestamp,
                )
            )

        nb_region = len(list_region)
        list_nb_interval = [0] * nb_region
        list_nb_dropped = [0] * nb_region
        list_set_row: list[set] = [set() for _ in range(nb_region)]
        for boundary_pair in boundary_pairing.list_pair:
            pair_region_id = map_enter[boundary_pair.start.site_id][0]
            list_nb_interval[pair_region_id] += 1
            list_set_row[pair_region_id].add(boundary_pair.end.global_tid)
        for record in boundary_pairing.list_unpaired_start:
            list_nb_dropped[map_enter[record.site_id][0]] += 1
        for record in boundary_pairing.list_unpaired_end:
            list_nb_dropped[map_leave[record.site_id][0]] += 1

        list_region_stat = []
        for region_id, region in enumerate(list_region):
            # every boundary the region saw either paired or was dropped
            nb_boundary = 2 * list_nb_interval[region_id] + list_nb_dropped[region_id]
            list_region_stat.append(
                RegionStat(
                    label=region.label,
                    enter_site=region.enter_site,
                    leave_site=region.leave_site,
                    intervals=list_nb_interval[region_id],
                    rows=len(list_set_row[region_id]),
                    dropped=list_nb_dropped[region_id],
                    pair_rate=2 * list_nb_interval[region_id] / nb_boundary if nb_boundary else 0.0,
                )
            )
        return cls(
            list_record=list_out,
            list_region_stat=list_region_stat,
        )


class SiteRecords:
    """The records a machine-code run's sites wrote, read as points rather than scope boundaries."""

    @staticmethod
    def are_sites(list_record: list[TraceRecord]) -> bool:
        """Whether a run's records are site points rather than scope boundaries.

        Args:
            list_record: the run's trace records

        Returns:
            True when the first record is a site point
        """
        if not list_record:
            return False
        return isinstance(
            list_record[0],   # obj
            SiteTraceRecord,  # class_or_tuple
        )

    @staticmethod
    def by_row(list_record: list[SiteTraceRecord]) -> dict[int, list[SiteTraceRecord]]:
        """The records of each row, in the order the instruction reached them.

        Args:
            list_record: the run's site records

        Returns:
            thread id -> its records, in timestamp order
        """
        return BoundaryPairing.group_by_row(
            list_record=list_record,
            row_of=lambda record: record.global_tid,
            sort_key=lambda record: record.timestamp,
        )
