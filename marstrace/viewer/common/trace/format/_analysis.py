import statistics
from dataclasses import dataclass, field
from typing import Any
from marstrace.viewer.common.launch import LaunchShape
from marstrace.viewer.common.pairing import BoundaryPairing
from marstrace.viewer.common.trace._result import ScopeTraceRecord, TraceCompileResults, TraceResult
from ._sites import (
    RegionStat,
    SitePairing,
    SiteRecords,
    SiteRegion,
    SiteStat,
)


@dataclass
class ScopeInterval:
    """One closed scope on one row: where it started and ended, and how deep it nests."""

    tid: int
    scope_id: int
    scope_label: str
    t_start: int
    t_end: int
    depth: int = 0

    @property
    def duration(self) -> int:
        """The scope's length, in the run's clock units."""
        return self.t_end - self.t_start


@dataclass
class ScopePairing:
    """A run's scope records paired into intervals, and the boundaries left unpaired."""

    map_tid_intervals: dict[int, list[ScopeInterval]]
    map_label_intervals: dict[str, list[ScopeInterval]]
    unmatched_starts: int
    unmatched_ends: int

    @classmethod
    def pair(
        cls,
        list_record: list[ScopeTraceRecord],
    ) -> "ScopePairing":
        """Pair each row's scope-start and scope-end records into intervals.

        An end closes the latest still-open start of the same scope on its row.

        Args:
            list_record: the run's scope records

        Returns:
            the intervals by row and by scope label, and how many starts and how many ends
            were left unpaired
        """
        map_row_record = BoundaryPairing.group_by_row(
            list_record=list_record,
            row_of=lambda record: record.global_tid,
            sort_key=lambda record: record.payload,
        )
        boundary_pairing = BoundaryPairing.pair(
            list_record=[record for list_row_record in map_row_record.values() for record in list_row_record],
            row_of=lambda record: record.global_tid,
            is_start=lambda record: record.type_str == "SCOPE_START",
            is_end=lambda record: record.type_str == "SCOPE_END",
            key_of=lambda record: record.scope_id,
        )

        map_tid_intervals: dict[int, list[ScopeInterval]] = {}
        map_label_intervals: dict[str, list[ScopeInterval]] = {}
        for boundary_pair in boundary_pairing.list_pair:
            start = boundary_pair.start
            label = str(start.scope_label or start.scope_str or str(start.scope_id))
            interval = ScopeInterval(
                tid=start.global_tid,
                scope_id=start.scope_id,
                scope_label=label,
                t_start=start.payload,
                t_end=boundary_pair.end.payload,
                depth=boundary_pair.depth,
            )
            map_tid_intervals.setdefault(
                start.global_tid,  # key
                [],                # default
            ).append(interval)
            map_label_intervals.setdefault(
                label,  # key
                [],     # default
            ).append(interval)
        for list_thread_interval in map_tid_intervals.values():
            list_thread_interval.sort(key=lambda interval: interval.t_start)
        return cls(
            map_tid_intervals=map_tid_intervals,
            map_label_intervals=map_label_intervals,
            unmatched_starts=len(boundary_pairing.list_unpaired_start),
            unmatched_ends=len(boundary_pairing.list_unpaired_end),
        )


@dataclass
class ScopeStat:
    """How often one scope ran, over how many rows, and the spread of its durations."""

    # the percentiles a scope's duration spread is summarised by, out of the full range
    PERCENT_FULL = 100.0
    PERCENTILE_LOWER_QUARTILE = 25
    PERCENTILE_MEDIAN = 50
    PERCENTILE_UPPER_QUARTILE = 75
    PERCENTILE_TAIL = 95

    scope_label: str
    count: int
    lanes: int
    duration_min: int
    duration_p25: int
    duration_p50: int
    duration_mean: float
    duration_p75: int
    duration_p95: int
    duration_max: int
    duration_total: int

    def to_json(self) -> dict[str, Any]:
        """The statistics as the JSON archive keeps them.

        Returns:
            the statistics, keyed as the archive names them
        """
        return {
            "scope_label": self.scope_label,
            "count": self.count,
            "lanes": self.lanes,
            "min": self.duration_min,
            "p25": self.duration_p25,
            "p50": self.duration_p50,
            "mean": self.duration_mean,
            "p75": self.duration_p75,
            "p95": self.duration_p95,
            "max": self.duration_max,
            "total": self.duration_total,
        }

    @staticmethod
    def percentile(
        list_value: list[int],
        pct: float,
    ) -> int:
        """The nearest-rank percentile of a list of durations.

        Args:
            list_value: the durations
            pct: the percentile, from 0 to 100

        Returns:
            the value at that rank, or 0 for an empty list
        """
        if not list_value:
            return 0
        list_sorted = sorted(list_value)
        if len(list_sorted) == 1:
            return list_sorted[0]
        rank = max(
            0,
            min(
                len(list_sorted) - 1,
                int(round((len(list_sorted) - 1) * pct / ScopeStat.PERCENT_FULL)),
            ),
        )
        return list_sorted[rank]


@dataclass
class ThreadBubble:
    """How long one row was inside a scope, and how long it sat idle between scopes."""

    active: int  # the merged coverage of the row's outermost scopes
    bubble: int  # the lifetime the row spent outside them
    lifetime: int  # from the row's first start to its last end
    nb_intervals: int

    def to_json(self) -> dict[str, int]:
        """The figures as the JSON report keeps them.

        Returns:
            the figures, keyed as the report names them
        """
        return {
            "active": self.active,
            "bubble": self.bubble,
            "lifetime": self.lifetime,
            "n_intervals": self.nb_intervals,
        }


@dataclass
class BubblyThread:
    """One of the rows that sat idle longest."""

    tid: int
    bubble: int
    active: int
    lifetime: int

    def to_json(self) -> dict[str, int]:
        """The row's figures as the JSON archive keeps them.

        Returns:
            the figures, keyed by field name
        """
        return {
            "tid": self.tid,
            "bubble": self.bubble,
            "active": self.active,
            "lifetime": self.lifetime,
        }


@dataclass
class PipelineSummary:
    """The run's active, bubble and lifetime spread across rows."""

    # the rows that sat idle longest, which the summary names
    BUBBLIEST_COUNT = 3

    nb_threads: int
    active_mean: float
    active_min: int
    active_max: int
    bubble_mean: float
    bubble_min: int
    bubble_max: int
    lifetime_mean: float
    lifetime_min: int
    lifetime_max: int
    bubble_ratio: float
    list_bubbliest: list[BubblyThread]

    def to_json(self) -> dict[str, Any]:
        """The summary as the JSON archive keeps it.

        Returns:
            the summary, keyed as the archive names it
        """
        return {
            "n_threads": self.nb_threads,
            "active_mean": self.active_mean,
            "active_min": self.active_min,
            "active_max": self.active_max,
            "bubble_mean": self.bubble_mean,
            "bubble_min": self.bubble_min,
            "bubble_max": self.bubble_max,
            "lifetime_mean": self.lifetime_mean,
            "lifetime_min": self.lifetime_min,
            "lifetime_max": self.lifetime_max,
            "bubble_ratio": self.bubble_ratio,
            "top_bubbliest": [thread.to_json() for thread in self.list_bubbliest],
        }


@dataclass
class ScopeGap:
    """An idle gap between two consecutive outermost scopes on one row."""

    # the widest gaps a run reports
    GAP_COUNT_DEFAULT = 5

    tid: int
    scope_before: str
    scope_after: str
    gap: int

    def to_json(self) -> dict[str, Any]:
        """The gap as the JSON archive keeps it.

        Returns:
            the gap, keyed as the archive names it
        """
        return {
            "tid": self.tid,
            "before": self.scope_before,
            "after": self.scope_after,
            "gap": self.gap,
        }


@dataclass
class Concurrency:
    """How many intervals are open over the run's span, bin by bin."""

    # the equal bins the run's span is cut into
    BIN_COUNT_DEFAULT = 60

    list_bin: list[int]  # the open-interval count of each bin
    peak: int
    span: int

    def to_json(self) -> dict[str, Any]:
        """The histogram as the JSON archive keeps it.

        Returns:
            the histogram, keyed as the archive names it
        """
        return {
            "bins": self.list_bin,
            "peak": self.peak,
            "span": self.span,
        }


@dataclass
class ScopeRoleGroup:
    """The scopes one warp role executes, by their position in the run."""

    role: str | None  # None for the scopes no role was named for
    list_scope_index: list[int] = field(default_factory=list)

    @staticmethod
    def group(
        list_scope_label: list[str],
        map_role: dict[str, str],
    ) -> list["ScopeRoleGroup"]:
        """Scopes under the warp role that executes them, first appearance first.

        A scope's ink follows its position in the run, so grouping never recolours: a scope
        carries its position with it into whichever group it lands in.

        Args:
            list_scope_label: the run's scope labels, in ink order
            map_role: scope label -> the warp role that executes it

        Returns:
            one group per role, in the order the roles first appear
        """
        map_group: dict[str | None, ScopeRoleGroup] = {}
        for index, label in enumerate(list_scope_label):
            role = map_role.get(label)
            if role not in map_group:
                map_group[role] = ScopeRoleGroup(role=role)
            map_group[role].list_scope_index.append(index)
        return list(map_group.values())


@dataclass
class TraceAnalysis:
    """One run's trace, paired into scope intervals, with the launch it came from."""

    kernel_prototype: str
    compile_results: TraceCompileResults
    map_oe_results: dict[str, Any]
    map_tid_intervals: dict[int, list[ScopeInterval]] = field(default_factory=dict)
    map_label_intervals: dict[str, list[ScopeInterval]] = field(default_factory=dict)
    list_tid: list[int] = field(default_factory=list)
    list_scope_label: list[str] = field(default_factory=list)
    t_min: int = 0
    t_max: int = 0
    unmatched_starts: int = 0
    unmatched_ends: int = 0
    total_records: int = 0
    grid_dim: tuple[int, int, int] | None = None
    block_dim: tuple[int, int, int] | None = None
    # the clock the payloads were read from, and what one record observes
    clock_type: str = "gpu"
    record_shape: str = "scope"
    record_granularity: str = "thread"
    # per-site aggregates, on a run whose records are points
    list_site_stat: list[SiteStat] = field(default_factory=list)
    # the warp role each scope belongs to, when the caller named them
    map_scope_role: dict[str, str] = field(default_factory=dict)
    # the regions the caller read the sites as, and what pairing them managed
    list_site_region: list[SiteRegion] = field(default_factory=list)
    list_region_stat: list[RegionStat] = field(default_factory=list)

    @property
    def span(self) -> int:
        """The time from the earliest interval start to the latest end."""
        return max(
            0,
            self.t_max - self.t_min,
        )

    @property
    def launch_threads(self) -> int | None:
        """Threads the launch ran, or None when it did not carry its shape."""
        if (
            not self.grid_dim
            or not self.block_size
        ):
            return None
        grid_x, grid_y, grid_z = self.grid_dim
        nb_thread = (
            LaunchShape.count_blocks(
                (
                    int(grid_x),
                    int(grid_y),
                    int(grid_z),
                )
            )
            * self.block_size
        )
        return nb_thread if nb_thread > 0 else None

    @property
    def block_size(self) -> int | None:
        """Threads per block, or None when the run did not carry its block shape."""
        if not self.block_dim:
            return None
        block_x, block_y, block_z = self.block_dim
        nb_thread = LaunchShape.count_block_threads(
            (
                int(block_x),
                int(block_y),
                int(block_z),
            )
        )
        return nb_thread if nb_thread > 0 else None

    @classmethod
    def from_result(
        cls,
        run: TraceResult,
        site_regions: Any | None = None,
        scope_roles: dict[str, str] | None = None,
    ) -> "TraceAnalysis":
        """Aggregate one run, pairing its site records when regions name them.

        ``site_regions`` is a list of ``(enter_site, leave_site)`` or
        ``(enter_site, leave_site, label)``. Which two sites bound a region is the
        analyst's reading of the kernel, never a guess this makes for them.

        Args:
            run: one run's result, as a trace returned it or a report saved it
            site_regions: the regions to pair site records into; a re-read report's own
                regions when None
            scope_roles: scope label -> the warp role it belongs to

        Returns:
            the run's analysis

        Raises:
            ValueError: a region names the same site as its enter and its leave
        """
        # a run re-read from a report carries its launch in the block it was given
        map_analysis = run.map_saved_analysis or {}
        list_record = run.list_record
        list_site_stat: list[SiteStat] = []
        list_region: list[SiteRegion] = []
        list_region_stat: list[RegionStat] = []
        if SiteRecords.are_sites(list_record):
            list_site_stat = SiteStat.collect(list_record)
            # a re-read report pairs its sites with the regions it was saved with
            list_region = SiteRegion.normalise_all(
                site_regions if site_regions is not None else map_analysis.get("site_regions")
            )
            if list_region:
                pairing = SitePairing.pair(
                    list_record=list_record,
                    list_region=list_region,
                )
                list_record = pairing.list_record
                list_region_stat = pairing.list_region_stat
            else:
                list_record = []
        pairing = ScopePairing.pair(list_record)

        t_min: int | None = None
        t_max: int | None = None
        for list_interval in pairing.map_tid_intervals.values():
            for interval in list_interval:
                if (
                    t_min is None
                    or interval.t_start < t_min
                ):
                    t_min = interval.t_start
                if (
                    t_max is None
                    or interval.t_end > t_max
                ):
                    t_max = interval.t_end

        compile_results = run.compile_results
        return cls(
            kernel_prototype=str(run.kernel_prototype or "(unknown)"),
            compile_results=compile_results,
            map_oe_results=dict(run.map_oe_results or {}),
            map_tid_intervals=pairing.map_tid_intervals,
            map_label_intervals=pairing.map_label_intervals,
            list_tid=sorted(pairing.map_tid_intervals.keys()),
            list_scope_label=sorted(pairing.map_label_intervals.keys()),
            t_min=t_min if t_min is not None else 0,
            t_max=t_max if t_max is not None else 0,
            unmatched_starts=pairing.unmatched_starts,
            unmatched_ends=pairing.unmatched_ends,
            total_records=len(run.list_record),
            grid_dim=cls._as_dim3(run.grid_dim or map_analysis.get("grid_dim")),
            block_dim=cls._as_dim3(run.block_dim or map_analysis.get("block_dim")),
            clock_type=str(compile_results.clock_type or "gpu"),
            record_shape=str(compile_results.record_shape or "scope"),
            record_granularity=str(compile_results.record_granularity or "thread"),
            list_site_stat=list_site_stat,
            list_site_region=list_region,
            list_region_stat=list_region_stat,
            map_scope_role=dict(scope_roles or map_analysis.get("scope_roles") or {}),
        )

    @staticmethod
    def _as_dim3(v: Any) -> tuple[int, int, int] | None:
        """The launch dimension as three integers, or None when it is not one."""
        if v is None:
            return None
        try:
            list_dim = list(v)
            return (
                int(list_dim[0]),
                int(list_dim[1]),
                int(list_dim[2]),
            )
        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            return None

    def scope_stats(self) -> list[ScopeStat]:
        """Per scope label: how often it ran, over how many rows, and its duration spread.

        Args:

        Returns:
            one entry per scope label, largest total time first
        """
        list_stat: list[ScopeStat] = []
        for label in self.list_scope_label:
            list_duration = [
                interval.duration
                for interval in self.map_label_intervals.get(
                    label,  # key
                    [],     # default
                )
            ]
            if not list_duration:
                continue
            list_stat.append(
                ScopeStat(
                    scope_label=label,
                    count=len(list_duration),
                    lanes=len(
                        {
                            interval.tid
                            for interval in self.map_label_intervals.get(
                                label,  # key
                                [],     # default
                            )
                        }
                    ),
                    duration_min=min(list_duration),
                    duration_p25=ScopeStat.percentile(
                        list_value=list_duration,
                        pct=ScopeStat.PERCENTILE_LOWER_QUARTILE,
                    ),
                    duration_p50=ScopeStat.percentile(
                        list_value=list_duration,
                        pct=ScopeStat.PERCENTILE_MEDIAN,
                    ),
                    duration_mean=statistics.fmean(list_duration),
                    duration_p75=ScopeStat.percentile(
                        list_value=list_duration,
                        pct=ScopeStat.PERCENTILE_UPPER_QUARTILE,
                    ),
                    duration_p95=ScopeStat.percentile(
                        list_value=list_duration,
                        pct=ScopeStat.PERCENTILE_TAIL,
                    ),
                    duration_max=max(list_duration),
                    duration_total=sum(list_duration),
                )
            )
        list_stat.sort(
            key=lambda stat: stat.duration_total,
            reverse=True,
        )
        return list_stat

    def bubble_per_thread(self) -> dict[int, ThreadBubble]:
        """Per thread: how long it was inside a scope, and how long it sat idle between.

        Active is the merged coverage of the outermost scopes, lifetime the time from the
        first start to the last end, and bubble lifetime minus active.

        Args:

        Returns:
            thread id -> its figures
        """
        map_tid_bubble: dict[int, ThreadBubble] = {}
        for tid, list_interval in self.map_tid_intervals.items():
            if not list_interval:
                continue
            list_outer = [
                interval for interval in list_interval if interval.depth == 0
            ] or list_interval
            list_outer_sorted = sorted(
                list_outer,  # iterable
                key=lambda interval: interval.t_start,
            )
            list_merged: list[tuple[int, int]] = []
            for interval in list_outer_sorted:
                if (
                    list_merged
                    and interval.t_start <= list_merged[-1][1]
                ):
                    list_merged[-1] = (
                        list_merged[-1][0],
                        max(
                            list_merged[-1][1],
                            interval.t_end,
                        ),
                    )
                else:
                    list_merged.append(
                        (
                            interval.t_start,
                            interval.t_end,
                        )
                    )
            active = sum(span_end - span_start for span_start, span_end in list_merged)
            lifetime = list_merged[-1][1] - list_merged[0][0]
            map_tid_bubble[tid] = ThreadBubble(
                active=active,
                bubble=lifetime - active,
                lifetime=lifetime,
                nb_intervals=len(list_interval),
            )
        return map_tid_bubble

    def pipeline_summary(
        self,
        map_bubble: dict[int, ThreadBubble] | None = None,
    ) -> PipelineSummary | None:
        """The run's active, bubble and lifetime spread across threads.

        Args:
            map_bubble: the per-thread figures `bubble_per_thread` computed; computed here when None

        Returns:
            the means, extremes and bubble ratio, and the three bubbliest threads; None when
            no thread closed a scope
        """
        map_bubble = self.bubble_per_thread() if map_bubble is None else map_bubble
        if not map_bubble:
            return None
        list_active = [v.active for v in map_bubble.values()]
        list_bubble = [v.bubble for v in map_bubble.values()]
        list_lifetime = [v.lifetime for v in map_bubble.values()]
        total_life = sum(list_lifetime)
        list_bubbliest_item = sorted(
            map_bubble.items(),  # iterable
            key=lambda entry: entry[1].bubble,
            reverse=True,
        )[: PipelineSummary.BUBBLIEST_COUNT]
        return PipelineSummary(
            nb_threads=len(map_bubble),
            active_mean=statistics.fmean(list_active),
            active_min=min(list_active),
            active_max=max(list_active),
            bubble_mean=statistics.fmean(list_bubble),
            bubble_min=min(list_bubble),
            bubble_max=max(list_bubble),
            lifetime_mean=statistics.fmean(list_lifetime),
            lifetime_min=min(list_lifetime),
            lifetime_max=max(list_lifetime),
            bubble_ratio=(sum(list_bubble) / total_life) if total_life else 0.0,
            list_bubbliest=[
                BubblyThread(
                    tid=tid,
                    bubble=v.bubble,
                    active=v.active,
                    lifetime=v.lifetime,
                )
                for tid, v in list_bubbliest_item
            ],
        )

    def inter_scope_gaps(
        self,
        top: int = ScopeGap.GAP_COUNT_DEFAULT,
    ) -> list[ScopeGap]:
        """The widest idle gaps between consecutive outermost scopes on one row.

        Args:
            top: how many gaps to keep

        Returns:
            the gaps, widest first
        """
        list_gap: list[ScopeGap] = []
        for tid, list_interval in self.map_tid_intervals.items():
            list_outer = sorted(
                [interval for interval in list_interval if interval.depth == 0],  # iterable
                key=lambda interval: interval.t_start,
            )
            if len(list_outer) < 2:
                continue
            for prev, cur in zip(
                list_outer,
                list_outer[1:],
            ):
                gap = cur.t_start - prev.t_end
                if gap > 0:
                    list_gap.append(
                        ScopeGap(
                            tid=tid,
                            scope_before=prev.scope_label,
                            scope_after=cur.scope_label,
                            gap=gap,
                        )
                    )
        list_gap.sort(
            key=lambda record: record.gap,
            reverse=True,
        )
        return list_gap[:top]

    def concurrency(
        self,
        bins: int = Concurrency.BIN_COUNT_DEFAULT,
    ) -> Concurrency:
        """How many intervals are open over the run's span, bin by bin.

        Args:
            bins: how many equal bins the span is cut into

        Returns:
            the open-interval count per bin, its maximum, and the span
        """
        if (
            not self.map_tid_intervals
            or self.t_max <= self.t_min
        ):
            return Concurrency(
                list_bin=[],
                peak=0,
                span=0,
            )
        span = self.t_max - self.t_min
        bin_width = span / bins
        list_count = [0] * bins
        for list_interval in self.map_tid_intervals.values():
            for interval in list_interval:
                bin_low = max(
                    0,
                    min(
                        bins - 1,
                        int((interval.t_start - self.t_min) / bin_width),
                    ),
                )
                bin_high = max(
                    0,
                    min(
                        bins - 1,
                        int((interval.t_end - self.t_min) / bin_width),
                    ),
                )
                for i in range(
                    bin_low,
                    bin_high + 1,
                ):
                    list_count[i] += 1
        return Concurrency(
            list_bin=list_count,
            peak=max(list_count),
            span=span,
        )

    def to_json(self) -> dict[str, Any]:
        """Pre-computed aggregates for the JSON ``analysis`` block.

        Args:

        Returns:
            the ``analysis`` block of the run's JSON report
        """
        summary = self.pipeline_summary()
        return {
            "clock_type": self.clock_type,
            "record_shape": self.record_shape,
            "record_granularity": self.record_granularity,
            "n_threads": len(self.list_tid),
            "n_scopes": len(self.list_scope_label),
            "scope_labels": list(self.list_scope_label),
            "grid_dim": list(self.grid_dim) if self.grid_dim else None,
            "block_dim": list(self.block_dim) if self.block_dim else None,
            "block_size": self.block_size,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "span": self.span,
            "unmatched_starts": self.unmatched_starts,
            "unmatched_ends": self.unmatched_ends,
            "scope_stats": [stat.to_json() for stat in self.scope_stats()],
            "site_stats": [stat.to_json() for stat in self.list_site_stat],
            "site_regions": [region.to_json() for region in self.list_site_region],
            "scope_roles": dict(self.map_scope_role),
            "region_stats": [stat.to_json() for stat in self.list_region_stat],
            "pipeline_summary": summary.to_json() if summary is not None else {},
            "longest_bubbles": [
                gap.to_json()
                for gap in self.inter_scope_gaps(
                    top=ScopeGap.GAP_COUNT_DEFAULT,
                )
            ],
            "concurrency": self.concurrency().to_json(),
        }
