from dataclasses import dataclass
from typing import Any
from .._analysis import TraceAnalysis
from ._contract import PanelContract


@dataclass
class PanelScope:
    """One scope as the panel inks it: its position in the run, its label and its role."""

    scope_index: int
    label: str
    role: str | None

    def to_json(self) -> dict[str, Any]:
        """The scope as the panel's payload carries it.

        Returns:
            the scope, keyed as the panel reads it
        """
        return {
            "id": self.scope_index,
            "label": self.label,
            "role": self.role,
        }


@dataclass
class PanelLane:
    """One row of the panel, with the hardware groupings it rolls up into."""

    tid: int
    warp: int
    warpgroup: int
    block: int

    @staticmethod
    def of(
        tid: int,
        block_size: int | None,
        warp_size: int,
        warps_per_group: int,
    ) -> "PanelLane":
        """The hardware groupings a row rolls up into.

        The warp and the warpgroup are the launch's own, not the block's, so each
        level holds no more rows than the one below it and the axis can refine from
        block down to thread without a count going back up.

        Args:
            tid: the row's thread id
            block_size: threads per block, or None when the launch did not carry it
            warp_size: threads per warp
            warps_per_group: warps per warpgroup

        Returns:
            the row with its warp, warpgroup and block
        """
        warp = tid // warp_size
        return PanelLane(
            tid=tid,
            warp=warp,
            warpgroup=warp // warps_per_group,
            block=tid // block_size if block_size else 0,
        )

    def id_at(self, level: str) -> int:
        """The id this row has at one lane level.

        Args:
            level: the lane level: "thread", "warp", "warpgroup" or "block"

        Returns:
            the row's thread, warp, warpgroup or block id
        """
        if level == "thread":
            return self.tid
        if level == "warp":
            return self.warp
        if level == "warpgroup":
            return self.warpgroup
        return self.block

    def to_json(self) -> dict[str, int]:
        """The row as the panel's payload carries it.

        Returns:
            the row, keyed as the panel reads it
        """
        return {
            "tid": self.tid,
            "warp": self.warp,
            "wg": self.warpgroup,
            "block": self.block,
        }


@dataclass
class PanelInterval:
    """One interval as the panel draws it: its row, its scope and where it lies in time."""

    lane_index: int
    scope_index: int
    start: int  # from the run's first interval start
    duration: int

    def to_json(self) -> list[int]:
        """The interval as the panel's payload carries it, compactly.

        Returns:
            ``[lane, scope, start, duration]``
        """
        return [
            self.lane_index,
            self.scope_index,
            self.start,
            self.duration,
        ]


@dataclass
class PanelPayload:
    """One analysed run as the panel reads it."""

    # the rows a panel can hold before sampling; a canvas has fewer pixels than this
    LANE_MAX = 4096

    root_id: str
    kernel: str
    list_grid: list[int]
    list_block: list[int]
    block_size: int
    span: int
    clock: str
    unit: str  # the payload's unit, which an "sm" clock leaves in cycles
    granularity: str
    warp_label: str
    sampled_every: int
    # a row stands for itself: the report never claims threads it did not compare
    lane_repeat: int
    total_rows: int
    list_level: list[str]
    list_scope: list[PanelScope]
    list_lane: list[PanelLane]
    list_interval: list[PanelInterval]

    def to_json(self) -> dict[str, Any]:
        """The payload as the panel's script reads it.

        Returns:
            the payload, keyed as the panel reads it
        """
        return {
            "id": self.root_id,
            "kernel": self.kernel,
            "grid": self.list_grid,
            "block": self.list_block,
            "blockSize": self.block_size,
            "span": self.span,
            "clock": self.clock,
            "unit": self.unit,
            "granularity": self.granularity,
            "warpLabel": self.warp_label,
            "sampledEvery": self.sampled_every,
            "laneRepeat": self.lane_repeat,
            "totalRows": self.total_rows,
            "levels": self.list_level,
            "scopes": [scope.to_json() for scope in self.list_scope],
            "lanes": [lane.to_json() for lane in self.list_lane],
            "intervals": [interval.to_json() for interval in self.list_interval],
        }

    @classmethod
    def build(
        cls,
        analysis: TraceAnalysis,
        root_id: str,
        map_scope_role: dict[str, str] | None = None,
    ) -> "PanelPayload":
        """One analysed run as the panel reads it.

        ``map_scope_role`` maps a scope label to the warp role that executes it -- the
        roles are per-kernel data, and the legend groups scopes under them. A scope
        the map does not name carries no role, and a payload with no roles at all
        orders its lanes by thread id alone.

        Args:
            analysis: the run's analysis
            root_id: the id of the panel element the payload belongs to
            map_scope_role: scope label -> the warp role that executes it

        Returns:
            the panel's payload: launch shape, clock, lane levels, scopes, lanes and intervals
        """
        map_role = dict(map_scope_role or {})
        # the ink follows the scope's position here, so the order is fixed once
        list_scope = [
            PanelScope(
                scope_index=index,
                label=label,
                role=map_role.get(label),
            )
            for index, label in enumerate(analysis.list_scope_label)
        ]
        map_scope_id = {label: index for index, label in enumerate(analysis.list_scope_label)}

        list_tid = list(analysis.list_tid)
        stride = cls._stride_for(len(list_tid))
        list_kept = list_tid[::stride]
        map_lane_index = {tid: index for index, tid in enumerate(list_kept)}
        list_lane = [
            PanelLane.of(
                tid=tid,
                block_size=analysis.block_size,
                warp_size=analysis.compile_results.get_warp_size(),
                warps_per_group=analysis.compile_results.get_warps_per_group(),
            )
            for tid in list_kept
        ]

        list_interval: list[PanelInterval] = []
        for tid in list_kept:
            lane = map_lane_index[tid]
            for interval in analysis.map_tid_intervals.get(
                tid,  # key
                (),   # default
            ):
                scope_id = map_scope_id.get(interval.scope_label)
                if scope_id is None:
                    continue
                list_interval.append(
                    PanelInterval(
                        lane_index=lane,
                        scope_index=scope_id,
                        start=interval.t_start - analysis.t_min,
                        duration=interval.duration,
                    )
                )

        nb_row = analysis.launch_threads or (len(list_tid) * stride)
        return cls(
            root_id=root_id,
            kernel=analysis.kernel_prototype,
            list_grid=list(analysis.grid_dim) if analysis.grid_dim else [],
            list_block=list(analysis.block_dim) if analysis.block_dim else [],
            block_size=analysis.block_size or 0,
            span=analysis.span,
            clock=analysis.clock_type,
            unit="cyc" if analysis.clock_type == "sm" else "ns",
            granularity=analysis.record_granularity,
            warp_label=analysis.compile_results.get_warp_label(),
            sampled_every=stride,
            lane_repeat=1,
            total_rows=nb_row,
            list_level=cls._levels_of(
                list_lane=list_lane,
                granularity=analysis.record_granularity,
            ),
            list_scope=list_scope,
            list_lane=list_lane,
            list_interval=list_interval,
        )

    @staticmethod
    def _stride_for(nb_lane: int) -> int:
        """Keep every Nth row, so the payload stays under the lane cap."""
        if nb_lane <= PanelPayload.LANE_MAX:
            return 1
        return (nb_lane + PanelPayload.LANE_MAX - 1) // PanelPayload.LANE_MAX

    @staticmethod
    def _levels_of(
        list_lane: list[PanelLane],
        granularity: str,
    ) -> list[str]:
        """The lane levels this payload can honestly draw, finest first.

        A level is usable when it holds more than one row and strictly fewer rows
        than the level above it. A warp-level run has no per-thread axis to offer,
        so `thread` is dropped from it whatever the ids look like.
        """
        list_level: list[str] = []
        nb_lane_last = None
        for level in PanelContract.LANE_LEVELS:
            if (
                level == "thread"
                and granularity == "warp"
            ):
                continue
            nb_lane_at_level = len({lane.id_at(level) for lane in list_lane})
            if (
                nb_lane_at_level <= 1
                or (
                nb_lane_last is not None and nb_lane_at_level >= nb_lane_last
            )
            ):
                continue
            list_level.append(level)
            nb_lane_last = nb_lane_at_level
        return list_level
