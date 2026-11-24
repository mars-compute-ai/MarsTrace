from dataclasses import dataclass, field
from typing import Any
from marstrace.viewer.common.trace._schema import TraceSchema


@dataclass
class ScopeTraceRecord:
    """One scope boundary a thread (or, warp-level, a warp's leading lane) recorded.

    A site run paired into regions is read back in the same shape.
    """

    global_tid: int  # recovered from the slab row the record sits in
    type_id: int  # dense id of the boundary kind
    type_str: str  # SCOPE_START, SCOPE_END or POINT
    scope_id: int  # dense id of the scope
    scope_str: str  # the scope's name as the run gave it
    scope_label: str  # the caller's label for the scope, else its name
    payload: int  # the timestamp: the GPU-wide timer (gpu) or the core clock (sm)
    value: int = 0  # the value the site recorded, 0 where it names none

    def to_json(self) -> dict[str, Any]:
        """The record as the JSON archive keeps it.

        Returns:
            the record's fields, keyed as the archive names them
        """
        return {
            "global_tid": self.global_tid,
            "type_id": self.type_id,
            "type_str": self.type_str,
            "scope_id": self.scope_id,
            "scope_str": self.scope_str,
            "scope_label": self.scope_label,
            "payload": self.payload,
            "value": self.value,
        }


@dataclass
class SiteTraceRecord:
    """One record a site wrote: the site, the value it read and when."""

    global_tid: int  # the thread, or a warp's leading thread, that reached the site
    site_id: int  # the site's position in the site list
    value: int  # the value the site recorded, 0 for a timestamp-only site
    timestamp: int  # in the run's clock units
    smid: int | None = None  # the SM (CU on AMDGPU) a warp-level record's block ran on, None per thread

    def to_json(self) -> dict[str, Any]:
        """The record as the JSON archive keeps it.

        Returns:
            the record's fields, keyed as the archive names them; ``smid`` only on a
            warp-level record
        """
        map_record: dict[str, Any] = {
            "global_tid": self.global_tid,
            "site_id": self.site_id,
            "value": self.value,
        }
        if self.smid is not None:
            map_record["smid"] = self.smid
        map_record["timestamp"] = self.timestamp
        return map_record


@dataclass
class TraceCompileResults:
    """How a run was traced, and how its records are to be read."""

    instrumentation_level: str  # the kind of trace the run is
    origin_reg_used_avg: int  # the kernel's register count as built
    trace_reg_used_avg: int  # the kernel's register count while traced
    # the clock the payloads were read from, and what one record observes and whose
    clock_type: str | None = None
    record_shape: str | None = None
    record_granularity: str | None = None
    # the counts in use, where the run reports them
    origin_reg_used_overall: int | None = None
    trace_reg_used_overall: int | None = None
    # the record buffer's size, where the run reports it
    slab_bytes: int | None = None
    nb_warps: int | None = None
    # the threads of a warp, the warps of a warpgroup and the backend's name for a warp;
    # None reads as the defaults of TraceSchema
    warp_size: int | None = None
    warps_per_group: int | None = None
    warp_label: str | None = None
    # the records one row (a thread, or a warp at warp level) keeps before its ring wraps: a
    # count equal to it may have wrapped, one below it did not
    records_per_row: int | None = None

    # field -> the key the archive names it by, in the archive's order
    MAP_JSON_KEY = {
        "instrumentation_level": "instrumentation_level",
        "clock_type": "clock_type",
        "record_shape": "record_shape",
        "record_granularity": "record_granularity",
        "origin_reg_used_avg": "origin_reg_used [avg]",
        "trace_reg_used_avg": "trace_reg_used [avg]",
        "origin_reg_used_overall": "origin_reg_used [overall]",
        "trace_reg_used_overall": "trace_reg_used [overall]",
        "slab_bytes": "slab_bytes",
        "nb_warps": "nb_warps",
        "warp_size": "warp_size",
        "warps_per_group": "warps_per_group",
        "warp_label": "warp_label",
        "records_per_row": "records_per_row",
    }

    def to_json(self) -> dict[str, Any]:
        """The results as the JSON archive keeps them.

        Returns:
            every result the run carries, keyed as the archive names it
        """
        return {
            json_key: getattr(
                self,
                field_name,
            )
            for field_name, json_key in self.MAP_JSON_KEY.items()
            if getattr(
                self,
                field_name,
            )
            is not None
        }

    def get_warp_size(self) -> int:
        """The threads of one of the run's warps.

        Returns:
            the run's warp size, or the default when the run does not carry one
        """
        return self.warp_size or TraceSchema.WARP_SIZE

    def get_warps_per_group(self) -> int:
        """The warps of one of the run's warpgroups.

        Returns:
            the run's warps per group, or the default when the run does not carry one
        """
        return self.warps_per_group or TraceSchema.WARPS_PER_GROUP

    def get_warp_label(self) -> str:
        """What the run's backend calls a warp, for text a reader sees.

        Returns:
            the run's warp label, or the default when the run does not carry one
        """
        return self.warp_label or TraceSchema.WARP_LABEL

    @classmethod
    def from_json(cls, map_json: dict[str, Any]) -> "TraceCompileResults":
        """The results a JSON archive kept.

        Args:
            map_json: the results, keyed as the archive names them

        Returns:
            the results, with what the archive does not carry left None
        """
        return cls(
            **{
                field_name: map_json.get(json_key)
                for field_name, json_key in cls.MAP_JSON_KEY.items()
            }
        )


@dataclass
class SiteSpliceReport:
    """One site's placement details, as the run reported them."""

    # each entry as the run reported it
    list_map_splice: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """The report as the JSON archive keeps it.

        Returns:
            the entries, under ``splices``
        """
        return {"splices": self.list_map_splice}


@dataclass
class TraceSpliceResults:
    """Each site's placement details, and the setting they were made under."""

    schedule_mode: str  # the placement setting the run used
    schedule_prune: bool  # a second placement setting
    map_site_report: dict[int, SiteSpliceReport] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """The results as the JSON archive keeps them, before its keys are made strings.

        Returns:
            the setting and every site's details, keyed by site id
        """
        return {
            "schedule_mode": self.schedule_mode,
            "schedule_prune": self.schedule_prune,
            "sites": {
                site_id: report.to_json() for site_id, report in self.map_site_report.items()
            },
        }


TraceRecord = ScopeTraceRecord | SiteTraceRecord


@dataclass
class TraceDevice:
    """The device a run traced on, so a report read elsewhere need not guess it."""

    name: str
    sm_count: int  # streaming multiprocessors (compute units on AMDGPU)
    compute_capability: str  # "major.minor"
    sm_clock_khz: int | None = None  # the SM clock rate the driver reports
    sm_clock_mhz_at_trace: int | None = None  # the SM clock read right after the launch, None without NVML

    def to_json(self) -> dict[str, Any]:
        """The device as the JSON archive keeps it.

        Returns:
            the device's fields, keyed by field name; the clocks only when known
        """
        map_device: dict[str, Any] = {
            "name": self.name,
            "sm_count": self.sm_count,
            "compute_capability": self.compute_capability,
        }
        if self.sm_clock_khz is not None:
            map_device["sm_clock_khz"] = self.sm_clock_khz
        if self.sm_clock_mhz_at_trace is not None:
            map_device["sm_clock_mhz_at_trace"] = self.sm_clock_mhz_at_trace
        return map_device

    @classmethod
    def from_json(cls, map_device: Any) -> "TraceDevice | None":
        """A device a JSON archive kept, or None when the archive names none.

        Args:
            map_device: the run's ``device`` entry

        Returns:
            the device, or None when the entry is missing or names no SM count
        """
        if not isinstance(
            map_device,  # obj
            dict,        # class_or_tuple
        ) or not isinstance(
            map_device.get("sm_count"),  # obj
            int,                         # class_or_tuple
        ):
            return None
        return cls(
            name=str(map_device.get("name", "")),
            sm_count=map_device["sm_count"],
            compute_capability=str(map_device.get("compute_capability", "")),
            sm_clock_khz=map_device.get("sm_clock_khz"),
            sm_clock_mhz_at_trace=map_device.get("sm_clock_mhz_at_trace"),
        )


@dataclass
class TraceResult:
    """One traced launch: the kernel, how it was traced and the records it wrote."""

    kernel_prototype: str
    compile_results: TraceCompileResults
    grid_dim: tuple[int, int, int] | None
    block_dim: tuple[int, int, int] | None
    list_record: list[TraceRecord] = field(default_factory=list)
    # observer-effect results: always empty, as the evaluation is not implemented
    map_oe_results: dict[str, Any] = field(default_factory=dict)
    # each site's placement details, None where the run reports none
    splice_results: TraceSpliceResults | None = None
    # the analysis block a saved report kept beside the run, None for a fresh trace
    map_saved_analysis: dict[str, Any] | None = None
    # the device the launch ran on, None where the backend or an old report names none
    device: TraceDevice | None = None

    @staticmethod
    def record_from_json(map_record: Any) -> TraceRecord | None:
        """A record a JSON archive kept, read back into its kind.

        Args:
            map_record: the record as the archive keeps it

        Returns:
            the scope or site record, or None when the entry is not a record of either
            kind or names no integer row
        """
        if not isinstance(
            map_record,  # obj
            dict,        # class_or_tuple
        ):
            return None
        global_tid = map_record.get("global_tid")
        if not isinstance(
            global_tid,  # obj
            int,         # class_or_tuple
        ):
            return None
        if "type_str" in map_record:
            return ScopeTraceRecord(
                global_tid=global_tid,
                type_id=map_record.get(
                    "type_id",
                    0,
                ),
                type_str=map_record.get(
                    "type_str",
                    "",
                ),
                scope_id=map_record.get(
                    "scope_id",
                    -1,
                ),
                scope_str=map_record.get(
                    "scope_str",
                    "",
                ),
                scope_label=map_record.get(
                    "scope_label",
                    "",
                ),
                payload=map_record.get(
                    "payload",
                    0,
                ),
                value=map_record.get(
                    "value",
                    0,
                ),
            )
        site_id = map_record.get("site_id")
        timestamp = map_record.get("timestamp")
        if (
            not isinstance(
                site_id,  # obj
                int,      # class_or_tuple
            )
            or not isinstance(
                timestamp,  # obj
                int,        # class_or_tuple
            )
        ):
            return None
        return SiteTraceRecord(
            global_tid=global_tid,
            site_id=site_id,
            value=map_record.get(
                "value",
                0,
            ),
            timestamp=timestamp,
            smid=map_record.get("smid"),
        )

    @classmethod
    def from_json(cls, map_run: dict[str, Any]) -> "TraceResult":
        """A run a JSON archive kept, read back with the analysis block saved beside it.

        The archive keeps the launch's shape only in the analysis block, so it is read
        from there.

        Args:
            map_run: one entry of the archive's ``runs``

        Returns:
            the run, whose records are every entry that reads back as a record
        """
        map_analysis = map_run.get("analysis")
        if not isinstance(
            map_analysis,  # obj
            dict,          # class_or_tuple
        ):
            map_analysis = None
        list_record = []
        for map_record in map_run.get("records") or []:
            record = cls.record_from_json(map_record)
            if record is not None:
                list_record.append(record)
        map_splice = map_run.get("splice_results")
        splice_results = None
        if isinstance(
            map_splice,  # obj
            dict,        # class_or_tuple
        ):
            splice_results = TraceSpliceResults(
                schedule_mode=map_splice.get("schedule_mode"),
                schedule_prune=map_splice.get("schedule_prune"),
                map_site_report={
                    int(site_id): SiteSpliceReport(
                        list_map_splice=list(map_report.get("splices") or [])
                    )
                    for site_id, map_report in (map_splice.get("sites") or {}).items()
                },
            )
        return cls(
            kernel_prototype=map_run.get("kernel_prototype"),
            compile_results=TraceCompileResults.from_json(map_run.get("compile_results") or {}),
            grid_dim=map_run.get("grid_dim") or (map_analysis or {}).get("grid_dim"),
            block_dim=map_run.get("block_dim") or (map_analysis or {}).get("block_dim"),
            list_record=list_record,
            map_oe_results=dict(map_run.get("oe_results") or {}),
            splice_results=splice_results,
            map_saved_analysis=map_analysis,
            device=TraceDevice.from_json(map_run.get("device")),
        )
