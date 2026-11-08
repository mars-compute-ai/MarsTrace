import abc
import decimal
import gzip
import io
import json
import math
import os
import random
import re
import tarfile
import tomllib
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.binary import KernelBinary
from xtrace_harness.error import ManifestError
from xtrace_harness.listing import InspectionCache, KernelListing
from xtrace_harness.tensors import TensorEntry, TensorFile


@dataclass
class TraceSite:
    """One instruction of the traced kernel that writes a timestamp, and optionally a value.

    Args:
        index: the instruction index, as ``xtrace sass`` lists it
        value_register: the register recorded beside the timestamp, ``R<n>`` or ``UR<n>``
        value_width_bits: 32, or 64 to record the register pair starting at an even ``R<n>``
        value_is_smid: record the SM the block runs on instead of a register
        warp_level: one record per warp, written by an elected lane, instead of per thread
    """

    REGISTER_PATTERN = re.compile(r"^(UR|R)(\d+)$")
    # the highest register number of each register file a site may record
    MAP_REGISTER_MAX = {
        "R": 254,
        "UR": 62,
    }
    VALUE_WIDTH_DEFAULT = 32
    VALUE_WIDTH_WIDE = 64
    LIST_VALUE_WIDTH = (
        VALUE_WIDTH_DEFAULT,
        VALUE_WIDTH_WIDE,
    )
    SET_KEY = {
        "index",
        "value_register",
        "value_width_bits",
        "value_is_smid",
        "warp_level",
    }

    index: int | None = None
    value_register: str | None = None
    value_width_bits: int | None = None
    value_is_smid: bool = False
    warp_level: bool = False

    @classmethod
    def from_table(cls, map_site: Any) -> "TraceSite":
        """Read one ``[[trace.sites]]`` table.

        Args:
            map_site: the table

        Returns:
            the site

        Raises:
            ManifestError: the entry is not a table or has unknown keys
        """
        JobManifest.check_table(
            value=map_site,
            set_key=TraceSite.SET_KEY,
            where="[[trace.sites]]",
        )
        return cls(
            index=map_site.get("index"),
            value_register=map_site.get("value_register"),
            value_width_bits=map_site.get("value_width_bits"),
            value_is_smid=map_site.get(
                "value_is_smid",
                False,
            ),
            warp_level=map_site.get(
                "warp_level",
                False,
            ),
        )

    def validate(self, site_id: int) -> None:
        """Check that the site names an instruction and a value it can record.

        Args:
            site_id: the site's position in the manifest, for the message

        Raises:
            ManifestError: the site lacks its index or names an unknown register or width
        """
        where = f"site {site_id}"
        JobManifest.check_integer(
            value=self.index,
            where=f"{where} index",
            low=0,
        )
        if (
            not isinstance(
                self.value_is_smid,  # obj
                bool,                # class_or_tuple
            )
            or not isinstance(
                self.warp_level,  # obj
                bool,             # class_or_tuple
            )
        ):
            raise ManifestError(f"{where}: 'value_is_smid' and 'warp_level' are true or false")
        if (
            self.value_is_smid
            and self.value_register is not None
        ):
            raise ManifestError(f"{where}: record either the SM id or a register, not both")
        if (
            self.value_width_bits is not None
            and self.value_register is None
        ):
            raise ManifestError(f"{where}: 'value_width_bits' needs 'value_register'")
        if self.value_register is None:
            return
        match = TraceSite.REGISTER_PATTERN.match(str(self.value_register))
        if (
            match is None
            or int(match.group(2)) > TraceSite.MAP_REGISTER_MAX[match.group(1)]
        ):
            raise ManifestError(f"{where}: 'value_register' must be R0 to R254 or UR0 to UR62")
        width = self.width_bits()
        if width not in TraceSite.LIST_VALUE_WIDTH:
            raise ManifestError(f"{where}: 'value_width_bits' must be 32 or 64")
        if (
            width == TraceSite.VALUE_WIDTH_WIDE
            and (
                match.group(1) != "R"
                or int(match.group(2)) % 2 != 0
            )
        ):
            raise ManifestError(f"{where}: a 64-bit value needs an even R register, not {self.value_register}")

    def width_bits(self) -> int:
        """The width of the recorded register value.

        Returns:
            32 or 64
        """
        if self.value_width_bits is None:
            return TraceSite.VALUE_WIDTH_DEFAULT
        return self.value_width_bits

    def probe(self) -> str:
        """What the site records beside its timestamp.

        Returns:
            ``timing``, ``value`` or ``smid``
        """
        if self.value_is_smid:
            return "smid"
        if self.value_register is not None:
            return "value"
        return "timing"

    def to_api(self, site_id: int) -> dict[str, Any]:
        """The site as the trace plan carries it.

        Args:
            site_id: the site's position in the manifest

        Returns:
            the site
        """
        map_site: dict[str, Any] = {
            "site_id": f"s{site_id}",
            "instruction_index": self.index,
            "probe": self.probe(),
        }
        if self.value_register is not None:
            map_site["value_register"] = self.value_register
            map_site["value_width_bits"] = self.width_bits()
        return map_site


@dataclass
class TraceRegion:
    """A region of the kernel bounded by two sites: the one entering it and the one leaving it."""

    LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")

    enter: int
    leave: int
    label: str

    @classmethod
    def from_table(cls, map_region: Any) -> "TraceRegion":
        """Read one ``[[trace.regions]]`` table.

        Args:
            map_region: the table

        Returns:
            the region

        Raises:
            ManifestError: the entry lacks integer site ids
        """
        JobManifest.check_table(
            value=map_region,
            set_key={
                "enter",
                "leave",
                "label",
            },
            where="[[trace.regions]]",
        )
        for key in (
            "enter",
            "leave",
        ):
            JobManifest.check_integer(
                value=map_region.get(key),
                where=f"[[trace.regions]] {key}",
                low=0,
            )
        return cls(
            enter=map_region["enter"],
            leave=map_region["leave"],
            label=map_region.get(
                "label",
                f"s{map_region['enter']}-s{map_region['leave']}",
            ),
        )

    def validate(self, site_count: int) -> None:
        """Check the region's label and the sites it names.

        Args:
            site_count: how many sites the manifest declares

        Raises:
            ManifestError: the label is not an identifier or a site is absent
        """
        if not TraceRegion.LABEL_PATTERN.match(str(self.label)):
            raise ManifestError(
                f"region label '{self.label}' must be 1 to 200 letters, digits, '_', '.', ':' "
                "or '-'"
            )
        for site_id in (
            self.enter,
            self.leave,
        ):
            if site_id >= site_count:
                raise ManifestError(
                    f"region '{self.label}' names site {site_id}, but the manifest has "
                    f"{site_count} site(s)"
                )
        if self.enter == self.leave:
            raise ManifestError(f"region '{self.label}' enters and leaves at the same site")

    def to_api(self) -> dict[str, str]:
        """The region as the trace plan carries it.

        Returns:
            the region id and its two site ids
        """
        return {
            "region_id": self.label,
            "start_site_id": f"s{self.enter}",
            "end_site_id": f"s{self.leave}",
        }


@dataclass
class TraceOptions:
    """The ``[trace]`` table: which instructions record, how, and what else the GPU measures."""

    LIST_CLOCK = (
        "gpu",
        "sm",
        "anchor",
    )
    LIST_SCHEDULE = (
        "first_feasible",
        "least_makespan",
    )
    LIST_STAGE = (
        "xtrace",
        "range_profile",
        "pc_sampling",
    )
    SET_KEY = {
        "clock",
        "schedule",
        "per_thread_records",
        "stages",
        "sites",
        "regions",
        "scope_roles",
        "range_profile",
        "pc_sampling",
    }
    PER_THREAD_RECORDS_DEFAULT = 32
    PER_THREAD_RECORDS_MAX = 65536
    MAX_RECORDS_MAX = 1_000_000_000
    SITE_COUNT_MAX = 4096
    REGION_COUNT_MAX = 2048
    SCOPE_ROLE_COUNT_MAX = 64
    SCOPE_ROLE_CHARACTERS_MAX = 64
    METRIC_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,199}$")
    METRIC_COUNT_MAX = 16
    INVOCATIONS_MIN = 2
    INVOCATIONS_MAX = 128
    REPEATS_MAX = 100

    clock: str = "gpu"
    schedule: str = "first_feasible"
    per_thread_records: int = PER_THREAD_RECORDS_DEFAULT
    list_stage: list[str] = field(default_factory=lambda: ["xtrace"])
    list_site: list[TraceSite] = field(default_factory=list)
    list_region: list[TraceRegion] = field(default_factory=list)
    # region label -> the warp role that executes it, for the report's lanes
    map_scope_role: dict[str, str] = field(default_factory=dict)
    map_range_profile: dict[str, Any] | None = None
    map_pc_sampling: dict[str, Any] | None = None

    @classmethod
    def from_table(cls, map_trace: Any) -> "TraceOptions":
        """Read the ``[trace]`` table.

        Args:
            map_trace: the table

        Returns:
            the options

        Raises:
            ManifestError: the table has unknown keys
        """
        JobManifest.check_table(
            value=map_trace,
            set_key=TraceOptions.SET_KEY,
            where="[trace]",
        )
        map_range_profile = map_trace.get("range_profile")
        if map_range_profile is not None:
            JobManifest.check_table(
                value=map_range_profile,
                set_key={
                    "metrics",
                    "max_invocations",
                },
                where="[trace.range_profile]",
            )
        map_pc_sampling = map_trace.get("pc_sampling")
        if map_pc_sampling is not None:
            JobManifest.check_table(
                value=map_pc_sampling,
                set_key={"repeats"},
                where="[trace.pc_sampling]",
            )
        map_scope_role = map_trace.get(
            "scope_roles",
            {},
        )
        if not isinstance(
            map_scope_role,  # obj
            dict,            # class_or_tuple
        ):
            raise ManifestError("[trace.scope_roles] must be a table of region label = role")
        return cls(
            clock=map_trace.get(
                "clock",
                "gpu",
            ),
            schedule=map_trace.get(
                "schedule",
                "first_feasible",
            ),
            per_thread_records=map_trace.get(
                "per_thread_records",
                TraceOptions.PER_THREAD_RECORDS_DEFAULT,
            ),
            list_stage=list(
                map_trace.get(
                    "stages",
                    ["xtrace"],
                )
            ),
            list_site=[TraceSite.from_table(map_site) for map_site in map_trace.get("sites", [])],
            list_region=[
                TraceRegion.from_table(map_region) for map_region in map_trace.get("regions", [])
            ],
            map_scope_role=dict(map_scope_role),
            map_range_profile=map_range_profile,
            map_pc_sampling=map_pc_sampling,
        )

    def validate(self) -> None:
        """Check the options against each other and against the sites they name.

        Raises:
            ManifestError: an option is out of range or a region names an absent site
        """
        if self.clock not in TraceOptions.LIST_CLOCK:
            raise ManifestError(f"trace.clock must be one of {', '.join(TraceOptions.LIST_CLOCK)}")
        if self.schedule not in TraceOptions.LIST_SCHEDULE:
            raise ManifestError(
                f"trace.schedule must be one of {', '.join(TraceOptions.LIST_SCHEDULE)}"
            )
        JobManifest.check_integer(
            value=self.per_thread_records,
            where="trace.per_thread_records",
            low=1,
            high=TraceOptions.PER_THREAD_RECORDS_MAX,
        )
        self._validate_stages()
        if (
            "xtrace" in self.list_stage
            and not self.list_site
        ):
            raise ManifestError("the xtrace stage needs at least one [[trace.sites]] entry")
        if len(self.list_site) > TraceOptions.SITE_COUNT_MAX:
            raise ManifestError(f"a trace has at most {TraceOptions.SITE_COUNT_MAX} sites")
        for site_id, site in enumerate(self.list_site):
            site.validate(site_id)
        if len({site.warp_level for site in self.list_site}) > 1:
            raise ManifestError("set warp_level alike on every site: a trace records per thread or per warp")
        if (
            self.clock == "anchor"
            and self.granularity() != "warp"
        ):
            raise ManifestError("trace.clock = 'anchor' needs every site to be warp_level")
        if len(self.list_region) > TraceOptions.REGION_COUNT_MAX:
            raise ManifestError(f"a trace has at most {TraceOptions.REGION_COUNT_MAX} regions")
        set_label = set()
        for region in self.list_region:
            region.validate(len(self.list_site))
            if region.label in set_label:
                raise ManifestError(f"region label '{region.label}' is used twice")
            set_label.add(region.label)
        if len(self.map_scope_role) > TraceOptions.SCOPE_ROLE_COUNT_MAX:
            raise ManifestError(f"[trace.scope_roles] has at most {TraceOptions.SCOPE_ROLE_COUNT_MAX} entries")
        for label, role in self.map_scope_role.items():
            if label not in set_label:
                raise ManifestError(f"[trace.scope_roles] names region '{label}', which [[trace.regions]] lacks")
            if (
                not isinstance(
                    role,  # obj
                    str,   # class_or_tuple
                )
                or not 1 <= len(role) <= TraceOptions.SCOPE_ROLE_CHARACTERS_MAX
            ):
                raise ManifestError(
                    f"[trace.scope_roles] {label}: a role is 1 to "
                    f"{TraceOptions.SCOPE_ROLE_CHARACTERS_MAX} characters"
                )

    def _validate_stages(self) -> None:
        """Check the selected stages and that each profiler stage has exactly its own options."""
        if (
            not self.list_stage
            or len(set(self.list_stage)) != len(self.list_stage)
            or not set(self.list_stage) <= set(TraceOptions.LIST_STAGE)
        ):
            raise ManifestError(
                f"trace.stages lists distinct stages from {', '.join(TraceOptions.LIST_STAGE)}"
            )
        if ("range_profile" in self.list_stage) != (self.map_range_profile is not None):
            raise ManifestError("[trace.range_profile] is given exactly when trace.stages has 'range_profile'")
        if ("pc_sampling" in self.list_stage) != (self.map_pc_sampling is not None):
            raise ManifestError("[trace.pc_sampling] is given exactly when trace.stages has 'pc_sampling'")
        if self.map_range_profile is not None:
            list_metric = self.map_range_profile.get("metrics")
            if (
                not isinstance(
                    list_metric,  # obj
                    list,         # class_or_tuple
                )
                or not 1 <= len(list_metric) <= TraceOptions.METRIC_COUNT_MAX
                or len(set(list_metric)) != len(list_metric)
                or not all(
                    isinstance(
                        metric,  # obj
                        str,     # class_or_tuple
                    )
                    and TraceOptions.METRIC_PATTERN.match(metric)
                    for metric in list_metric
                )
            ):
                raise ManifestError(
                    f"[trace.range_profile] metrics lists 1 to {TraceOptions.METRIC_COUNT_MAX} "
                    "distinct metric names"
                )
            JobManifest.check_integer(
                value=self.map_range_profile.get("max_invocations"),
                where="[trace.range_profile] max_invocations",
                low=TraceOptions.INVOCATIONS_MIN,
                high=TraceOptions.INVOCATIONS_MAX,
            )
        if self.map_pc_sampling is not None:
            JobManifest.check_integer(
                value=self.map_pc_sampling.get("repeats"),
                where="[trace.pc_sampling] repeats",
                low=1,
                high=TraceOptions.REPEATS_MAX,
            )

    def granularity(self) -> str:
        """Whether the trace records per thread or per warp.

        Returns:
            ``thread`` or ``warp``
        """
        if (
            self.list_site
            and all(site.warp_level for site in self.list_site)
        ):
            return "warp"
        return "thread"

    def to_plan(
        self,
        listing: KernelListing | None,
        launch: "LaunchSpec",
    ) -> dict[str, Any]:
        """The trace plan the API takes.

        Args:
            listing: the inspected kernel, whose listing every site must name when it is cached
            launch: the launch, whose rows size the record budget

        Returns:
            the plan

        Raises:
            ManifestError: two sites probe one instruction
            HarnessError: the listing lacks a site's instruction
        """
        granularity = self.granularity()
        rows = launch.rows(granularity)
        max_records = self.per_thread_records * rows
        if max_records > TraceOptions.MAX_RECORDS_MAX:
            raise ManifestError(
                f"per_thread_records {self.per_thread_records} x {rows} {granularity} rows is "
                f"{max_records} records; the platform takes at most {TraceOptions.MAX_RECORDS_MAX}"
            )
        list_api_site = []
        map_site_of_index: dict[int, int] = {}
        for site_id, site in enumerate(self.list_site):
            instruction_index = int(site.index or 0)
            if listing is not None:
                listing.check_index(instruction_index)
            if instruction_index in map_site_of_index:
                raise ManifestError(
                    f"sites {map_site_of_index[instruction_index]} and {site_id} both probe "
                    f"instruction {instruction_index}"
                )
            map_site_of_index[instruction_index] = site_id
            list_api_site.append(site.to_api(site_id))
        map_plan: dict[str, Any] = {
            "tier": "sass",
            "clock": self.clock,
            "schedule": self.schedule,
            "sites": list_api_site,
            "regions": [region.to_api() for region in self.list_region],
            "granularity": granularity,
            "max_records": max_records,
            "capture_values": any(site.probe() != "timing" for site in self.list_site),
            "stages": list(self.list_stage),
        }
        if self.map_range_profile is not None:
            map_plan["range_profile"] = {
                "metrics": list(self.map_range_profile["metrics"]),
                "max_invocations": self.map_range_profile["max_invocations"],
            }
        if self.map_pc_sampling is not None:
            map_plan["pc_sampling"] = {"repeats": self.map_pc_sampling["repeats"]}
        if self.map_scope_role:
            map_plan["scope_roles"] = dict(self.map_scope_role)
        return map_plan


@dataclass
class PlannerOptions:
    """The ``[planner]`` table: the platform picks the trace sites from the intent, with a model."""

    MAX_CALLS_MAX = 20
    MAX_INPUT_TOKENS_MAX = 1_000_000
    MAX_OUTPUT_TOKENS_MAX = 100_000
    MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")

    model: str
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int

    @classmethod
    def from_table(cls, map_planner: Any) -> "PlannerOptions":
        """Read and check the ``[planner]`` table.

        Args:
            map_planner: the table

        Returns:
            the options

        Raises:
            ManifestError: a key is missing, unknown or out of range
        """
        JobManifest.check_table(
            value=map_planner,
            set_key={
                "model",
                "max_calls",
                "max_input_tokens",
                "max_output_tokens",
            },
            where="[planner]",
        )
        if not PlannerOptions.MODEL_PATTERN.match(str(map_planner.get("model", ""))):
            raise ManifestError("[planner] needs 'model', one of the models 'xtrace hardware' lists")
        for key, high in (
            ("max_calls", PlannerOptions.MAX_CALLS_MAX),
            ("max_input_tokens", PlannerOptions.MAX_INPUT_TOKENS_MAX),
            ("max_output_tokens", PlannerOptions.MAX_OUTPUT_TOKENS_MAX),
        ):
            JobManifest.check_integer(
                value=map_planner.get(key),
                where=f"[planner] {key}",
                low=1,
                high=high,
            )
        return cls(
            model=map_planner["model"],
            max_calls=map_planner["max_calls"],
            max_input_tokens=map_planner["max_input_tokens"],
            max_output_tokens=map_planner["max_output_tokens"],
        )

    def to_analysis(self) -> dict[str, Any]:
        """The planned analysis the API takes.

        Returns:
            the analysis, mode ``platform_planned``
        """
        return {
            "mode": "platform_planned",
            "model": self.model,
            "max_calls": self.max_calls,
            "max_input_tokens_total": self.max_input_tokens,
            "max_output_tokens_total": self.max_output_tokens,
            "capture_values": False,
        }


@dataclass
class JobLimits:
    """The ``[limits]`` table: the most a job may cost and use, over the platform's defaults."""

    MICRO_PER_UNIT = 1_000_000
    # manifest key -> (API key, highest value)
    MAP_KEY = {
        "max_gpu_seconds": ("max_active_gpu_seconds", 86400),
        "max_wall_seconds": ("max_wall_seconds", 172800),
        "max_attempts": ("max_attempts", 10),
        "max_trace_bytes": ("max_trace_bytes", 1099511627776),
    }
    COST_KEY = "max_cost_usd"

    map_limit: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_table(cls, map_limits: Any) -> "JobLimits":
        """Read and check the ``[limits]`` table.

        Args:
            map_limits: the table

        Returns:
            the limits

        Raises:
            ManifestError: a key is unknown or out of range
        """
        JobManifest.check_table(
            value=map_limits,
            set_key=set(JobLimits.MAP_KEY) | {JobLimits.COST_KEY},
            where="[limits]",
        )
        for key, (_, high) in JobLimits.MAP_KEY.items():
            if key in map_limits:
                JobManifest.check_integer(
                    value=map_limits[key],
                    where=f"[limits] {key}",
                    low=1,
                    high=high,
                )
        if JobLimits.COST_KEY in map_limits:
            JobLimits.micro_usd(map_limits[JobLimits.COST_KEY])
        return cls(map_limit=dict(map_limits))

    @staticmethod
    def micro_usd(usd: Any) -> str:
        """A dollar amount as the API's micro-USD money string, rounded down.

        Args:
            usd: the amount in dollars

        Returns:
            the decimal string of whole micro-USD

        Raises:
            ManifestError: the amount is not a positive number
        """
        if (
            isinstance(
                usd,   # obj
                bool,  # class_or_tuple
            )
            or not isinstance(
                usd,           # obj
                (int, float),  # class_or_tuple
            )
        ):
            raise ManifestError("[limits] max_cost_usd must be a number of dollars")
        micro = int(
            (decimal.Decimal(str(usd)) * JobLimits.MICRO_PER_UNIT).to_integral_value(
                rounding=decimal.ROUND_DOWN,
            )
        )
        if micro <= 0:
            raise ManifestError("[limits] max_cost_usd must be at least $0.000001")
        return str(micro)

    def to_api(self, map_default: dict[str, Any]) -> dict[str, Any]:
        """The limits the API takes: the manifest's values over the platform's defaults.

        Args:
            map_default: the capabilities' ``limits``

        Returns:
            the limits

        Raises:
            ManifestError: the GPU-time allowance exceeds the wall-clock limit
        """
        map_api = dict(map_default)
        for key, (api_key, _) in JobLimits.MAP_KEY.items():
            if key in self.map_limit:
                map_api[api_key] = self.map_limit[key]
        if JobLimits.COST_KEY in self.map_limit:
            map_api["max_total_cost_micro_usd"] = JobLimits.micro_usd(self.map_limit[JobLimits.COST_KEY])
        if map_api.get("max_active_gpu_seconds", 0) > map_api.get(
            "max_wall_seconds",
            math.inf,
        ):
            raise ManifestError(
                f"max_gpu_seconds {map_api['max_active_gpu_seconds']} exceeds max_wall_seconds "
                f"{map_api['max_wall_seconds']}"
            )
        return map_api


@dataclass
class KernelSpec:
    """The kernel a job traces: the cubin that holds it, its entry point and its sources.

    Args:
        binary: the cubin, relative to the job directory
        name: the kernel's entry point, as the cubin names it (mangled for C++)
        list_source: source files the viewer shows beside the listing; never uploaded
    """

    binary: str
    name: str
    list_source: list[str] = field(default_factory=list)

    @classmethod
    def from_table(cls, map_kernel: Any) -> "KernelSpec":
        """Read the ``[kernel]`` table.

        Args:
            map_kernel: the table

        Returns:
            the kernel

        Raises:
            ManifestError: the table lacks the binary or the name
        """
        JobManifest.check_table(
            value=map_kernel,
            set_key={
                "binary",
                "name",
                "sources",
            },
            where="[kernel]",
        )
        for key in (
            "binary",
            "name",
        ):
            if not isinstance(
                map_kernel.get(key),  # obj
                str,                  # class_or_tuple
            ):
                raise ManifestError(f"[kernel] needs '{key}'")
        return cls(
            binary=map_kernel["binary"],
            name=map_kernel["name"],
            list_source=list(map_kernel.get("sources", [])),
        )


@dataclass
class BufferSpec:
    """A device allocation the platform makes for the launch, and what it holds before it.

    A ``ones`` or ``random`` buffer starts from a tensor the harness prepares and uploads
    with the job's tensors, so the launch restores the same bytes before every invocation.

    Args:
        name: the buffer's name, which launch arguments refer to
        dtype: the element type, e.g. ``f16`` or ``i32``
        list_shape: the extents; the buffer holds their product of elements
        init: ``zeros``, ``ones``, ``random``, or ``tensor`` for the tensor of the same name
            in the job's tensors file
        seed: the seed of a ``random`` buffer
        is_output: the kernel writes it; the launch declares it an output and checks it
    """

    # element type -> (manifest name, bytes per element)
    MAP_DTYPE = {
        "bool": ("bool", 1),
        "u8": ("uint8", 1),
        "i8": ("int8", 1),
        "f8e4m3": ("float8_e4m3fn", 1),
        "f8e5m2": ("float8_e5m2", 1),
        "u16": ("uint16", 2),
        "i16": ("int16", 2),
        "f16": ("float16", 2),
        "bf16": ("bfloat16", 2),
        "u32": ("uint32", 4),
        "i32": ("int32", 4),
        "f32": ("float32", 4),
        "u64": ("uint64", 8),
        "i64": ("int64", 8),
        "f64": ("float64", 8),
    }
    # element type -> the dtype its output is declared and compared as, where it differs
    MAP_OUTPUT_DTYPE = {
        "f8e4m3": "uint8",
        "f8e5m2": "uint8",
    }
    # element type -> the bits of one element of value one
    MAP_ONE_BITS = {
        "bool": 0x01,
        "u8": 0x01,
        "i8": 0x01,
        "f8e4m3": 0x38,
        "f8e5m2": 0x3C,
        "u16": 0x0001,
        "i16": 0x0001,
        "f16": 0x3C00,
        "bf16": 0x3F80,
        "u32": 0x00000001,
        "i32": 0x00000001,
        "f32": 0x3F800000,
        "u64": 0x0000000000000001,
        "i64": 0x0000000000000001,
        "f64": 0x3FF0000000000000,
    }
    # element type -> (byte of the element, AND mask, OR mask) applied to random bytes, which
    # keeps the sign and mantissa and sets the exponent of a magnitude in [0.5, 1); integer
    # types keep every random bit
    MAP_RANDOM_BYTE_MASK = {
        "bool": ((0, 0x01, 0x00),),
        "f8e4m3": ((0, 0x87, 0x30),),
        "f8e5m2": ((0, 0x83, 0x38),),
        "f16": ((1, 0x83, 0x38),),
        "bf16": ((1, 0x80, 0x3F), (0, 0x7F, 0x00)),
        "f32": ((3, 0x80, 0x3F), (2, 0x7F, 0x00)),
        "f64": ((7, 0x80, 0x3F), (6, 0x0F, 0xE0)),
    }
    RANDOM_ALGORITHM = "mt19937-bytes-unit-magnitude-v1"
    BYTE_VALUES = 256
    LIST_INIT = (
        "zeros",
        "ones",
        "random",
        "tensor",
    )
    SET_PREPARED_INIT = {
        "ones",
        "random",
    }
    NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    RESERVED_PREFIX = "_xtrace"
    RANK_MAX = 8
    SEED_MAX = 2**32 - 1

    name: str
    dtype: str
    list_shape: list[int]
    init: str = "zeros"
    seed: int = 0
    is_output: bool = False

    @classmethod
    def from_table(cls, map_buffer: Any) -> "BufferSpec":
        """Read one ``[[buffers]]`` table.

        Args:
            map_buffer: the table

        Returns:
            the buffer

        Raises:
            ManifestError: the table lacks a field or has unknown keys
        """
        JobManifest.check_table(
            value=map_buffer,
            set_key={
                "name",
                "dtype",
                "shape",
                "init",
                "seed",
                "output",
            },
            where="[[buffers]]",
        )
        for key in (
            "name",
            "dtype",
            "shape",
        ):
            if key not in map_buffer:
                raise ManifestError(f"each [[buffers]] entry needs '{key}'")
        if (
            "seed" in map_buffer
            and map_buffer.get("init") != "random"
        ):
            raise ManifestError(f"buffer '{map_buffer['name']}': 'seed' belongs to init = 'random'")
        return cls(
            name=map_buffer["name"],
            dtype=map_buffer["dtype"],
            list_shape=list(map_buffer["shape"]),
            init=map_buffer.get(
                "init",
                "zeros",
            ),
            seed=map_buffer.get(
                "seed",
                0,
            ),
            is_output=bool(
                map_buffer.get(
                    "output",
                    False,
                )
            ),
        )

    def nbytes(self) -> int:
        """The buffer's size.

        Returns:
            the size in bytes
        """
        return BufferSpec.MAP_DTYPE[self.dtype][1] * TensorFile.element_count(self.list_shape)

    def validate(self, tensor_file: TensorFile | None) -> None:
        """Check the buffer's type and shape, and its tensor when it starts from the tensors file.

        Args:
            tensor_file: the job's tensors file, or None when it has none

        Raises:
            ManifestError: the buffer is malformed or does not match its tensor
        """
        where = f"buffer '{self.name}'"
        if (
            not BufferSpec.NAME_PATTERN.match(str(self.name))
            or str(self.name).startswith(BufferSpec.RESERVED_PREFIX)
        ):
            raise ManifestError(
                f"{where}: the name must be an identifier of 1 to 128 letters, digits and '_', "
                f"not starting with '{BufferSpec.RESERVED_PREFIX}'"
            )
        if self.dtype not in BufferSpec.MAP_DTYPE:
            raise ManifestError(f"{where}: dtype must be one of {', '.join(BufferSpec.MAP_DTYPE)}")
        if (
            not 1 <= len(self.list_shape) <= BufferSpec.RANK_MAX
            or not all(
                isinstance(
                    extent,  # obj
                    int,     # class_or_tuple
                )
                and extent > 0
                for extent in self.list_shape
            )
        ):
            raise ManifestError(
                f"{where}: shape must be 1 to {BufferSpec.RANK_MAX} positive integers"
            )
        if self.init not in BufferSpec.LIST_INIT:
            raise ManifestError(f"{where}: init must be one of {', '.join(BufferSpec.LIST_INIT)}")
        JobManifest.check_integer(
            value=self.seed,
            where=f"{where} seed",
            low=0,
            high=BufferSpec.SEED_MAX,
        )
        if self.init != "tensor":
            return
        if tensor_file is None:
            raise ManifestError(f"{where} starts from a tensor, but the job names no 'tensors' file")
        self.check_tensor(
            tensor_file=tensor_file,
            where=where,
        )

    def check_tensor(
        self,
        tensor_file: TensorFile,
        where: str,
    ) -> None:
        """Check that a safetensors file holds a tensor of the buffer's name, type and shape.

        Args:
            tensor_file: the file
            where: what the check is for, for the message

        Raises:
            ManifestError: the tensor is absent or differs
        """
        tensor = tensor_file.map_tensor.get(self.name)
        if tensor is None:
            raise ManifestError(f"{where}: {tensor_file.path} has no tensor '{self.name}'")
        if (
            tensor.dtype != self.dtype
            or tensor.list_shape != self.list_shape
        ):
            raise ManifestError(
                f"{where}: buffer '{self.name}' is {self.dtype}{self.list_shape}, but tensor "
                f"'{self.name}' of {tensor_file.path} is {tensor.dtype}{tensor.list_shape}"
            )

    def is_prepared(self) -> bool:
        """Whether the harness prepares the buffer's initial contents as a tensor.

        Returns:
            whether it starts as ``ones`` or ``random``
        """
        return self.init in BufferSpec.SET_PREPARED_INIT

    def tensor_entry(self) -> TensorEntry:
        """The tensor the buffer starts from, named after the buffer.

        Returns:
            the entry
        """
        return TensorEntry(
            name=self.name,
            dtype=self.dtype,
            list_shape=list(self.list_shape),
            nbytes=self.nbytes(),
        )

    def prepared_bytes(self) -> bytes:
        """The initial contents of a ``ones`` or ``random`` buffer.

        A random buffer holds the bytes of a Mersenne Twister seeded with ``seed``; a
        floating-point one has each element's exponent set to a magnitude in [0.5, 1).

        Returns:
            the bytes, little-endian
        """
        width = BufferSpec.MAP_DTYPE[self.dtype][1]
        count = TensorFile.element_count(self.list_shape)
        if self.init == "ones":
            return BufferSpec.MAP_ONE_BITS[self.dtype].to_bytes(
                length=width,
                byteorder="little",
            ) * count
        data = bytearray(random.Random(self.seed).randbytes(width * count))
        for position, and_mask, or_mask in BufferSpec.MAP_RANDOM_BYTE_MASK.get(self.dtype, ()):
            table = bytes((value & and_mask) | or_mask for value in range(BufferSpec.BYTE_VALUES))
            data[position::width] = data[position::width].translate(table)
        return bytes(data)

    def to_allocation(self) -> dict[str, Any]:
        """The buffer's storage as the launch manifest carries it.

        Returns:
            the allocation: its id, size and initializer
        """
        map_init: dict[str, Any] = {"kind": "zeros"}
        if self.init != "zeros":
            map_init = {
                "kind": "tensor",
                "tensor_key": self.name,
            }
        return {
            "id": self.name,
            "size_bytes": self.nbytes(),
            "init": map_init,
        }

    def to_view(self) -> dict[str, Any]:
        """The buffer as the launch manifest's view of its whole allocation.

        Returns:
            the view: its name, allocation, extent and access
        """
        return {
            "name": self.name,
            "allocation_id": self.name,
            "offset_bytes": 0,
            "size_bytes": self.nbytes(),
            "access": "read_write" if self.is_output else "read",
        }

    def to_output(self) -> dict[str, Any]:
        """The buffer as an output of the launch manifest.

        Returns:
            the output: its name, view, element type and shape
        """
        return {
            "name": self.name,
            "buffer": self.name,
            "dtype": BufferSpec.MAP_OUTPUT_DTYPE.get(
                self.dtype,
                BufferSpec.MAP_DTYPE[self.dtype][0],
            ),
            "shape": list(self.list_shape),
        }


@dataclass
class PointerRelocation:
    """A place in a by-value argument where the platform writes a buffer's device address.

    Args:
        offset: the byte offset in the argument, a multiple of 8
        buffer: the buffer whose address is written
        offset_bytes: added to the buffer's address
    """

    POINTER_BYTES = 8

    offset: int
    buffer: str
    offset_bytes: int = 0

    def to_manifest(self) -> dict[str, Any]:
        """The relocation as the launch manifest carries it.

        Returns:
            the offset, the buffer and the byte offset into it
        """
        return {
            "offset": self.offset,
            "buffer": self.buffer,
            "offset_bytes": self.offset_bytes,
        }


@dataclass
class LaunchArgument:
    """One kernel parameter, in the order of the kernel's signature.

    ``buffer``: a buffer's device address (plus ``offset_bytes``).
    ``scalar``: a number of type ``dtype``.
    ``tensor_map``: a TMA descriptor the platform encodes over a buffer with
    ``cuTensorMapEncodeTiled``'s parameters.
    ``bytes``: a by-value struct, given as hex, with ``relocations`` where buffer addresses go.
    """

    LIST_KIND = (
        "buffer",
        "scalar",
        "tensor_map",
        "bytes",
    )
    # scalar type -> (manifest ABI type, lowest value, highest value, bytes)
    MAP_SCALAR = {
        "i8": ("int8", -(2**7), 2**7 - 1, 1),
        "u8": ("uint8", 0, 2**8 - 1, 1),
        "i16": ("int16", -(2**15), 2**15 - 1, 2),
        "u16": ("uint16", 0, 2**16 - 1, 2),
        "i32": ("int32", -(2**31), 2**31 - 1, 4),
        "u32": ("uint32", 0, 2**32 - 1, 4),
        "i64": ("int64", -(2**63), 2**63 - 1, 8),
        "u64": ("uint64", 0, 2**64 - 1, 8),
        "f32": ("float32", None, None, 4),
        "f64": ("float64", None, None, 8),
    }
    FLOAT32_MAX = 3.4028234663852886e38
    SET_TENSOR_MAP_KEY = {
        "data_type",
        "global_dims",
        "global_strides_bytes",
        "box_dims",
        "element_strides",
        "interleave",
        "swizzle",
        "l2_promotion",
        "oob_fill",
    }
    # tensor map element type -> bytes per element
    MAP_TENSOR_MAP_WIDTH = {
        "u8": 1,
        "u16": 2,
        "u32": 4,
        "i32": 4,
        "u64": 8,
        "i64": 8,
        "f16": 2,
        "f32": 4,
        "f64": 8,
        "bf16": 2,
        "f32_ftz": 4,
        "tf32": 4,
        "tf32_ftz": 4,
    }
    # the choices of each enumerated tensor map field
    MAP_TENSOR_MAP_CHOICE = {
        "interleave": (
            "none",
            "16b",
            "32b",
        ),
        "swizzle": (
            "none",
            "32b",
            "64b",
            "128b",
        ),
        "l2_promotion": (
            "none",
            "64b",
            "128b",
            "256b",
        ),
        "oob_fill": (
            "none",
            "nan",
        ),
    }
    TENSOR_MAP_RANK_MAX = 5
    GLOBAL_EXTENT_MAX = 2**32
    GLOBAL_STRIDE_MAX = 2**40 - 1
    BOX_EXTENT_MAX = 256
    ELEMENT_STRIDE_MAX = 8
    TENSOR_MAP_BYTES = 128
    # alignment of a buffer offset and of a tensor map's strides
    ADDRESS_ALIGNMENT_BYTES = 16
    # the largest by-value argument the launch manifest carries
    BYTES_MAX = 4096

    kind: str
    buffer: str | None = None
    offset_bytes: int = 0
    dtype: str | None = None
    value: int | float | None = None
    map_tensor_map: dict[str, Any] = field(default_factory=dict)
    data_hex: str | None = None
    list_relocation: list[PointerRelocation] = field(default_factory=list)

    @classmethod
    def from_table(cls, map_argument: Any) -> "LaunchArgument":
        """Read one entry of ``launch.args``.

        Args:
            map_argument: the inline table

        Returns:
            the argument

        Raises:
            ManifestError: the entry is not a table or has unknown keys
        """
        JobManifest.check_table(
            value=map_argument,
            set_key={
                "kind",
                "buffer",
                "offset_bytes",
                "dtype",
                "value",
                "hex",
                "relocations",
            } | LaunchArgument.SET_TENSOR_MAP_KEY,
            where="launch.args",
        )
        list_relocation = []
        for map_relocation in map_argument.get("relocations", []):
            JobManifest.check_table(
                value=map_relocation,
                set_key={
                    "offset",
                    "buffer",
                    "offset_bytes",
                },
                where="launch.args relocations",
            )
            list_relocation.append(
                PointerRelocation(
                    offset=map_relocation.get(
                        "offset",
                        -1,
                    ),
                    buffer=map_relocation.get(
                        "buffer",
                        "",
                    ),
                    offset_bytes=map_relocation.get(
                        "offset_bytes",
                        0,
                    ),
                )
            )
        return cls(
            kind=map_argument.get(
                "kind",
                "",
            ),
            buffer=map_argument.get("buffer"),
            offset_bytes=map_argument.get(
                "offset_bytes",
                0,
            ),
            dtype=map_argument.get("dtype"),
            value=map_argument.get("value"),
            map_tensor_map={
                key: map_argument[key]
                for key in LaunchArgument.SET_TENSOR_MAP_KEY
                if key in map_argument
            },
            data_hex=map_argument.get("hex"),
            list_relocation=list_relocation,
        )

    def validate(
        self,
        argument_id: int,
        map_buffer: dict[str, BufferSpec],
    ) -> None:
        """Check the argument against its kind and the buffers it names.

        Args:
            argument_id: the argument's position, for the message
            map_buffer: the job's buffers by name

        Raises:
            ManifestError: the argument is malformed or names an absent buffer
        """
        where = f"launch argument {argument_id}"
        if self.kind not in LaunchArgument.LIST_KIND:
            raise ManifestError(f"{where}: kind must be one of {', '.join(LaunchArgument.LIST_KIND)}")
        if self.kind in (
            "buffer",
            "tensor_map",
        ):
            if self.buffer not in map_buffer:
                raise ManifestError(f"{where}: names buffer '{self.buffer}', which [[buffers]] lacks")
            JobManifest.check_integer(
                value=self.offset_bytes,
                where=f"{where} offset_bytes",
                low=0,
                high=map_buffer[self.buffer].nbytes() - 1,
            )
            if self.offset_bytes % LaunchArgument.ADDRESS_ALIGNMENT_BYTES != 0:
                raise ManifestError(
                    f"{where}: offset_bytes must be a multiple of "
                    f"{LaunchArgument.ADDRESS_ALIGNMENT_BYTES}"
                )
        if self.kind == "scalar":
            self._validate_scalar(where)
        if self.kind == "tensor_map":
            self._validate_tensor_map(
                where=where,
                buffer_bytes=map_buffer[str(self.buffer)].nbytes(),
            )
        if self.kind == "bytes":
            self._validate_bytes(
                where=where,
                map_buffer=map_buffer,
            )

    def _validate_scalar(self, where: str) -> None:
        """Check a scalar's type and that its value fits it."""
        if self.dtype not in LaunchArgument.MAP_SCALAR:
            raise ManifestError(
                f"{where}: a scalar's dtype is one of {', '.join(LaunchArgument.MAP_SCALAR)}; pass "
                "a 16-bit float as its bits in a u16"
            )
        _, value_min, value_max, _ = LaunchArgument.MAP_SCALAR[self.dtype]
        if value_min is None:
            if (
                isinstance(
                    self.value,  # obj
                    bool,        # class_or_tuple
                )
                or not isinstance(
                    self.value,    # obj
                    (int, float),  # class_or_tuple
                )
                or not math.isfinite(self.value)
            ):
                raise ManifestError(f"{where}: value must be a finite number")
            if (
                self.dtype == "f32"
                and abs(self.value) > LaunchArgument.FLOAT32_MAX
            ):
                raise ManifestError(f"{where}: value {self.value} overflows f32")
            return
        if (
            isinstance(
                self.value,  # obj
                bool,        # class_or_tuple
            )
            or not isinstance(
                self.value,  # obj
                int,         # class_or_tuple
            )
            or not value_min <= self.value <= value_max
        ):
            raise ManifestError(f"{where}: value must be an integer that fits {self.dtype}")

    def _validate_tensor_map(
        self,
        where: str,
        buffer_bytes: int,
    ) -> None:
        """Check a TMA descriptor's enumerations, extents and strides, and that it fits its buffer."""
        for key in (
            "data_type",
            "global_dims",
            "global_strides_bytes",
            "box_dims",
        ):
            if key not in self.map_tensor_map:
                raise ManifestError(f"{where}: a tensor_map needs '{key}'")
        if self.map_tensor_map["data_type"] not in LaunchArgument.MAP_TENSOR_MAP_WIDTH:
            raise ManifestError(
                f"{where}: data_type must be one of "
                f"{', '.join(LaunchArgument.MAP_TENSOR_MAP_WIDTH)}"
            )
        for key, list_choice in LaunchArgument.MAP_TENSOR_MAP_CHOICE.items():
            if self.map_tensor_map.get(
                key,
                "none",
            ) not in list_choice:
                raise ManifestError(f"{where}: {key} must be one of {', '.join(list_choice)}")
        rank = len(self.map_tensor_map["global_dims"])
        if not 1 <= rank <= LaunchArgument.TENSOR_MAP_RANK_MAX:
            raise ManifestError(
                f"{where}: global_dims must have 1 to {LaunchArgument.TENSOR_MAP_RANK_MAX} entries"
            )
        list_stride = self.map_tensor_map["global_strides_bytes"]
        list_box = self.map_tensor_map["box_dims"]
        list_element_stride = self.map_tensor_map.get(
            "element_strides",
            [1] * rank,
        )
        if len(list_stride) != rank - 1:
            raise ManifestError(f"{where}: global_strides_bytes needs one entry per dimension after the first")
        if len(list_box) != rank:
            raise ManifestError(f"{where}: box_dims needs one entry per dimension")
        if len(list_element_stride) != rank:
            raise ManifestError(f"{where}: element_strides needs one entry per dimension")
        for extent in self.map_tensor_map["global_dims"]:
            JobManifest.check_integer(
                value=extent,
                where=f"{where} global_dims",
                low=1,
                high=LaunchArgument.GLOBAL_EXTENT_MAX,
            )
        for stride in list_stride:
            JobManifest.check_integer(
                value=stride,
                where=f"{where} global_strides_bytes",
                low=LaunchArgument.ADDRESS_ALIGNMENT_BYTES,
                high=LaunchArgument.GLOBAL_STRIDE_MAX,
            )
            if stride % LaunchArgument.ADDRESS_ALIGNMENT_BYTES != 0:
                raise ManifestError(
                    f"{where}: global_strides_bytes entries are multiples of "
                    f"{LaunchArgument.ADDRESS_ALIGNMENT_BYTES}"
                )
        for extent in list_box:
            JobManifest.check_integer(
                value=extent,
                where=f"{where} box_dims",
                low=1,
                high=LaunchArgument.BOX_EXTENT_MAX,
            )
        for stride in list_element_stride:
            JobManifest.check_integer(
                value=stride,
                where=f"{where} element_strides",
                low=1,
                high=LaunchArgument.ELEMENT_STRIDE_MAX,
            )
        width = LaunchArgument.MAP_TENSOR_MAP_WIDTH[self.map_tensor_map["data_type"]]
        extent_bytes = width + sum(
            (extent - 1) * stride
            for extent, stride in zip(
                self.map_tensor_map["global_dims"],
                [width, *list_stride],
            )
        )
        if extent_bytes > buffer_bytes - self.offset_bytes:
            raise ManifestError(
                f"{where}: the tensor map spans {extent_bytes} bytes, but buffer '{self.buffer}' "
                f"holds {buffer_bytes - self.offset_bytes} from offset_bytes"
            )

    def _validate_bytes(
        self,
        where: str,
        map_buffer: dict[str, BufferSpec],
    ) -> None:
        """Check a by-value struct's bytes and that its relocations fall inside it."""
        try:
            data = bytes.fromhex(self.data_hex or "")
        except ValueError as err:
            raise ManifestError(f"{where}: hex is not hexadecimal") from err
        if not 0 < len(data) <= LaunchArgument.BYTES_MAX:
            raise ManifestError(f"{where}: a bytes argument holds 1 to {LaunchArgument.BYTES_MAX} bytes")
        set_offset = set()
        for relocation in self.list_relocation:
            if (
                not isinstance(
                    relocation.offset,  # obj
                    int,                # class_or_tuple
                )
                or relocation.offset < 0
                or relocation.offset % PointerRelocation.POINTER_BYTES != 0
                or relocation.offset + PointerRelocation.POINTER_BYTES > len(data)
            ):
                raise ManifestError(
                    f"{where}: relocation offset {relocation.offset} is not 8-aligned inside "
                    f"the {len(data)} bytes"
                )
            if relocation.offset in set_offset:
                raise ManifestError(f"{where}: two relocations write offset {relocation.offset}")
            if any(data[relocation.offset:relocation.offset + PointerRelocation.POINTER_BYTES]):
                raise ManifestError(
                    f"{where}: the {PointerRelocation.POINTER_BYTES} bytes at relocation offset "
                    f"{relocation.offset} must be zero; the platform writes the address there"
                )
            set_offset.add(relocation.offset)
            if relocation.buffer not in map_buffer:
                raise ManifestError(
                    f"{where}: a relocation names buffer '{relocation.buffer}', which [[buffers]] lacks"
                )
            JobManifest.check_integer(
                value=relocation.offset_bytes,
                where=f"{where} relocation offset_bytes",
                low=0,
                high=map_buffer[relocation.buffer].nbytes() - 1,
            )

    def size_bytes(self) -> int:
        """The bytes the argument occupies in the kernel's parameter space.

        Returns:
            the size
        """
        if self.kind == "scalar":
            return LaunchArgument.MAP_SCALAR[str(self.dtype)][3]
        if self.kind == "tensor_map":
            return LaunchArgument.TENSOR_MAP_BYTES
        if self.kind == "bytes":
            return len(bytes.fromhex(self.data_hex or ""))
        return PointerRelocation.POINTER_BYTES

    def is_naming(self, buffer_name: str) -> bool:
        """Whether the argument hands a buffer's address to the kernel.

        Args:
            buffer_name: the buffer

        Returns:
            whether it does, directly, through a tensor map or through a relocation
        """
        return (
            self.buffer == buffer_name
            or any(relocation.buffer == buffer_name for relocation in self.list_relocation)
        )

    def to_manifest(self, index: int) -> dict[str, Any]:
        """The argument as the launch manifest carries it.

        Args:
            index: the argument's position in the kernel's signature

        Returns:
            the argument
        """
        map_argument: dict[str, Any] = {
            "index": index,
            "kind": self.kind,
        }
        if self.kind == "buffer":
            map_argument["buffer"] = self.buffer
            map_argument["offset_bytes"] = self.offset_bytes
        if self.kind == "scalar":
            map_argument["abi_type"] = LaunchArgument.MAP_SCALAR[str(self.dtype)][0]
            map_argument["value"] = str(self.value)
        if self.kind == "tensor_map":
            rank = len(self.map_tensor_map["global_dims"])
            map_argument.update(
                {
                    "buffer": self.buffer,
                    "offset_bytes": self.offset_bytes,
                    "data_type": self.map_tensor_map["data_type"],
                    "global_dims": list(self.map_tensor_map["global_dims"]),
                    "global_strides_bytes": list(self.map_tensor_map["global_strides_bytes"]),
                    "box_dims": list(self.map_tensor_map["box_dims"]),
                    "element_strides": list(
                        self.map_tensor_map.get(
                            "element_strides",
                            [1] * rank,
                        )
                    ),
                }
            )
            for key in LaunchArgument.MAP_TENSOR_MAP_CHOICE:
                map_argument[key] = self.map_tensor_map.get(
                    key,
                    "none",
                )
        if self.kind == "bytes":
            map_argument["hex"] = (self.data_hex or "").lower()
            map_argument["relocations"] = [
                relocation.to_manifest() for relocation in self.list_relocation
            ]
        return map_argument


@dataclass
class LaunchSpec:
    """How the platform launches the kernel: its shape and its arguments."""

    THREADS_PER_BLOCK_MAX = 1024
    DIMENSION_COUNT = 3
    GRID_EXTENT_MAX = (
        2**31 - 1,
        65535,
        65535,
    )
    BLOCK_EXTENT_MAX = (
        1024,
        1024,
        64,
    )
    CLUSTER_BLOCKS_MAX = 16
    SHARED_BYTES_MAX = 233472
    ARGUMENT_COUNT_MAX = 128
    # the most bytes the arguments occupy together, and the most relocations they carry
    ARGUMENT_BYTES_MAX = 4096
    RELOCATION_COUNT_MAX = 128
    THREADS_PER_WARP = 32

    list_grid: list[int]
    list_block: list[int]
    list_cluster: list[int] | None = None
    shared_bytes: int = 0
    list_argument: list[LaunchArgument] = field(default_factory=list)

    @classmethod
    def from_table(cls, map_launch: Any) -> "LaunchSpec":
        """Read the ``[launch]`` table.

        Args:
            map_launch: the table

        Returns:
            the launch

        Raises:
            ManifestError: the table lacks the grid or the block
        """
        JobManifest.check_table(
            value=map_launch,
            set_key={
                "grid",
                "block",
                "cluster",
                "shared_bytes",
                "args",
            },
            where="[launch]",
        )
        for key in (
            "grid",
            "block",
        ):
            if key not in map_launch:
                raise ManifestError(f"[launch] needs '{key}'")
        return cls(
            list_grid=LaunchSpec._dimensions(
                value=map_launch["grid"],
                where="grid",
            ),
            list_block=LaunchSpec._dimensions(
                value=map_launch["block"],
                where="block",
            ),
            list_cluster=(
                LaunchSpec._dimensions(
                    value=map_launch["cluster"],
                    where="cluster",
                )
                if "cluster" in map_launch
                else None
            ),
            shared_bytes=map_launch.get(
                "shared_bytes",
                0,
            ),
            list_argument=[
                LaunchArgument.from_table(map_argument) for map_argument in map_launch.get("args", [])
            ],
        )

    @staticmethod
    def _dimensions(
        value: Any,
        where: str,
    ) -> list[int]:
        """A grid, block or cluster shape padded to three dimensions."""
        if isinstance(
            value,  # obj
            int,    # class_or_tuple
        ):
            value = [value]
        if (
            not isinstance(
                value,  # obj
                list,   # class_or_tuple
            )
            or not 1 <= len(value) <= LaunchSpec.DIMENSION_COUNT
            or not all(
                isinstance(
                    extent,  # obj
                    int,     # class_or_tuple
                )
                and not isinstance(
                    extent,  # obj
                    bool,    # class_or_tuple
                )
                and extent > 0
                for extent in value
            )
        ):
            raise ManifestError(f"[launch] {where} is 1 to 3 positive integers")
        return list(value) + [1] * (LaunchSpec.DIMENSION_COUNT - len(value))

    def validate(self, map_buffer: dict[str, BufferSpec]) -> None:
        """Check the launch shape and every argument.

        Args:
            map_buffer: the job's buffers by name

        Raises:
            ManifestError: the shape is out of range or an argument is malformed
        """
        for name, list_extent, list_extent_max in (
            ("grid", self.list_grid, LaunchSpec.GRID_EXTENT_MAX),
            ("block", self.list_block, LaunchSpec.BLOCK_EXTENT_MAX),
        ):
            for extent, extent_max in zip(
                list_extent,
                list_extent_max,
            ):
                if extent > extent_max:
                    raise ManifestError(f"[launch] {name} {list_extent} exceeds {list(list_extent_max)}")
        threads_per_block = TensorFile.element_count(self.list_block)
        if threads_per_block > LaunchSpec.THREADS_PER_BLOCK_MAX:
            raise ManifestError(
                f"a block has at most {LaunchSpec.THREADS_PER_BLOCK_MAX} threads, not {threads_per_block}"
            )
        JobManifest.check_integer(
            value=self.shared_bytes,
            where="[launch] shared_bytes",
            low=0,
            high=LaunchSpec.SHARED_BYTES_MAX,
        )
        if self.list_cluster is not None:
            for grid_extent, cluster_extent in zip(
                self.list_grid,
                self.list_cluster,
            ):
                if grid_extent % cluster_extent != 0:
                    raise ManifestError(
                        f"[launch] grid {self.list_grid} is not a multiple of cluster {self.list_cluster}"
                    )
            if TensorFile.element_count(self.list_cluster) > LaunchSpec.CLUSTER_BLOCKS_MAX:
                raise ManifestError(f"a cluster holds at most {LaunchSpec.CLUSTER_BLOCKS_MAX} blocks")
        if not 1 <= len(self.list_argument) <= LaunchSpec.ARGUMENT_COUNT_MAX:
            raise ManifestError(
                f"[launch] args lists 1 to {LaunchSpec.ARGUMENT_COUNT_MAX} kernel parameters"
            )
        for argument_id, argument in enumerate(self.list_argument):
            argument.validate(
                argument_id=argument_id,
                map_buffer=map_buffer,
            )
        argument_bytes = sum(argument.size_bytes() for argument in self.list_argument)
        if argument_bytes > LaunchSpec.ARGUMENT_BYTES_MAX:
            raise ManifestError(
                f"[launch] args occupy {argument_bytes} bytes; a launch passes at most "
                f"{LaunchSpec.ARGUMENT_BYTES_MAX}"
            )
        if sum(len(argument.list_relocation) for argument in self.list_argument) > LaunchSpec.RELOCATION_COUNT_MAX:
            raise ManifestError(f"[launch] args carry at most {LaunchSpec.RELOCATION_COUNT_MAX} relocations")

    def rows(self, granularity: str) -> int:
        """The rows a trace records into: every thread, or every warp, of the launch.

        Args:
            granularity: ``thread`` or ``warp``

        Returns:
            the row count

        Raises:
            ManifestError: warp-level recording on blocks of partial warps
        """
        block_count = TensorFile.element_count(self.list_grid)
        threads_per_block = TensorFile.element_count(self.list_block)
        if granularity == "thread":
            return block_count * threads_per_block
        if threads_per_block % LaunchSpec.THREADS_PER_WARP != 0:
            raise ManifestError(
                f"warp_level sites need blocks of whole warps; a block has {threads_per_block} threads"
            )
        return block_count * (threads_per_block // LaunchSpec.THREADS_PER_WARP)

    def to_target(
        self,
        architecture: str,
        kernel_symbol: str,
    ) -> dict[str, Any]:
        """The launch target as the launch manifest carries it.

        Args:
            architecture: the hardware's architecture
            kernel_symbol: the kernel's entry point

        Returns:
            the target
        """
        return {
            "architecture": architecture,
            "kernel_symbol": kernel_symbol,
            "grid": list(self.list_grid),
            "block": list(self.list_block),
            "cluster": None if self.list_cluster is None else list(self.list_cluster),
            "dynamic_shared_memory_bytes": self.shared_bytes,
        }


@dataclass
class ValidationSpec:
    """The ``[validation]`` table: an output buffer checked against the job's reference file."""

    TOLERANCE_MAX = 1

    output: str
    atol: float
    rtol: float

    @classmethod
    def from_table(cls, map_validation: Any) -> "ValidationSpec":
        """Read and check the ``[validation]`` table.

        Args:
            map_validation: the table

        Returns:
            the validation

        Raises:
            ManifestError: a key is missing, unknown or out of range
        """
        JobManifest.check_table(
            value=map_validation,
            set_key={
                "output",
                "atol",
                "rtol",
            },
            where="[validation]",
        )
        if not isinstance(
            map_validation.get("output"),  # obj
            str,                           # class_or_tuple
        ):
            raise ManifestError("[validation] needs 'output', the buffer checked against the reference")
        for key in (
            "atol",
            "rtol",
        ):
            value = map_validation.get(
                key,
                0,
            )
            if (
                isinstance(
                    value,  # obj
                    bool,   # class_or_tuple
                )
                or not isinstance(
                    value,         # obj
                    (int, float),  # class_or_tuple
                )
                or not 0 <= value <= ValidationSpec.TOLERANCE_MAX
            ):
                raise ManifestError(f"[validation] {key} must be a number in [0, 1]")
        return cls(
            output=map_validation["output"],
            atol=map_validation.get(
                "atol",
                0,
            ),
            rtol=map_validation.get(
                "rtol",
                0,
            ),
        )

    def to_manifest(self) -> dict[str, Any]:
        """The validation as the launch manifest carries it.

        Returns:
            the output, the reference tensor of the same name, and the tolerances
        """
        return {
            "output_key": self.output,
            "reference_key": self.output,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass
class JobArtifact:
    """A file the job uploads as an artifact: one of the job directory, or one the harness prepares.

    Args:
        kind: the artifact kind, e.g. ``cubin``
        path: the file, or for prepared bytes the path whose name the artifact carries
        content_type: the media type it is uploaded with
        data: the prepared bytes, or None to upload the file
    """

    kind: str
    path: str
    content_type: str = "application/octet-stream"
    data: bytes | None = None

    def read(self) -> bytes:
        """The bytes the artifact uploads.

        Returns:
            the prepared bytes, or the file's
        """
        if self.data is not None:
            return self.data
        with open(
            file=self.path,
            mode="rb",
        ) as artifact_file:
            return artifact_file.read()


@dataclass(kw_only=True)
class JobManifest(abc.ABC):
    """A job directory's ``xtrace.toml``, checked, and its translation into the platform API.

    A job either launches one kernel of a cubin under a trace plan (``LaunchJobManifest``),
    or hands a workspace to an agent on the GPU host (``AgentJobManifest``).

    Args:
        directory: the job directory
        hardware: the environment id the job runs on
        intent: what the trace should answer, the job's prompt
        limits: the most the job may cost and use
    """

    FILE_NAME = "xtrace.toml"
    SUBMIT_SCHEMA = "marscompute.xtrace.submit.v1"
    PROMPT_CHARACTERS_MAX = 32000
    # the manifest's required text keys and what each holds
    MAP_REQUIRED_TEXT = {
        "hardware": "the environment id from 'xtrace hardware'",
        "intent": "what the trace should answer, in your words",
    }
    SET_COMMON_KEY = {
        "hardware",
        "intent",
        "limits",
    }
    # the tables that make a job an agent-mode job
    SET_AGENT_KEY = {
        "workspace",
        "agent",
    }

    directory: str
    hardware: str
    intent: str
    limits: JobLimits = field(default_factory=JobLimits)

    @staticmethod
    def check_table(
        value: Any,
        set_key: set[str],
        where: str,
    ) -> None:
        """Check that a manifest entry is a table with known keys only.

        Args:
            value: the entry
            set_key: the keys it may hold
            where: the entry's place in the manifest, for the message

        Raises:
            ManifestError: the entry is not a table or has an unknown key
        """
        if not isinstance(
            value,  # obj
            dict,   # class_or_tuple
        ):
            raise ManifestError(f"each {where} entry must be a table")
        set_unknown = set(value) - set_key
        if set_unknown:
            raise ManifestError(f"{where}: unknown key(s) {', '.join(sorted(set_unknown))}")

    @staticmethod
    def check_integer(
        value: Any,
        where: str,
        low: int,
        high: int | None = None,
    ) -> None:
        """Check that a manifest value is an integer in a range.

        Args:
            value: the value
            where: the value's place in the manifest, for the message
            low: the lowest value allowed
            high: the highest value allowed, or None for no bound

        Raises:
            ManifestError: the value is not an integer in the range
        """
        if (
            isinstance(
                value,  # obj
                bool,   # class_or_tuple
            )
            or not isinstance(
                value,  # obj
                int,    # class_or_tuple
            )
            or value < low
            or (
                high is not None
                and value > high
            )
        ):
            bound = f"from {low}" if high is None else f"in [{low}, {high}]"
            raise ManifestError(f"{where} must be an integer {bound}, not {value!r}")

    @staticmethod
    def _check_path(
        directory: str,
        relative_path: Any,
        where: str,
        is_directory: bool = False,
    ) -> str:
        """Check that a file, or a directory, the manifest names lies inside the job directory and exists."""
        if not isinstance(
            relative_path,  # obj
            str,            # class_or_tuple
        ):
            raise ManifestError(f"{where} must be a path")
        if (
            os.path.isabs(relative_path)
            or os.path.normpath(relative_path).startswith("..")
        ):
            raise ManifestError(f"{where}: {relative_path} must lie inside the job directory")
        path = os.path.join(
            directory,
            relative_path,
        )
        if is_directory:
            if not os.path.isdir(path):
                raise ManifestError(f"{where}: {relative_path} is not a directory of the job directory")
        elif not os.path.isfile(path):
            raise ManifestError(f"{where}: {relative_path} does not exist in the job directory")
        return path

    @classmethod
    def load(cls, directory: str) -> "JobManifest":
        """Read and check the manifest of a job directory.

        Args:
            directory: the job directory, holding ``xtrace.toml``

        Returns:
            an agent-mode manifest when it has ``[workspace]`` or ``[agent]``, else a launch
            manifest

        Raises:
            ManifestError: the manifest is missing, malformed or incomplete, or names an absent
                file
            HarnessError: a binary is not a cubin
        """
        path = os.path.join(
            directory,
            JobManifest.FILE_NAME,
        )
        if not os.path.isfile(path):
            raise ManifestError(f"no {JobManifest.FILE_NAME} in {directory}")
        with open(
            file=path,
            mode="rb",
        ) as manifest_file:
            try:
                map_manifest = tomllib.load(manifest_file)
            except tomllib.TOMLDecodeError as err:
                raise ManifestError(f"{path}: {err}") from err
        for key, meaning in JobManifest.MAP_REQUIRED_TEXT.items():
            if (
                not isinstance(
                    map_manifest.get(key),  # obj
                    str,                    # class_or_tuple
                )
                or not map_manifest[key].strip()
            ):
                raise ManifestError(f"{path}: '{key}' is required: {meaning}")
        if len(map_manifest["intent"]) > JobManifest.PROMPT_CHARACTERS_MAX:
            raise ManifestError(f"{path}: 'intent' holds at most {JobManifest.PROMPT_CHARACTERS_MAX} characters")
        if JobManifest.SET_AGENT_KEY & set(map_manifest):
            return AgentJobManifest.from_table(
                directory=directory,
                path=path,
                map_manifest=map_manifest,
            )
        return LaunchJobManifest.from_table(
            directory=directory,
            path=path,
            map_manifest=map_manifest,
        )

    def select_compute(self, map_capabilities: dict[str, Any]) -> dict[str, Any]:
        """The capabilities entry of the manifest's hardware, checked to run the job.

        Args:
            map_capabilities: the platform's capabilities

        Returns:
            the entry, holding ``compute``, ``architecture`` and what it runs

        Raises:
            ManifestError: the hardware is unknown or cannot run the job
        """
        if not map_capabilities.get("enabled"):
            raise ManifestError("Xtrace is not enabled for this account; see the console's Xtrace settings")
        list_entry = map_capabilities.get("compute", [])
        for map_entry in list_entry:
            if map_entry.get("environment_id") == self.hardware:
                self.check_entry(map_entry)
                return map_entry
        raise ManifestError(
            f"no hardware '{self.hardware}'; 'xtrace hardware' lists: "
            f"{', '.join(map_entry.get('environment_id', '?') for map_entry in list_entry) or 'none'}"
        )

    @abc.abstractmethod
    def check_entry(self, map_entry: dict[str, Any]) -> None:
        """Check that a capabilities entry runs the job.

        Args:
            map_entry: the entry of the manifest's hardware

        Raises:
            ManifestError: the entry cannot run the job
        """

    def cached_listing(self) -> KernelListing | None:
        """The inspection the job is checked against, when one is cached.

        Returns:
            the listing, or None
        """
        return None

    def check_listing(self, listing: KernelListing | None) -> None:
        """Check the job against a cached inspection before anything is uploaded.

        Args:
            listing: the cached inspection, or None

        Raises:
            ManifestError: the job does not fit the inspected kernel
        """

    @abc.abstractmethod
    def list_artifact(
        self,
        architecture: str,
        listing: KernelListing | None,
    ) -> list[JobArtifact]:
        """The files the job uploads.

        Args:
            architecture: the hardware's architecture
            listing: the cached inspection, or None

        Returns:
            the artifacts, in upload order
        """

    @abc.abstractmethod
    def bundle(self, map_artifact_ids: dict[str, list[str]]) -> dict[str, Any]:
        """The bundle of the uploaded files.

        Args:
            map_artifact_ids: artifact kind to the ids of the uploaded artifacts of that kind

        Returns:
            the bundle
        """

    @abc.abstractmethod
    def analysis(self, listing: KernelListing | None) -> dict[str, Any]:
        """What the platform does with the bundle.

        Args:
            listing: the cached inspection, or None

        Returns:
            the analysis
        """

    def estimate_input(
        self,
        map_capabilities: dict[str, Any],
        map_artifact_ids: dict[str, list[str]],
        listing: KernelListing | None,
    ) -> dict[str, Any]:
        """The request a quote and a submission take.

        Args:
            map_capabilities: the platform's capabilities
            map_artifact_ids: artifact kind to the ids of the uploaded artifacts of that kind
            listing: the cached inspection, or None

        Returns:
            the estimate input, ``marscompute.xtrace.submit.v1``
        """
        return {
            "schema_version": JobManifest.SUBMIT_SCHEMA,
            "prompt": self.intent,
            "bundle": self.bundle(map_artifact_ids),
            "compute": self.select_compute(map_capabilities)["compute"],
            "analysis": self.analysis(listing),
            "limits": self.limits.to_api(map_capabilities.get("limits", {})),
        }


@dataclass(kw_only=True)
class LaunchJobManifest(JobManifest):
    """The manifest of a job that launches one kernel of a cubin and traces it: a
    ``cubin_launch`` bundle with an explicit trace plan or the platform planner."""

    BUNDLE_KIND = "cubin_launch"
    LAUNCH_SCHEMA = "marscompute.kernel-launch"
    # the architectures a launch manifest targets
    LIST_LAUNCH_ARCHITECTURE = (
        "sm_80",
        "sm_86",
        "sm_89",
        "sm_90",
        "sm_90a",
        "sm_100",
        "sm_100a",
        "sm_103",
        "sm_103a",
    )
    PREPARED_TENSORS_NAME = "tensors.safetensors"
    RESET_POLICY = "restore_pristine_before_every_invocation"
    BUFFER_COUNT_MAX = 64
    MANIFEST_NAME = "manifest.json"
    SET_KEY = {
        "tensors",
        "reference",
        "kernel",
        "buffers",
        "launch",
        "validation",
        "trace",
        "planner",
    }

    kernel: KernelSpec
    binary: KernelBinary
    launch: LaunchSpec
    list_buffer: list[BufferSpec] = field(default_factory=list)
    tensors: str | None = None
    reference: str | None = None
    validation: ValidationSpec | None = None
    trace: TraceOptions | None = None
    planner: PlannerOptions | None = None

    @classmethod
    def from_table(
        cls,
        directory: str,
        path: str,
        map_manifest: dict[str, Any],
    ) -> "LaunchJobManifest":
        """Read and check the manifest of a job that launches one kernel of a cubin.

        Args:
            directory: the job directory
            path: the manifest file, for the message
            map_manifest: the manifest's table, whose common keys are checked

        Returns:
            the manifest

        Raises:
            ManifestError: a table is missing, malformed or incomplete, names an absent file,
                or names a kernel its cubin lacks
            HarnessError: the binary is not a cubin
        """
        JobManifest.check_table(
            value=map_manifest,
            set_key=JobManifest.SET_COMMON_KEY | LaunchJobManifest.SET_KEY,
            where=JobManifest.FILE_NAME,
        )
        for key in (
            "kernel",
            "launch",
        ):
            if key not in map_manifest:
                raise ManifestError(f"{path}: a [{key}] table is required")
        if ("trace" in map_manifest) == ("planner" in map_manifest):
            raise ManifestError(
                f"{path}: give either [trace] with the sites to probe or [planner] to have the "
                "platform pick them"
            )
        if ("reference" in map_manifest) != ("validation" in map_manifest):
            raise ManifestError(f"{path}: 'reference' and [validation] go together")
        kernel = KernelSpec.from_table(map_manifest["kernel"])
        binary_path = JobManifest._check_path(
            directory=directory,
            relative_path=kernel.binary,
            where="[kernel] binary",
        )
        for relative_path in kernel.list_source:
            JobManifest._check_path(
                directory=directory,
                relative_path=relative_path,
                where="[kernel] sources",
            )
        manifest = cls(
            directory=directory,
            hardware=map_manifest["hardware"],
            intent=map_manifest["intent"],
            limits=JobLimits.from_table(
                map_manifest.get(
                    "limits",
                    {},
                )
            ),
            kernel=kernel,
            binary=KernelBinary.load(binary_path),
            launch=LaunchSpec.from_table(map_manifest["launch"]),
            list_buffer=[BufferSpec.from_table(map_buffer) for map_buffer in map_manifest.get("buffers", [])],
            tensors=map_manifest.get("tensors"),
            reference=map_manifest.get("reference"),
            validation=(
                ValidationSpec.from_table(map_manifest["validation"])
                if "validation" in map_manifest
                else None
            ),
            trace=TraceOptions.from_table(map_manifest["trace"]) if "trace" in map_manifest else None,
            planner=(
                PlannerOptions.from_table(map_manifest["planner"])
                if "planner" in map_manifest
                else None
            ),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Check the manifest's parts against each other and against the files it names.

        Raises:
            ManifestError: a part is malformed or inconsistent
        """
        if self.kernel.name not in self.binary.list_kernel:
            raise ManifestError(
                f"{self.kernel.binary} defines no kernel '{self.kernel.name}'; it defines: "
                f"{', '.join(self.binary.list_kernel) or 'none'}"
            )
        tensor_file = None
        if self.tensors is not None:
            tensor_file = TensorFile.load(
                JobManifest._check_path(
                    directory=self.directory,
                    relative_path=self.tensors,
                    where="tensors",
                )
            )
            if not any(buffer.init == "tensor" for buffer in self.list_buffer):
                raise ManifestError("'tensors' is named, but no buffer has init = 'tensor'")
        if len(self.list_buffer) > LaunchJobManifest.BUFFER_COUNT_MAX:
            raise ManifestError(f"a job has at most {LaunchJobManifest.BUFFER_COUNT_MAX} buffers")
        map_buffer: dict[str, BufferSpec] = {}
        for buffer in self.list_buffer:
            buffer.validate(tensor_file)
            if buffer.name in map_buffer:
                raise ManifestError(f"buffer '{buffer.name}' is defined twice")
            map_buffer[buffer.name] = buffer
        self.launch.validate(map_buffer)
        for buffer in self.list_buffer:
            if (
                buffer.is_output
                and not any(argument.is_naming(buffer.name) for argument in self.launch.list_argument)
            ):
                raise ManifestError(f"output buffer '{buffer.name}' is named by no launch argument")
        if self.validation is not None:
            buffer = map_buffer.get(self.validation.output)
            if (
                buffer is None
                or not buffer.is_output
            ):
                raise ManifestError(
                    f"[validation] output '{self.validation.output}' must be a buffer with output = true"
                )
            if buffer.dtype in BufferSpec.MAP_OUTPUT_DTYPE:
                raise ManifestError(
                    f"[validation] output '{buffer.name}' is {buffer.dtype}, which the launch compares "
                    "as raw bytes only; validate an output of another type"
                )
            buffer.check_tensor(
                tensor_file=TensorFile.load(
                    JobManifest._check_path(
                        directory=self.directory,
                        relative_path=self.reference,
                        where="reference",
                    )
                ),
                where="reference",
            )
        if self.trace is not None:
            self.trace.validate()
            self.launch.rows(self.trace.granularity())

    def _tensors_artifact(self) -> JobArtifact | None:
        """The tensors the buffers start from: the job's tensors file as it is, or a file the
        harness writes when a buffer starts as ``ones`` or ``random``."""
        list_prepared = [buffer for buffer in self.list_buffer if buffer.is_prepared()]
        tensors_path = None if self.tensors is None else os.path.join(
            self.directory,
            self.tensors,
        )
        if not list_prepared:
            if tensors_path is None:
                return None
            return JobArtifact(
                kind="safetensors",
                path=tensors_path,
            )
        tensor_file = None if tensors_path is None else TensorFile.load(tensors_path)
        list_tensor = []
        map_metadata = {}
        for buffer in self.list_buffer:
            if buffer.init == "tensor" and tensor_file is not None:
                list_tensor.append((buffer.tensor_entry(), tensor_file.read_tensor(buffer.name)))
            if buffer.is_prepared():
                list_tensor.append((buffer.tensor_entry(), buffer.prepared_bytes()))
                map_metadata[buffer.name] = (
                    f"random seed={buffer.seed} {BufferSpec.RANDOM_ALGORITHM}"
                    if buffer.init == "random"
                    else buffer.init
                )
        return JobArtifact(
            kind="safetensors",
            path=tensors_path or os.path.join(
                self.directory,
                LaunchJobManifest.PREPARED_TENSORS_NAME,
            ),
            data=TensorFile.encode(
                list_tensor=list_tensor,
                map_metadata=map_metadata,
            ),
        )

    def cached_listing(self) -> KernelListing | None:
        """The inspection of the job's kernel cached beside its cubin.

        Returns:
            the listing, or None when none is cached
        """
        return InspectionCache.load(
            binary_path=os.path.join(
                self.directory,
                self.kernel.binary,
            ),
            sha256=self.binary.sha256,
            kernel_symbol=self.kernel.name,
        )

    def check_listing(self, listing: KernelListing | None) -> None:
        """Check the launch and the trace sites against the kernel's cached inspection.

        Args:
            listing: the cached inspection, or None

        Raises:
            ManifestError: an argument or a site does not fit the kernel
            HarnessError: the listing lacks a site's instruction
        """
        if listing is not None:
            self.check_parameters(listing)
        self.analysis(listing)

    def list_artifact(
        self,
        architecture: str,
        listing: KernelListing | None,
    ) -> list[JobArtifact]:
        """The files the job uploads.

        Args:
            architecture: the hardware's architecture
            listing: the cached inspection, whose parameter layout the launch manifest carries

        Returns:
            the cubin, the launch manifest, the tensors when a buffer starts from one, and the
            reference file when the job has one
        """
        list_artifact = [
            JobArtifact(
                kind="cubin",
                path=os.path.join(
                    self.directory,
                    self.kernel.binary,
                ),
            )
        ]
        list_artifact.append(
            JobArtifact(
                kind="launch_manifest",
                path=os.path.join(
                    self.directory,
                    LaunchJobManifest.MANIFEST_NAME,
                ),
                content_type="application/json",
                data=self.launch_manifest_bytes(
                    architecture=architecture,
                    listing=listing,
                ),
            )
        )
        tensors_artifact = self._tensors_artifact()
        if tensors_artifact is not None:
            list_artifact.append(tensors_artifact)
        if self.reference is not None:
            list_artifact.append(
                JobArtifact(
                    kind="reference",
                    path=os.path.join(
                        self.directory,
                        self.reference,
                    ),
                )
            )
        return list_artifact

    def check_parameters(self, listing: KernelListing) -> None:
        """Check the launch arguments against the parameter layout an inspection read.

        Args:
            listing: the inspected kernel

        Raises:
            ManifestError: the count or a size of the arguments differs from the kernel's
        """
        kernel = listing.inspected_kernel()
        if kernel.list_parameter is None:
            return
        if len(kernel.list_parameter) != len(self.launch.list_argument):
            raise ManifestError(
                f"{self.kernel.name} takes {len(kernel.list_parameter)} parameters, but [launch] "
                f"args lists {len(self.launch.list_argument)}"
            )
        for parameter, argument in zip(
            sorted(
                kernel.list_parameter,
                key=lambda listed: listed.index,
            ),
            self.launch.list_argument,
        ):
            if parameter.size_bytes != argument.size_bytes():
                raise ManifestError(
                    f"launch argument {parameter.index} ({argument.kind}) is "
                    f"{argument.size_bytes()} bytes, but parameter {parameter.index} of "
                    f"{self.kernel.name} is {parameter.size_bytes}"
                )

    def check_entry(self, map_entry: dict[str, Any]) -> None:
        """Check that a capabilities entry runs the cubin, cubin launches and every selected stage.

        Args:
            map_entry: the entry of the manifest's hardware

        Raises:
            ManifestError: the entry cannot run the job
        """
        architecture = map_entry["architecture"]
        if not self.binary.runs_on(architecture):
            raise ManifestError(
                f"{self.kernel.binary} targets {self.binary.target()}, but {self.hardware} is "
                f"{architecture}; rebuild it for {architecture}"
            )
        if LaunchJobManifest.BUNDLE_KIND not in map_entry["bundle_kinds"]:
            raise ManifestError(f"{self.hardware} does not run cubin launches")
        list_stage = self.trace.list_stage if self.trace is not None else ["xtrace"]
        map_architecture = map_entry.get("architectures") or {}
        for stage in list_stage:
            if stage not in map_entry["stages"]:
                raise ManifestError(f"{self.hardware} does not run the {stage} stage")
            if (
                stage in map_architecture
                and architecture not in map_architecture[stage]
            ):
                raise ManifestError(f"the {stage} stage does not support {architecture}")

    def launch_manifest(
        self,
        architecture: str,
        listing: KernelListing | None,
    ) -> dict[str, Any]:
        """The launch manifest, ``marscompute.kernel-launch``, uploaded as an artifact.

        Args:
            architecture: the hardware's architecture
            listing: the inspected kernel when it is cached, whose parameter layout the
                manifest carries

        Returns:
            the manifest

        Raises:
            ManifestError: the architecture is not one a launch manifest targets
        """
        if architecture not in LaunchJobManifest.LIST_LAUNCH_ARCHITECTURE:
            raise ManifestError(
                f"a launch manifest targets {', '.join(LaunchJobManifest.LIST_LAUNCH_ARCHITECTURE)}; "
                f"{self.hardware} is {architecture}"
            )
        list_parameter = None
        if listing is not None:
            list_parameter = listing.inspected_kernel().list_parameter
        map_launch: dict[str, Any] = {
            "schema_version": LaunchJobManifest.LAUNCH_SCHEMA,
            "target": self.launch.to_target(
                architecture=architecture,
                kernel_symbol=self.kernel.name,
            ),
            "parameter_layout": None if list_parameter is None else [
                {
                    "index": parameter.index,
                    "offset": parameter.offset_bytes,
                    "size_bytes": parameter.size_bytes,
                }
                for parameter in list_parameter
            ],
            "allocations": [buffer.to_allocation() for buffer in self.list_buffer],
            "buffers": [buffer.to_view() for buffer in self.list_buffer],
            "arguments": [
                argument.to_manifest(index) for index, argument in enumerate(self.launch.list_argument)
            ],
            "outputs": [buffer.to_output() for buffer in self.list_buffer if buffer.is_output],
            "runtime_profile": None,
            "reset": {"policy": LaunchJobManifest.RESET_POLICY},
        }
        if self.validation is not None:
            map_launch["validation"] = self.validation.to_manifest()
        return map_launch

    def launch_manifest_bytes(
        self,
        architecture: str,
        listing: KernelListing | None,
    ) -> bytes:
        """The launch manifest as the bytes of its artifact.

        Args:
            architecture: the hardware's architecture
            listing: the inspected kernel when it is cached

        Returns:
            canonical JSON: sorted keys, no whitespace
        """
        return json.dumps(
            self.launch_manifest(
                architecture=architecture,
                listing=listing,
            ),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def analysis(self, listing: KernelListing | None) -> dict[str, Any]:
        """The job's analysis: an explicit trace plan, or the platform planner.

        Args:
            listing: the inspected kernel when it is cached, whose listing every site must name

        Returns:
            the analysis

        Raises:
            ManifestError: two sites probe one instruction
            HarnessError: the listing lacks a site's instruction
        """
        if self.planner is not None:
            return self.planner.to_analysis()
        if self.trace is None:
            raise ManifestError("the job has neither [trace] nor [planner]")
        return {
            "mode": "explicit_plan",
            "trace_plan": self.trace.to_plan(
                listing=listing,
                launch=self.launch,
            ),
        }

    def bundle(self, map_artifact_ids: dict[str, list[str]]) -> dict[str, Any]:
        """The ``cubin_launch`` bundle of the uploaded files.

        Args:
            map_artifact_ids: artifact kind to the ids of the uploaded artifacts of that kind

        Returns:
            the bundle
        """
        map_bundle = {
            "kind": LaunchJobManifest.BUNDLE_KIND,
            "cubin_artifact_id": map_artifact_ids["cubin"][0],
            "manifest_artifact_id": map_artifact_ids["launch_manifest"][0],
        }
        if "safetensors" in map_artifact_ids:
            map_bundle["tensors_artifact_id"] = map_artifact_ids["safetensors"][0]
        if "reference" in map_artifact_ids:
            map_bundle["reference_artifact_id"] = map_artifact_ids["reference"][0]
        return map_bundle


@dataclass
class AgentOptions:
    """The ``[agent]`` table: the model that traces the kernel on the GPU host, and the
    budget of its session.

    Args:
        model: one of the agent models of the capabilities
        max_turns: the most turns the session takes
        max_input_tokens: the most input tokens it reads, cache reads and writes included
        max_output_tokens: the most tokens it writes
        max_wall_seconds: the most seconds the session runs
    """

    MODEL_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
    # key -> (lowest value, highest value)
    MAP_BOUND = {
        "max_turns": (1, 500),
        "max_input_tokens": (1, 50_000_000),
        "max_output_tokens": (1, 1_000_000),
        "max_wall_seconds": (60, 2580),
    }

    model: str
    max_turns: int
    max_input_tokens: int
    max_output_tokens: int
    max_wall_seconds: int

    @classmethod
    def from_table(cls, map_agent: Any) -> "AgentOptions":
        """Read and check the ``[agent]`` table.

        Args:
            map_agent: the table

        Returns:
            the options

        Raises:
            ManifestError: a key is missing, unknown or out of range
        """
        JobManifest.check_table(
            value=map_agent,
            set_key={"model"} | set(AgentOptions.MAP_BOUND),
            where="[agent]",
        )
        if not AgentOptions.MODEL_PATTERN.match(str(map_agent.get("model", ""))):
            raise ManifestError("[agent] needs 'model', one of the agent models 'xtrace hardware' lists")
        for key, (low, high) in AgentOptions.MAP_BOUND.items():
            JobManifest.check_integer(
                value=map_agent.get(key),
                where=f"[agent] {key}",
                low=low,
                high=high,
            )
        return cls(
            model=map_agent["model"],
            max_turns=map_agent["max_turns"],
            max_input_tokens=map_agent["max_input_tokens"],
            max_output_tokens=map_agent["max_output_tokens"],
            max_wall_seconds=map_agent["max_wall_seconds"],
        )

    def to_analysis(self) -> dict[str, Any]:
        """The agent analysis the API takes.

        Returns:
            the analysis, mode ``agent``
        """
        return {
            "mode": "agent",
            "model": self.model,
            "max_turns": self.max_turns,
            "max_input_tokens_total": self.max_input_tokens,
            "max_output_tokens_total": self.max_output_tokens,
            "max_wall_seconds": self.max_wall_seconds,
        }


@dataclass
class WorkspaceSpec:
    """The ``[workspace]`` table: what the agent on the GPU host works with, relative to the
    job directory.

    Args:
        list_cubin: the cubins, 1 to 8
        source: a directory of source the agent may read, build and run, or None
        manifest: a ``marscompute.kernel-launch`` file describing a launch, or None
        list_tensors: safetensors files of input data, 0 to 8
        list_file: other files, such as a host driver script or a build file, 0 to 8
    """

    CUBINS_MAX = 8
    TENSORS_MAX = 8
    FILES_MAX = 8
    SET_KEY = {
        "cubins",
        "source",
        "manifest",
        "tensors",
        "files",
    }
    # names left out of the source archive: version control and the harness's own caches
    SET_EXCLUDED_NAME = {
        ".git",
        ".xtrace-cache",
        "xtrace-results",
    }
    SOURCE_ARCHIVE_NAME = "source.tar.gz"
    MODE_EXECUTABLE = 0o755
    MODE_REGULAR = 0o644
    EXECUTABLE_BITS = 0o111

    list_cubin: list[str]
    source: str | None = None
    manifest: str | None = None
    list_tensors: list[str] = field(default_factory=list)
    list_file: list[str] = field(default_factory=list)

    @staticmethod
    def _list_of_paths(
        value: Any,
        where: str,
        low: int,
        high: int,
    ) -> list[str]:
        """A list of ``low`` to ``high`` paths."""
        if (
            not isinstance(
                value,  # obj
                list,   # class_or_tuple
            )
            or not low <= len(value) <= high
            or not all(
                isinstance(
                    path,  # obj
                    str,   # class_or_tuple
                )
                for path in value
            )
        ):
            raise ManifestError(f"{where} lists {low} to {high} paths")
        return list(value)

    @classmethod
    def from_table(cls, map_workspace: Any) -> "WorkspaceSpec":
        """Read the ``[workspace]`` table.

        Args:
            map_workspace: the table

        Returns:
            the workspace

        Raises:
            ManifestError: a key is unknown or a list has the wrong length
        """
        JobManifest.check_table(
            value=map_workspace,
            set_key=WorkspaceSpec.SET_KEY,
            where="[workspace]",
        )
        return cls(
            list_cubin=WorkspaceSpec._list_of_paths(
                value=map_workspace.get("cubins"),
                where="[workspace] cubins",
                low=1,
                high=WorkspaceSpec.CUBINS_MAX,
            ),
            source=map_workspace.get("source"),
            manifest=map_workspace.get("manifest"),
            list_tensors=WorkspaceSpec._list_of_paths(
                value=map_workspace.get(
                    "tensors",
                    [],
                ),
                where="[workspace] tensors",
                low=0,
                high=WorkspaceSpec.TENSORS_MAX,
            ),
            list_file=WorkspaceSpec._list_of_paths(
                value=map_workspace.get(
                    "files",
                    [],
                ),
                where="[workspace] files",
                low=0,
                high=WorkspaceSpec.FILES_MAX,
            ),
        )

    def validate(self, directory: str) -> list[KernelBinary]:
        """Check every file the workspace names, and that it names each once.

        Args:
            directory: the job directory

        Returns:
            the cubins

        Raises:
            ManifestError: a file is absent, outside the job directory, named twice or not
                what its key says
            HarnessError: a cubin is not a cubin
        """
        list_path = [
            *self.list_cubin,
            *([self.manifest] if self.manifest is not None else []),
            *self.list_tensors,
            *self.list_file,
        ]
        for relative_path in list_path:
            JobManifest._check_path(
                directory=directory,
                relative_path=relative_path,
                where="[workspace]",
            )
        set_seen = set()
        for relative_path in list_path:
            normalized_path = os.path.normpath(relative_path)
            if normalized_path in set_seen:
                raise ManifestError(f"[workspace] names {relative_path} twice; each file uploads once")
            set_seen.add(normalized_path)
        if self.source is not None:
            JobManifest._check_path(
                directory=directory,
                relative_path=self.source,
                where="[workspace] source",
                is_directory=True,
            )
        if self.manifest is not None:
            manifest_path = os.path.join(
                directory,
                self.manifest,
            )
            with open(
                file=manifest_path,
                mode="r",
                encoding="utf-8",
            ) as launch_file:
                try:
                    map_launch = json.load(launch_file)
                except ValueError as err:
                    raise ManifestError(f"[workspace] manifest: {self.manifest} is not JSON") from err
            if (
                not isinstance(
                    map_launch,  # obj
                    dict,        # class_or_tuple
                )
                or map_launch.get("schema_version") != LaunchJobManifest.LAUNCH_SCHEMA
            ):
                raise ManifestError(
                    f"[workspace] manifest: {self.manifest} is not a {LaunchJobManifest.LAUNCH_SCHEMA} manifest"
                )
        for relative_path in self.list_tensors:
            TensorFile.load(
                os.path.join(
                    directory,
                    relative_path,
                )
            )
        return [
            KernelBinary.load(
                os.path.join(
                    directory,
                    relative_path,
                )
            )
            for relative_path in self.list_cubin
        ]

    @staticmethod
    def _normalize(member: tarfile.TarInfo) -> tarfile.TarInfo:
        """A member of the source archive with its owner, times and mode made reproducible."""
        member.uid = 0
        member.gid = 0
        member.uname = ""
        member.gname = ""
        member.mtime = 0
        if member.isdir() or member.mode & WorkspaceSpec.EXECUTABLE_BITS:
            member.mode = WorkspaceSpec.MODE_EXECUTABLE
        else:
            member.mode = WorkspaceSpec.MODE_REGULAR
        return member

    def archive_source(self, directory: str) -> bytes:
        """The source directory as a reproducible ``.tar.gz``, without version control.

        Args:
            directory: the job directory

        Returns:
            the archive's bytes; the same tree always gives the same bytes
        """
        source_path = os.path.join(
            directory,
            str(self.source),
        )
        archive = io.BytesIO()
        with gzip.GzipFile(
            fileobj=archive,
            mode="wb",
            mtime=0,
        ) as compressed, tarfile.open(
            fileobj=compressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as tar:
            for root, list_directory, list_name in os.walk(source_path):
                list_directory[:] = sorted(
                    name for name in list_directory if name not in WorkspaceSpec.SET_EXCLUDED_NAME
                )
                for name in sorted([*list_directory, *list_name]):
                    if name in WorkspaceSpec.SET_EXCLUDED_NAME:
                        continue
                    path = os.path.join(
                        root,
                        name,
                    )
                    tar.add(
                        name=path,
                        arcname=os.path.relpath(
                            path,
                            source_path,
                        ),
                        recursive=False,
                        filter=WorkspaceSpec._normalize,
                    )
        return archive.getvalue()


@dataclass(kw_only=True)
class AgentJobManifest(JobManifest):
    """The manifest of an agent-mode job: an agent on the GPU host reads the workspace,
    instruments and runs the kernel within its budget, and returns one ``xtrace.json``.

    Args:
        workspace: the files the agent works with
        agent: the model and the session's budget
        list_binary: the workspace's cubins
    """

    BUNDLE_KIND = "agent_workspace"
    SET_KEY = {
        "workspace",
        "agent",
    }
    # artifact kind of each workspace part
    MAP_KIND = {
        "cubin": "cubin",
        "source": "source",
        "manifest": "launch_manifest",
        "tensors": "safetensors",
        "files": "workspace_file",
    }

    workspace: WorkspaceSpec
    agent: AgentOptions
    list_binary: list[KernelBinary] = field(default_factory=list)

    @classmethod
    def from_table(
        cls,
        directory: str,
        path: str,
        map_manifest: dict[str, Any],
    ) -> "AgentJobManifest":
        """Read and check the manifest of an agent-mode job.

        Args:
            directory: the job directory
            path: the manifest file, for the message
            map_manifest: the manifest's table, whose common keys are checked

        Returns:
            the manifest

        Raises:
            ManifestError: a table is missing, malformed or names an absent file
            HarnessError: a cubin is not a cubin
        """
        JobManifest.check_table(
            value=map_manifest,
            set_key=JobManifest.SET_COMMON_KEY | AgentJobManifest.SET_KEY,
            where=f"{JobManifest.FILE_NAME} in agent mode",
        )
        for key in AgentJobManifest.SET_KEY:
            if key not in map_manifest:
                raise ManifestError(f"{path}: agent mode needs [workspace] and [agent]; [{key}] is missing")
        workspace = WorkspaceSpec.from_table(map_manifest["workspace"])
        return cls(
            directory=directory,
            hardware=map_manifest["hardware"],
            intent=map_manifest["intent"],
            limits=JobLimits.from_table(
                map_manifest.get(
                    "limits",
                    {},
                )
            ),
            workspace=workspace,
            agent=AgentOptions.from_table(map_manifest["agent"]),
            list_binary=workspace.validate(directory),
        )

    def select_compute(self, map_capabilities: dict[str, Any]) -> dict[str, Any]:
        """The capabilities entry of the manifest's hardware, checked to run the agent.

        Args:
            map_capabilities: the platform's capabilities

        Returns:
            the entry

        Raises:
            ManifestError: the hardware cannot run the workspace, or the model is not offered
        """
        map_entry = super().select_compute(map_capabilities)
        list_model = map_capabilities.get("agent_models", [])
        if self.agent.model not in list_model:
            raise ManifestError(
                f"agent model '{self.agent.model}' is not offered; 'xtrace hardware' lists: "
                f"{', '.join(list_model) or 'none'}"
            )
        return map_entry

    def check_entry(self, map_entry: dict[str, Any]) -> None:
        """Check that a capabilities entry runs agent workspaces and every cubin.

        Args:
            map_entry: the entry of the manifest's hardware

        Raises:
            ManifestError: the entry cannot run the job
        """
        if AgentJobManifest.BUNDLE_KIND not in map_entry["bundle_kinds"]:
            raise ManifestError(
                f"{self.hardware} does not run agent mode; pick an environment whose bundle kinds "
                f"list {AgentJobManifest.BUNDLE_KIND}"
            )
        architecture = map_entry["architecture"]
        for relative_path, binary in zip(
            self.workspace.list_cubin,
            self.list_binary,
        ):
            if not binary.runs_on(architecture):
                raise ManifestError(
                    f"{relative_path} targets {binary.target()}, but {self.hardware} is "
                    f"{architecture}; rebuild it for {architecture}"
                )

    def list_artifact(
        self,
        architecture: str,
        listing: KernelListing | None,
    ) -> list[JobArtifact]:
        """The files the workspace uploads.

        Args:
            architecture: the hardware's architecture
            listing: unused; an agent inspects the kernel itself

        Returns:
            the cubins, the source archive, the launch manifest, the tensors and the files
        """
        list_artifact = [
            JobArtifact(
                kind=AgentJobManifest.MAP_KIND["cubin"],
                path=os.path.join(
                    self.directory,
                    relative_path,
                ),
            )
            for relative_path in self.workspace.list_cubin
        ]
        if self.workspace.source is not None:
            list_artifact.append(
                JobArtifact(
                    kind=AgentJobManifest.MAP_KIND["source"],
                    path=os.path.join(
                        self.directory,
                        WorkspaceSpec.SOURCE_ARCHIVE_NAME,
                    ),
                    content_type="application/gzip",
                    data=self.workspace.archive_source(self.directory),
                )
            )
        if self.workspace.manifest is not None:
            list_artifact.append(
                JobArtifact(
                    kind=AgentJobManifest.MAP_KIND["manifest"],
                    path=os.path.join(
                        self.directory,
                        self.workspace.manifest,
                    ),
                    content_type="application/json",
                )
            )
        for key, list_relative_path in (
            ("tensors", self.workspace.list_tensors),
            ("files", self.workspace.list_file),
        ):
            list_artifact.extend(
                JobArtifact(
                    kind=AgentJobManifest.MAP_KIND[key],
                    path=os.path.join(
                        self.directory,
                        relative_path,
                    ),
                )
                for relative_path in list_relative_path
            )
        return list_artifact

    def bundle(self, map_artifact_ids: dict[str, list[str]]) -> dict[str, Any]:
        """The ``agent_workspace`` bundle of the uploaded files.

        Args:
            map_artifact_ids: artifact kind to the ids of the uploaded artifacts of that kind

        Returns:
            the bundle
        """
        map_bundle: dict[str, Any] = {
            "kind": AgentJobManifest.BUNDLE_KIND,
            "cubin_artifact_ids": list(map_artifact_ids[AgentJobManifest.MAP_KIND["cubin"]]),
            "tensor_artifact_ids": list(map_artifact_ids.get(AgentJobManifest.MAP_KIND["tensors"], [])),
            "file_artifact_ids": list(map_artifact_ids.get(AgentJobManifest.MAP_KIND["files"], [])),
        }
        for key, bundle_key in (
            ("source", "source_archive_artifact_id"),
            ("manifest", "launch_manifest_artifact_id"),
        ):
            list_id = map_artifact_ids.get(AgentJobManifest.MAP_KIND[key])
            if list_id:
                map_bundle[bundle_key] = list_id[0]
        return map_bundle

    def analysis(self, listing: KernelListing | None) -> dict[str, Any]:
        """The agent analysis.

        Args:
            listing: unused; an agent inspects the kernel itself

        Returns:
            the analysis, mode ``agent``
        """
        return self.agent.to_analysis()
