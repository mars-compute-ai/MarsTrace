import argparse
import json
import os
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.viewer.common.launch import LaunchShape
from xtrace_harness.viewer.common.xtrace._result import TraceCompileResults, TraceResult
from xtrace_harness.viewer.common.xtrace.format._analysis import ScopeInterval, ScopePairing, TraceAnalysis
from xtrace_harness.viewer.common.xtrace.format._common import TraceClock
from xtrace_harness.viewer.common.xtrace.format._sites import SitePairing, SiteRecords, SiteRegion
from xtrace_harness.viewer.common.xtrace.format.panel import SelectionToken


@dataclass
class IdRange:
    """A closed range of thread, block or warp ids a filter keeps."""

    # a bracketed range names its first and last id
    RANGE_ENDS = 2

    first: int
    last: int

    def contains(self, value: int) -> bool:
        """Report whether an id falls in the range.

        Args:
            value: the id

        Returns:
            True when the id lies between the range's ends, inclusive
        """
        return self.first <= value <= self.last

    @staticmethod
    def any_contains(
        list_range: list["IdRange"],
        value: int,
    ) -> bool:
        """Report whether an id falls in any of several ranges.

        Args:
            list_range: the ranges
            value: the id

        Returns:
            True when at least one range holds the id
        """
        return any(id_range.contains(value) for id_range in list_range)

    @staticmethod
    def parse_list(spec: str) -> list["IdRange"]:
        """Parse a string like ``"[0,31], [64,95]"`` into the ranges 0..31 and 64..95.

        Outside brackets the string is read as a comma-separated list of ids and
        ``a-b`` ranges; the two forms may be mixed.

        Args:
            spec: the ranges as the command line gives them

        Returns:
            the ranges

        Raises:
            ValueError: a range is not an id, an ``a-b`` pair or a bracketed pair
        """
        list_range: list[IdRange] = []
        for chunk in re.findall(
            pattern=r"\[([^\]]*)\]",
            string=spec,
        ):
            list_bound = [bound.strip() for bound in chunk.split(",") if bound.strip()]
            if not 1 <= len(list_bound) <= IdRange.RANGE_ENDS:
                raise ValueError(f"Bad id range: '[{chunk}]'")
            list_range.append(
                IdRange._from_bounds(
                    text=f"[{chunk}]",
                    first_text=list_bound[0],
                    last_text=list_bound[-1],
                )
            )
        bare_spec = re.sub(
            pattern=r"\[[^\]]*\]",
            repl=",",
            string=spec,
        )
        for token in bare_spec.split(","):
            token = token.strip()
            if not token:
                continue
            first_text, _, last_text = token.partition("-")
            list_range.append(
                IdRange._from_bounds(
                    text=token,
                    first_text=first_text,
                    last_text=last_text if "-" in token else first_text,
                )
            )
        return list_range

    @staticmethod
    def _from_bounds(
        text: str,
        first_text: str,
        last_text: str,
    ) -> "IdRange":
        """Build a range from its two bound strings, naming the range text when one is not an id."""
        try:
            return IdRange(
                first=int(first_text),
                last=int(last_text),
            )
        except ValueError as err:
            raise ValueError(f"Bad id range: '{text}'") from err


@dataclass
class TraceData:
    """One traced kernel run prepared for display: its scope intervals per thread and per scope, the time span and the launch shape."""

    # the default number of longest bubbles and of concurrency time bins
    LONGEST_BUBBLES_DEFAULT = 5
    CONCURRENCY_BINS_DEFAULT = 60

    kernel_prototype: str
    map_compile_results: dict[str, Any]
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
    # pre-computed aggregate block from the report (``runs[].analysis``), used
    # for the aggregate views so large reports skip interval reconstruction.
    map_analysis: dict[str, Any] | None = None
    intervals_built: bool = True
    n_threads: int = 0
    # the clock the payloads were read from, and what one record observes
    clock_type: str = "gpu"
    record_shape: str = "scope"
    record_granularity: str = "thread"
    # per-site aggregates, on a run whose records are points
    list_site_stat: list[dict[str, Any]] = field(default_factory=list)
    # launch scale (for warp/block grouping)
    grid_dim: list[int] | None = None
    block_dim: list[int] | None = None
    block_size: int | None = None

    def get_compile_results(self) -> TraceCompileResults:
        """The run's compile results, as the report layer models them.

        Returns:
            the compile results the run carries
        """
        return TraceCompileResults.from_json(self.map_compile_results)

    def as_analysis(self) -> TraceAnalysis:
        """Express this run as the report layer models it, to aggregate it there.

        Returns:
            the run as a report-layer trace analysis
        """
        return TraceAnalysis(
            kernel_prototype=self.kernel_prototype,
            compile_results=TraceCompileResults.from_json(self.map_compile_results),
            map_oe_results=self.map_oe_results,
            map_tid_intervals=self.map_tid_intervals,
            map_label_intervals=self.map_label_intervals,
            list_tid=self.list_tid,
            list_scope_label=self.list_scope_label,
            t_min=self.t_min,
            t_max=self.t_max,
            clock_type=self.clock_type,
            record_shape=self.record_shape,
            record_granularity=self.record_granularity,
        )

    def use_analysis(self) -> bool:
        """Report whether the aggregate views may read the report's precomputed ``analysis``.

        That holds only when intervals were not (re)built, i.e. on the
        unfiltered path. After a tid/time/block/warp filter the full-run
        analysis no longer matches, so the aggregates are recomputed.

        Returns:
            True when the precomputed analysis matches the displayed records
        """
        return (not self.intervals_built) and self.map_analysis is not None

    def block_of(self, tid: int) -> int:
        """The block a thread id falls in, 0 without the launch scale.

        Args:
            tid: the thread id

        Returns:
            the block id
        """
        return tid // self.block_size if self.block_size else 0

    def warp_of(self, tid: int) -> int:
        """The warp a thread id falls in: within its block, or across the grid without the launch scale.

        Args:
            tid: the thread id

        Returns:
            the warp id
        """
        warp_size = self.get_compile_results().get_warp_size()
        if self.block_size:
            return (tid % self.block_size) // warp_size
        return tid // warp_size

    def locator(self, tid: int) -> str:
        """Compact locator for a record's owner.

        A warp-level run stores one record per warp under the id of its first lane, so
        the thread part of the locator would name a lane that did not do the storing.

        Args:
            tid: the thread id the record is stored under

        Returns:
            the owner as ``b<block> w<warp> t<thread>``, without the parts the run cannot tell
        """
        if self.record_granularity == "warp":
            if self.block_size:
                return f"b{self.block_of(tid)} w{self.warp_of(tid)}"
            return f"w{self.warp_of(tid)}"
        if self.block_size:
            return f"b{self.block_of(tid)} w{self.warp_of(tid)} t{tid}"
        return f"t{tid}"

    def unit_label(self) -> str:
        """What one row of records belongs to: a thread, or a warp.

        Returns:
            ``"warp"`` for a warp-level run, else ``"thread"``
        """
        return "warp" if self.record_granularity == "warp" else "thread"

    @staticmethod
    def _block_size_from(
        block_dim: list[int] | None,
        map_analysis: dict[str, Any] | None,
    ) -> int | None:
        """The threads per block the report's analysis or the launch's block shape gives, else None."""
        if (
            map_analysis
            and map_analysis.get("block_size")
        ):
            return int(map_analysis["block_size"])
        if block_dim:
            try:
                num_threads = LaunchShape.count_block_threads(
                    (
                        int(block_dim[0]),
                        int(block_dim[1]),
                        int(block_dim[2]),
                    )
                )
                return num_threads if num_threads > 0 else None
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                return None
        return None

    @classmethod
    def from_run(
        cls,
        map_run: dict[str, Any],
        need_intervals: bool = True,
    ) -> "TraceData":
        """Build a ``TraceData`` from a report run.

        When ``need_intervals`` is False and the run carries a pre-computed
        ``analysis`` block, the per-thread intervals are NOT reconstructed (a big
        saving for million-record reports) and the aggregate views read the
        ``analysis`` block directly. Per-thread views (timeline, filtering,
        outliers, longest) require ``need_intervals=True``.

        Args:
            map_run: one run of the report's intra-kernel trace section
            need_intervals: rebuild the per-thread intervals even when the run carries an analysis

        Returns:
            the run prepared for display
        """
        map_analysis = (
            map_run.get("analysis")
            if isinstance(
                map_run.get("analysis"),  # obj
                dict,                     # class_or_tuple
            )
            else None
        )
        list_map_record = map_run.get("trace_results") or []
        nb_record = len(list_map_record)
        list_record = [
            record
            for record in (
                TraceResult.record_from_json(map_record) for map_record in list_map_record
            )
            if record is not None
        ]
        # a site run carries points; the regions the report names turn them into intervals
        if SiteRecords.are_sites(list_record):
            list_region = [
                SiteRegion(
                    enter_site=int(region[0]),
                    leave_site=int(region[1]),
                    label=str(region[2]),
                )
                for region in ((map_analysis or {}).get("site_regions") or [])
            ]
            list_record = (
                SitePairing.pair(
                    list_record=list_record,
                    list_region=list_region,
                ).list_record
                if list_region
                else []
            )
        block_dim = (map_analysis or {}).get("block_dim") or map_run.get("block_dim")
        grid_dim = (map_analysis or {}).get("grid_dim") or map_run.get("grid_dim")
        block_size = TraceData._block_size_from(
            block_dim=block_dim,
            map_analysis=map_analysis,
        )

        map_compile_results = dict(map_run.get("compile_results") or {})
        map_common_field = dict(
            kernel_prototype=str(map_run.get("kernel_prototype") or "(unknown)"),
            map_compile_results=map_compile_results,
            map_oe_results=dict(map_run.get("oe_results") or {}),
            total_records=nb_record,
            map_analysis=map_analysis,
            grid_dim=list(grid_dim) if grid_dim else None,
            block_dim=list(block_dim) if block_dim else None,
            block_size=block_size,
            list_site_stat=list((map_analysis or {}).get("site_stats") or []),
            clock_type=str(map_compile_results.get("clock_type") or "gpu"),
            record_shape=str(map_compile_results.get("record_shape") or "scope"),
            record_granularity=str(map_compile_results.get("record_granularity") or "thread"),
        )

        if (
            not need_intervals
            and map_analysis is not None
        ):
            return cls(
                list_tid=[],
                list_scope_label=list(map_analysis.get("scope_labels") or []),
                t_min=int(
                    map_analysis.get(
                        "t_min",
                        0,
                    )
                ),
                t_max=int(
                    map_analysis.get(
                        "t_max",
                        0,
                    )
                ),
                unmatched_starts=int(
                    map_analysis.get(
                        "unmatched_starts",
                        0,
                    )
                ),
                unmatched_ends=int(
                    map_analysis.get(
                        "unmatched_ends",
                        0,
                    )
                ),
                intervals_built=False,
                n_threads=int(
                    map_analysis.get(
                        "n_threads",
                        0,
                    )
                ),
                **map_common_field,
            )

        pairing = ScopePairing.pair(list_record)
        map_tid_intervals = pairing.map_tid_intervals
        map_label_intervals = pairing.map_label_intervals
        unmatched_starts = pairing.unmatched_starts
        unmatched_ends = pairing.unmatched_ends
        t_min: int | None = None
        t_max: int | None = None
        for list_interval in map_tid_intervals.values():
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

        return cls(
            map_tid_intervals=map_tid_intervals,
            map_label_intervals=map_label_intervals,
            list_tid=sorted(map_tid_intervals.keys()),
            list_scope_label=sorted(map_label_intervals.keys()),
            t_min=t_min if t_min is not None else 0,
            t_max=t_max if t_max is not None else 0,
            unmatched_starts=unmatched_starts,
            unmatched_ends=unmatched_ends,
            intervals_built=True,
            n_threads=len(map_tid_intervals),
            **map_common_field,
        )

    def filtered(
        self,
        list_tid_range: list[IdRange],
        stime: int | None,
        etime: int | None,
        list_block_range: list[IdRange] | None = None,
        list_warp_range: list[IdRange] | None = None,
        set_tid: set[int] | None = None,
    ) -> "TraceData":
        """The run restricted to the threads, blocks, warps and time window the filters keep.

        Args:
            list_tid_range: the thread ids to keep, or empty for every thread
            stime: drop intervals ending before this tick, or None
            etime: drop intervals starting after this tick, or None
            list_block_range: the block ids to keep, or None
            list_warp_range: the warp-in-block ids to keep, or None
            set_tid: the only thread ids to keep, or None

        Returns:
            this run when no filter is set, else the subset with its time span recomputed
        """
        if (
            not list_tid_range
            and not list_block_range
            and not list_warp_range
            and set_tid is None
            and stime is None
            and etime is None
        ):
            return self

        map_kept_tid_intervals: dict[int, list[ScopeInterval]] = {}
        map_kept_label_intervals: dict[str, list[ScopeInterval]] = {}
        t_min: int | None = None
        t_max: int | None = None

        for tid, list_interval in self.map_tid_intervals.items():
            if (
                set_tid is not None
                and tid not in set_tid
            ):
                continue
            if (
                list_tid_range
                and not IdRange.any_contains(
                    list_range=list_tid_range,
                    value=tid,
                )
            ):
                continue
            if (
                list_block_range
                and not IdRange.any_contains(
                    list_range=list_block_range,
                    value=self.block_of(tid),
                )
            ):
                continue
            if (
                list_warp_range
                and not IdRange.any_contains(
                    list_range=list_warp_range,
                    value=self.warp_of(tid),
                )
            ):
                continue
            list_kept_interval: list[ScopeInterval] = []
            for interval in list_interval:
                if (
                    stime is not None
                    and interval.t_end < stime
                ):
                    continue
                if (
                    etime is not None
                    and interval.t_start > etime
                ):
                    continue
                list_kept_interval.append(interval)
                map_kept_label_intervals.setdefault(
                    interval.scope_label,  # key
                    [],                    # default
                ).append(interval)
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
            if list_kept_interval:
                map_kept_tid_intervals[tid] = list_kept_interval

        return TraceData(
            kernel_prototype=self.kernel_prototype,
            map_compile_results=self.map_compile_results,
            map_oe_results=self.map_oe_results,
            map_tid_intervals=map_kept_tid_intervals,
            map_label_intervals=map_kept_label_intervals,
            list_tid=sorted(map_kept_tid_intervals.keys()),
            list_scope_label=sorted(map_kept_label_intervals.keys()),
            t_min=t_min if t_min is not None else 0,
            t_max=t_max if t_max is not None else 0,
            unmatched_starts=self.unmatched_starts,
            unmatched_ends=self.unmatched_ends,
            total_records=self.total_records,
            # the clock and the record kind are the run's, not the subset's
            clock_type=self.clock_type,
            record_shape=self.record_shape,
            record_granularity=self.record_granularity,
            # a site aggregate counts the whole run, so a subset carries none
            list_site_stat=[],
            map_analysis=None,  # filtered subset no longer matches full-run analysis
            intervals_built=True,
            n_threads=len(map_kept_tid_intervals),
            grid_dim=self.grid_dim,
            block_dim=self.block_dim,
            block_size=self.block_size,
        )

    def scope_stats(self) -> list[dict[str, Any]]:
        """Per-scope duration statistics, from the ``analysis`` block when it applies, else computed.

        Returns:
            one row per scope label, keyed as the report names them
        """
        if (
            self.use_analysis()
            and self.map_analysis.get("scope_stats") is not None
        ):
            return list(self.map_analysis["scope_stats"])
        return [stat.to_json() for stat in self.as_analysis().scope_stats()]

    def bubble_per_thread(self) -> dict[int, dict[str, int]]:
        """Each thread's active, bubble and lifetime ticks, computed from its intervals.

        Returns:
            the times of each thread, keyed by thread id
        """
        return {
            tid: bubble.to_json() for tid, bubble in self.as_analysis().bubble_per_thread().items()
        }

    def pipeline_summary(self) -> dict[str, Any] | None:
        """Per-thread active/bubble aggregate — from the ``analysis`` block when
        available, else computed from intervals.

        Returns:
            the aggregate keyed as the report names it, or None when there is none
        """
        if self.use_analysis():
            return self.map_analysis.get("pipeline_summary") or None
        summary = self.as_analysis().pipeline_summary()
        return summary.to_json() if summary is not None else None

    def longest_bubbles(
        self,
        top: int = LONGEST_BUBBLES_DEFAULT,
    ) -> list[dict[str, Any]]:
        """Largest inter-scope gaps — from ``analysis`` when available, else
        computed.

        Args:
            top: the number of gaps to keep

        Returns:
            the gaps, largest first, keyed as the report names them
        """
        if (
            self.use_analysis()
            and self.map_analysis.get("longest_bubbles") is not None
        ):
            return list(self.map_analysis["longest_bubbles"])[:top]
        return [gap.to_json() for gap in self.as_analysis().inter_scope_gaps(top=top)]

    def concurrency(
        self,
        bins: int = CONCURRENCY_BINS_DEFAULT,
    ) -> dict[str, Any] | None:
        """Active-threads-per-time-bin histogram — from ``analysis`` when
        available, else computed.

        Args:
            bins: the number of time bins

        Returns:
            the histogram keyed as the report names it, or None without intervals
        """
        if (
            self.use_analysis()
            and self.map_analysis.get("concurrency")
        ):
            return self.map_analysis["concurrency"]
        if (
            not self.map_tid_intervals
            or self.t_max <= self.t_min
        ):
            return None
        return self.as_analysis().concurrency(bins=bins).to_json()


class ShowCommand:
    """The ``show`` command: render the pipeline analysis of an intra-kernel trace report in the terminal."""

    # ANSI palette
    ANSI_RESET = "\033[0m"
    ANSI_BOLD = "\033[1m"
    ANSI_FG_GREY = "\033[38;5;245m"
    ANSI_FG_RED = "\033[38;5;196m"
    ANSI_FG_YELLOW = "\033[38;5;226m"

    # 24 distinguishable background colours for scope blocks
    LIST_SCOPE_BACKGROUND = [
        "\033[48;5;33m",
        "\033[48;5;208m",
        "\033[48;5;76m",
        "\033[48;5;201m",
        "\033[48;5;220m",
        "\033[48;5;39m",
        "\033[48;5;160m",
        "\033[48;5;105m",
        "\033[48;5;46m",
        "\033[48;5;202m",
        "\033[48;5;93m",
        "\033[48;5;226m",
        "\033[48;5;51m",
        "\033[48;5;199m",
        "\033[48;5;130m",
        "\033[48;5;87m",
        "\033[48;5;124m",
        "\033[48;5;112m",
        "\033[48;5;57m",
        "\033[48;5;214m",
        "\033[48;5;141m",
        "\033[48;5;82m",
        "\033[48;5;167m",
        "\033[48;5;72m",
    ]

    CHAR_BUBBLE = "·"  # · — middle dot

    # Per-scope fill characters used when colour is off (no --human) so multi-scope
    # traces remain distinguishable.  In --human mode every scope uses the solid
    # block "█" and is differentiated by background colour.
    LIST_SCOPE_GLYPH = [
        "█",
        "▓",
        "▒",
        "░",
        "▚",
        "▞",
        "◆",
        "◇",
        "●",
        "○",
        "▲",
        "▽",
    ]
    CHAR_SCOPE_COLORED = "█"
    # the concurrency sparkline's levels, lowest first
    SPARK_LEVELS = "▁▂▃▄▅▆▇█"

    # the terminal size assumed when it cannot be read, and the narrowest one laid out for
    TERMINAL_WIDTH_DEFAULT = 100
    TERMINAL_HEIGHT_DEFAULT = 24
    TERMINAL_WIDTH_MIN = 60
    # the width of the header's field names
    HEADER_LABEL_WIDTH = 15
    # the timeline's default row count, its narrowest bar, and its label column's default width and padding
    TIMELINE_SAMPLE_DEFAULT = 32
    TIMELINE_BAR_MIN = 20
    TIMELINE_LABEL_WIDTH_DEFAULT = 4
    TIMELINE_LABEL_PADDING = 2
    # outliers: the default cut-off in standard deviations, the fewest rows to compare, the most listed
    OUTLIER_SIGMAS_DEFAULT = 2.0
    OUTLIER_ROWS_MIN = 2
    OUTLIER_SHOWN_MAX = 20
    # a ratio as a percentage
    PERCENT_SCALE = 100
    # the JSON view's indent
    JSON_INDENT = 2
    # the exit status of an unusable report or argument
    EXIT_USAGE = 2

    @staticmethod
    def _load_report(path: str) -> Any:
        """Load a ``Section_IntraKernelTrace`` JSON report."""
        if path.lower().endswith(
            (
                ".yaml",
                ".yml",
            )
        ):
            raise ValueError(
                "YAML reports are no longer supported — re-dump the report as .json "
                "(File.render('report.json')). JSON loads orders of magnitude faster."
            )
        with open(
            file=path,
            mode="r",
            encoding="utf-8",
        ) as report_file:
            return json.load(report_file)

    @staticmethod
    def _find_intra_kernel_section(doc: Any) -> dict[str, Any] | None:
        """Locate a ``Section_IntraKernelTrace`` payload inside a File envelope.

        The expected envelope is the structure produced by
        the trace report's JSON renderer::

            {title: ..., sections: [{title, type, data}, ...]}
        """
        if not isinstance(
            doc,   # obj
            dict,  # class_or_tuple
        ):
            return None
        list_section = doc.get("sections")
        if not isinstance(
            list_section,  # obj
            list,          # class_or_tuple
        ):
            return None
        for map_section in list_section:
            if (
                isinstance(
                    map_section,  # obj
                    dict,         # class_or_tuple
                )
                and map_section.get("type") == "Section_IntraKernelTrace"
            ):
                map_data = map_section.get("data")
                if isinstance(
                    map_data,  # obj
                    dict,      # class_or_tuple
                ):
                    return map_data
        return None

    @staticmethod
    def _select_run(
        map_section_data: dict[str, Any],
        run_index: int,
    ) -> dict[str, Any]:
        list_run = map_section_data.get("runs") or []
        if (
            not isinstance(
                list_run,  # obj
                list,      # class_or_tuple
            )
            or not list_run
        ):
            raise ValueError("Section_IntraKernelTrace has no runs.")
        if (
            run_index < 0
            or run_index >= len(list_run)
        ):
            raise ValueError(f"--run {run_index} out of range; report has {len(list_run)} run(s).")
        return list_run[run_index]

    @staticmethod
    def _terminal_width(default: int = TERMINAL_WIDTH_DEFAULT) -> int:
        """The terminal's width in columns, never below the narrowest laid out for."""
        try:
            return max(
                ShowCommand.TERMINAL_WIDTH_MIN,
                shutil.get_terminal_size(
                    (
                        default,
                        ShowCommand.TERMINAL_HEIGHT_DEFAULT,
                    )
                ).columns,
            )
        except OSError:
            return default

    @staticmethod
    def _scope_color_map(list_label: list[str]) -> dict[str, str]:
        return {
            label: ShowCommand.LIST_SCOPE_BACKGROUND[index % len(ShowCommand.LIST_SCOPE_BACKGROUND)]
            for index, label in enumerate(list_label)
        }

    @staticmethod
    def _scope_glyph_map(list_label: list[str]) -> dict[str, str]:
        """Per-scope fill character for the no-colour timeline.

        All scopes get a distinguishable glyph so multi-scope traces remain
        readable when ``--human`` is off.
        """
        return {
            label: ShowCommand.LIST_SCOPE_GLYPH[index % len(ShowCommand.LIST_SCOPE_GLYPH)]
            for index, label in enumerate(list_label)
        }

    @staticmethod
    def _format_legend(
        list_label: list[str],
        human: bool,
    ) -> str:
        """Build the timeline legend.

        With ``--human`` (colour) every scope is shown as a coloured block followed
        by its label; without it, each scope's distinct glyph precedes the label.
        """
        if not list_label:
            return ""
        if human:
            map_label_color = ShowCommand._scope_color_map(list_label)
            list_item = [
                f"{map_label_color[label]} {ShowCommand.CHAR_SCOPE_COLORED} {ShowCommand.ANSI_RESET} {label}"
                for label in list_label
            ]
        else:
            map_label_glyph = ShowCommand._scope_glyph_map(list_label)
            list_item = [f"{map_label_glyph[label]} {label}" for label in list_label]
        return "   ".join(list_item)

    @staticmethod
    def _print_selection(
        selected: SelectionToken,
        data: TraceData,
    ) -> None:
        """What the token asked for, and what the report had to answer with."""
        span = TraceClock.format_ticks(
            ticks=selected.t1 - selected.t0,
            clock_type=data.clock_type,
        )
        print(
            f"{ShowCommand.ANSI_BOLD}Selection:{ShowCommand.ANSI_RESET}     "
            f"{len(selected.list_id)} {selected.level}(s) → {len(data.list_tid):,} "
            f"{data.unit_label()}(s)  ·  "
            f"[{selected.t0}, {selected.t1}] = {span}"
        )
        if selected.list_scope:
            print(
                f"{ShowCommand.ANSI_BOLD}Token scopes:{ShowCommand.ANSI_RESET}  {', '.join(selected.list_scope)}"
            )
        print()

    @staticmethod
    def _print_header(
        data: TraceData,
        run_index: int,
        n_runs: int,
    ) -> None:
        print(
            f"{ShowCommand.ANSI_BOLD}Kernel:{ShowCommand.ANSI_RESET}        {data.kernel_prototype}"
        )
        print(
            f"{ShowCommand.ANSI_BOLD}Run:{ShowCommand.ANSI_RESET}           {run_index} of {n_runs}"
        )
        map_compile_results = data.map_compile_results
        if map_compile_results:
            level = map_compile_results.get(
                "instrumentation_level",  # key
                "?",                      # default
            )
            list_bit = [
                f"{level}",
                f"{data.record_shape} records",
                f"per {data.record_granularity}",
                f"clock={data.clock_type}",
            ]
            for key in (
                "origin_reg_used [avg]",
                "trace_reg_used [avg]",
                "origin_reg_used [overall]",
                "trace_reg_used [overall]",
            ):
                if (
                    key in map_compile_results
                    and map_compile_results[key] is not None
                ):
                    list_bit.append(f"{key}={map_compile_results[key]}")
            print(
                f"{ShowCommand.ANSI_BOLD}Compile:{ShowCommand.ANSI_RESET}       {'  '.join(list_bit)}"
            )
        if (
            data.grid_dim
            or data.block_dim
        ):

            def _dim(dim: list[int] | None) -> str:
                return "x".join(str(int(value)) for value in dim) if dim else "?"

            block_size_text = (
                f"  ({data.block_size} threads/block, {data.get_compile_results().get_warp_label()}={data.get_compile_results().get_warp_size()})"
                if data.block_size
                else ""
            )
            print(
                f"{ShowCommand.ANSI_BOLD}Launch:{ShowCommand.ANSI_RESET}        grid={_dim(data.grid_dim)}  "
                f"block={_dim(data.block_dim)}{block_size_text}"
            )
        print(
            f"{ShowCommand.ANSI_BOLD}Records:{ShowCommand.ANSI_RESET}       {data.total_records:,}"
        )
        unit = data.unit_label().capitalize() + "s:"
        print(
            f"{ShowCommand.ANSI_BOLD}{unit}{ShowCommand.ANSI_RESET}{' ' * (ShowCommand.HEADER_LABEL_WIDTH - len(unit))}{data.n_threads:,}"
        )
        print(
            f"{ShowCommand.ANSI_BOLD}Scopes:{ShowCommand.ANSI_RESET}        {len(data.list_scope_label)}  "
            f"({', '.join(data.list_scope_label) or '-'})"
        )
        if data.t_max > data.t_min:
            print(
                f"{ShowCommand.ANSI_BOLD}Time range:{ShowCommand.ANSI_RESET}    [{data.t_min}, {data.t_max}]  "
                f"= {TraceClock.format_ticks(data.t_max - data.t_min, data.clock_type)}"
            )
        if (
            data.unmatched_starts
            or data.unmatched_ends
        ):
            print(
                f"{ShowCommand.ANSI_FG_YELLOW}Warning:{ShowCommand.ANSI_RESET}      "
                f"unmatched SCOPE_START={data.unmatched_starts}, "
                f"SCOPE_END={data.unmatched_ends} "
                f"(records may be truncated or out of order)"
            )
        print()

    @staticmethod
    def _new_table(
        list_header: list[str],
        left_aligned_column: str | None = None,
    ) -> Any | None:
        """A right-aligned table with the given header, or None when prettytable is not installed."""
        try:
            from prettytable import PrettyTable  # noqa: PLC0415
        except ImportError:
            return None
        table = PrettyTable()
        table.field_names = list_header
        table.align = "r"
        if left_aligned_column is not None:
            table.align[left_aligned_column] = "l"
        return table

    @staticmethod
    def _print_table(
        list_header: list[str],
        list_row: list[list[Any]],
        left_aligned_column: str | None = None,
    ) -> None:
        """Print rows under a header, as a table when prettytable is installed, else as plain lines."""
        table = ShowCommand._new_table(
            list_header=list_header,
            left_aligned_column=left_aligned_column,
        )
        if table is not None:
            for list_cell in list_row:
                table.add_row(list_cell)
            print(table)
        else:
            print("  " + "  ".join(list_header))
            for list_cell in list_row:
                print("  " + "  ".join(str(cell) for cell in list_cell))

    @staticmethod
    def _print_site_stats(data: TraceData) -> None:
        """What each site recorded, on a run whose records are points."""
        if not data.list_site_stat:
            return
        unit = data.unit_label()
        print(f"{ShowCommand.ANSI_BOLD}Per-site records:{ShowCommand.ANSI_RESET}")
        list_header = [
            "site",
            "records",
            f"{unit}s",
            "per row",
            "repeat min",
            "repeat mean",
            "repeat max",
        ]
        list_row = [
            [
                map_site_stat["site_id"],
                f'{map_site_stat["count"]:,}',
                f'{map_site_stat["rows"]:,}',
                f'{map_site_stat["records_per_row"]:.1f}',
                TraceClock.format_ticks(
                    ticks=map_site_stat["repeat_min"],
                    clock_type=data.clock_type,
                ),
                TraceClock.format_ticks(
                    ticks=int(map_site_stat["repeat_mean"]),
                    clock_type=data.clock_type,
                ),
                TraceClock.format_ticks(
                    ticks=map_site_stat["repeat_max"],
                    clock_type=data.clock_type,
                ),
            ]
            for map_site_stat in data.list_site_stat
        ]
        ShowCommand._print_table(
            list_header=list_header,
            list_row=list_row,
        )
        print()

    @staticmethod
    def _print_region_stats(data: TraceData) -> None:
        """What pairing the named regions managed, and what it had to drop."""
        list_stat = list((data.map_analysis or {}).get("region_stats") or [])
        if not list_stat:
            return
        unit = data.unit_label()
        print(f"{ShowCommand.ANSI_BOLD}Paired regions:{ShowCommand.ANSI_RESET}")
        list_header = [
            "region",
            "sites",
            "intervals",
            f"{unit}s",
            "dropped",
            "pair rate",
        ]
        list_row = [
            [
                map_region_stat["label"],
                f'{map_region_stat["enter_site"]} -> {map_region_stat["leave_site"]}',
                f'{map_region_stat["intervals"]:,}',
                f'{map_region_stat["rows"]:,}',
                f'{map_region_stat["dropped"]:,}',
                f'{map_region_stat["pair_rate"] * ShowCommand.PERCENT_SCALE:.2f}%',
            ]
            for map_region_stat in list_stat
        ]
        ShowCommand._print_table(
            list_header=list_header,
            list_row=list_row,
            left_aligned_column="region",
        )
        worst = min(map_region_stat["pair_rate"] for map_region_stat in list_stat)
        if worst < 1.0:
            print(
                f"{ShowCommand.ANSI_FG_YELLOW}  worst pair rate {worst * ShowCommand.PERCENT_SCALE:.2f}%: boundaries were dropped{ShowCommand.ANSI_RESET}"
            )
        print()

    @staticmethod
    def _print_scope_stats(data: TraceData) -> None:
        ShowCommand._print_site_stats(data)
        ShowCommand._print_region_stats(data)
        list_scope_stat = data.scope_stats()
        if not list_scope_stat:
            if data.record_shape == "site":
                print(
                    f"{ShowCommand.ANSI_FG_GREY}This run records sites, which are points: an interval "
                    f"needs a pair of them{ShowCommand.ANSI_RESET}\n"
                    f"{ShowCommand.ANSI_FG_GREY}named through add_run(..., site_regions="
                    f"[(enter, leave, label)]).{ShowCommand.ANSI_RESET}\n"
                )
                return
            print(
                f"{ShowCommand.ANSI_FG_GREY}No scope intervals to summarise.{ShowCommand.ANSI_RESET}\n"
            )
            return

        table = ShowCommand._new_table(
            list_header=[
                "scope_label",
                "count",
                "min",
                "p50",
                "mean",
                "p95",
                "max",
                "total",
            ],
            left_aligned_column="scope_label",
        )
        if table is not None:
            for map_scope_stat in list_scope_stat:
                table.add_row(
                    [
                        map_scope_stat["scope_label"],
                        f'{map_scope_stat["count"]:,}',
                        TraceClock.format_ticks(
                            ticks=map_scope_stat["min"],
                            clock_type=data.clock_type,
                        ),
                        TraceClock.format_ticks(
                            ticks=map_scope_stat["p50"],
                            clock_type=data.clock_type,
                        ),
                        TraceClock.format_ticks(
                            ticks=int(map_scope_stat["mean"]),
                            clock_type=data.clock_type,
                        ),
                        TraceClock.format_ticks(
                            ticks=map_scope_stat["p95"],
                            clock_type=data.clock_type,
                        ),
                        TraceClock.format_ticks(
                            ticks=map_scope_stat["max"],
                            clock_type=data.clock_type,
                        ),
                        TraceClock.format_ticks(
                            ticks=map_scope_stat["total"],
                            clock_type=data.clock_type,
                        ),
                    ]
                )
            print(
                f"{ShowCommand.ANSI_BOLD}Per-scope stats (sorted by total time):{ShowCommand.ANSI_RESET}"
            )
            print(table)
        else:
            print(f"{ShowCommand.ANSI_BOLD}Per-scope stats:{ShowCommand.ANSI_RESET}")
            for map_scope_stat in list_scope_stat:
                print(
                    f"  {map_scope_stat['scope_label']:<20s} count={map_scope_stat['count']}  "
                    f"mean={TraceClock.format_ticks(int(map_scope_stat['mean']), data.clock_type)}  "
                    f"total={TraceClock.format_ticks(map_scope_stat['total'], data.clock_type)}"
                )
        print()

    @staticmethod
    def _print_pipeline_summary(data: TraceData) -> None:
        map_summary = data.pipeline_summary()
        if not map_summary:
            return

        def _line(
            label: str,
            mean: float,
            minimum: float | None = None,
            maximum: float | None = None,
        ) -> None:
            list_part = [f"avg={TraceClock.format_ticks(int(mean), data.clock_type)}"]
            if minimum is not None:
                list_part.append(f"min={TraceClock.format_ticks(int(minimum), data.clock_type)}")
            if maximum is not None:
                list_part.append(f"max={TraceClock.format_ticks(int(maximum), data.clock_type)}")
            print(f"  {label:<16s}{', '.join(list_part)}")

        print(
            f"{ShowCommand.ANSI_BOLD}Pipeline summary (across {map_summary.get('n_threads', 0):,} "
            f"{data.unit_label()}(s)):{ShowCommand.ANSI_RESET}"
        )
        if "active_mean" in map_summary:
            _line(
                label="active time",
                mean=map_summary["active_mean"],
                minimum=map_summary.get("active_min"),
                maximum=map_summary.get("active_max"),
            )
        if "bubble_mean" in map_summary:
            _line(
                label="bubble time",
                mean=map_summary["bubble_mean"],
                minimum=map_summary.get("bubble_min"),
                maximum=map_summary.get("bubble_max"),
            )
        if "lifetime_mean" in map_summary:
            _line(
                label=f"{data.unit_label()} lifetime",
                mean=map_summary["lifetime_mean"],
                minimum=map_summary.get("lifetime_min"),
                maximum=map_summary.get("lifetime_max"),
            )
        if "bubble_ratio" in map_summary:
            print(
                f"  {'bubble ratio':<16s}{map_summary['bubble_ratio'] * ShowCommand.PERCENT_SCALE:.1f}%"
            )

        list_bubbliest = map_summary.get("top_bubbliest") or []
        if list_bubbliest:
            print(f"  bubbliest {data.unit_label()}s:")
            for map_thread in list_bubbliest:
                life = map_thread.get(
                    "lifetime",
                    0,
                )
                bubble_percent = (
                    (
                        map_thread.get(
                            "bubble",
                            0,
                        )
                        / life
                    )
                    * ShowCommand.PERCENT_SCALE
                    if life
                    else 0
                )
                print(
                    f"    {data.locator(map_thread['tid'])}  "
                    f"bubble={TraceClock.format_ticks(map_thread.get('bubble', 0), data.clock_type)} / "
                    f"lifetime={TraceClock.format_ticks(life, data.clock_type)}  ({bubble_percent:.1f}%)"
                )
        print()

    @staticmethod
    def _sample_tids(
        list_tid: list[int],
        count: int,
    ) -> list[int]:
        if (
            count <= 0
            or len(list_tid) <= count
        ):
            return list(list_tid)
        step = len(list_tid) / count
        return [list_tid[int(index * step)] for index in range(count)]

    @staticmethod
    def _render_timeline(
        data: TraceData,
        sample: int,
        human: bool,
        sort_by: str | None,
    ) -> None:
        if not data.map_tid_intervals:
            if not data.intervals_built:
                print(
                    f"{ShowCommand.ANSI_FG_GREY}Timeline needs per-thread records; this report path "
                    f"used the aggregate block only.{ShowCommand.ANSI_RESET}\n"
                )
            else:
                print(
                    f"{ShowCommand.ANSI_FG_GREY}No intervals to render.{ShowCommand.ANSI_RESET}\n"
                )
            return

        # Choose threads
        list_tid = list(data.list_tid)
        if sort_by:
            map_tid_bubble = data.bubble_per_thread()
            if sort_by == "bubble":
                list_tid.sort(
                    key=lambda sampled_tid: map_tid_bubble.get(
                        sampled_tid,  # key
                        {},           # default
                    ).get(
                        "bubble",  # key
                        0,         # default
                    ),
                    reverse=True,
                )
            elif sort_by == "active":
                list_tid.sort(
                    key=lambda sampled_tid: map_tid_bubble.get(
                        sampled_tid,  # key
                        {},           # default
                    ).get(
                        "active",  # key
                        0,         # default
                    ),
                    reverse=True,
                )
            elif sort_by == "lifetime":
                list_tid.sort(
                    key=lambda sampled_tid: map_tid_bubble.get(
                        sampled_tid,  # key
                        {},           # default
                    ).get(
                        "lifetime",  # key
                        0,           # default
                    ),
                    reverse=True,
                )
            elif sort_by == "tid":
                list_tid.sort()
            else:
                # Sort by total time spent in a specific scope
                list_tid.sort(
                    key=lambda sampled_tid: sum(
                        interval.duration
                        for interval in data.map_tid_intervals.get(
                            sampled_tid,  # key
                            [],           # default
                        )
                        if interval.scope_label == sort_by
                    ),
                    reverse=True,
                )

        list_sampled_tid = ShowCommand._sample_tids(
            list_tid=list_tid,
            count=sample,
        )
        map_label_color = ShowCommand._scope_color_map(data.list_scope_label)
        map_label_glyph = ShowCommand._scope_glyph_map(data.list_scope_label)

        map_tid_label = {sampled_tid: data.locator(sampled_tid) for sampled_tid in list_sampled_tid}
        width = ShowCommand._terminal_width()
        label_w = (
            max(
                (len(label) for label in map_tid_label.values()),
                default=ShowCommand.TIMELINE_LABEL_WIDTH_DEFAULT,
            )
        ) + ShowCommand.TIMELINE_LABEL_PADDING
        bar_w = max(
            ShowCommand.TIMELINE_BAR_MIN,
            width - label_w - ShowCommand.TIMELINE_LABEL_PADDING,
        )

        span = max(
            1,
            data.t_max - data.t_min,
        )

        print(
            f"{ShowCommand.ANSI_BOLD}Timeline{ShowCommand.ANSI_RESET}  (showing {len(list_sampled_tid)} of {len(list_tid)} threads, "
            f"width={bar_w} cols, span={TraceClock.format_ticks(span, data.clock_type)})"
        )
        print(
            f"  {ShowCommand.ANSI_BOLD}Legend:{ShowCommand.ANSI_RESET}  {ShowCommand._format_legend(data.list_scope_label, human)}"
        )
        print()

        for tid in list_sampled_tid:
            list_interval = data.map_tid_intervals.get(
                tid,  # key
                [],   # default
            )
            # Background row
            list_cell = [ShowCommand.CHAR_BUBBLE] * bar_w
            list_background: list[str | None] = [None] * bar_w
            for interval in list_interval:
                column_start = int((interval.t_start - data.t_min) / span * (bar_w - 1))
                column_end = int((interval.t_end - data.t_min) / span * (bar_w - 1))
                column_start = max(
                    0,
                    min(
                        bar_w - 1,
                        column_start,
                    ),
                )
                column_end = max(
                    0,
                    min(
                        bar_w - 1,
                        column_end,
                    ),
                )
                if column_end < column_start:
                    column_end = column_start
                fill = (
                    ShowCommand.CHAR_SCOPE_COLORED
                    if human
                    else map_label_glyph.get(
                        interval.scope_label,  # key
                        "█",                   # default
                    )
                )
                for column in range(
                    column_start,
                    column_end + 1,
                ):
                    list_cell[column] = fill
                    list_background[column] = (
                        map_label_color.get(interval.scope_label) if human else None
                    )

            # Render with optional colour
            if human:
                list_part: list[str] = []
                cur_color: str | None = None
                for char, color in zip(
                    list_cell,
                    list_background,
                ):
                    if color != cur_color:
                        if cur_color is not None:
                            list_part.append(ShowCommand.ANSI_RESET)
                        if color is not None:
                            list_part.append(color)
                        cur_color = color
                    list_part.append(char)
                if cur_color is not None:
                    list_part.append(ShowCommand.ANSI_RESET)
                bar = "".join(list_part)
            else:
                bar = "".join(list_cell)

            print(
                f"  {ShowCommand.ANSI_FG_GREY}{map_tid_label[tid]:<{label_w}}{ShowCommand.ANSI_RESET}{bar}"
            )
        print()

    @staticmethod
    def _print_longest_scope(
        data: TraceData,
        scope: str,
    ) -> None:
        list_interval = data.map_label_intervals.get(scope)
        if not list_interval:
            print(
                f"{ShowCommand.ANSI_FG_RED}No scope named {scope!r} in this run.{ShowCommand.ANSI_RESET}\n"
            )
            return
        interval = max(
            list_interval,
            key=lambda candidate: candidate.duration,
        )
        print(
            f"{ShowCommand.ANSI_BOLD}Longest {scope!r} instance:{ShowCommand.ANSI_RESET}\n"
            f"  {data.locator(interval.tid)}  "
            f"duration={TraceClock.format_ticks(interval.duration, data.clock_type)}  "
            f"[{interval.t_start}, {interval.t_end}]  depth={interval.depth}\n"
        )

    @staticmethod
    def _print_longest_bubble(
        data: TraceData,
        top: int = TraceData.LONGEST_BUBBLES_DEFAULT,
    ) -> None:
        list_gap = data.longest_bubbles(top)
        if not list_gap:
            print(
                f"{ShowCommand.ANSI_FG_GREY}No inter-scope bubbles found (single-scope kernel?).{ShowCommand.ANSI_RESET}\n"
            )
            return
        print(
            f"{ShowCommand.ANSI_BOLD}Longest bubbles (top {len(list_gap)}):{ShowCommand.ANSI_RESET}"
        )
        for map_gap in list_gap:
            print(
                f"  {data.locator(map_gap['tid'])}  "
                f"{map_gap['before']} → {map_gap['after']}  gap={TraceClock.format_ticks(map_gap['gap'], data.clock_type)}"
            )
        print()

    @staticmethod
    def _print_outliers(
        data: TraceData,
        sigmas: float = OUTLIER_SIGMAS_DEFAULT,
    ) -> None:
        map_tid_bubble = data.bubble_per_thread()
        if len(map_tid_bubble) < ShowCommand.OUTLIER_ROWS_MIN:
            return
        list_active = [map_bubble["active"] for map_bubble in map_tid_bubble.values()]
        mean = statistics.fmean(list_active)
        std_dev = statistics.pstdev(list_active) if len(list_active) > 1 else 0.0
        if std_dev == 0:
            print(
                f"{ShowCommand.ANSI_FG_GREY}No active-time variance — no outliers.{ShowCommand.ANSI_RESET}\n"
            )
            return
        # thread -> how many standard deviations its active time lies from the mean
        map_flagged_score: dict[int, float] = {}
        for tid, map_bubble in map_tid_bubble.items():
            z_score = (map_bubble["active"] - mean) / std_dev
            if abs(z_score) >= sigmas:
                map_flagged_score[tid] = z_score
        list_flagged_tid = sorted(
            map_flagged_score,  # iterable
            key=lambda flagged_tid: abs(map_flagged_score[flagged_tid]),
            reverse=True,
        )
        print(
            f"{ShowCommand.ANSI_BOLD}Outlier {data.unit_label()}s "
            f"(|z| ≥ {sigmas}, mean={TraceClock.format_ticks(int(mean), data.clock_type)}, "
            f"sd={TraceClock.format_ticks(int(std_dev), data.clock_type)}):{ShowCommand.ANSI_RESET}"
        )
        if not list_flagged_tid:
            print(f"  {ShowCommand.ANSI_FG_GREY}none{ShowCommand.ANSI_RESET}\n")
            return
        for tid in list_flagged_tid[: ShowCommand.OUTLIER_SHOWN_MAX]:
            active = map_tid_bubble[tid]["active"]
            z_score = map_flagged_score[tid]
            marker = "+" if z_score > 0 else "-"
            print(
                f"  {data.locator(tid)}  "
                f"active={TraceClock.format_ticks(active, data.clock_type)}  "
                f"z={marker}{abs(z_score):.2f}"
            )
        if len(list_flagged_tid) > ShowCommand.OUTLIER_SHOWN_MAX:
            print(f"  ... +{len(list_flagged_tid) - ShowCommand.OUTLIER_SHOWN_MAX} more")
        print()

    @staticmethod
    def _print_concurrency(
        data: TraceData,
        bins: int = TraceData.CONCURRENCY_BINS_DEFAULT,
    ) -> None:
        map_concurrency = data.concurrency(bins)
        if (
            not map_concurrency
            or not map_concurrency.get("bins")
        ):
            return
        list_count = map_concurrency["bins"]
        max_count = map_concurrency.get("peak") or (max(list_count) if list_count else 0)
        print(
            f"{ShowCommand.ANSI_BOLD}Concurrency (active {data.unit_label()}s per time bin, "
            f"{len(list_count)} bins):{ShowCommand.ANSI_RESET}"
        )
        print(
            f"  peak={max_count:,}  span={TraceClock.format_ticks(map_concurrency.get('span', 0), data.clock_type)}\n"
        )
        chars = ShowCommand.SPARK_LEVELS
        list_char = []
        for count in list_count:
            if max_count == 0:
                list_char.append(" ")
            else:
                level_index = min(
                    len(chars) - 1,
                    int(count / max_count * (len(chars) - 1)),
                )
                list_char.append(chars[level_index])
        print("  " + "".join(list_char))
        print()

    @staticmethod
    def _emit_json(data: TraceData) -> None:
        map_payload = {
            "kernel_prototype": data.kernel_prototype,
            "compile_results": data.map_compile_results,
            "launch": {
                "grid_dim": data.grid_dim,
                "block_dim": data.block_dim,
                "block_size": data.block_size,
                "warp_size": data.get_compile_results().get_warp_size(),
            },
            "scope_labels": data.list_scope_label,
            "n_threads": data.n_threads,
            "n_records": data.total_records,
            "t_min": data.t_min,
            "t_max": data.t_max,
            "scope_stats": data.scope_stats(),
            "pipeline_summary": data.pipeline_summary(),
            "longest_bubbles": data.longest_bubbles(),
            "concurrency": data.concurrency(),
            "unmatched_starts": data.unmatched_starts,
            "unmatched_ends": data.unmatched_ends,
        }
        print(
            json.dumps(
                obj=map_payload,
                indent=ShowCommand.JSON_INDENT,
                default=str,
            )
        )

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        """Add the ``show`` subparser and its view and filter options to the command line.

        Args:
            subparsers: the subcommand registry of the parser
        """
        self.parser = subparsers.add_parser(
            name="show",
            help="Visualize a Section_IntraKernelTrace JSON report",
            usage="xtrace show <report.json> [options]",
            description=(
                "Render pipeline analysis for a Section_IntraKernelTrace report:\n"
                "  - aggregate per-scope statistics\n"
                "  - per-thread ASCII timeline (Gantt-style)\n"
                "  - bubble (idle-gap) detection\n"
                "  - longest scope / longest bubble lookup\n"
                "  - concurrency over time histogram\n"
                "  - outlier-thread detection\n\n"
                "Only JSON trace reports are accepted."
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser = self.parser
        parser.add_argument(
            "report",
            help="Path to a .json report produced by Section_IntraKernelTrace",
        )
        parser.add_argument(
            "--run",
            type=int,
            default=0,
            help="Run index to visualize when multiple runs are present (default: 0)",
        )

        # Filtering
        parser.add_argument(
            "--select",
            type=str,
            default=None,
            metavar="TOKEN",
            help="A selection token copied from the report's trace "
            "panel ('gwtrace/1 ...'): show only the rows and the "
            "stretch of time it names. Paste it verbatim.",
        )
        parser.add_argument(
            "--tid",
            type=str,
            default=None,
            help="Thread ID ranges, e.g. '[0,31],[64,95]' or '0,1,2'",
        )
        parser.add_argument(
            "--block",
            type=str,
            default=None,
            help="Block ID ranges (needs launch scale), e.g. '0-3,5' or '[0,3]'",
        )
        parser.add_argument(
            "--warp",
            type=str,
            default=None,
            help="Warp-in-block ID ranges (needs launch scale), e.g. '0,1'",
        )
        parser.add_argument(
            "--stime",
            type=int,
            default=None,
            help="Start time (payload tick) — drop intervals ending before this",
        )
        parser.add_argument(
            "--etime",
            type=int,
            default=None,
            help="End time (payload tick) — drop intervals starting after this",
        )

        # Views
        parser.add_argument(
            "--stats",
            action="store_true",
            help="Show per-scope stats only",
        )
        parser.add_argument(
            "--timeline",
            action="store_true",
            help="Show per-thread ASCII timeline only",
        )
        parser.add_argument(
            "--bubbles",
            action="store_true",
            help="Show bubble summary + longest bubbles",
        )
        parser.add_argument(
            "--concurrency",
            action="store_true",
            help="Show concurrency histogram over time",
        )
        parser.add_argument(
            "--outliers",
            action="store_true",
            help="Show threads whose total active time is an outlier",
        )
        parser.add_argument(
            "--longest-bubble",
            action="store_true",
            help="List the longest inter-scope bubbles across all threads",
        )
        parser.add_argument(
            "--longest",
            type=str,
            default=None,
            metavar="SCOPE",
            help="Show the single longest instance of SCOPE (scope_label)",
        )

        # Rendering
        parser.add_argument(
            "--sample",
            type=int,
            default=ShowCommand.TIMELINE_SAMPLE_DEFAULT,
            help=f"Number of threads to render in --timeline (default: {ShowCommand.TIMELINE_SAMPLE_DEFAULT}; 0 means all)",
        )
        parser.add_argument(
            "--sort-by",
            type=str,
            default=None,
            help="Sort threads in --timeline by 'tid' / 'active' / "
            "'bubble' / 'lifetime' / <scope_label>",
        )
        parser.add_argument(
            "--human",
            action="store_true",
            help="Use ANSI colour for the timeline",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Emit machine-readable JSON instead of pretty output",
        )
        parser.set_defaults(func=self.run)

    # ---- entry point ------------------------------------------------------

    def run(self, args: argparse.Namespace) -> None:
        """Load the report, select a run, apply the filters and print the requested views.

        With no view option the default overview is printed; ``--json`` emits
        the result as JSON instead.

        Args:
            args: the parsed ``show`` arguments

        Raises:
            SystemExit: the report is missing or malformed, holds no intra-kernel
                trace section, the run index or a filter is invalid, or the
                selection token names a different kernel
        """
        if not os.path.isfile(args.report):
            print(
                f"{ShowCommand.ANSI_FG_RED}File not found:{ShowCommand.ANSI_RESET} {args.report}",
                file=sys.stderr,
            )
            sys.exit(ShowCommand.EXIT_USAGE)

        try:
            doc = ShowCommand._load_report(args.report)
        except (
            json.JSONDecodeError,
            ValueError,
        ) as e:
            print(
                f"{ShowCommand.ANSI_FG_RED}Report parse error:{ShowCommand.ANSI_RESET} {e}",
                file=sys.stderr,
            )
            sys.exit(ShowCommand.EXIT_USAGE)

        map_section = ShowCommand._find_intra_kernel_section(doc)
        if map_section is None:
            print(
                f"{ShowCommand.ANSI_FG_RED}No Section_IntraKernelTrace section found in this report.{ShowCommand.ANSI_RESET}\n"
                "  Expected envelope: {sections: [{type: 'Section_IntraKernelTrace', data: {runs: [...]}}]}",
                file=sys.stderr,
            )
            sys.exit(ShowCommand.EXIT_USAGE)

        list_run = map_section.get("runs") or []
        n_runs = (
            len(list_run)
            if isinstance(
                list_run,  # obj
                list,      # class_or_tuple
            )
            else 0
        )

        try:
            map_run = ShowCommand._select_run(
                map_section_data=map_section,
                run_index=args.run,
            )
        except ValueError as e:
            print(
                f"{ShowCommand.ANSI_FG_RED}{e}{ShowCommand.ANSI_RESET}",
                file=sys.stderr,
            )
            sys.exit(ShowCommand.EXIT_USAGE)

        # Parse filters
        try:
            list_tid_range = IdRange.parse_list(args.tid) if args.tid else []
            list_block_range = IdRange.parse_list(args.block) if args.block else []
            list_warp_range = IdRange.parse_list(args.warp) if args.warp else []
        except ValueError as e:
            print(
                f"{ShowCommand.ANSI_FG_RED}{e}{ShowCommand.ANSI_RESET}",
                file=sys.stderr,
            )
            sys.exit(ShowCommand.EXIT_USAGE)
        selected = None
        stime, etime = args.stime, args.etime
        if args.select:
            try:
                selected = SelectionToken.parse(args.select)
            except ValueError as e:
                print(
                    f"{ShowCommand.ANSI_FG_RED}{e}{ShowCommand.ANSI_RESET}",
                    file=sys.stderr,
                )
                sys.exit(ShowCommand.EXIT_USAGE)
            # the tag is what says the token and the report are about one kernel
            actual = SelectionToken.kernel_tag(str(map_run.get("kernel_prototype") or ""))
            if selected.kernel != actual:
                print(
                    f"{ShowCommand.ANSI_FG_RED}this token was copied from a panel over a different "
                    f"kernel{ShowCommand.ANSI_RESET}\n"
                    f"  token: {selected.kernel}\n"
                    f"  report: {actual}  ({map_run.get('kernel_prototype')})",
                    file=sys.stderr,
                )
                sys.exit(ShowCommand.EXIT_USAGE)
            # the panel rebases time to the run's first interval, so its token does too
            base = int((map_run.get("analysis") or {}).get("t_min") or 0)
            stime = base + selected.t0 if stime is None else stime
            etime = base + selected.t1 if etime is None else etime
        filtering = bool(
            list_tid_range
            or list_block_range
            or list_warp_range
            or selected
            or stime is not None
            or etime is not None
        )

        explicit_views = (
            args.stats
            or args.timeline
            or args.bubbles
            or args.concurrency
            or args.outliers
            or args.longest_bubble
            or args.longest is not None
        )

        # per-thread intervals are rebuilt only for views that show a thread's records
        need_intervals = (
            filtering
            or args.timeline
            or args.outliers
            or (args.longest is not None)
            or (not explicit_views and not args.json)
        )
        data = TraceData.from_run(
            map_run=map_run,
            need_intervals=need_intervals,
        )
        set_tid = None
        if selected:
            # the token names rows at one level; the run answers with its own ids
            set_tid = set(
                selected.rows_of(
                    list_tid=data.list_tid,
                    block_size=data.block_size or 0,
                    warp_size=data.get_compile_results().get_warp_size(),
                    warps_per_group=data.get_compile_results().get_warps_per_group(),
                )
            )
            if not set_tid:
                print(
                    f"{ShowCommand.ANSI_FG_YELLOW}the token names no {selected.level} this report holds{ShowCommand.ANSI_RESET}",
                    file=sys.stderr,
                )
        data = data.filtered(
            list_tid_range=list_tid_range,
            stime=stime,
            etime=etime,
            list_block_range=list_block_range,
            list_warp_range=list_warp_range,
            set_tid=set_tid,
        )
        if selected:
            ShowCommand._print_selection(
                selected=selected,
                data=data,
            )

        if args.json:
            ShowCommand._emit_json(data)
            return

        ShowCommand._print_header(
            data=data,
            run_index=args.run,
            n_runs=n_runs or 1,
        )

        if not explicit_views:
            ShowCommand._print_scope_stats(data)
            ShowCommand._print_pipeline_summary(data)
            ShowCommand._render_timeline(
                data=data,
                sample=args.sample,
                human=args.human,
                sort_by=args.sort_by,
            )
            return

        if args.stats:
            ShowCommand._print_scope_stats(data)
        if args.bubbles:
            ShowCommand._print_pipeline_summary(data)
            ShowCommand._print_longest_bubble(data)
        if args.timeline:
            ShowCommand._render_timeline(
                data=data,
                sample=args.sample,
                human=args.human,
                sort_by=args.sort_by,
            )
        if args.concurrency:
            ShowCommand._print_concurrency(data)
        if args.outliers:
            ShowCommand._print_outliers(data)
        if (
            args.longest_bubble
            and not args.bubbles
        ):
            ShowCommand._print_longest_bubble(data)
        if args.longest is not None:
            ShowCommand._print_longest_scope(
                data=data,
                scope=args.longest,
            )


# ---------------------------------------------------------------------------
# Standalone entry point (kept for ad-hoc invocation; ``xtrace show`` is the
# normal path).
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the ``show`` command on its own, outside the ``xtrace`` command line."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    command = ShowCommand()
    command.register(subparsers)
    args = parser.parse_args()
    if hasattr(
        args,    # obj
        "func",  # name
    ):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
