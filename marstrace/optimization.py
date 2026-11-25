import argparse
import functools
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from marstrace.error import ServiceError


@dataclass
class DeliveredOptimization:
    """The optimization a thinking job delivers in ``marstrace.json`` after its trace, read for
    a person: the outcome, the time of every configuration before and after, the iterations
    the agent kept or reverted, and the deliverables, which the client writes as files.

    Args:
        status: ``improved``, ``no_gain`` or ``incomplete``
        correctness: whether the optimized kernel passed ``verify`` on every configuration
        best_speedup: the geometric mean over the configurations of baseline time over best
            time, or None
        list_config: per configuration its ``name``, ``baseline_us``, ``best_us``,
            ``speedup`` and ``correct``
        list_iteration: per iteration its ``name``, ``verdict``, ``speedup`` and ``summary``
        list_file: the deliverables, each with ``path``, ``sha256``, ``size_bytes``,
            ``media_type`` and its UTF-8 ``text``
    """

    SECTION_TYPE = "marscompute.optimization"
    DIRECTORY = "optimization"
    PATCH_PATH = "optimized.patch"
    REPORT_SUFFIX = ".md"
    ITERATIONS_DIRECTORY = "iterations"
    NO_GAIN_STATUS = "no_gain"
    PASSED = "passed"
    # a finished run that measured every change and kept none, told apart from a failed job
    NO_GAIN_NOTE = (
        "  no faster version found: every change was measured and reverted, so the uploaded code "
        "stays the fastest; the job is complete, and its quality is partial because nothing got faster"
    )
    PATH_PATTERN = re.compile(r"[A-Za-z0-9._/-]{1,200}")
    PARENT_REFERENCE = ".."
    # path segments that name no file
    SET_EMPTY_SEGMENT = {
        "",
        ".",
    }
    SEPARATOR = "/"
    # column title -> width of the configuration table, after the name column
    MAP_COLUMN_WIDTH = {
        "baseline us": 12,
        "best us": 10,
        "speedup": 9,
        "correct": 8,
    }

    status: str
    correctness: str
    best_speedup: float | None = None
    list_config: list[dict[str, Any]] = field(default_factory=list)
    list_iteration: list[dict[str, Any]] = field(default_factory=list)
    list_file: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _list_of_object(value: Any) -> list[dict[str, Any]]:
        """The objects of a list, or none when the value is not a list."""
        if not isinstance(
            value,  # obj
            list,   # class_or_tuple
        ):
            return []
        return [
            item
            for item in value
            if isinstance(
                item,  # obj
                dict,  # class_or_tuple
            )
        ]

    @staticmethod
    def _number(value: Any) -> float | None:
        """A JSON number, or None for anything else."""
        if (
            isinstance(
                value,  # obj
                bool,   # class_or_tuple
            )
            or not isinstance(
                value,         # obj
                (int, float),  # class_or_tuple
            )
        ):
            return None
        return float(value)

    @staticmethod
    def from_document(document: Any) -> "DeliveredOptimization | None":
        """The optimization section of a decoded ``marstrace.json``.

        Args:
            document: the decoded report

        Returns:
            the optimization, or None when the report carries none
        """
        if (
            not isinstance(
                document,  # obj
                dict,      # class_or_tuple
            )
            or not isinstance(
                document.get("sections"),  # obj
                list,                      # class_or_tuple
            )
        ):
            return None
        for section in DeliveredOptimization._list_of_object(document["sections"]):
            if (
                section.get("type") == DeliveredOptimization.SECTION_TYPE
                and isinstance(
                    section.get("data"),  # obj
                    dict,                 # class_or_tuple
                )
            ):
                data = section["data"]
                summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
                return DeliveredOptimization(
                    status=str(summary.get("status") or ""),
                    correctness=str(summary.get("correctness") or ""),
                    best_speedup=DeliveredOptimization._number(summary.get("best_speedup")),
                    list_config=DeliveredOptimization._list_of_object(data.get("list_config")),
                    list_iteration=DeliveredOptimization._list_of_object(data.get("list_iteration")),
                    list_file=DeliveredOptimization._list_of_object(data.get("list_file")),
                )
        return None

    @staticmethod
    def from_file(path: str) -> "DeliveredOptimization | None":
        """The optimization of a ``marstrace.json`` on disk.

        Args:
            path: the report

        Returns:
            the optimization, or None when the report carries none or does not read as JSON
        """
        try:
            with open(
                file=path,
                mode="r",
                encoding="utf-8",
            ) as trace_file:
                document = json.load(trace_file)
        except (
            OSError,
            ValueError,
        ):
            return None
        return DeliveredOptimization.from_document(document)

    @staticmethod
    def directory_of(trace_path: str) -> str:
        """Where the deliverables of a ``marstrace.json`` are written.

        Args:
            trace_path: the report

        Returns:
            the ``optimization`` directory beside it
        """
        return os.path.join(
            os.path.dirname(trace_path),
            DeliveredOptimization.DIRECTORY,
        )

    @staticmethod
    def check_path(path: Any) -> str:
        """Check a deliverable's path before anything is written under it.

        Args:
            path: the ``path`` of a ``list_file`` entry

        Returns:
            the path

        Raises:
            ServiceError: the path is not a relative path of letters, digits, ``.``, ``_``,
                ``-`` and ``/``, or it holds ``..``
        """
        if (
            not isinstance(
                path,  # obj
                str,   # class_or_tuple
            )
            or path.startswith(DeliveredOptimization.SEPARATOR)
            or DeliveredOptimization.PARENT_REFERENCE in path
            or DeliveredOptimization.PATH_PATTERN.fullmatch(path) is None
            or any(
                segment in DeliveredOptimization.SET_EMPTY_SEGMENT
                for segment in path.split(DeliveredOptimization.SEPARATOR)
            )
        ):
            raise ServiceError(f"the optimization delivered a file with an unsafe path {path!r}; nothing was written")
        return path

    @staticmethod
    def check_file(map_file: dict[str, Any]) -> bytes:
        """Check a deliverable's text against its digest and size.

        Args:
            map_file: a ``list_file`` entry

        Returns:
            the text's UTF-8 bytes

        Raises:
            ServiceError: the path is unsafe, or the bytes differ from ``sha256`` or
                ``size_bytes``
        """
        path = DeliveredOptimization.check_path(map_file.get("path"))
        text = map_file.get("text")
        if not isinstance(
            text,  # obj
            str,   # class_or_tuple
        ):
            raise ServiceError(f"the optimization delivered {path} without its text; nothing was written")
        data = text.encode("utf-8")
        if (
            len(data) != map_file.get("size_bytes")
            or hashlib.sha256(data).hexdigest() != map_file.get("sha256")
        ):
            raise ServiceError(
                f"the optimization delivered {path} with different bytes than its sha256 and "
                "size_bytes name; nothing was written"
            )
        return data

    def write_files(self, output_directory: str) -> list[str]:
        """Write every deliverable under ``<output_directory>/optimization/``, once all are checked.

        Args:
            output_directory: the directory the job's results land in

        Returns:
            the paths written, in the order delivered

        Raises:
            ServiceError: a deliverable's path is unsafe or named twice, or its bytes differ
                from its digest or size; nothing is written then
        """
        directory = os.path.join(
            output_directory,
            DeliveredOptimization.DIRECTORY,
        )
        map_data: dict[str, bytes] = {}
        for map_file in self.list_file:
            data = DeliveredOptimization.check_file(map_file)
            if map_file["path"] in map_data:
                raise ServiceError(f"the optimization delivered {map_file['path']} twice; nothing was written")
            map_data[map_file["path"]] = data
        list_path = []
        for relative_path, data in map_data.items():
            path = os.path.join(
                directory,
                *relative_path.split(DeliveredOptimization.SEPARATOR),
            )
            os.makedirs(
                name=os.path.dirname(path),
                exist_ok=True,
            )
            with open(
                file=path,
                mode="wb",
            ) as deliverable_file:
                deliverable_file.write(data)
            list_path.append(path)
        return list_path

    @staticmethod
    def speedup_text(speedup: Any) -> str:
        """A speedup as a reader quotes it.

        Args:
            speedup: baseline time over optimized time, or None

        Returns:
            e.g. ``1.37x``, or ``none yet`` when there is none
        """
        number = DeliveredOptimization._number(speedup)
        return "none yet" if number is None else f"{number:.2f}x"

    @staticmethod
    def _time_text(time_us: Any) -> str:
        """A time in microseconds, or ``-`` when absent."""
        number = DeliveredOptimization._number(time_us)
        return "-" if number is None else f"{number:.1f}"

    def config_lines(self) -> list[str]:
        """The configuration table: each configuration's time before and after, and its check.

        Returns:
            the header and one line per configuration, or none without configurations
        """
        if not self.list_config:
            return []
        name_width = max(
            len("config"),
            *(len(str(map_config.get("name", ""))) for map_config in self.list_config),
        )
        list_line = [
            f"  {'config':<{name_width}}  "
            + "".join(
                f"{title:>{width}}"
                for title, width in DeliveredOptimization.MAP_COLUMN_WIDTH.items()
            )
        ]
        for map_config in self.list_config:
            correct = map_config.get("correct")
            list_value = [
                DeliveredOptimization._time_text(map_config.get("baseline_us")),
                DeliveredOptimization._time_text(map_config.get("best_us")),
                DeliveredOptimization.speedup_text(map_config.get("speedup")),
                "-" if correct is None else ("yes" if correct is True else "no"),
            ]
            list_line.append(
                f"  {str(map_config.get('name', '')):<{name_width}}  "
                + "".join(
                    f"{value:>{width}}"
                    for value, width in zip(
                        list_value,
                        DeliveredOptimization.MAP_COLUMN_WIDTH.values(),
                    )
                )
            )
        return list_line

    def text(self, directory: str | None = None) -> str:
        """The optimization as text: outcome, configurations, iterations, and where its files are.

        Args:
            directory: the ``optimization`` directory the deliverables were written to, or
                None when they were not written

        Returns:
            the text
        """
        list_line = [
            f"optimization: {self.status or 'no status'}, correctness {self.correctness or 'unknown'}, "
            f"best speedup {DeliveredOptimization.speedup_text(self.best_speedup)}"
        ]
        if (
            self.status == DeliveredOptimization.NO_GAIN_STATUS
            and self.correctness == DeliveredOptimization.PASSED
        ):
            list_line.append(DeliveredOptimization.NO_GAIN_NOTE)
        list_line.extend(self.config_lines())
        if self.list_iteration:
            list_line.append("iterations:")
        for map_iteration in self.list_iteration:
            summary = f": {map_iteration['summary']}" if map_iteration.get("summary") else ""
            list_line.append(
                f"  {str(map_iteration.get('verdict', '')):<9} {map_iteration.get('name', '')} "
                f"{DeliveredOptimization.speedup_text(map_iteration.get('speedup'))}{summary}"
            )
        list_relative_path = [str(map_file.get("path", "")) for map_file in self.list_file]
        if directory is None:
            if list_relative_path:
                list_line.append(f"files: {len(list_relative_path)} delivered, not written")
            return "\n".join(list_line) + "\n"
        list_report = [
            relative_path
            for relative_path in list_relative_path
            if (
                relative_path.endswith(DeliveredOptimization.REPORT_SUFFIX)
                and DeliveredOptimization.SEPARATOR not in relative_path
            )
        ]
        if DeliveredOptimization.PATCH_PATH in list_relative_path:
            list_line.append(f"patch     {os.path.join(directory, DeliveredOptimization.PATCH_PATH)}")
        else:
            list_line.append("patch     none delivered")
        for index, relative_path in enumerate(list_report):
            label = "reports" if index == 0 else ""
            list_line.append(f"{label:<9} {os.path.join(directory, relative_path)}")
        nb_iteration_file = sum(
            1
            for relative_path in list_relative_path
            if relative_path.startswith(DeliveredOptimization.ITERATIONS_DIRECTORY + DeliveredOptimization.SEPARATOR)
        )
        if nb_iteration_file:
            list_line.append(
                f"iteration {nb_iteration_file} file(s) under "
                f"{os.path.join(directory, DeliveredOptimization.ITERATIONS_DIRECTORY)}"
            )
        return "\n".join(list_line) + "\n"

    def to_json(self) -> dict[str, Any]:
        """The section's data for a machine reader, each deliverable named without its text.

        Returns:
            the status, correctness, best speedup, configurations, iterations and files
        """
        return {
            "status": self.status,
            "correctness": self.correctness,
            "best_speedup": self.best_speedup,
            "list_config": self.list_config,
            "list_iteration": self.list_iteration,
            "list_file": [
                {
                    key: value
                    for key, value in map_file.items()
                    if key != "text"
                }
                for map_file in self.list_file
            ],
        }

    @staticmethod
    def extend_show(subparsers: argparse._SubParsersAction) -> None:
        """Give the viewer's ``show`` command the ``--optimization`` view, which the client owns.

        Args:
            subparsers: the subcommands, after the viewer registered ``show`` among them
        """
        show = subparsers.choices["show"]
        show.add_argument(
            "--optimization",
            action="store_true",
            help="Show the optimization a thinking job delivered: the outcome, each configuration's "
            "time before and after, the iterations, and where the patch and the reports were written",
        )
        show.set_defaults(
            func=functools.partial(
                DeliveredOptimization._show,
                viewer=show.get_default("func"),
            )
        )

    @staticmethod
    def _show(
        args: argparse.Namespace,
        viewer: Callable[[argparse.Namespace], None],
    ) -> None:
        """Print the report's optimization for ``--optimization``; any other view is the viewer's."""
        if not args.optimization:
            viewer(args)
            return
        optimization = DeliveredOptimization.from_file(args.report)
        if args.json:
            print(
                json.dumps(
                    None if optimization is None else optimization.to_json(),
                    indent=2,
                )
            )
            return
        if optimization is None:
            print("optimization: this report carries none")
            return
        directory = DeliveredOptimization.directory_of(args.report)
        print(
            optimization.text(directory if os.path.isdir(directory) else None),
            end="",
        )
