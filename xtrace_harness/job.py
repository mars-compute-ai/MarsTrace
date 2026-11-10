import decimal
import gzip
import io
import json
import math
import os
import re
import tarfile
import tomllib
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.binary import KernelBinary
from xtrace_harness.error import ManifestError
from xtrace_harness.tensors import TensorFile


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
    LAUNCH_SCHEMA = "marscompute.kernel-launch"
    MODE_EXECUTABLE = 0o755
    MODE_REGULAR = 0o644
    EXECUTABLE_BITS = 0o111

    list_cubin: list[str]
    source: str | None = None
    manifest: str | None = None
    list_tensors: list[str] = field(default_factory=list)
    list_file: list[str] = field(default_factory=list)

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
            WorkspaceSpec._check_path(
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
            WorkspaceSpec._check_path(
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
                or map_launch.get("schema_version") != WorkspaceSpec.LAUNCH_SCHEMA
            ):
                raise ManifestError(
                    f"[workspace] manifest: {self.manifest} is not a {WorkspaceSpec.LAUNCH_SCHEMA} manifest"
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
class JobManifest:
    """A job directory's ``xtrace.toml``, checked, and its translation into the platform API:
    an agent on the GPU host reads the workspace, instruments and runs the kernel within its
    budget, and returns one ``xtrace.json``.

    Args:
        directory: the job directory
        hardware: the environment id the job runs on
        intent: what the trace should answer, the job's prompt
        workspace: the files the agent works with
        agent: the model and the session's budget
        limits: the most the job may cost and use
        list_binary: the workspace's cubins
    """

    FILE_NAME = "xtrace.toml"
    SUBMIT_SCHEMA = "marscompute.xtrace.submit.v1"
    BUNDLE_KIND = "agent_workspace"
    PROMPT_CHARACTERS_MAX = 32000
    # the manifest's required text keys and what each holds
    MAP_REQUIRED_TEXT = {
        "hardware": "the environment id from 'xtrace hardware'",
        "intent": "what the trace should answer, in your words",
    }
    # the manifest's required tables
    SET_REQUIRED_TABLE = {
        "workspace",
        "agent",
    }
    SET_KEY = {
        "hardware",
        "intent",
        "workspace",
        "agent",
        "limits",
    }
    # artifact kind of each workspace part
    MAP_KIND = {
        "cubin": "cubin",
        "source": "source",
        "manifest": "launch_manifest",
        "tensors": "safetensors",
        "files": "workspace_file",
    }

    directory: str
    hardware: str
    intent: str
    workspace: WorkspaceSpec
    agent: AgentOptions
    limits: JobLimits = field(default_factory=JobLimits)
    list_binary: list[KernelBinary] = field(default_factory=list)

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

    @classmethod
    def load(cls, directory: str) -> "JobManifest":
        """Read and check the manifest of a job directory.

        Args:
            directory: the job directory, holding ``xtrace.toml``

        Returns:
            the manifest

        Raises:
            ManifestError: the manifest is missing, malformed or incomplete, or names an absent
                file
            HarnessError: a cubin is not a cubin
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
        JobManifest.check_table(
            value=map_manifest,
            set_key=JobManifest.SET_KEY,
            where=JobManifest.FILE_NAME,
        )
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
        for key in sorted(JobManifest.SET_REQUIRED_TABLE):
            if key not in map_manifest:
                raise ManifestError(f"{path}: [{key}] is required; a job needs [workspace] and [agent]")
        workspace = WorkspaceSpec.from_table(map_manifest["workspace"])
        return cls(
            directory=directory,
            hardware=map_manifest["hardware"],
            intent=map_manifest["intent"],
            workspace=workspace,
            agent=AgentOptions.from_table(map_manifest["agent"]),
            limits=JobLimits.from_table(
                map_manifest.get(
                    "limits",
                    {},
                )
            ),
            list_binary=workspace.validate(directory),
        )

    def select_compute(self, map_capabilities: dict[str, Any]) -> dict[str, Any]:
        """The capabilities entry of the manifest's hardware, checked to run the job.

        Args:
            map_capabilities: the platform's capabilities

        Returns:
            the entry, holding ``compute`` and ``architecture``

        Raises:
            ManifestError: the hardware is unknown, a cubin targets another architecture, or
                the model is not offered
        """
        if not map_capabilities.get("enabled"):
            raise ManifestError("Xtrace is not enabled for this account; see the console's Xtrace settings")
        list_entry = map_capabilities.get("compute", [])
        map_entry = next(
            (
                map_candidate
                for map_candidate in list_entry
                if map_candidate.get("environment_id") == self.hardware
            ),
            None,
        )
        if map_entry is None:
            raise ManifestError(
                f"no hardware '{self.hardware}'; 'xtrace hardware' lists: "
                f"{', '.join(map_candidate.get('environment_id', '?') for map_candidate in list_entry) or 'none'}"
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
        list_model = map_capabilities.get("agent_models", [])
        if self.agent.model not in list_model:
            raise ManifestError(
                f"agent model '{self.agent.model}' is not offered; 'xtrace hardware' lists: "
                f"{', '.join(list_model) or 'none'}"
            )
        return map_entry

    def list_artifact(self) -> list[JobArtifact]:
        """The files the workspace uploads.

        Returns:
            the cubins, the source archive, the launch manifest, the tensors and the files
        """
        list_artifact = [
            JobArtifact(
                kind=JobManifest.MAP_KIND["cubin"],
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
                    kind=JobManifest.MAP_KIND["source"],
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
                    kind=JobManifest.MAP_KIND["manifest"],
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
                    kind=JobManifest.MAP_KIND[key],
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
            "kind": JobManifest.BUNDLE_KIND,
            "cubin_artifact_ids": list(map_artifact_ids[JobManifest.MAP_KIND["cubin"]]),
            "tensor_artifact_ids": list(map_artifact_ids.get(JobManifest.MAP_KIND["tensors"], [])),
            "file_artifact_ids": list(map_artifact_ids.get(JobManifest.MAP_KIND["files"], [])),
        }
        for key, bundle_key in (
            ("source", "source_archive_artifact_id"),
            ("manifest", "launch_manifest_artifact_id"),
        ):
            list_id = map_artifact_ids.get(JobManifest.MAP_KIND[key])
            if list_id:
                map_bundle[bundle_key] = list_id[0]
        return map_bundle

    def estimate_input(
        self,
        map_capabilities: dict[str, Any],
        map_artifact_ids: dict[str, list[str]],
    ) -> dict[str, Any]:
        """The request a quote and a submission take.

        Args:
            map_capabilities: the platform's capabilities
            map_artifact_ids: artifact kind to the ids of the uploaded artifacts of that kind

        Returns:
            the estimate input, ``marscompute.xtrace.submit.v1``
        """
        return {
            "schema_version": JobManifest.SUBMIT_SCHEMA,
            "prompt": self.intent,
            "bundle": self.bundle(map_artifact_ids),
            "compute": self.select_compute(map_capabilities)["compute"],
            "analysis": self.agent.to_analysis(),
            "limits": self.limits.to_api(map_capabilities.get("limits", {})),
        }
