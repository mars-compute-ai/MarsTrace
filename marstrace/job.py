import gzip
import io
import json
import os
import re
import tarfile
import tomllib
from dataclasses import dataclass, field
from typing import Any
from marstrace.binary import KernelBinary
from marstrace.error import ManifestError
from marstrace.tensors import TensorFile


@dataclass
class JobLimits:
    """The optional ``[limits]`` table: how many attempts a job may take. A job is charged its
    actual usage over every attempt, so ``max_attempts`` and ``[agent]`` bound what it can cost.

    Args:
        max_attempts: the most attempts, or None for the service's default
    """

    ATTEMPTS_KEY = "max_attempts"
    ATTEMPTS_MAX = 3
    # attempts the service gives a request that leaves max_attempts out
    ATTEMPTS_DEFAULT = 2

    max_attempts: int | None = None

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
            set_key={JobLimits.ATTEMPTS_KEY},
            where="[limits]",
        )
        if JobLimits.ATTEMPTS_KEY in map_limits:
            JobManifest.check_integer(
                value=map_limits[JobLimits.ATTEMPTS_KEY],
                where=f"[limits] {JobLimits.ATTEMPTS_KEY}",
                low=1,
                high=JobLimits.ATTEMPTS_MAX,
            )
        return cls(max_attempts=map_limits.get(JobLimits.ATTEMPTS_KEY))

    def attempts(self) -> int:
        """The most attempts the job takes.

        Returns:
            ``max_attempts``, or the service's default when the manifest leaves it out
        """
        return JobLimits.ATTEMPTS_DEFAULT if self.max_attempts is None else self.max_attempts

    def to_api(self) -> dict[str, Any] | None:
        """The limits the job request takes.

        Returns:
            ``{"max_attempts": n}``, or None when the manifest leaves it to the service
        """
        if self.max_attempts is None:
            return None
        return {JobLimits.ATTEMPTS_KEY: self.max_attempts}


@dataclass
class JobInput:
    """A file the job uploads: one of the job directory, or one the client prepares.

    Args:
        role: the input's role, e.g. ``cubin``
        path: the file, or for prepared bytes the path whose name the upload carries
        content_type: the media type it is uploaded with
        data: the prepared bytes, or None to upload the file
    """

    role: str
    path: str
    content_type: str = "application/octet-stream"
    data: bytes | None = None

    def size_bytes(self) -> int:
        """The size of the bytes the input uploads.

        Returns:
            the size
        """
        if self.data is not None:
            return len(self.data)
        return os.path.getsize(self.path)

    def read(self) -> bytes:
        """The bytes the input uploads.

        Returns:
            the prepared bytes, or the file's
        """
        if self.data is not None:
            return self.data
        with open(
            file=self.path,
            mode="rb",
        ) as input_file:
            return input_file.read()


@dataclass
class AgentOptions:
    """The ``[agent]`` table: the budget of the session that traces the kernel on the GPU
    host. An absent key takes the service's default.

    Args:
        map_option: the keys the manifest sets
    """

    # key -> (lowest value, highest value)
    MAP_BOUND = {
        "max_turns": (1, 500),
        "max_input_tokens": (1, 50_000_000),
        "max_output_tokens": (1, 1_000_000),
        "max_wall_seconds": (60, 2580),
    }

    map_option: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_table(cls, map_agent: Any) -> "AgentOptions":
        """Read and check the ``[agent]`` table.

        Args:
            map_agent: the table

        Returns:
            the options

        Raises:
            ManifestError: a key is unknown or out of range
        """
        JobManifest.check_table(
            value=map_agent,
            set_key=set(AgentOptions.MAP_BOUND),
            where="[agent]",
        )
        for key, (low, high) in AgentOptions.MAP_BOUND.items():
            if key in map_agent:
                JobManifest.check_integer(
                    value=map_agent[key],
                    where=f"[agent] {key}",
                    low=low,
                    high=high,
                )
        return cls(map_option=dict(map_agent))

    def to_api(self, map_agent_hardware: dict[str, Any]) -> dict[str, int]:
        """The agent budget the job request takes: the manifest's values over the defaults.

        Args:
            map_agent_hardware: the ``agent`` of the service's hardware list

        Returns:
            every budget key

        Raises:
            ManifestError: a value exceeds the service's maximum
        """
        map_api = {key: map_agent_hardware["defaults"][key] for key in AgentOptions.MAP_BOUND}
        map_api.update(self.map_option)
        for key, value in map_api.items():
            maximum = map_agent_hardware["maximums"][key]
            if value > maximum:
                raise ManifestError(f"[agent] {key} {value} exceeds the service's maximum {maximum}")
        return map_api


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
    # names left out of the source archive: version control and the fetched results
    SET_EXCLUDED_NAME = {
        ".git",
        "marstrace-results",
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
            MarsTraceError: a cubin is not a cubin
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
    """A job directory's ``marstrace.toml``, checked, and its translation into a MarsTrace
    job request: an agent on the GPU host reads the workspace, instruments and runs the
    kernel within its budget, and returns one ``marstrace.json``.

    Args:
        directory: the job directory
        hardware: the hardware id the job runs on
        cuda_version: the CUDA toolkit version, ``M.m``, the job runs with; None for the
            hardware's default
        intent: what the trace should answer, the job's prompt
        workspace: the files the agent works with
        agent: the session's budget
        limits: the attempts
        list_binary: the workspace's cubins
    """

    FILE_NAME = "marstrace.toml"
    PROMPT_CHARACTERS_MAX = 32000
    MEBIBYTE = 1 << 20
    # the most bytes of every input of a job together
    INPUTS_BYTES_MAX = 256 * MEBIBYTE
    # the manifest's required text keys and what each holds
    MAP_REQUIRED_TEXT = {
        "hardware": "the hardware id from 'marstrace hardware'",
        "intent": "what the trace should answer, in your words",
    }
    # the manifest's required tables
    SET_REQUIRED_TABLE = {
        "workspace",
    }
    SET_KEY = {
        "hardware",
        "cuda_version",
        "intent",
        "workspace",
        "agent",
        "limits",
    }
    CUDA_VERSION_PATTERN = re.compile(r"[1-9][0-9]*\.[0-9]+")
    # upload role of each workspace key
    MAP_ROLE = {
        "cubins": "cubin",
        "source": "source_archive",
        "manifest": "launch_manifest",
        "tensors": "tensor",
        "files": "file",
    }
    # the most bytes of one input of each role
    MAP_ROLE_BYTES_MAX = {
        "cubin": 64 * MEBIBYTE,
        "source_archive": 256 * MEBIBYTE,
        "launch_manifest": 1 * MEBIBYTE,
        "tensor": 128 * MEBIBYTE,
        "file": 16 * MEBIBYTE,
    }

    directory: str
    hardware: str
    intent: str
    cuda_version: str | None = None
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

    @staticmethod
    def check_cuda_version(
        value: Any,
        where: str,
    ) -> None:
        """Check the spelling of a CUDA toolkit version.

        Args:
            value: the version
            where: the value's place, for the message

        Raises:
            ManifestError: the value is not a major and minor version such as ``13.0``
        """
        if (
            not isinstance(
                value,  # obj
                str,    # class_or_tuple
            )
            or JobManifest.CUDA_VERSION_PATTERN.fullmatch(value) is None
        ):
            raise ManifestError(f"{where} must be a CUDA version such as \"13.0\", not {value!r}")

    @classmethod
    def load(cls, directory: str) -> "JobManifest":
        """Read and check the manifest of a job directory.

        Args:
            directory: the job directory, holding ``marstrace.toml``

        Returns:
            the manifest

        Raises:
            ManifestError: the manifest is missing, malformed or incomplete, or names an absent
                file
            MarsTraceError: a cubin is not a cubin
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
        if "\0" in map_manifest["intent"]:
            raise ManifestError(f"{path}: 'intent' may not hold a NUL character")
        for key in sorted(JobManifest.SET_REQUIRED_TABLE):
            if key not in map_manifest:
                raise ManifestError(f"{path}: [{key}] is required: the files the agent works with")
        workspace = WorkspaceSpec.from_table(map_manifest["workspace"])
        cuda_version = map_manifest.get("cuda_version")
        if cuda_version is not None:
            JobManifest.check_cuda_version(
                value=cuda_version,
                where=f"{path}: 'cuda_version'",
            )
        return cls(
            directory=directory,
            hardware=map_manifest["hardware"],
            cuda_version=cuda_version,
            intent=map_manifest["intent"],
            workspace=workspace,
            agent=AgentOptions.from_table(
                map_manifest.get(
                    "agent",
                    {},
                )
            ),
            limits=JobLimits.from_table(
                map_manifest.get(
                    "limits",
                    {},
                )
            ),
            list_binary=workspace.validate(directory),
        )

    def select_hardware(self, map_hardware: dict[str, Any]) -> dict[str, Any]:
        """The hardware entry of the manifest's hardware id, checked to run the job.

        Args:
            map_hardware: the service's hardware list

        Returns:
            the entry, holding ``gpu``, ``architecture``, ``cuda_versions`` and ``rates``

        Raises:
            ManifestError: the hardware is unknown or not offered now, does not offer the
                manifest's CUDA version, or a cubin targets another architecture
        """
        list_entry = map_hardware["hardware"]
        map_entry = next(
            (
                map_candidate
                for map_candidate in list_entry
                if map_candidate["id"] == self.hardware
            ),
            None,
        )
        if map_entry is None:
            raise ManifestError(
                f"no hardware '{self.hardware}'; 'marstrace hardware' lists: "
                f"{', '.join(map_candidate['id'] for map_candidate in list_entry) or 'none'}"
            )
        if not map_entry["available"]:
            raise ManifestError(f"{self.hardware} is not offered now; see 'marstrace hardware'")
        if (
            self.cuda_version is not None
            and self.cuda_version not in map_entry["cuda_versions"]
        ):
            raise ManifestError(
                f"{self.hardware} does not offer CUDA {self.cuda_version}; it offers "
                f"{', '.join(map_entry['cuda_versions'])}, see 'marstrace hardware'"
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
        return map_entry

    def list_input(self) -> list[JobInput]:
        """The files the workspace uploads, checked against the service's size limits.

        Returns:
            the cubins, the source archive, the launch manifest, the tensors and the files

        Raises:
            ManifestError: an input, or all of them together, exceed a size limit
        """
        list_input = [
            JobInput(
                role=JobManifest.MAP_ROLE["cubins"],
                path=os.path.join(
                    self.directory,
                    relative_path,
                ),
            )
            for relative_path in self.workspace.list_cubin
        ]
        if self.workspace.source is not None:
            list_input.append(
                JobInput(
                    role=JobManifest.MAP_ROLE["source"],
                    path=os.path.join(
                        self.directory,
                        WorkspaceSpec.SOURCE_ARCHIVE_NAME,
                    ),
                    content_type="application/gzip",
                    data=self.workspace.archive_source(self.directory),
                )
            )
        if self.workspace.manifest is not None:
            list_input.append(
                JobInput(
                    role=JobManifest.MAP_ROLE["manifest"],
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
            list_input.extend(
                JobInput(
                    role=JobManifest.MAP_ROLE[key],
                    path=os.path.join(
                        self.directory,
                        relative_path,
                    ),
                )
                for relative_path in list_relative_path
            )
        total_bytes = 0
        for job_input in list_input:
            size_bytes = job_input.size_bytes()
            bytes_max = JobManifest.MAP_ROLE_BYTES_MAX[job_input.role]
            if size_bytes > bytes_max:
                raise ManifestError(
                    f"{job_input.path} is {size_bytes} bytes; a {job_input.role} holds at most "
                    f"{bytes_max // JobManifest.MEBIBYTE} MiB"
                )
            total_bytes += size_bytes
        if total_bytes > JobManifest.INPUTS_BYTES_MAX:
            raise ManifestError(
                f"the workspace is {total_bytes} bytes; a job's inputs hold at most "
                f"{JobManifest.INPUTS_BYTES_MAX // JobManifest.MEBIBYTE} MiB together"
            )
        return list_input

    @staticmethod
    def inputs(map_upload_ids: dict[str, list[str]]) -> dict[str, Any]:
        """The ``inputs`` of a job request.

        Args:
            map_upload_ids: upload role to the ids of the verified uploads of that role

        Returns:
            the inputs; the source archive and the launch manifest only when uploaded
        """
        map_inputs: dict[str, Any] = {
            "cubins": list(map_upload_ids[JobManifest.MAP_ROLE["cubins"]]),
            "tensors": list(map_upload_ids.get(JobManifest.MAP_ROLE["tensors"], [])),
            "files": list(map_upload_ids.get(JobManifest.MAP_ROLE["files"], [])),
        }
        for key in (
            "source",
            "manifest",
        ):
            role = JobManifest.MAP_ROLE[key]
            list_id = map_upload_ids.get(role)
            if list_id:
                map_inputs[role] = list_id[0]
        return map_inputs

    def request(
        self,
        map_hardware: dict[str, Any],
        map_upload_ids: dict[str, list[str]],
    ) -> dict[str, Any]:
        """The job request a quote and a submission take.

        Args:
            map_hardware: the service's hardware list
            map_upload_ids: upload role to the ids of the verified uploads of that role

        Returns:
            ``hardware``, ``prompt``, ``inputs``, ``agent``, ``cuda_version`` when the
            manifest sets it, and ``limits`` when the manifest sets ``max_attempts``
        """
        map_request = {
            "hardware": self.select_hardware(map_hardware)["id"],
            "prompt": self.intent,
            "inputs": JobManifest.inputs(map_upload_ids),
            "agent": self.agent.to_api(map_hardware["agent"]),
        }
        if self.cuda_version is not None:
            map_request["cuda_version"] = self.cuda_version
        map_limits = self.limits.to_api()
        if map_limits is not None:
            map_request["limits"] = map_limits
        return map_request
