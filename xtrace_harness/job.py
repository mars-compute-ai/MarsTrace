import gzip
import hashlib
import io
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
class TraceSite:
    """One instruction of the traced kernel that writes a timestamp, and optionally a value.

    A site names its instruction either by its index in the kernel's SASS listing, or by a
    source line, which the platform resolves to the first instruction emitted for it.

    Args:
        index: the instruction index, as ``xtrace sass`` lists it
        line: ``<file>:<line>`` of the kernel source, when the cubin carries line info
        value_register: the register recorded beside the timestamp, ``R<n>`` or ``UR<n>``
        value_width_bits: 32, or 64 to record the register pair starting there
        value_is_smid: record the SM the block runs on instead of a register
        warp_level: one record per warp, written by an elected lane, instead of per thread
    """

    # the register names a site may record
    REGISTER_PATTERN = re.compile(r"^(U?)R(\d+)$")
    # the value widths a site may record
    LIST_VALUE_WIDTH = (
        32,
        64,
    )
    SET_KEY = {
        "index",
        "line",
        "value_register",
        "value_width_bits",
        "value_is_smid",
        "warp_level",
    }

    index: int | None = None
    line: str | None = None
    value_register: str | None = None
    value_width_bits: int = 32
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
            line=map_site.get("line"),
            value_register=map_site.get("value_register"),
            value_width_bits=map_site.get(
                "value_width_bits",
                32,
            ),
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
        """Check that the site names exactly one instruction and a value it can record.

        Args:
            site_id: the site's position in the manifest, for the message

        Raises:
            ManifestError: the site is ambiguous or names an unknown register or width
        """
        if (self.index is None) == (self.line is None):
            raise ManifestError(f"site {site_id}: give exactly one of 'index' and 'line'")
        if (
            self.index is not None
            and self.index < 0
        ):
            raise ManifestError(f"site {site_id}: 'index' must not be negative")
        if (
            self.line is not None
            and not re.match(
                r"^.+:\d+$",  # pattern
                self.line,    # string
            )
        ):
            raise ManifestError(f"site {site_id}: 'line' must read '<file>:<line>'")
        if (
            self.value_register is not None
            and not TraceSite.REGISTER_PATTERN.match(self.value_register)
        ):
            raise ManifestError(f"site {site_id}: 'value_register' must read R<n> or UR<n>")
        if self.value_width_bits not in TraceSite.LIST_VALUE_WIDTH:
            raise ManifestError(f"site {site_id}: 'value_width_bits' must be 32 or 64")
        if (
            self.value_is_smid
            and self.value_register is not None
        ):
            raise ManifestError(f"site {site_id}: record either the SM id or a register, not both")

    def to_json(self) -> dict[str, Any]:
        """The site as the job specification carries it.

        Returns:
            the site's fields, without the ones left unset
        """
        map_site: dict[str, Any] = {
            "value_width_bits": self.value_width_bits,
            "value_is_smid": self.value_is_smid,
            "warp_level": self.warp_level,
        }
        if self.index is not None:
            map_site["index"] = self.index
        if self.line is not None:
            map_site["line"] = self.line
        if self.value_register is not None:
            map_site["value_register"] = self.value_register
        return map_site


@dataclass
class TraceRegion:
    """A region of the kernel bounded by two sites: the one entering it and the one leaving it."""

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
        if (
            not isinstance(
                map_region.get("enter"),  # obj
                int,                      # class_or_tuple
            )
            or not isinstance(
                map_region.get("leave"),  # obj
                int,                      # class_or_tuple
            )
        ):
            raise ManifestError("each [[trace.regions]] entry needs integer 'enter' and 'leave'")
        return cls(
            enter=map_region["enter"],
            leave=map_region["leave"],
            label=str(
                map_region.get(
                    "label",
                    f"{map_region['enter']}->{map_region['leave']}",
                )
            ),
        )

    def to_json(self) -> list[Any]:
        """The region as the trace report keeps it.

        Returns:
            ``[enter, leave, label]``
        """
        return [
            self.enter,
            self.leave,
            self.label,
        ]


@dataclass
class TraceOptions:
    """How a trace job instruments the kernel and what its report pairs into intervals."""

    LIST_CLOCK = (
        "gpu",
        "sm",
        "anchor",
    )
    LIST_SCHEDULE = (
        "first_feasible",
        "least_makespan",
    )
    PER_THREAD_RECORDS_MAX = 4096

    clock: str = "gpu"
    per_thread_records: int | None = None
    schedule: str = "first_feasible"
    list_site: list[TraceSite] = field(default_factory=list)
    list_region: list[TraceRegion] = field(default_factory=list)
    # region label -> the warp role that executes it, for the report's lane grouping
    map_scope_role: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_table(cls, map_trace: dict[str, Any]) -> "TraceOptions":
        """Read the ``[trace]`` table.

        Args:
            map_trace: the table

        Returns:
            the options
        """
        return cls(
            clock=map_trace.get(
                "clock",
                "gpu",
            ),
            per_thread_records=map_trace.get("per_thread_records"),
            schedule=map_trace.get(
                "schedule",
                "first_feasible",
            ),
            list_site=[TraceSite.from_table(map_site) for map_site in map_trace.get("sites", [])],
            list_region=[
                TraceRegion.from_table(map_region) for map_region in map_trace.get("regions", [])
            ],
            map_scope_role=dict(map_trace.get("scope_roles", {})),
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
        if (
            self.per_thread_records is not None
            and not 1 <= self.per_thread_records <= TraceOptions.PER_THREAD_RECORDS_MAX
        ):
            raise ManifestError(
                f"trace.per_thread_records must lie in [1, {TraceOptions.PER_THREAD_RECORDS_MAX}]"
            )
        if not self.list_site:
            raise ManifestError("a trace job needs at least one [[trace.sites]] entry")
        for site_id, site in enumerate(self.list_site):
            site.validate(site_id)
        if (
            self.clock == "anchor"
            and not all(site.warp_level for site in self.list_site)
        ):
            raise ManifestError("trace.clock = 'anchor' needs every site to be warp_level")
        for region in self.list_region:
            for site_id in (
                region.enter,
                region.leave,
            ):
                if not 0 <= site_id < len(self.list_site):
                    raise ManifestError(
                        f"region '{region.label}' names site {site_id}, "
                        f"but the manifest has {len(self.list_site)} site(s)"
                    )
            if region.enter == region.leave:
                raise ManifestError(f"region '{region.label}' enters and leaves at the same site")

    def to_json(self) -> dict[str, Any]:
        """The options as the job specification carries them.

        Returns:
            the clock, record depth, schedule, sites, regions and scope roles
        """
        return {
            "clock": self.clock,
            "per_thread_records": self.per_thread_records,
            "schedule": self.schedule,
            "sites": [site.to_json() for site in self.list_site],
            "regions": [region.to_json() for region in self.list_region],
            "scope_roles": dict(self.map_scope_role),
        }


@dataclass
class KernelSpec:
    """The kernel a job traces: the binary that holds it, its entry point and its sources.

    Args:
        binary: the cubin or PTX file, relative to the job directory
        name: the kernel's entry point, as the binary names it (mangled for C++)
        list_source: source files shown beside the SASS listing, relative to the job directory
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

    def to_json(self) -> dict[str, Any]:
        """The kernel as the job specification carries it.

        Returns:
            the binary, the entry point and the sources
        """
        return {
            "binary": self.binary,
            "name": self.name,
            "sources": list(self.list_source),
        }


@dataclass
class BufferSpec:
    """A device allocation the platform makes for the launch, and what it holds before it.

    Args:
        name: the buffer's name, which launch arguments refer to
        dtype: the element type, e.g. ``f16`` or ``i32``
        list_shape: the extents; the buffer holds their product of elements
        init: ``zeros``, ``ones``, ``random``, or ``file`` for the tensor of the same name in
            the job's tensors file
        is_output: the platform returns the buffer after the launch and checks that the traced
            launch wrote the same bytes as the untraced one
    """

    # element type -> bytes per element
    MAP_DTYPE_BYTES = {
        "bool": 1,
        "u8": 1,
        "i8": 1,
        "f8e4m3": 1,
        "f8e5m2": 1,
        "u16": 2,
        "i16": 2,
        "f16": 2,
        "bf16": 2,
        "u32": 4,
        "i32": 4,
        "f32": 4,
        "u64": 8,
        "i64": 8,
        "f64": 8,
    }
    LIST_INIT = (
        "zeros",
        "ones",
        "random",
        "file",
    )

    name: str
    dtype: str
    list_shape: list[int]
    init: str = "zeros"
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
        return cls(
            name=map_buffer["name"],
            dtype=map_buffer["dtype"],
            list_shape=list(map_buffer["shape"]),
            init=map_buffer.get(
                "init",
                "zeros",
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
        return BufferSpec.MAP_DTYPE_BYTES[self.dtype] * TensorFile.element_count(self.list_shape)

    def validate(self, tensor_file: TensorFile | None) -> None:
        """Check the buffer's type and shape, and its tensor when it starts from the tensors file.

        Args:
            tensor_file: the job's tensors file, or None when it has none

        Raises:
            ManifestError: the buffer is malformed or does not match its tensor
        """
        if not re.match(
            r"^[A-Za-z_]\w*$",  # pattern
            self.name,          # string
        ):
            raise ManifestError(f"buffer '{self.name}': the name must be an identifier")
        if self.dtype not in BufferSpec.MAP_DTYPE_BYTES:
            raise ManifestError(
                f"buffer '{self.name}': dtype must be one of {', '.join(BufferSpec.MAP_DTYPE_BYTES)}"
            )
        if (
            not self.list_shape
            or not all(
                isinstance(
                    extent,  # obj
                    int,     # class_or_tuple
                )
                and extent > 0
                for extent in self.list_shape
            )
        ):
            raise ManifestError(f"buffer '{self.name}': shape must be a list of positive integers")
        if self.init not in BufferSpec.LIST_INIT:
            raise ManifestError(
                f"buffer '{self.name}': init must be one of {', '.join(BufferSpec.LIST_INIT)}"
            )
        if self.init != "file":
            return
        if tensor_file is None:
            raise ManifestError(f"buffer '{self.name}' starts from a file, but the job names no 'tensors'")
        tensor = tensor_file.map_tensor.get(self.name)
        if tensor is None:
            raise ManifestError(f"buffer '{self.name}': {tensor_file.path} has no tensor of that name")
        if (
            tensor.dtype != self.dtype
            or tensor.list_shape != self.list_shape
        ):
            raise ManifestError(
                f"buffer '{self.name}' is {self.dtype}{self.list_shape}, but its tensor is "
                f"{tensor.dtype}{tensor.list_shape}"
            )

    def to_json(self) -> dict[str, Any]:
        """The buffer as the job specification carries it.

        Returns:
            the name, element type, shape, initial contents and output flag
        """
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.list_shape),
            "init": self.init,
            "output": self.is_output,
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

    def to_json(self) -> dict[str, Any]:
        """The relocation as the job specification carries it.

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
    MAP_SCALAR_RANGE = {
        "i8": (-(2**7), 2**7 - 1),
        "u8": (0, 2**8 - 1),
        "i16": (-(2**15), 2**15 - 1),
        "u16": (0, 2**16 - 1),
        "i32": (-(2**31), 2**31 - 1),
        "u32": (0, 2**32 - 1),
        "i64": (-(2**63), 2**63 - 1),
        "u64": (0, 2**64 - 1),
    }
    SET_SCALAR_FLOAT = {
        "f16",
        "bf16",
        "f32",
        "f64",
    }
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
    # the choices of each enumerated tensor map field, as cuTensorMapEncodeTiled names them
    MAP_TENSOR_MAP_CHOICE = {
        "data_type": (
            "u8",
            "u16",
            "u32",
            "i32",
            "u64",
            "i64",
            "f16",
            "f32",
            "f64",
            "bf16",
            "f32_ftz",
            "tf32",
            "tf32_ftz",
        ),
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
    # the largest by-value argument a kernel takes
    BYTES_MAX = 32 * 1024

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
            ManifestError: the entry is not a table or has an unknown kind
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
            if not 0 <= self.offset_bytes < map_buffer[self.buffer].nbytes():
                raise ManifestError(f"{where}: offset_bytes lies outside buffer '{self.buffer}'")
        if self.kind == "scalar":
            self._validate_scalar(where)
        if self.kind == "tensor_map":
            self._validate_tensor_map(where)
        if self.kind == "bytes":
            self._validate_bytes(
                where=where,
                map_buffer=map_buffer,
            )

    def _validate_scalar(self, where: str) -> None:
        """Check a scalar's type and that its value fits it."""
        if self.dtype in LaunchArgument.MAP_SCALAR_RANGE:
            value_min, value_max = LaunchArgument.MAP_SCALAR_RANGE[self.dtype]
            if (
                not isinstance(
                    self.value,  # obj
                    int,         # class_or_tuple
                )
                or isinstance(
                    self.value,  # obj
                    bool,        # class_or_tuple
                )
                or not value_min <= self.value <= value_max
            ):
                raise ManifestError(f"{where}: value must be an integer that fits {self.dtype}")
            return
        if self.dtype in LaunchArgument.SET_SCALAR_FLOAT:
            if not isinstance(
                self.value,  # obj
                (int, float),  # class_or_tuple
            ):
                raise ManifestError(f"{where}: value must be a number")
            return
        raise ManifestError(f"{where}: a scalar needs dtype, one of the integer or float types")

    def _validate_tensor_map(self, where: str) -> None:
        """Check that a TMA descriptor's extents agree in rank."""
        for key in (
            "data_type",
            "global_dims",
            "global_strides_bytes",
            "box_dims",
        ):
            if key not in self.map_tensor_map:
                raise ManifestError(f"{where}: a tensor_map needs '{key}'")
        for key, list_choice in LaunchArgument.MAP_TENSOR_MAP_CHOICE.items():
            if (
                key in self.map_tensor_map
                and self.map_tensor_map[key] not in list_choice
            ):
                raise ManifestError(f"{where}: {key} must be one of {', '.join(list_choice)}")
        rank = len(self.map_tensor_map["global_dims"])
        if not 1 <= rank <= LaunchArgument.TENSOR_MAP_RANK_MAX:
            raise ManifestError(f"{where}: global_dims must have 1 to {LaunchArgument.TENSOR_MAP_RANK_MAX} entries")
        if len(self.map_tensor_map["global_strides_bytes"]) != rank - 1:
            raise ManifestError(f"{where}: global_strides_bytes needs one entry per dimension after the first")
        if len(self.map_tensor_map["box_dims"]) != rank:
            raise ManifestError(f"{where}: box_dims needs one entry per dimension")
        if len(self.map_tensor_map.get("element_strides", [1] * rank)) != rank:
            raise ManifestError(f"{where}: element_strides needs one entry per dimension")

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
        for relocation in self.list_relocation:
            if (
                relocation.offset < 0
                or relocation.offset % PointerRelocation.POINTER_BYTES != 0
                or relocation.offset + PointerRelocation.POINTER_BYTES > len(data)
            ):
                raise ManifestError(f"{where}: relocation offset {relocation.offset} is unaligned or outside the bytes")
            if relocation.buffer not in map_buffer:
                raise ManifestError(f"{where}: a relocation names buffer '{relocation.buffer}', which [[buffers]] lacks")

    def to_json(self) -> dict[str, Any]:
        """The argument as the job specification carries it.

        Returns:
            the kind and the fields that kind uses
        """
        map_argument: dict[str, Any] = {"kind": self.kind}
        if self.kind in (
            "buffer",
            "tensor_map",
        ):
            map_argument["buffer"] = self.buffer
            map_argument["offset_bytes"] = self.offset_bytes
        if self.kind == "scalar":
            map_argument["dtype"] = self.dtype
            map_argument["value"] = self.value
        if self.kind == "tensor_map":
            map_argument.update(self.map_tensor_map)
        if self.kind == "bytes":
            map_argument["hex"] = self.data_hex
            map_argument["relocations"] = [relocation.to_json() for relocation in self.list_relocation]
        return map_argument


@dataclass
class LaunchSpec:
    """How the platform launches the kernel: its shape and its arguments."""

    THREADS_PER_BLOCK_MAX = 1024
    DIMENSION_COUNT = 3

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
            list_grid=LaunchSpec._dimensions(map_launch["grid"]),
            list_block=LaunchSpec._dimensions(map_launch["block"]),
            list_cluster=(
                LaunchSpec._dimensions(map_launch["cluster"])
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
    def _dimensions(value: Any) -> list[int]:
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
                and extent > 0
                for extent in value
            )
        ):
            raise ManifestError("grid, block and cluster are 1 to 3 positive integers")
        return list(value) + [1] * (LaunchSpec.DIMENSION_COUNT - len(value))

    def validate(self, map_buffer: dict[str, BufferSpec]) -> None:
        """Check the launch shape and every argument.

        Args:
            map_buffer: the job's buffers by name

        Raises:
            ManifestError: the shape is out of range or an argument is malformed
        """
        threads_per_block = TensorFile.element_count(self.list_block)
        if threads_per_block > LaunchSpec.THREADS_PER_BLOCK_MAX:
            raise ManifestError(f"a block has at most {LaunchSpec.THREADS_PER_BLOCK_MAX} threads, not {threads_per_block}")
        if self.shared_bytes < 0:
            raise ManifestError("shared_bytes must not be negative")
        if self.list_cluster is not None:
            for grid_extent, cluster_extent in zip(
                self.list_grid,
                self.list_cluster,
            ):
                if grid_extent % cluster_extent != 0:
                    raise ManifestError("each grid extent must be a multiple of the cluster's")
        for argument_id, argument in enumerate(self.list_argument):
            argument.validate(
                argument_id=argument_id,
                map_buffer=map_buffer,
            )

    def to_json(self) -> dict[str, Any]:
        """The launch as the job specification carries it.

        Returns:
            the grid, block, cluster, dynamic shared memory and arguments
        """
        return {
            "grid": list(self.list_grid),
            "block": list(self.list_block),
            "cluster": list(self.list_cluster) if self.list_cluster is not None else None,
            "shared_bytes": self.shared_bytes,
            "args": [argument.to_json() for argument in self.list_argument],
        }


@dataclass
class JobSpec:
    """What one Xtrace job asks of the platform: the kernel, how to launch it and what to trace.

    An ``inspect`` job reads the kernel's binary and returns its SASS listing and parameter
    layout, from which trace sites are picked; a ``trace`` job launches the kernel, first
    untraced and then with those sites instrumented, and returns the trace report.
    """

    SCHEMA = "xtrace.job/1"
    LIST_KIND = (
        "inspect",
        "trace",
    )
    TIMEOUT_SECONDS_DEFAULT = 300
    INTENT_CHARACTERS_MAX = 2000

    kind: str
    kernel: KernelSpec
    tensors: str | None = None
    list_buffer: list[BufferSpec] = field(default_factory=list)
    launch: LaunchSpec | None = None
    trace: TraceOptions | None = None
    timeout_seconds: int = TIMEOUT_SECONDS_DEFAULT
    intent: str | None = None

    def validate(self, tensor_file: TensorFile | None) -> None:
        """Check the specification before it is uploaded.

        Args:
            tensor_file: the job's tensors file, or None when it has none

        Raises:
            ManifestError: a field is missing, out of range or inconsistent
        """
        if self.kind not in JobSpec.LIST_KIND:
            raise ManifestError(f"kind must be one of {', '.join(JobSpec.LIST_KIND)}")
        if self.timeout_seconds <= 0:
            raise ManifestError("'timeout_seconds' must be positive")
        if (
            self.intent is not None
            and len(self.intent) > JobSpec.INTENT_CHARACTERS_MAX
        ):
            raise ManifestError(f"'intent' holds at most {JobSpec.INTENT_CHARACTERS_MAX} characters")
        map_buffer: dict[str, BufferSpec] = {}
        for buffer in self.list_buffer:
            buffer.validate(tensor_file)
            if buffer.name in map_buffer:
                raise ManifestError(f"buffer '{buffer.name}' is defined twice")
            map_buffer[buffer.name] = buffer
        if self.launch is not None:
            self.launch.validate(map_buffer)
        if self.kind == "inspect":
            return
        if self.launch is None:
            raise ManifestError("a trace job needs a [launch] table")
        if self.trace is None:
            raise ManifestError("a trace job needs a [trace] table")
        self.trace.validate()

    def to_json(self) -> dict[str, Any]:
        """The specification as the platform's job API takes it.

        Returns:
            the specification, with its schema tag
        """
        map_spec: dict[str, Any] = {
            "schema": JobSpec.SCHEMA,
            "kind": self.kind,
            "kernel": self.kernel.to_json(),
            "tensors": self.tensors,
            "buffers": [buffer.to_json() for buffer in self.list_buffer],
            "launch": self.launch.to_json() if self.launch is not None else None,
            "timeout_seconds": self.timeout_seconds,
            "intent": self.intent,
        }
        if (
            self.kind == "trace"
            and self.trace is not None
        ):
            map_spec["trace"] = self.trace.to_json()
        return map_spec


@dataclass
class JobManifest:
    """A job directory's ``xtrace.toml``: the job specification, the hardware and the files it names."""

    FILE_NAME = "xtrace.toml"
    SET_KEY = {
        "kind",
        "hardware",
        "timeout_seconds",
        "intent",
        "tensors",
        "kernel",
        "buffers",
        "launch",
        "trace",
    }

    directory: str
    hardware: str | None
    spec: JobSpec
    binary: KernelBinary

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

    @classmethod
    def load(
        cls,
        directory: str,
        kind: str | None = None,
    ) -> "JobManifest":
        """Read and check the manifest of a job directory.

        Args:
            directory: the job directory, holding ``xtrace.toml``
            kind: ``inspect`` or ``trace`` to override the manifest's kind, or None

        Returns:
            the manifest

        Raises:
            ManifestError: the manifest is missing or malformed, names an absent file, or
                names a kernel its binary lacks
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
        if "kernel" not in map_manifest:
            raise ManifestError(f"{path}: a [kernel] table is required")
        spec = JobSpec(
            kind=kind or map_manifest.get(
                "kind",
                "trace",
            ),
            kernel=KernelSpec.from_table(map_manifest["kernel"]),
            tensors=map_manifest.get("tensors"),
            list_buffer=[BufferSpec.from_table(map_buffer) for map_buffer in map_manifest.get("buffers", [])],
            launch=LaunchSpec.from_table(map_manifest["launch"]) if "launch" in map_manifest else None,
            trace=TraceOptions.from_table(map_manifest["trace"]) if "trace" in map_manifest else None,
            timeout_seconds=map_manifest.get(
                "timeout_seconds",
                JobSpec.TIMEOUT_SECONDS_DEFAULT,
            ),
            intent=map_manifest.get("intent"),
        )
        for relative_path in [
            spec.kernel.binary,
            spec.tensors,
            *spec.kernel.list_source,
        ]:
            if relative_path is None:
                continue
            if (
                os.path.isabs(relative_path)
                or os.path.normpath(relative_path).startswith("..")
            ):
                raise ManifestError(f"{path}: {relative_path} must lie inside the job directory")
            if not os.path.isfile(
                os.path.join(
                    directory,
                    relative_path,
                )
            ):
                raise ManifestError(f"{path}: {relative_path} does not exist in the job directory")
        tensor_file = None
        if spec.tensors is not None:
            tensor_file = TensorFile.load(
                os.path.join(
                    directory,
                    spec.tensors,
                )
            )
        spec.validate(tensor_file)
        binary = KernelBinary.load(
            os.path.join(
                directory,
                spec.kernel.binary,
            )
        )
        if spec.kernel.name not in binary.list_kernel:
            raise ManifestError(
                f"{spec.kernel.binary} defines no kernel '{spec.kernel.name}'; it defines: "
                f"{', '.join(binary.list_kernel) or 'none'}"
            )
        return cls(
            directory=directory,
            hardware=map_manifest.get("hardware"),
            spec=spec,
            binary=binary,
        )

    def list_files(self) -> list[str]:
        """The files the bundle carries: the manifest and every file it names, sorted.

        Returns:
            the paths, relative to the job directory
        """
        set_file = {
            JobManifest.FILE_NAME,
            self.spec.kernel.binary,
            *self.spec.kernel.list_source,
        }
        if self.spec.tensors is not None:
            set_file.add(self.spec.tensors)
        return sorted(os.path.normpath(relative_path) for relative_path in set_file)


@dataclass
class JobBundle:
    """The gzip'd tar of the files a manifest names, as it is uploaded."""

    # the largest bundle the platform accepts
    BUNDLE_BYTES_MAX = 512 * 1024 * 1024

    data: bytes
    sha256: str
    list_file: list[str]

    @classmethod
    def pack(cls, manifest: JobManifest) -> "JobBundle":
        """Pack the files a manifest names, with stable order and timestamps.

        Args:
            manifest: the job directory's manifest

        Returns:
            the bundle

        Raises:
            ManifestError: the bundle exceeds the platform's limit
        """
        list_file = manifest.list_files()
        buffer = io.BytesIO()
        with gzip.GzipFile(
            fileobj=buffer,
            mode="wb",
            mtime=0,
        ) as compressed, tarfile.open(
            fileobj=compressed,
            mode="w",
            dereference=True,
        ) as archive:
            for relative_path in list_file:
                path = os.path.join(
                    manifest.directory,
                    relative_path,
                )
                info = archive.gettarinfo(
                    name=path,
                    arcname=relative_path,
                )
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with open(
                    file=path,
                    mode="rb",
                ) as member_file:
                    archive.addfile(
                        tarinfo=info,
                        fileobj=member_file,
                    )
        data = buffer.getvalue()
        if len(data) > JobBundle.BUNDLE_BYTES_MAX:
            raise ManifestError(
                f"the bundle is {len(data)} bytes; the platform accepts at most "
                f"{JobBundle.BUNDLE_BYTES_MAX}. Start large buffers with init = 'random' "
                "instead of a tensors file"
            )
        return cls(
            data=data,
            sha256=hashlib.sha256(data).hexdigest(),
            list_file=list_file,
        )
