import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.error import HarnessError


@dataclass
class ListedInstruction:
    """One instruction of an inspected kernel: where it sits, what it is and the source it came from.

    Args:
        index: the instruction index that trace sites name
        pc: the byte offset of the instruction in the kernel
        sass: the instruction's SASS text
        file: the source file it was emitted for, or None without line info
        line: the source line it was emitted for, or None without line info
        block: the index of the basic block that holds it, or None
        is_eligible: a probe can be placed on it
    """

    index: int
    pc: int
    sass: str
    file: str | None = None
    line: int | None = None
    block: int | None = None
    is_eligible: bool = False

    def source_of(self) -> str:
        """The instruction's source position, or an empty string without line info.

        Returns:
            ``<file>:<line>``, or ""
        """
        if (
            self.file is None
            or self.line is None
        ):
            return ""
        return f"{self.file}:{self.line}"

    def is_at_source(self, source: str) -> bool:
        """Whether the instruction was emitted for a source line, given with or without its file.

        Args:
            source: ``<file>:<line>`` or ``<line>``; the file matches by its trailing path

        Returns:
            whether it was
        """
        if self.line is None:
            return False
        if ":" not in source:
            return str(self.line) == source
        file_name, _, line = source.rpartition(":")
        return (
            str(self.line) == line
            and (self.file or "").endswith(file_name)
        )


@dataclass
class ListedParameter:
    """One parameter of an inspected kernel, as the binary lays it out."""

    index: int
    offset_bytes: int
    size_bytes: int


@dataclass
class ListedKernel:
    """The inspected kernel: its resources, parameter layout and SASS listing."""

    symbol: str
    demangled: str | None
    registers: int | None = None
    static_shared_bytes: int | None = None
    list_parameter: list[ListedParameter] = field(default_factory=list)
    list_instruction: list[ListedInstruction] = field(default_factory=list)


@dataclass
class KernelListing:
    """An ``inspection.json``: a cubin's kernels, and the inspected kernel's layout and listing.

    Args:
        architecture: the cubin's target, e.g. ``sm_90a``
        map_kernel_name: every kernel of the cubin, symbol to demangled name
        kernel: the inspected kernel, or None when the inspection named none
    """

    SCHEMA = "marscompute.xtrace.inspection.v1"

    architecture: str
    map_kernel_name: dict[str, str | None]
    kernel: ListedKernel | None

    @classmethod
    def from_json(
        cls,
        map_inspection: Any,
        where: str,
    ) -> "KernelListing":
        """Read an inspection result.

        Args:
            map_inspection: the parsed ``inspection.json``
            where: the file or job it came from, for the message

        Returns:
            the listing

        Raises:
            HarnessError: the value is not an inspection result
        """
        if (
            not isinstance(
                map_inspection,  # obj
                dict,            # class_or_tuple
            )
            or map_inspection.get("schema_version") != KernelListing.SCHEMA
        ):
            raise HarnessError(f"{where} is not a {KernelListing.SCHEMA} result")
        try:
            map_kernel = map_inspection.get("kernel")
            kernel = None
            if map_kernel is not None:
                set_eligible = set(map_kernel.get("eligible_sites", []))
                kernel = ListedKernel(
                    symbol=map_kernel["symbol"],
                    demangled=map_kernel.get("demangled"),
                    registers=map_kernel.get("registers"),
                    static_shared_bytes=map_kernel.get("static_shared_bytes"),
                    list_parameter=[
                        ListedParameter(
                            index=map_parameter["index"],
                            offset_bytes=map_parameter["offset_bytes"],
                            size_bytes=map_parameter["size_bytes"],
                        )
                        for map_parameter in map_kernel.get("parameters", [])
                    ],
                    list_instruction=[
                        ListedInstruction(
                            index=map_instruction["index"],
                            pc=map_instruction["pc"],
                            sass=map_instruction["sass"],
                            file=map_instruction.get("file"),
                            line=map_instruction.get("line"),
                            block=map_instruction.get("block"),
                            is_eligible=map_instruction["index"] in set_eligible,
                        )
                        for map_instruction in map_kernel.get("instructions", [])
                    ],
                )
            return cls(
                architecture=map_inspection["architecture"],
                map_kernel_name={
                    map_entry["symbol"]: map_entry.get("demangled")
                    for map_entry in map_inspection.get("kernels", [])
                },
                kernel=kernel,
            )
        except (
            KeyError,
            TypeError,
        ) as err:
            raise HarnessError(f"{where} is a malformed inspection result: {err}") from err

    @classmethod
    def load(cls, path: str) -> "KernelListing":
        """Read an ``inspection.json`` file.

        Args:
            path: the file

        Returns:
            the listing

        Raises:
            HarnessError: the file is not an inspection result
        """
        with open(
            file=path,
            mode="r",
            encoding="utf-8",
        ) as listing_file:
            try:
                map_inspection = json.load(listing_file)
            except ValueError as err:
                raise HarnessError(f"{path} is not JSON") from err
        return cls.from_json(
            map_inspection=map_inspection,
            where=path,
        )

    def inspected_kernel(self) -> ListedKernel:
        """The kernel the inspection named, whose layout and listing it carries.

        Returns:
            the kernel

        Raises:
            HarnessError: the inspection named no kernel
        """
        if self.kernel is None:
            raise HarnessError(
                "the inspection lists the cubin's kernels only; inspect again with --kernel "
                f"naming one of: {', '.join(self.map_kernel_name) or 'none'}"
            )
        return self.kernel

    def resolve_line(self, source: str) -> int:
        """The first instruction of a source line that a probe can be placed on.

        Args:
            source: ``<file>:<line>``

        Returns:
            the instruction index

        Raises:
            HarnessError: the line emitted no instruction, or none of them takes a probe
        """
        kernel = self.inspected_kernel()
        list_at_line = [
            instruction
            for instruction in kernel.list_instruction
            if instruction.is_at_source(source)
        ]
        if not list_at_line:
            if all(instruction.line is None for instruction in kernel.list_instruction):
                raise HarnessError(
                    f"site '{source}': the cubin carries no line info; rebuild it with "
                    "-lineinfo or name the site by index"
                )
            raise HarnessError(
                f"site '{source}': no instruction of {kernel.symbol} was emitted for that line; "
                "run 'xtrace sass' to list the lines that have instructions"
            )
        for instruction in list_at_line:
            if instruction.is_eligible:
                return instruction.index
        list_index = ", ".join(str(instruction.index) for instruction in list_at_line)
        raise HarnessError(
            f"site '{source}': none of its instructions ({list_index}) takes a probe; name a "
            "nearby eligible instruction by index (marked '+' in 'xtrace sass')"
        )

    def check_index(self, index: int) -> None:
        """Check that an instruction index exists in the kernel and takes a probe.

        Args:
            index: the instruction index

        Raises:
            HarnessError: the index is out of range or not eligible
        """
        kernel = self.inspected_kernel()
        for instruction in kernel.list_instruction:
            if instruction.index != index:
                continue
            if not instruction.is_eligible:
                raise HarnessError(
                    f"site index {index} ({instruction.sass.strip()}) does not take a probe; "
                    "pick an instruction marked '+' in 'xtrace sass'"
                )
            return
        raise HarnessError(
            f"site index {index} lies outside {kernel.symbol}, which has "
            f"{len(kernel.list_instruction)} instructions"
        )

    def render_kernels(self) -> str:
        """The cubin's kernels, one per line, the inspected one marked and followed by its layout.

        Returns:
            the text
        """
        list_line = [f"architecture {self.architecture}"]
        for symbol, demangled in self.map_kernel_name.items():
            is_inspected = (
                self.kernel is not None
                and self.kernel.symbol == symbol
            )
            marker = "*" if is_inspected else " "
            list_line.append(f"{marker} {symbol}" + (f"  ({demangled})" if demangled else ""))
            if (
                not is_inspected
                or self.kernel is None
            ):
                continue
            list_line.append(
                f"    registers {self.kernel.registers}, static shared "
                f"{self.kernel.static_shared_bytes} bytes"
            )
            for parameter in self.kernel.list_parameter:
                list_line.append(
                    f"    param {parameter.index:<3} offset {parameter.offset_bytes:<5} "
                    f"size {parameter.size_bytes}"
                )
        return "\n".join(list_line)

    def render_listing(
        self,
        source: str | None = None,
        opcode_pattern: str | None = None,
        first_index: int | None = None,
        last_index: int | None = None,
    ) -> str:
        """The inspected kernel's instructions, narrowed by source line, opcode or index range.

        Args:
            source: keep instructions of this source line, ``<file>:<line>`` or ``<line>``
            opcode_pattern: keep instructions whose text matches this regex
            first_index: keep instructions from this index on
            last_index: keep instructions up to this index

        Returns:
            the listing, one instruction per line: index, '+' when it takes a probe, block,
            source position and SASS text
        """
        kernel = self.inspected_kernel()
        opcode_regex = re.compile(opcode_pattern) if opcode_pattern else None
        list_line = []
        for instruction in kernel.list_instruction:
            if (
                first_index is not None
                and instruction.index < first_index
            ):
                continue
            if (
                last_index is not None
                and instruction.index > last_index
            ):
                continue
            if (
                source is not None
                and not instruction.is_at_source(source)
            ):
                continue
            if (
                opcode_regex is not None
                and not opcode_regex.search(instruction.sass)
            ):
                continue
            block = f"B{instruction.block}" if instruction.block is not None else ""
            eligible = "+" if instruction.is_eligible else " "
            list_line.append(
                f"{instruction.index:>6} {eligible} {block:<5} {instruction.source_of():<28} "
                f"{instruction.sass}"
            )
        return "\n".join(list_line)


class InspectionCache:
    """Inspection results kept beside the cubin they describe, keyed by its digest and kernel.

    A listing describes the exact bytes it was made from, so a cubin rebuilt with the same
    name gets a different key and is inspected again.
    """

    DIRECTORY_NAME = ".xtrace-cache"
    # the kernel key of an inspection that names no kernel
    KERNELS_ONLY_KEY = "kernels"
    KERNEL_KEY_CHARACTERS = 16

    @staticmethod
    def path_of(
        binary_path: str,
        sha256: str,
        kernel_symbol: str | None,
    ) -> str:
        """Where the inspection of a cubin and kernel is cached.

        Args:
            binary_path: the cubin
            sha256: the cubin's digest
            kernel_symbol: the inspected kernel, or None for the kernel list only

        Returns:
            the cache file's path
        """
        kernel_key = InspectionCache.KERNELS_ONLY_KEY
        if kernel_symbol is not None:
            kernel_key = hashlib.sha256(kernel_symbol.encode("utf-8")).hexdigest()[
                :InspectionCache.KERNEL_KEY_CHARACTERS
            ]
        return os.path.join(
            os.path.dirname(os.path.abspath(binary_path)),
            InspectionCache.DIRECTORY_NAME,
            f"{sha256}.{kernel_key}.inspection.json",
        )

    @staticmethod
    def load(
        binary_path: str,
        sha256: str,
        kernel_symbol: str | None,
    ) -> KernelListing | None:
        """The cached inspection of a cubin and kernel.

        Args:
            binary_path: the cubin
            sha256: the cubin's digest
            kernel_symbol: the inspected kernel, or None for the kernel list only

        Returns:
            the listing, or None when none is cached
        """
        path = InspectionCache.path_of(
            binary_path=binary_path,
            sha256=sha256,
            kernel_symbol=kernel_symbol,
        )
        if not os.path.isfile(path):
            return None
        return KernelListing.load(path)

    @staticmethod
    def store(
        binary_path: str,
        sha256: str,
        kernel_symbol: str | None,
        data: bytes,
    ) -> str:
        """Cache an inspection result.

        Args:
            binary_path: the cubin
            sha256: the cubin's digest
            kernel_symbol: the inspected kernel, or None for the kernel list only
            data: the ``inspection.json`` bytes

        Returns:
            the cache file's path
        """
        path = InspectionCache.path_of(
            binary_path=binary_path,
            sha256=sha256,
            kernel_symbol=kernel_symbol,
        )
        os.makedirs(
            name=os.path.dirname(path),
            exist_ok=True,
        )
        with open(
            file=path,
            mode="wb",
        ) as cache_file:
            cache_file.write(data)
        return path
