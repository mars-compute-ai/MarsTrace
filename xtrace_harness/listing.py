import json
import re
from dataclasses import dataclass, field
from xtrace_harness.error import HarnessError


@dataclass
class ListedInstruction:
    """One instruction of an inspected kernel: where it sits, what it is and the source it came from."""

    index: int
    pc: int
    text: str
    file: str | None = None
    line: int | None = None
    # the index of the basic block that holds the instruction
    block: int | None = None

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


@dataclass
class ListedParameter:
    """One parameter of an inspected kernel, as the binary lays it out."""

    index: int
    offset: int
    size: int


@dataclass
class ListedKernel:
    """A kernel of the inspected binary, with its parameters and SASS listing when it was the one named."""

    name: str
    demangled: str | None
    is_matched: bool
    registers: int | None = None
    static_shared_bytes: int | None = None
    list_parameter: list[ListedParameter] = field(default_factory=list)
    list_instruction: list[ListedInstruction] = field(default_factory=list)


@dataclass
class KernelListing:
    """The result of an ``inspect`` job: the binary's kernels, and the named one's layout and listing."""

    SCHEMA = "xtrace.inspect/1"

    list_kernel: list[ListedKernel]

    @classmethod
    def load(cls, path: str) -> "KernelListing":
        """Read an ``inspect.json`` the platform returned.

        Args:
            path: the file

        Returns:
            the listing

        Raises:
            HarnessError: the file is not an inspect result
        """
        with open(
            file=path,
            mode="r",
            encoding="utf-8",
        ) as listing_file:
            map_listing = json.load(listing_file)
        if map_listing.get("schema") != KernelListing.SCHEMA:
            raise HarnessError(f"{path} is not an {KernelListing.SCHEMA} result")
        list_kernel = []
        for map_kernel in map_listing.get("kernels", []):
            list_kernel.append(
                ListedKernel(
                    name=map_kernel["name"],
                    demangled=map_kernel.get("demangled"),
                    is_matched=bool(map_kernel.get("matched")),
                    registers=map_kernel.get("registers"),
                    static_shared_bytes=map_kernel.get("static_shared_bytes"),
                    list_parameter=[
                        ListedParameter(
                            index=map_parameter["index"],
                            offset=map_parameter["offset"],
                            size=map_parameter["size"],
                        )
                        for map_parameter in map_kernel.get("params", [])
                    ],
                    list_instruction=[
                        ListedInstruction(
                            index=map_instruction["index"],
                            pc=map_instruction["pc"],
                            text=map_instruction["text"],
                            file=map_instruction.get("file"),
                            line=map_instruction.get("line"),
                            block=map_instruction.get("block"),
                        )
                        for map_instruction in map_kernel.get("instructions", [])
                    ],
                )
            )
        return cls(list_kernel=list_kernel)

    def matched_kernel(self) -> ListedKernel:
        """The kernel the job named, whose layout and listing the result carries.

        Returns:
            the kernel

        Raises:
            HarnessError: the result holds no listing
        """
        for kernel in self.list_kernel:
            if kernel.is_matched:
                return kernel
        raise HarnessError(
            "the result holds no kernel listing; run 'xtrace sass --kernels' to list the "
            "binary's kernels"
        )

    def render_kernels(self) -> str:
        """The binary's kernels, one per line, the named one marked and followed by its parameters.

        Returns:
            the text
        """
        list_line = []
        for kernel in self.list_kernel:
            marker = "*" if kernel.is_matched else " "
            list_line.append(f"{marker} {kernel.demangled or kernel.name}")
            if not kernel.is_matched:
                continue
            list_line.append(f"    registers {kernel.registers}, static shared {kernel.static_shared_bytes} bytes")
            for parameter in kernel.list_parameter:
                list_line.append(
                    f"    param {parameter.index:<3} offset {parameter.offset:<5} size {parameter.size}"
                )
        return "\n".join(list_line)

    def render_listing(
        self,
        source: str | None = None,
        opcode_pattern: str | None = None,
        first_index: int | None = None,
        last_index: int | None = None,
    ) -> str:
        """The matched kernel's instructions, narrowed by source line, opcode or index range.

        Args:
            source: keep instructions of this source line, ``<file>:<line>`` or ``<line>``
            opcode_pattern: keep instructions whose text matches this regex
            first_index: keep instructions from this index on
            last_index: keep instructions up to this index

        Returns:
            the listing, one instruction per line: index, block, source position, SASS text
        """
        kernel = self.matched_kernel()
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
                and not KernelListing._is_at_source(
                    instruction=instruction,
                    source=source,
                )
            ):
                continue
            if (
                opcode_regex is not None
                and not opcode_regex.search(instruction.text)
            ):
                continue
            block = f"B{instruction.block}" if instruction.block is not None else ""
            list_line.append(
                f"{instruction.index:>6}  {block:<5} {instruction.source_of():<28} {instruction.text}"
            )
        return "\n".join(list_line)

    @staticmethod
    def _is_at_source(
        instruction: ListedInstruction,
        source: str,
    ) -> bool:
        """Whether an instruction came from a source line, given with or without its file."""
        if instruction.line is None:
            return False
        if ":" not in source:
            return str(instruction.line) == source
        file_name, _, line = source.rpartition(":")
        return (
            str(instruction.line) == line
            and (instruction.file or "").endswith(file_name)
        )
