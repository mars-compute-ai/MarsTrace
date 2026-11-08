import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.error import HarnessError


@dataclass
class ListedInstruction:
    """One instruction of an inspected kernel's static listing.

    Args:
        index: the instruction's slot in the kernel, which trace sites name
        address: the byte offset of the instruction in the kernel's section
        opcode: the opcode with its modifiers, e.g. ``LDG.E``
        operands: the operands as the disassembler prints them
        predicate: the guarding predicate, e.g. ``@P0``, or None when unguarded
    """

    index: int
    address: int
    opcode: str
    operands: str
    predicate: str | None = None

    def text(self) -> str:
        """The instruction as one line of SASS.

        Returns:
            the predicate, opcode and operands
        """
        list_part = [self.opcode]
        if self.predicate:
            list_part.insert(0, self.predicate)
        if self.operands:
            list_part.append(self.operands)
        return " ".join(list_part)


@dataclass
class ListedParameter:
    """One parameter of an inspected kernel, as the binary lays it out."""

    index: int
    offset_bytes: int
    size_bytes: int


@dataclass
class ListedKernel:
    """A kernel of an inspection result: its parameter layout and, once downloaded, its listing.

    Args:
        symbol: the kernel's entry point
        architecture: the target the inspection read, e.g. ``sm_90a``
        registers: registers per thread, or None when the inspection did not establish it
        static_shared_bytes: static shared memory, or None when not established
        list_parameter: the parameter layout, or None when the inspection could not extract it
        layout_reason: why the layout is absent, or None when it is extracted
        map_instructions_artifact: the result's descriptor of the instruction listing artifact
        instruction_count: how many instructions the listing holds
        is_listing_complete: the listing holds every instruction of the kernel
        list_instruction: the instructions, empty until the listing is downloaded
    """

    symbol: str
    architecture: str
    registers: int | None
    static_shared_bytes: int | None
    list_parameter: list[ListedParameter] | None
    layout_reason: str | None
    map_instructions_artifact: dict[str, Any]
    instruction_count: int
    is_listing_complete: bool
    list_instruction: list[ListedInstruction] = field(default_factory=list)


@dataclass
class KernelListing:
    """An inspection result, ``marscompute.xtrace.inspection-result.v1``, with the static
    instruction listing of its inspected kernel when that was downloaded.

    Args:
        inspection_id: the inspection that produced the result
        binary_sha256: the digest of the cubin it describes
        map_kernel: every kernel of the result by symbol
        kernel: the kernel whose listing is loaded, or None for the kernel catalog only
    """

    RESULT_SCHEMA = "marscompute.xtrace.inspection-result.v1"
    INSTRUCTIONS_SCHEMA = "marscompute.xtrace.static-instructions.v1"

    inspection_id: str
    binary_sha256: str
    map_kernel: dict[str, ListedKernel]
    kernel: ListedKernel | None

    @staticmethod
    def _kernel_of(map_kernel: dict[str, Any]) -> ListedKernel:
        """One kernel of the result's catalog."""
        map_layout = map_kernel["parameter_layout"]
        list_parameter = None
        layout_reason = map_layout.get("reason")
        if map_layout["state"] == "extracted":
            list_parameter = [
                ListedParameter(
                    index=map_parameter["index"],
                    offset_bytes=map_parameter["offset"],
                    size_bytes=map_parameter["size"],
                )
                for map_parameter in map_layout["parameters"]
            ]
        map_coverage = map_kernel["instructions"]["coverage"]
        return ListedKernel(
            symbol=map_kernel["symbol"],
            architecture=map_kernel["architecture"],
            registers=map_kernel["resources"]["registers"],
            static_shared_bytes=map_kernel["resources"]["static_shared_memory_bytes"],
            list_parameter=list_parameter,
            layout_reason=layout_reason,
            map_instructions_artifact=dict(map_kernel["instructions"]["artifact"]),
            instruction_count=map_coverage["returned_count"],
            is_listing_complete=map_coverage["complete"],
        )

    @classmethod
    def from_documents(
        cls,
        map_result: Any,
        map_instructions: Any,
        where: str,
    ) -> "KernelListing":
        """Read an inspection result and, when given, the instruction listing of one of its kernels.

        Args:
            map_result: the parsed inspection result
            map_instructions: the parsed static instruction listing, or None
            where: the file or inspection it came from, for the message

        Returns:
            the listing

        Raises:
            HarnessError: a document is not what it claims, or the listing names another kernel
        """
        if (
            not isinstance(
                map_result,  # obj
                dict,        # class_or_tuple
            )
            or map_result.get("schema_version") != KernelListing.RESULT_SCHEMA
        ):
            raise HarnessError(f"{where} is not a {KernelListing.RESULT_SCHEMA} document")
        try:
            map_kernel = {
                map_entry["symbol"]: KernelListing._kernel_of(map_entry)
                for map_entry in map_result["kernel_catalog"]["kernels"]
            }
            listing = cls(
                inspection_id=map_result["inspection_id"],
                binary_sha256=map_result["binary"]["sha256"],
                map_kernel=map_kernel,
                kernel=None,
            )
            if map_instructions is None:
                return listing
            if (
                not isinstance(
                    map_instructions,  # obj
                    dict,              # class_or_tuple
                )
                or map_instructions.get("schema_version") != KernelListing.INSTRUCTIONS_SCHEMA
            ):
                raise HarnessError(f"{where}: the listing is not a {KernelListing.INSTRUCTIONS_SCHEMA} document")
            kernel = map_kernel.get(map_instructions["symbol"])
            if kernel is None:
                raise HarnessError(f"{where}: the listing names {map_instructions['symbol']}, which the result lacks")
            kernel.list_instruction = [
                ListedInstruction(
                    index=map_instruction["function_index"],
                    address=map_instruction["address"],
                    opcode=map_instruction["opcode"],
                    operands=map_instruction["operands"],
                    predicate=map_instruction["predicate"],
                )
                for map_instruction in map_instructions["instructions"]
            ]
            listing.kernel = kernel
            return listing
        except (
            KeyError,
            TypeError,
        ) as err:
            raise HarnessError(f"{where} is a malformed inspection document: {err}") from err

    @staticmethod
    def _read_json(path: str) -> Any:
        """A JSON file's value."""
        with open(
            file=path,
            mode="r",
            encoding="utf-8",
        ) as json_file:
            try:
                return json.load(json_file)
            except ValueError as err:
                raise HarnessError(f"{path} is not JSON") from err

    @classmethod
    def load(cls, path: str) -> "KernelListing":
        """Read a cached inspection result and the instruction listing cached beside it.

        Args:
            path: the result file, ``<name>.inspection-result.json``; its listing, when one was
                downloaded, is ``<name>.instructions.json``

        Returns:
            the listing

        Raises:
            HarnessError: a file is not an inspection document
        """
        instructions_path = InspectionCache.instructions_path_of(path)
        return cls.from_documents(
            map_result=KernelListing._read_json(path),
            map_instructions=(
                KernelListing._read_json(instructions_path)
                if os.path.isfile(instructions_path)
                else None
            ),
            where=path,
        )

    def inspected_kernel(self) -> ListedKernel:
        """The kernel whose listing is loaded.

        Returns:
            the kernel

        Raises:
            HarnessError: no listing is loaded
        """
        if self.kernel is None:
            raise HarnessError(
                "the inspection holds the cubin's kernel catalog only; inspect again with --kernel "
                f"naming one of: {', '.join(self.map_kernel) or 'none'}"
            )
        return self.kernel

    def check_index(self, index: int) -> None:
        """Check that an instruction index exists in the inspected kernel's listing.

        Args:
            index: the instruction index

        Raises:
            HarnessError: the index is not in the listing
        """
        kernel = self.inspected_kernel()
        if any(instruction.index == index for instruction in kernel.list_instruction):
            return
        raise HarnessError(
            f"site index {index} is not an instruction of {kernel.symbol}, whose listing holds "
            f"{len(kernel.list_instruction)} instructions"
        )

    def render_kernels(self) -> str:
        """The result's kernels, one per line, the inspected one marked and followed by its layout.

        Returns:
            the text
        """
        list_line = [f"inspection {self.inspection_id}"]
        for symbol, kernel in self.map_kernel.items():
            marker = "*" if self.kernel is kernel else " "
            list_line.append(f"{marker} {symbol}  ({kernel.architecture}, {kernel.instruction_count} instructions)")
            if kernel.registers is not None:
                list_line.append(f"    registers {kernel.registers}")
            if kernel.static_shared_bytes is not None:
                list_line.append(f"    static shared {kernel.static_shared_bytes} bytes")
            if kernel.list_parameter is None:
                list_line.append(f"    parameter layout unknown: {kernel.layout_reason}")
                continue
            for parameter in kernel.list_parameter:
                list_line.append(
                    f"    param {parameter.index:<3} offset {parameter.offset_bytes:<5} "
                    f"size {parameter.size_bytes}"
                )
        return "\n".join(list_line)

    def render_listing(
        self,
        opcode_pattern: str | None = None,
        first_index: int | None = None,
        last_index: int | None = None,
    ) -> str:
        """The inspected kernel's instructions, narrowed by opcode or index range.

        Args:
            opcode_pattern: keep instructions whose text matches this regex
            first_index: keep instructions from this index on
            last_index: keep instructions up to this index

        Returns:
            the listing, one instruction per line: index, address and SASS text
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
                opcode_regex is not None
                and not opcode_regex.search(instruction.text())
            ):
                continue
            list_line.append(f"{instruction.index:>6}  {instruction.address:#08x}  {instruction.text()}")
        return "\n".join(list_line)


class InspectionCache:
    """Inspection results kept beside the cubin they describe, keyed by its digest and kernel.

    A result describes the exact bytes it was made from, so a cubin rebuilt with the same
    name gets a different key and is inspected again.
    """

    DIRECTORY_NAME = ".xtrace-cache"
    # the kernel key of an inspection of the whole kernel catalog
    KERNELS_ONLY_KEY = "kernels"
    KERNEL_KEY_CHARACTERS = 16
    RESULT_SUFFIX = ".inspection-result.json"
    INSTRUCTIONS_SUFFIX = ".instructions.json"

    @staticmethod
    def path_of(
        binary_path: str,
        sha256: str,
        kernel_symbol: str | None,
    ) -> str:
        """Where the inspection result of a cubin and kernel is cached.

        Args:
            binary_path: the cubin
            sha256: the cubin's digest
            kernel_symbol: the inspected kernel, or None for the kernel catalog only

        Returns:
            the result file's path
        """
        kernel_key = InspectionCache.KERNELS_ONLY_KEY
        if kernel_symbol is not None:
            kernel_key = hashlib.sha256(kernel_symbol.encode("utf-8")).hexdigest()[
                :InspectionCache.KERNEL_KEY_CHARACTERS
            ]
        return os.path.join(
            os.path.dirname(os.path.abspath(binary_path)),
            InspectionCache.DIRECTORY_NAME,
            f"{sha256}.{kernel_key}{InspectionCache.RESULT_SUFFIX}",
        )

    @staticmethod
    def instructions_path_of(result_path: str) -> str:
        """Where the instruction listing that belongs to a cached result lies.

        Args:
            result_path: the result file

        Returns:
            the listing file's path beside it
        """
        stem = result_path
        if stem.endswith(InspectionCache.RESULT_SUFFIX):
            stem = stem[:-len(InspectionCache.RESULT_SUFFIX)]
        return stem + InspectionCache.INSTRUCTIONS_SUFFIX

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
            kernel_symbol: the inspected kernel, or None for the kernel catalog only

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
        result_data: bytes,
        instructions_data: bytes | None,
    ) -> str:
        """Cache an inspection result and the instruction listing of its kernel.

        Args:
            binary_path: the cubin
            sha256: the cubin's digest
            kernel_symbol: the inspected kernel, or None for the kernel catalog only
            result_data: the result's bytes
            instructions_data: the listing's bytes, or None for the kernel catalog only

        Returns:
            the result file's path
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
        if instructions_data is not None:
            with open(
                file=InspectionCache.instructions_path_of(path),
                mode="wb",
            ) as instructions_file:
                instructions_file.write(instructions_data)
        with open(
            file=path,
            mode="wb",
        ) as result_file:
            result_file.write(result_data)
        return path
