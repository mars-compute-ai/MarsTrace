import getpass
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass, field
from typing import Any
from xtrace_harness.error import HarnessError


@dataclass
class KernelBinary:
    """A CUDA cubin a job loads.

    Args:
        path: the file
        sm: the SM version the cubin targets, e.g. 90
        sha256: the digest of the file's bytes, which identifies the cubin to the platform
        list_kernel: the kernel entry points the cubin defines
    """

    ELF_MAGIC = b"\x7fELF"
    # ELF machine of CUDA binaries
    MACHINE_CUDA = 190
    # the cubin ABI version from which the SM sits in the second byte of e_flags
    ABI_VERSION_SM_BYTE = 8
    BYTE_MASK = 0xFF
    BITS_PER_BYTE = 8
    SECTION_SYMBOL_TABLE = 2
    SYMBOL_TYPE_FUNCTION = 2
    SYMBOL_TYPE_MASK = 0x0F
    # st_other flag of a kernel entry point
    SYMBOL_CUDA_ENTRY = 0x10
    SYMBOL_ENTRY_BYTES = 24
    PTX_TARGET_PATTERN = re.compile(rb"^\s*\.target\s+sm_", re.MULTILINE)
    # the leading bytes searched for a PTX ``.target`` line
    PTX_PROBE_BYTES = 4096

    path: str
    sm: int
    sha256: str
    list_kernel: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "KernelBinary":
        """Read a cubin.

        Args:
            path: the file

        Returns:
            the binary

        Raises:
            HarnessError: the file is not a CUDA cubin
        """
        with open(
            file=path,
            mode="rb",
        ) as binary_file:
            data = binary_file.read()
        if KernelBinary.PTX_TARGET_PATTERN.search(data[:KernelBinary.PTX_PROBE_BYTES]):
            raise HarnessError(
                f"{path} is PTX; the platform loads cubins only. Build one with "
                "nvcc -cubin -arch=<target>, or ptxas -arch=<target>"
            )
        if not data.startswith(KernelBinary.ELF_MAGIC):
            raise HarnessError(f"{path} is not a cubin")
        return cls._load_cubin(
            path=path,
            data=data,
        )

    @classmethod
    def _load_cubin(
        cls,
        path: str,
        data: bytes,
    ) -> "KernelBinary":
        """Read a cubin's target and kernel symbols from its ELF header and symbol table."""
        try:
            (machine,) = struct.unpack_from(
                "<H",  # format
                data,  # buffer
                0x12,  # offset
            )
            (flags,) = struct.unpack_from(
                "<I",  # format
                data,  # buffer
                0x30,  # offset
            )
            list_kernel = KernelBinary._kernel_symbols(data)
        except struct.error as err:
            raise HarnessError(f"{path} is a truncated ELF file") from err
        if machine != KernelBinary.MACHINE_CUDA:
            raise HarnessError(f"{path} is an ELF file, but not a CUDA cubin")
        abi_version = data[8]
        if abi_version >= KernelBinary.ABI_VERSION_SM_BYTE:
            sm = (flags >> KernelBinary.BITS_PER_BYTE) & KernelBinary.BYTE_MASK
        else:
            sm = flags & KernelBinary.BYTE_MASK
        return cls(
            path=path,
            sm=sm,
            sha256=hashlib.sha256(data).hexdigest(),
            list_kernel=list_kernel,
        )

    @staticmethod
    def _kernel_symbols(data: bytes) -> list[str]:
        """The names of a cubin's kernel entry points."""
        (section_offset,) = struct.unpack_from(
            "<Q",  # format
            data,  # buffer
            0x28,  # offset
        )
        section_entry_bytes, section_count = struct.unpack_from(
            "<HH",  # format
            data,   # buffer
            0x3A,   # offset
        )
        list_section = [
            struct.unpack_from(
                "<IIQQQQIIQQ",                                     # format
                data,                                              # buffer
                section_offset + section_index * section_entry_bytes,  # offset
            )
            for section_index in range(section_count)
        ]
        list_kernel = []
        for section in list_section:
            if section[1] != KernelBinary.SECTION_SYMBOL_TABLE:
                continue
            string_offset = list_section[section[6]][4]
            for symbol_index in range(section[5] // KernelBinary.SYMBOL_ENTRY_BYTES):
                name_offset, info, other, _, _, _ = struct.unpack_from(
                    "<IBBHQQ",  # format
                    data,       # buffer
                    section[4] + symbol_index * KernelBinary.SYMBOL_ENTRY_BYTES,  # offset
                )
                if (
                    info & KernelBinary.SYMBOL_TYPE_MASK == KernelBinary.SYMBOL_TYPE_FUNCTION
                    and other & KernelBinary.SYMBOL_CUDA_ENTRY
                ):
                    name_end = data.index(
                        b"\0",                          # sub
                        string_offset + name_offset,    # start
                    )
                    list_kernel.append(data[string_offset + name_offset:name_end].decode("utf-8"))
        return sorted(list_kernel)

    def target(self) -> str:
        """The cubin's target as the hardware architectures name it.

        Returns:
            e.g. ``sm_90``
        """
        return f"sm_{self.sm}"

    def runs_on(self, architecture: str) -> bool:
        """Whether the cubin loads on a GPU of an architecture: a cubin runs on its own SM only.

        Args:
            architecture: the hardware's architecture, e.g. ``sm_90a``

        Returns:
            whether it loads
        """
        match = re.match(
            r"^sm_(\d+)",  # pattern
            architecture,  # string
        )
        return (
            match is not None
            and int(match.group(1)) == self.sm
        )


@dataclass
class CachedKernel:
    """A kernel found in a compiler's cache, with the launch facts the cache recorded.

    Args:
        binary: the cubin that holds it
        name: the kernel's entry point
        origin: the cache it came from, e.g. ``triton``
        map_hint: what the cache knows of its launch: block size, shared memory, signature
    """

    binary: KernelBinary
    name: str
    origin: str
    map_hint: dict[str, Any] = field(default_factory=dict)


class DslCache:
    """The on-disk caches where kernel DSLs and compilers leave the cubins they build."""

    THREADS_PER_WARP = 32
    TRITON_FUNCTION_PATTERN = re.compile(r"tt\.func public @[\w$]+\((.*?)\)\s*(?:attributes|\{)", re.DOTALL)
    TRITON_ARGUMENT_PATTERN = re.compile(r"%(\w+): ([^ ,{]+)")

    @staticmethod
    def default_roots() -> list[tuple[str, str]]:
        """The cache directories searched by default, each with the name of its compiler.

        Returns:
            ``(origin, directory)`` pairs; directories that do not exist are left out
        """
        list_root = [
            (
                "triton",
                os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache"),
            ),
            (
                "torch.compile",
                os.environ.get("TORCHINDUCTOR_CACHE_DIR")
                or os.path.join(
                    tempfile.gettempdir(),
                    f"torchinductor_{getpass.getuser()}",
                ),
            ),
            (
                "tilelang",
                os.environ.get("TILELANG_CACHE_DIR") or os.path.expanduser("~/.tilelang/cache"),
            ),
        ]
        return [
            (origin, directory)
            for origin, directory in list_root
            if os.path.isdir(directory)
        ]

    @staticmethod
    def find(
        pattern: str,
        list_root: list[tuple[str, str]],
    ) -> list[CachedKernel]:
        """Find the kernels whose name matches a pattern in cache directories.

        Args:
            pattern: a regex searched in the kernel names
            list_root: ``(origin, directory)`` pairs to search

        Returns:
            the matching kernels, newest cubin first
        """
        regex = re.compile(pattern)
        list_found = []
        for origin, directory in list_root:
            for root, _, list_name in os.walk(directory):
                for name in list_name:
                    if not name.endswith(".cubin"):
                        continue
                    path = os.path.join(
                        root,
                        name,
                    )
                    try:
                        binary = KernelBinary.load(path)
                    except (HarnessError, OSError):
                        continue
                    for kernel_name in binary.list_kernel:
                        if regex.search(kernel_name):
                            list_found.append(
                                CachedKernel(
                                    binary=binary,
                                    name=kernel_name,
                                    origin=origin,
                                    map_hint=DslCache._triton_hint(path),
                                )
                            )
        list_found.sort(
            key=lambda cached: os.path.getmtime(cached.binary.path),
            reverse=True,
        )
        return list_found

    @staticmethod
    def search(
        pattern: str,
        list_directory: list[str],
        copy_to: str | None,
        pick: int,
    ) -> dict[str, Any]:
        """Search the compiler caches and some directories for kernels, and copy one into a job.

        Args:
            pattern: a regex searched in the kernel names
            list_directory: more directories of cubins to search
            copy_to: a job directory to copy the picked cubin into, or None
            pick: the position of the match to copy

        Returns:
            ``kernels`` found, newest first, and ``copied_to``, the copy's path or None

        Raises:
            HarnessError: nothing matches, or the pick is out of range
        """
        list_root = DslCache.default_roots() + [
            (
                "directory",
                directory,
            )
            for directory in list_directory
        ]
        if not list_root:
            raise HarnessError("no compiler cache found; name a directory of cubins with --dir")
        list_found = DslCache.find(
            pattern=pattern,
            list_root=list_root,
        )
        if not list_found:
            raise HarnessError(
                f"no cached kernel matches '{pattern}' under "
                f"{', '.join(directory for _, directory in list_root)}"
            )
        copied_to = None
        if copy_to is not None:
            if not 0 <= pick < len(list_found):
                raise HarnessError(f"--pick {pick} is not one of the {len(list_found)} matches")
            os.makedirs(
                name=copy_to,
                exist_ok=True,
            )
            copied_to = os.path.join(
                copy_to,
                os.path.basename(list_found[pick].binary.path),
            )
            shutil.copyfile(
                src=list_found[pick].binary.path,
                dst=copied_to,
            )
        return {
            "kernels": [
                {
                    "name": cached.name,
                    "origin": cached.origin,
                    "target": cached.binary.target(),
                    "path": cached.binary.path,
                    "hints": cached.map_hint,
                }
                for cached in list_found
            ],
            "copied_to": copied_to,
        }

    @staticmethod
    def _triton_hint(cubin_path: str) -> dict[str, Any]:
        """The launch facts a Triton cache entry keeps beside its cubin, or none."""
        stem = cubin_path[:-len(".cubin")]
        map_hint: dict[str, Any] = {}
        if os.path.isfile(f"{stem}.json"):
            with open(
                file=f"{stem}.json",
                mode="r",
                encoding="utf-8",
            ) as metadata_file:
                map_metadata = json.load(metadata_file)
            map_hint["block"] = [
                map_metadata.get("num_warps", 0) * map_metadata.get(
                    "warp_size",
                    DslCache.THREADS_PER_WARP,
                ),
                1,
                1,
            ]
            map_hint["shared_bytes"] = map_metadata.get("shared")
            for key in (
                "num_ctas",
                "cluster_dims",
                "global_scratch_size",
                "profile_scratch_size",
                "launch_cooperative_grid",
                "launch_pdl",
            ):
                if map_metadata.get(key):
                    map_hint[key] = map_metadata[key]
        if os.path.isfile(f"{stem}.ttir"):
            with open(
                file=f"{stem}.ttir",
                mode="r",
                encoding="utf-8",
            ) as ttir_file:
                match = DslCache.TRITON_FUNCTION_PATTERN.search(ttir_file.read())
            if match is not None:
                map_hint["signature"] = [
                    f"{argument_name}: {argument_type}"
                    for argument_name, argument_type in DslCache.TRITON_ARGUMENT_PATTERN.findall(
                        match.group(1)
                    )
                ]
        return map_hint
