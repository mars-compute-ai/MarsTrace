import json
import math
import struct
from dataclasses import dataclass, field
from xtrace_harness.error import ManifestError


@dataclass
class TensorEntry:
    """One tensor of a safetensors file: its element type, shape and where its bytes lie.

    Args:
        name: the tensor's name
        dtype: the element type, e.g. ``f16``
        list_shape: the extents
        nbytes: the size of its data
        offset_bytes: where its data starts, counted from the end of the header
    """

    name: str
    dtype: str
    list_shape: list[int]
    nbytes: int
    offset_bytes: int = 0


@dataclass
class TensorFile:
    """The header of a safetensors file, which names the tensors a job's buffers start from."""

    HEADER_SIZE_BYTES = 8
    # the largest header a safetensors file may carry
    HEADER_BYTES_MAX = 100 * 1024 * 1024
    METADATA_KEY = "__metadata__"
    # safetensors element types and the job's names for them
    MAP_DTYPE = {
        "BOOL": "bool",
        "U8": "u8",
        "I8": "i8",
        "F8_E4M3": "f8e4m3",
        "F8_E5M2": "f8e5m2",
        "U16": "u16",
        "I16": "i16",
        "F16": "f16",
        "BF16": "bf16",
        "U32": "u32",
        "I32": "i32",
        "F32": "f32",
        "U64": "u64",
        "I64": "i64",
        "F64": "f64",
    }

    # the alignment of the data section that follows the header
    HEADER_ALIGNMENT_BYTES = 8

    path: str
    map_tensor: dict[str, TensorEntry] = field(default_factory=dict)
    data_offset_bytes: int = 0

    @classmethod
    def load(cls, path: str) -> "TensorFile":
        """Read the tensor table of a safetensors file.

        Args:
            path: the file

        Returns:
            the file's tensors

        Raises:
            ManifestError: the file is not a safetensors file
        """
        with open(
            file=path,
            mode="rb",
        ) as tensor_file:
            prefix = tensor_file.read(TensorFile.HEADER_SIZE_BYTES)
            if len(prefix) != TensorFile.HEADER_SIZE_BYTES:
                raise ManifestError(f"{path} is not a safetensors file")
            (header_bytes,) = struct.unpack(
                "<Q",    # format
                prefix,  # buffer
            )
            if header_bytes > TensorFile.HEADER_BYTES_MAX:
                raise ManifestError(f"{path} is not a safetensors file")
            try:
                map_header = json.loads(tensor_file.read(header_bytes))
            except (json.JSONDecodeError, UnicodeDecodeError) as err:
                raise ManifestError(f"{path} is not a safetensors file") from err
        map_tensor = {}
        for name, map_entry in map_header.items():
            if name == TensorFile.METADATA_KEY:
                continue
            dtype = TensorFile.MAP_DTYPE.get(map_entry.get("dtype"))
            if dtype is None:
                raise ManifestError(f"{path}: tensor '{name}' has unsupported dtype {map_entry.get('dtype')}")
            offset_begin, offset_end = map_entry["data_offsets"]
            map_tensor[name] = TensorEntry(
                name=name,
                dtype=dtype,
                list_shape=list(map_entry["shape"]),
                nbytes=offset_end - offset_begin,
                offset_bytes=offset_begin,
            )
        return cls(
            path=path,
            map_tensor=map_tensor,
            data_offset_bytes=TensorFile.HEADER_SIZE_BYTES + header_bytes,
        )

    def read_tensor(self, name: str) -> bytes:
        """The data of one tensor of the file.

        Args:
            name: the tensor

        Returns:
            its bytes

        Raises:
            ManifestError: the file is shorter than its header says
        """
        tensor = self.map_tensor[name]
        with open(
            file=self.path,
            mode="rb",
        ) as tensor_file:
            tensor_file.seek(self.data_offset_bytes + tensor.offset_bytes)
            data = tensor_file.read(tensor.nbytes)
        if len(data) != tensor.nbytes:
            raise ManifestError(f"{self.path}: tensor '{name}' is truncated")
        return data

    @staticmethod
    def encode(
        list_tensor: list[tuple[TensorEntry, bytes]],
        map_metadata: dict[str, str] | None = None,
    ) -> bytes:
        """A safetensors file holding the given tensors, in the given order.

        Args:
            list_tensor: each tensor's entry and data
            map_metadata: text the file carries under ``__metadata__``, or None

        Returns:
            the file's bytes
        """
        map_safetensors_dtype = {dtype: name for name, dtype in TensorFile.MAP_DTYPE.items()}
        map_header: dict[str, object] = {}
        if map_metadata:
            map_header[TensorFile.METADATA_KEY] = dict(map_metadata)
        offset = 0
        for tensor, data in list_tensor:
            map_header[tensor.name] = {
                "dtype": map_safetensors_dtype[tensor.dtype],
                "shape": list(tensor.list_shape),
                "data_offsets": [
                    offset,
                    offset + len(data),
                ],
            }
            offset += len(data)
        header = json.dumps(
            map_header,
            separators=(",", ":"),
        ).encode("utf-8")
        header += b" " * (-len(header) % TensorFile.HEADER_ALIGNMENT_BYTES)
        return b"".join(
            [
                struct.pack(
                    "<Q",         # format
                    len(header),  # v1
                ),
                header,
                *[data for _, data in list_tensor],
            ]
        )

    @staticmethod
    def element_count(list_shape: list[int]) -> int:
        """The number of elements of a shape.

        Args:
            list_shape: the shape

        Returns:
            the product of its extents, 1 for a scalar
        """
        return math.prod(list_shape)
