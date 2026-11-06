import io
import json
import os
import shutil
import struct
import tarfile
import tempfile
import unittest
from xtrace_harness.binary import KernelBinary
from xtrace_harness.error import ManifestError
from xtrace_harness.job import JobBundle, JobManifest

EXAMPLE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "matmul",
)

MANIFEST_HEAD = '[kernel]\nbinary = "matmul.cubin"\nname = "square_matrix_mul"\n'
BUFFERS = '[[buffers]]\nname = "a"\ndtype = "f16"\nshape = [64, 32]\n'
LAUNCH = '[launch]\ngrid = 4\nblock = [128]\nargs = [{ kind = "buffer", buffer = "a" }]\n'
TRACE = "[trace]\n[[trace.sites]]\nindex = 3\n[[trace.sites]]\nindex = 9\n"


def write_job(
    directory: str,
    manifest: str,
) -> None:
    """Write the example cubin and a manifest into a job directory."""
    shutil.copyfile(
        os.path.join(EXAMPLE_DIRECTORY, "matmul.cubin"),
        os.path.join(directory, "matmul.cubin"),
    )
    with open(os.path.join(directory, "xtrace.toml"), "w") as manifest_file:
        manifest_file.write(manifest)


def write_safetensors(
    path: str,
    map_header: dict,
) -> None:
    """Write a safetensors file with zeroed data behind a header."""
    header = json.dumps(map_header).encode("utf-8")
    data_bytes = max(
        [entry["data_offsets"][1] for entry in map_header.values() if "data_offsets" in entry],
        default=0,
    )
    with open(path, "wb") as tensor_file:
        tensor_file.write(struct.pack("<Q", len(header)))
        tensor_file.write(header)
        tensor_file.write(bytes(data_bytes))


class JobManifestTest(unittest.TestCase):
    def load(self, manifest: str) -> JobManifest:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        write_job(directory, manifest)
        return JobManifest.load(directory)

    def test_example_loads(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        map_spec = manifest.spec.to_json()
        self.assertEqual(map_spec["schema"], "xtrace.job/1")
        self.assertEqual(map_spec["trace"]["regions"], [[0, 1, "dot product"]])
        self.assertEqual(map_spec["launch"]["grid"], [16, 16, 1])
        self.assertEqual(map_spec["launch"]["args"][3], {"kind": "scalar", "dtype": "i32", "value": 256})
        self.assertEqual(manifest.hardware, "h100-sxm")
        self.assertTrue(manifest.binary.runs_on("sm_90a"))
        self.assertFalse(manifest.binary.runs_on("sm_100a"))

    def test_inspect_needs_only_the_kernel(self) -> None:
        self.assertEqual(self.load('kind = "inspect"\n' + MANIFEST_HEAD).spec.kind, "inspect")

    def test_kernel_must_be_in_the_binary(self) -> None:
        with self.assertRaisesRegex(ManifestError, "square_matrix_mul"):
            self.load('kind = "inspect"\n[kernel]\nbinary = "matmul.cubin"\nname = "other"\n')

    def test_trace_needs_launch_and_sites(self) -> None:
        with self.assertRaisesRegex(ManifestError, "launch"):
            self.load(MANIFEST_HEAD + TRACE)
        with self.assertRaisesRegex(ManifestError, "sites"):
            self.load(MANIFEST_HEAD + BUFFERS + LAUNCH + "[trace]\n")

    def test_trace_loads(self) -> None:
        manifest = self.load(MANIFEST_HEAD + BUFFERS + LAUNCH + TRACE)
        self.assertEqual(manifest.spec.launch.list_block, [128, 1, 1])
        self.assertEqual(manifest.spec.list_buffer[0].nbytes(), 64 * 32 * 2)

    def test_unknown_key(self) -> None:
        with self.assertRaisesRegex(ManifestError, "entry"):
            self.load('entry = "run.py"\n' + MANIFEST_HEAD)

    def test_argument_names_present_buffer(self) -> None:
        launch = '[launch]\ngrid = 1\nblock = 1\nargs = [{ kind = "buffer", buffer = "b" }]\n'
        with self.assertRaisesRegex(ManifestError, "'b'"):
            self.load(MANIFEST_HEAD + BUFFERS + launch + TRACE)

    def test_scalar_fits_type(self) -> None:
        launch = '[launch]\ngrid = 1\nblock = 1\nargs = [{ kind = "scalar", dtype = "u8", value = 300 }]\n'
        with self.assertRaisesRegex(ManifestError, "u8"):
            self.load(MANIFEST_HEAD + launch + TRACE)

    def test_block_size_limit(self) -> None:
        with self.assertRaisesRegex(ManifestError, "1024"):
            self.load(MANIFEST_HEAD + "[launch]\ngrid = 1\nblock = [64, 32]\n" + TRACE)

    def test_bytes_relocation_inside(self) -> None:
        launch = (
            '[launch]\ngrid = 1\nblock = 1\nargs = [{ kind = "bytes", hex = "0000000000000000", '
            'relocations = [{ offset = 8, buffer = "a" }] }]\n'
        )
        with self.assertRaisesRegex(ManifestError, "relocation"):
            self.load(MANIFEST_HEAD + BUFFERS + launch + TRACE)

    def test_tensor_map_rank(self) -> None:
        launch = (
            '[launch]\ngrid = 1\nblock = 1\nargs = [{ kind = "tensor_map", buffer = "a", '
            'data_type = "f16", global_dims = [32, 64], global_strides_bytes = [64], '
            "box_dims = [32] }]\n"
        )
        with self.assertRaisesRegex(ManifestError, "box_dims"):
            self.load(MANIFEST_HEAD + BUFFERS + launch + TRACE)

    def test_buffer_from_tensor_file(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        write_job(
            directory,
            'kind = "inspect"\ntensors = "inputs.safetensors"\n' + MANIFEST_HEAD
            + '[[buffers]]\nname = "a"\ndtype = "f16"\nshape = [64, 32]\ninit = "file"\n',
        )
        write_safetensors(
            os.path.join(directory, "inputs.safetensors"),
            {"a": {"dtype": "F16", "shape": [64, 16], "data_offsets": [0, 2048]}},
        )
        with self.assertRaisesRegex(ManifestError, "its tensor"):
            JobManifest.load(directory)
        write_safetensors(
            os.path.join(directory, "inputs.safetensors"),
            {"a": {"dtype": "F16", "shape": [64, 32], "data_offsets": [0, 4096]}},
        )
        self.assertEqual(JobManifest.load(directory).list_files()[0], "inputs.safetensors")

    def test_paths_stay_inside(self) -> None:
        with self.assertRaisesRegex(ManifestError, "inside"):
            self.load(MANIFEST_HEAD + 'sources = ["../secret.cu"]\n')

    def test_kind_override(self) -> None:
        manifest = JobManifest.load(
            directory=EXAMPLE_DIRECTORY,
            kind="inspect",
        )
        self.assertNotIn("trace", manifest.spec.to_json())


class KernelBinaryTest(unittest.TestCase):
    def test_ptx_target_and_entries(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as ptx_file:
            ptx_file.write(".version 8.5\n.target sm_90a\n.visible .entry k_one(\n)\n{}\n.entry k_two(\n)\n")
        self.addCleanup(os.unlink, ptx_file.name)
        binary = KernelBinary.load(ptx_file.name)
        self.assertEqual(binary.target(), "sm_90a")
        self.assertEqual(binary.list_kernel, ["k_one", "k_two"])
        self.assertTrue(binary.runs_on("sm_90a"))
        self.assertFalse(binary.runs_on("sm_100a"))


class JobBundleTest(unittest.TestCase):
    def test_bundle_holds_the_named_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_job(directory, 'kind = "inspect"\n' + MANIFEST_HEAD)
            with open(os.path.join(directory, "notes.txt"), "w") as notes_file:
                notes_file.write("x")
            bundle = JobBundle.pack(JobManifest.load(directory))
            self.assertEqual(bundle.list_file, ["matmul.cubin", "xtrace.toml"])
            with tarfile.open(fileobj=io.BytesIO(bundle.data), mode="r:gz") as archive:
                self.assertEqual(sorted(archive.getnames()), bundle.list_file)

    def test_bundle_is_reproducible(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        self.assertEqual(JobBundle.pack(manifest).sha256, JobBundle.pack(manifest).sha256)


if __name__ == "__main__":
    unittest.main()
