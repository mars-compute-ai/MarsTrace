import copy
import json
import os
import shutil
import struct
import tempfile
import unittest
from fake_control_plane import CAPABILITIES, EXAMPLE_DIRECTORY, matmul_documents
from schema_check import SchemaCheck
from xtrace_harness.binary import KernelBinary
from xtrace_harness.error import HarnessError, ManifestError
from xtrace_harness.job import JobManifest
from xtrace_harness.listing import KernelListing
from xtrace_harness.tensors import TensorFile

TOP = 'hardware = "h100-sxm"\nintent = "time the loop"\n'
KERNEL = '[kernel]\nbinary = "matmul.cubin"\nname = "square_matrix_mul"\nsources = ["matmul.cu"]\n'
BUFFERS = (
    '[[buffers]]\nname = "a"\ndtype = "f16"\nshape = [64, 32]\n'
    '[[buffers]]\nname = "c"\ndtype = "f32"\nshape = [64]\noutput = true\n'
)
LAUNCH = (
    '[launch]\ngrid = 4\nblock = [128]\n'
    'args = [{ kind = "buffer", buffer = "a" }, { kind = "buffer", buffer = "c" }]\n'
)
TRACE = "[trace]\n[[trace.sites]]\nindex = 6\n[[trace.sites]]\nindex = 11\n"
ARTIFACT_IDS = {"cubin": "cubin-1", "launch_manifest": "manifest-1"}
LISTING = KernelListing.from_documents(*matmul_documents()[:2], "fixture")


def write_safetensors(path: str, map_header: dict) -> None:
    """Write a safetensors file with zeroed data behind a header."""
    header = json.dumps(map_header).encode("utf-8")
    data_bytes = max(entry["data_offsets"][1] for entry in map_header.values())
    with open(path, "wb") as tensor_file:
        tensor_file.write(struct.pack("<Q", len(header)))
        tensor_file.write(header)
        tensor_file.write(bytes(data_bytes))


class TranslationTest(unittest.TestCase):
    """One test, or a few, per row of the mapping from xtrace.toml onto the API."""

    def directory(self, manifest: str) -> str:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        for name in ("matmul.cubin", "matmul.cu"):
            shutil.copyfile(os.path.join(EXAMPLE_DIRECTORY, name), os.path.join(directory, name))
        with open(os.path.join(directory, "xtrace.toml"), "w") as manifest_file:
            manifest_file.write(manifest)
        return directory

    def load(self, top: str = "", tables: str = BUFFERS + LAUNCH + TRACE) -> JobManifest:
        return JobManifest.load(self.directory(TOP + top + KERNEL + tables))

    def request(self, manifest: JobManifest, listing: KernelListing | None = None) -> dict:
        return manifest.estimate_input(
            map_capabilities=CAPABILITIES,
            map_artifact_id=dict(ARTIFACT_IDS),
            listing=listing,
        )

    def launch(self, manifest: JobManifest, listing: KernelListing | None = None) -> dict:
        """The launch manifest, checked against the vendored schema."""
        map_launch = manifest.launch_manifest("sm_90a", listing)
        SchemaCheck.validate(SchemaCheck.LAUNCH_MANIFEST, map_launch)
        return map_launch

    def plan(self, manifest: JobManifest, listing: KernelListing | None = None) -> dict:
        return self.request(manifest, listing)["analysis"]["trace_plan"]

    def test_example_translates(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        request = self.request(manifest, LISTING)
        self.assertEqual(request["analysis"]["trace_plan"]["regions"][0]["region_id"], "dot_product")
        launch = self.launch(manifest, LISTING)
        self.assertEqual(launch["arguments"][3], {"index": 3, "kind": "scalar", "abi_type": "int32", "value": "256"})
        self.assertEqual(launch["parameter_layout"][3], {"index": 3, "offset": 24, "size_bytes": 4})
        self.assertEqual(launch["outputs"], [{"name": "c", "buffer": "c", "dtype": "int32", "shape": [256, 256]}])
        manifest.check_parameters(LISTING)

    def test_hardware_selects_the_compute_entry(self) -> None:
        self.assertEqual(self.request(self.load())["compute"], CAPABILITIES["compute"][0]["compute"])
        with self.assertRaisesRegex(ManifestError, "no hardware 'a100'.*h100-sxm, b200"):
            self.request(JobManifest.load(self.directory(TOP.replace("h100-sxm", "a100") + KERNEL + BUFFERS + LAUNCH + TRACE)))
        with self.assertRaisesRegex(ManifestError, "sm_90, but b200 is sm_100a"):
            self.request(JobManifest.load(self.directory(TOP.replace("h100-sxm", "b200") + KERNEL + BUFFERS + LAUNCH + TRACE)))

    def test_hardware_must_run_the_stages(self) -> None:
        capabilities = copy.deepcopy(CAPABILITIES)
        capabilities["compute"][0]["architectures"]["pc_sampling"] = ["sm_100a"]
        trace = TRACE.replace("[trace]\n", '[trace]\nstages = ["xtrace", "pc_sampling"]\n[trace.pc_sampling]\nrepeats = 3\n')
        manifest = self.load(tables=BUFFERS + LAUNCH + trace)
        with self.assertRaisesRegex(ManifestError, "pc_sampling stage does not support sm_90a"):
            manifest.select_compute(capabilities)

    def test_intent_is_the_prompt(self) -> None:
        self.assertEqual(self.request(self.load())["prompt"], "time the loop")
        with self.assertRaisesRegex(ManifestError, "'intent' is required"):
            JobManifest.load(self.directory('hardware = "h100-sxm"\n' + KERNEL + BUFFERS + LAUNCH + TRACE))

    def test_binary_is_the_cubin_artifact(self) -> None:
        manifest = self.load()
        self.assertEqual([(artifact.kind, os.path.basename(artifact.path)) for artifact in manifest.list_artifact()], [("cubin", "matmul.cubin")])
        self.assertEqual(self.request(manifest)["bundle"]["cubin_artifact_id"], "cubin-1")

    def test_ptx_is_refused(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as ptx_file:
            ptx_file.write(".version 8.5\n.target sm_90a\n.visible .entry k(\n)\n{}\n")
        self.addCleanup(os.unlink, ptx_file.name)
        with self.assertRaisesRegex(HarnessError, "PTX"):
            KernelBinary.load(ptx_file.name)

    def test_name_is_the_kernel_symbol(self) -> None:
        self.assertEqual(self.launch(self.load())["target"]["kernel_symbol"], "square_matrix_mul")
        with self.assertRaisesRegex(ManifestError, "defines no kernel 'other'"):
            JobManifest.load(self.directory(TOP + KERNEL.replace("square_matrix_mul", "other") + BUFFERS + LAUNCH + TRACE))

    def test_sources_are_not_uploaded(self) -> None:
        self.assertNotIn("source", [artifact.kind for artifact in self.load().list_artifact()])

    def test_tensors_are_the_safetensors_artifact(self) -> None:
        directory = self.directory(
            TOP + 'tensors = "inputs.safetensors"\n' + KERNEL
            + '[[buffers]]\nname = "a"\ndtype = "f16"\nshape = [64, 32]\ninit = "tensor"\n'
            + '[[buffers]]\nname = "c"\ndtype = "f32"\nshape = [64]\noutput = true\n' + LAUNCH + TRACE
        )
        write_safetensors(os.path.join(directory, "inputs.safetensors"), {"a": {"dtype": "F16", "shape": [64, 16], "data_offsets": [0, 2048]}})
        with self.assertRaisesRegex(ManifestError, r"is f16\[64, 32\], but tensor 'a'"):
            JobManifest.load(directory)
        write_safetensors(os.path.join(directory, "inputs.safetensors"), {"a": {"dtype": "F16", "shape": [64, 32], "data_offsets": [0, 4096]}})
        manifest = JobManifest.load(directory)
        self.assertEqual([artifact.kind for artifact in manifest.list_artifact()], ["cubin", "safetensors"])
        self.assertEqual(self.launch(manifest)["allocations"][0], {"id": "a", "size_bytes": 4096, "init": {"kind": "tensor", "tensor_key": "a"}})
        self.assertIsNone(manifest.list_artifact()[1].data)
        request = manifest.estimate_input(CAPABILITIES, {**ARTIFACT_IDS, "safetensors": "tensors-1"}, None)
        self.assertEqual(request["bundle"]["tensors_artifact_id"], "tensors-1")

    def test_tensors_need_a_tensor_buffer(self) -> None:
        directory = self.directory(TOP + 'tensors = "inputs.safetensors"\n' + KERNEL + BUFFERS + LAUNCH + TRACE)
        write_safetensors(os.path.join(directory, "inputs.safetensors"), {"a": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}})
        with self.assertRaisesRegex(ManifestError, "no buffer has init = 'tensor'"):
            JobManifest.load(directory)

    def test_reference_and_validation(self) -> None:
        validation = '[validation]\noutput = "c"\natol = 0.01\nrtol = 0.001\n'
        directory = self.directory(TOP + 'reference = "reference.safetensors"\n' + KERNEL + BUFFERS + LAUNCH + validation + TRACE)
        write_safetensors(os.path.join(directory, "reference.safetensors"), {"c": {"dtype": "F32", "shape": [64], "data_offsets": [0, 256]}})
        manifest = JobManifest.load(directory)
        self.assertEqual(manifest.list_artifact()[-1].kind, "reference")
        self.assertEqual(self.launch(manifest)["validation"], {"output_key": "c", "reference_key": "c", "atol": 0.01, "rtol": 0.001})
        request = manifest.estimate_input(CAPABILITIES, {**ARTIFACT_IDS, "reference": "reference-1"}, None)
        self.assertEqual(request["bundle"]["reference_artifact_id"], "reference-1")
        with self.assertRaisesRegex(ManifestError, "must be a buffer with output = true"):
            JobManifest.load(self.directory(TOP + 'reference = "reference.safetensors"\n' + KERNEL + BUFFERS + LAUNCH + validation.replace('"c"', '"a"') + TRACE))
        with self.assertRaisesRegex(ManifestError, "go together"):
            self.load(tables=BUFFERS + LAUNCH + validation + TRACE)

    def test_output_dtypes_map_to_manifest_names(self) -> None:
        map_expected = {
            "bool": "bool", "u8": "uint8", "i8": "int8", "f8e4m3": "uint8", "f8e5m2": "uint8",
            "u16": "uint16", "i16": "int16", "f16": "float16", "bf16": "bfloat16", "u32": "uint32",
            "i32": "int32", "f32": "float32", "u64": "uint64", "i64": "int64", "f64": "float64",
        }
        buffers = "".join(f'[[buffers]]\nname = "b_{dtype}"\ndtype = "{dtype}"\nshape = [16]\noutput = true\n' for dtype in map_expected)
        launch = LAUNCH.replace(" }]\n", " }" + "".join(f', {{ kind = "buffer", buffer = "b_{dtype}" }}' for dtype in map_expected) + "]\n")
        list_output = self.launch(self.load(tables=BUFFERS + buffers + launch + TRACE))["outputs"]
        self.assertEqual({output["name"][2:]: output["dtype"] for output in list_output[1:]}, map_expected)
        self.assertEqual(list_output[4], {"name": "b_f8e4m3", "buffer": "b_f8e4m3", "dtype": "uint8", "shape": [16]})

    def test_buffers_are_allocations_views_and_outputs(self) -> None:
        launch = self.launch(self.load())
        self.assertEqual(
            launch["allocations"],
            [
                {"id": "a", "size_bytes": 4096, "init": {"kind": "zeros"}},
                {"id": "c", "size_bytes": 256, "init": {"kind": "zeros"}},
            ],
        )
        self.assertEqual(
            launch["buffers"],
            [
                {"name": "a", "allocation_id": "a", "offset_bytes": 0, "size_bytes": 4096, "access": "read"},
                {"name": "c", "allocation_id": "c", "offset_bytes": 0, "size_bytes": 256, "access": "read_write"},
            ],
        )
        self.assertEqual(launch["outputs"], [{"name": "c", "buffer": "c", "dtype": "float32", "shape": [64]}])
        self.assertEqual((launch["runtime_profile"], launch["parameter_layout"]), (None, None))
        with self.assertRaisesRegex(ManifestError, "output buffer 'c' is named by no launch argument"):
            self.load(tables=BUFFERS + LAUNCH.replace('buffer = "c"', 'buffer = "a"') + TRACE)
        with self.assertRaisesRegex(ManifestError, "not starting with '_xtrace'"):
            self.load(tables=BUFFERS.replace('"a"', '"_xtrace_a"') + LAUNCH.replace('"a"', '"_xtrace_a"') + TRACE)

    def test_ones_and_random_buffers_start_from_prepared_tensors(self) -> None:
        buffers = (
            '[[buffers]]\nname = "r"\ndtype = "f32"\nshape = [4, 64]\ninit = "random"\nseed = 7\n'
            '[[buffers]]\nname = "h"\ndtype = "bf16"\nshape = [256]\ninit = "random"\n'
            '[[buffers]]\nname = "o"\ndtype = "f16"\nshape = [4]\ninit = "ones"\n'
            '[[buffers]]\nname = "n"\ndtype = "i32"\nshape = [4]\ninit = "random"\nseed = 7\n'
        )
        manifest = self.load(tables=BUFFERS + buffers + LAUNCH + TRACE)
        list_allocation = self.launch(manifest)["allocations"]
        self.assertEqual(list_allocation[2], {"id": "r", "size_bytes": 1024, "init": {"kind": "tensor", "tensor_key": "r"}})
        self.assertEqual(list_allocation[0]["init"], {"kind": "zeros"})
        artifact = manifest.list_artifact()[1]
        self.assertEqual((artifact.kind, os.path.basename(artifact.path)), ("safetensors", "tensors.safetensors"))
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tensors_file:
            tensors_file.write(artifact.data)
            tensors_file.flush()
            tensor_file = TensorFile.load(tensors_file.name)
            self.assertEqual(sorted(tensor_file.map_tensor), ["h", "n", "o", "r"])
            list_random = struct.unpack("<256f", tensor_file.read_tensor("r"))
            self.assertTrue(all(0.5 <= abs(value) < 1 for value in list_random))
            self.assertTrue(any(value < 0 for value in list_random) and any(value > 0 for value in list_random))
            data = tensor_file.read_tensor("h")
            list_bfloat = [struct.unpack("<f", b"\0\0" + data[2 * index:2 * index + 2])[0] for index in range(256)]
            self.assertTrue(all(0.5 <= abs(value) < 1 for value in list_bfloat))
            self.assertEqual(struct.unpack("<4e", tensor_file.read_tensor("o")), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(manifest.list_artifact()[1].data, artifact.data)
        reseeded = self.load(tables=BUFFERS + buffers.replace("seed = 7", "seed = 8", 1) + LAUNCH + TRACE)
        self.assertNotEqual(reseeded.list_artifact()[1].data, artifact.data)
        with self.assertRaisesRegex(ManifestError, "'seed' belongs to init = 'random'"):
            self.load(tables=BUFFERS + '[[buffers]]\nname = "z"\ndtype = "f32"\nshape = [4]\nseed = 1\n' + LAUNCH + TRACE)

    def test_prepared_tensors_join_the_tensors_file(self) -> None:
        directory = self.directory(
            TOP + 'tensors = "inputs.safetensors"\n' + KERNEL
            + '[[buffers]]\nname = "a"\ndtype = "f16"\nshape = [64, 32]\ninit = "tensor"\n'
            + '[[buffers]]\nname = "c"\ndtype = "f32"\nshape = [64]\noutput = true\ninit = "random"\n' + LAUNCH + TRACE
        )
        write_safetensors(
            os.path.join(directory, "inputs.safetensors"),
            {
                "a": {"dtype": "F16", "shape": [64, 32], "data_offsets": [0, 4096]},
                "unused": {"dtype": "U8", "shape": [8], "data_offsets": [4096, 4104]},
            },
        )
        artifact = JobManifest.load(directory).list_artifact()[1]
        self.assertEqual(os.path.basename(artifact.path), "inputs.safetensors")
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tensors_file:
            tensors_file.write(artifact.data)
            tensors_file.flush()
            tensor_file = TensorFile.load(tensors_file.name)
            self.assertEqual(sorted(tensor_file.map_tensor), ["a", "c"])
            self.assertEqual(tensor_file.read_tensor("a"), bytes(4096))

    def test_validation_refuses_byte_compared_outputs(self) -> None:
        validation = '[validation]\noutput = "c"\natol = 0.01\nrtol = 0.001\n'
        buffers = BUFFERS.replace('dtype = "f32"', 'dtype = "f8e4m3"')
        directory = self.directory(TOP + 'reference = "reference.safetensors"\n' + KERNEL + buffers + LAUNCH + validation + TRACE)
        write_safetensors(os.path.join(directory, "reference.safetensors"), {"c": {"dtype": "F8_E4M3", "shape": [64], "data_offsets": [0, 64]}})
        with self.assertRaisesRegex(ManifestError, "compares as raw bytes only"):
            JobManifest.load(directory)

    def test_launch_shape_is_the_target(self) -> None:
        launch = LAUNCH.replace("grid = 4\n", "grid = [4, 2]\ncluster = [2, 2]\nshared_bytes = 1024\n")
        target = self.launch(self.load(tables=BUFFERS + launch + TRACE))["target"]
        self.assertEqual(
            target,
            {
                "architecture": "sm_90a", "kernel_symbol": "square_matrix_mul", "grid": [4, 2, 1],
                "block": [128, 1, 1], "cluster": [2, 2, 1], "dynamic_shared_memory_bytes": 1024,
            },
        )
        self.assertIsNone(self.launch(self.load())["target"]["cluster"])
        with self.assertRaisesRegex(ManifestError, "a launch manifest targets sm_80, .*; h100-sxm is sm_120a"):
            self.load().launch_manifest("sm_120a", None)
        self.assertEqual(self.load().launch_manifest("sm_103a", None)["target"]["architecture"], "sm_103a")
        with self.assertRaisesRegex(ManifestError, "at most 16 blocks"):
            self.load(tables=BUFFERS + LAUNCH.replace("grid = 4", "grid = [32]\ncluster = [32]") + TRACE)
        with self.assertRaisesRegex(ManifestError, "not a multiple of cluster"):
            self.load(tables=BUFFERS + LAUNCH.replace("grid = 4", "grid = [6]\ncluster = [4]") + TRACE)
        with self.assertRaisesRegex(ManifestError, "1024 threads"):
            self.load(tables=BUFFERS + LAUNCH.replace("block = [128]", "block = [64, 32]") + TRACE)

    def test_arguments_are_indexed_by_position(self) -> None:
        launch = (
            '[launch]\ngrid = 1\nblock = 32\nargs = [\n'
            '  { kind = "buffer", buffer = "a", offset_bytes = 32 },\n'
            '  { kind = "buffer", buffer = "c" },\n'
            '  { kind = "scalar", dtype = "u64", value = 18446744073709551615 },\n'
            '  { kind = "scalar", dtype = "f32", value = 0.5 },\n'
            '  { kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [32, 64], '
            'global_strides_bytes = [64], box_dims = [32, 16], swizzle = "64b" },\n'
            '  { kind = "bytes", hex = "00000000000000001122334455667788", relocations = [{ offset = 0, buffer = "c", offset_bytes = 4 }] },\n'
            ']\n'
        )
        list_argument = self.launch(self.load(tables=BUFFERS + launch + TRACE))["arguments"]
        self.assertEqual([argument["index"] for argument in list_argument], list(range(6)))
        self.assertEqual(list_argument[0], {"index": 0, "kind": "buffer", "buffer": "a", "offset_bytes": 32})
        self.assertEqual(list_argument[2], {"index": 2, "kind": "scalar", "abi_type": "uint64", "value": "18446744073709551615"})
        self.assertEqual(list_argument[3], {"index": 3, "kind": "scalar", "abi_type": "float32", "value": "0.5"})
        self.assertEqual(
            list_argument[4],
            {
                "index": 4, "kind": "tensor_map", "buffer": "a", "offset_bytes": 0, "data_type": "f16",
                "global_dims": [32, 64], "global_strides_bytes": [64], "box_dims": [32, 16],
                "element_strides": [1, 1], "interleave": "none", "swizzle": "64b", "l2_promotion": "none",
                "oob_fill": "none",
            },
        )
        self.assertEqual(list_argument[5]["relocations"], [{"offset": 0, "buffer": "c", "offset_bytes": 4}])

    def test_argument_errors_are_precise(self) -> None:
        def launch_with(argument: str) -> str:
            return f'[launch]\ngrid = 1\nblock = 32\nargs = [{{ kind = "buffer", buffer = "c" }}, {argument}]\n'

        for argument, message in (
            ('{ kind = "buffer", buffer = "b" }', "names buffer 'b', which"),
            ('{ kind = "buffer", buffer = "a", offset_bytes = 8 }', "multiple of 16"),
            ('{ kind = "buffer", buffer = "a", offset_bytes = 4096 }', r"offset_bytes must be an integer in \[0, 4095\]"),
            ('{ kind = "scalar", dtype = "u8", value = 300 }', "fits u8"),
            ('{ kind = "scalar", dtype = "f16", value = 1.0 }', "16-bit float"),
            ('{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [32, 64], global_strides_bytes = [64], box_dims = [32] }', "box_dims needs one entry"),
            ('{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [32, 64], global_strides_bytes = [24], box_dims = [32, 1] }', "multiples of 16"),
            ('{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [32], global_strides_bytes = [], box_dims = [512] }', r"box_dims must be an integer in \[1, 256\]"),
            ('{ kind = "bytes", hex = "0000000000000000", relocations = [{ offset = 8, buffer = "a" }] }', "not 8-aligned inside"),
            ('{ kind = "bytes", hex = "00000000000000000000000000000000", relocations = [{ offset = 0, buffer = "a" }, { offset = 0, buffer = "c" }] }', "two relocations"),
            ('{ kind = "bytes", hex = "0100000000000000", relocations = [{ offset = 0, buffer = "a" }] }', "at relocation offset 0 must be zero"),
            ('{ kind = "bytes", hex = "' + "00" * 4097 + '" }', "1 to 4096 bytes"),
            ('{ kind = "scalar", dtype = "f32", value = 1e39 }', "overflows f32"),
            ('{ kind = "tensor_map", buffer = "a", data_type = "f16", global_dims = [32, 65], global_strides_bytes = [64], box_dims = [32, 1] }', "spans 4160 bytes, but buffer 'a' holds 4096"),
        ):
            with self.subTest(argument=argument), self.assertRaisesRegex(ManifestError, message):
                self.load(tables=BUFFERS + launch_with(argument) + TRACE)

    def test_parameters_are_checked_against_the_listing(self) -> None:
        with self.assertRaisesRegex(ManifestError, "takes 4 parameters, but \\[launch\\] args lists 2"):
            self.load().check_parameters(LISTING)
        launch = (
            '[launch]\ngrid = 1\nblock = 32\nargs = [{ kind = "buffer", buffer = "a" }, { kind = "buffer", buffer = "a" }, '
            '{ kind = "buffer", buffer = "c" }, { kind = "scalar", dtype = "i64", value = 1 }]\n'
        )
        with self.assertRaisesRegex(ManifestError, "argument 3 \\(scalar\\) is 8 bytes, but parameter 3"):
            self.load(tables=BUFFERS + launch + TRACE).check_parameters(LISTING)

    def test_sites_by_index(self) -> None:
        self.assertEqual(
            self.plan(self.load())["sites"],
            [
                {"site_id": "s0", "instruction_index": 6, "probe": "timing"},
                {"site_id": "s1", "instruction_index": 11, "probe": "timing"},
            ],
        )
        with self.assertRaisesRegex(HarnessError, "site index 216 is not an instruction of square_matrix_mul"):
            self.plan(self.load(tables=BUFFERS + LAUNCH + TRACE.replace("11", "216")), LISTING)
        with self.assertRaisesRegex(ManifestError, "both probe instruction 6"):
            self.plan(self.load(tables=BUFFERS + LAUNCH + TRACE.replace("11", "6")))

    def test_sites_need_an_index(self) -> None:
        with self.assertRaisesRegex(ManifestError, "site 0 index must be an integer from 0, not None"):
            self.load(tables=BUFFERS + LAUNCH + "[trace]\n[[trace.sites]]\nwarp_level = false\n")
        with self.assertRaisesRegex(ManifestError, r"\[\[trace.sites\]\]: unknown key\(s\) line"):
            self.load(tables=BUFFERS + LAUNCH + '[trace]\n[[trace.sites]]\nline = "matmul.cu:8"\n')

    def test_value_sites(self) -> None:
        trace = (
            '[trace]\n[[trace.sites]]\nindex = 6\nvalue_register = "R12"\nvalue_width_bits = 64\n'
            '[[trace.sites]]\nindex = 11\nvalue_is_smid = true\n[[trace.sites]]\nindex = 9\nvalue_register = "UR4"\n'
        )
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + trace))
        self.assertTrue(plan["capture_values"])
        self.assertEqual(
            plan["sites"],
            [
                {"site_id": "s0", "instruction_index": 6, "probe": "value", "value_register": "R12", "value_width_bits": 64},
                {"site_id": "s1", "instruction_index": 11, "probe": "smid"},
                {"site_id": "s2", "instruction_index": 9, "probe": "value", "value_register": "UR4", "value_width_bits": 32},
            ],
        )
        self.assertFalse(self.plan(self.load())["capture_values"])
        for site, message in (
            ('value_register = "R13"\nvalue_width_bits = 64', "even R register"),
            ('value_register = "UR4"\nvalue_width_bits = 64', "even R register"),
            ('value_register = "R255"', "R0 to R254"),
            ('value_register = "UR63"', "UR0 to UR62"),
            ('value_width_bits = 64', "needs 'value_register'"),
            ('value_register = "R2"\nvalue_is_smid = true', "not both"),
        ):
            with self.subTest(site=site), self.assertRaisesRegex(ManifestError, message):
                self.load(tables=BUFFERS + LAUNCH + f"[trace]\n[[trace.sites]]\nindex = 6\n{site}\n")

    def test_warp_level_is_the_granularity(self) -> None:
        trace = TRACE.replace("index = 6\n", "index = 6\nwarp_level = true\n").replace("index = 11\n", "index = 11\nwarp_level = true\n")
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + trace))
        self.assertEqual(plan["granularity"], "warp")
        self.assertEqual(plan["max_records"], 32 * 4 * 4)
        with self.assertRaisesRegex(ManifestError, "warp_level alike"):
            self.load(tables=BUFFERS + LAUNCH + TRACE.replace("index = 6\n", "index = 6\nwarp_level = true\n"))
        with self.assertRaisesRegex(ManifestError, "whole warps"):
            self.load(tables=BUFFERS + LAUNCH.replace("[128]", "[48]") + trace)

    def test_per_thread_records_size_max_records(self) -> None:
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + TRACE.replace("[trace]\n", "[trace]\nper_thread_records = 100\n")))
        self.assertEqual(plan["granularity"], "thread")
        self.assertEqual(plan["max_records"], 100 * 4 * 128)
        self.assertEqual(self.plan(self.load())["max_records"], 32 * 4 * 128)
        with self.assertRaisesRegex(ManifestError, "the platform takes at most 1000000000"):
            self.plan(self.load(tables=BUFFERS + LAUNCH.replace("grid = 4", "grid = [65536, 4]") + TRACE.replace("[trace]\n", "[trace]\nper_thread_records = 65536\n")))

    def test_clock_schedule_and_scope_roles(self) -> None:
        trace = (
            '[trace]\nclock = "sm"\nschedule = "least_makespan"\n[trace.scope_roles]\nloop = "consumer"\n'
            "[[trace.sites]]\nindex = 6\n[[trace.sites]]\nindex = 11\n"
            '[[trace.regions]]\nenter = 0\nleave = 1\nlabel = "loop"\n'
        )
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + trace))
        self.assertEqual((plan["clock"], plan["schedule"], plan["scope_roles"]), ("sm", "least_makespan", {"loop": "consumer"}))
        self.assertEqual((plan["clock"], plan["schedule"]), ("sm", "least_makespan"))
        defaults = self.plan(self.load())
        self.assertEqual((defaults["clock"], defaults["schedule"], defaults["tier"]), ("gpu", "first_feasible", "sass"))
        self.assertNotIn("scope_roles", defaults)
        with self.assertRaisesRegex(ManifestError, "anchor' needs every site to be warp_level"):
            self.load(tables=BUFFERS + LAUNCH + TRACE.replace("[trace]\n", '[trace]\nclock = "anchor"\n'))
        with self.assertRaisesRegex(ManifestError, "names region 'other'"):
            self.load(tables=BUFFERS + LAUNCH + trace.replace("loop = ", "other = "))

    def test_regions_name_sites_and_labels(self) -> None:
        regions = '[[trace.regions]]\nenter = 0\nleave = 1\nlabel = "main.loop"\n[[trace.regions]]\nenter = 1\nleave = 0\n'
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + TRACE + regions))
        self.assertEqual(
            plan["regions"],
            [
                {"region_id": "main.loop", "start_site_id": "s0", "end_site_id": "s1"},
                {"region_id": "s1-s0", "start_site_id": "s1", "end_site_id": "s0"},
            ],
        )
        with self.assertRaisesRegex(ManifestError, "region label 'dot product' must be"):
            self.load(tables=BUFFERS + LAUNCH + TRACE + regions.replace("main.loop", "dot product"))
        with self.assertRaisesRegex(ManifestError, "names site 2, but the manifest has 2"):
            self.load(tables=BUFFERS + LAUNCH + TRACE + "[[trace.regions]]\nenter = 0\nleave = 2\n")

    def test_stages_and_profilers(self) -> None:
        trace = TRACE.replace(
            "[trace]\n",
            '[trace]\nstages = ["xtrace", "range_profile", "pc_sampling"]\n'
            '[trace.range_profile]\nmetrics = ["sm__cycles_elapsed.avg"]\nmax_invocations = 4\n'
            "[trace.pc_sampling]\nrepeats = 3\n",
        )
        plan = self.plan(self.load(tables=BUFFERS + LAUNCH + trace))
        self.assertEqual(plan["stages"], ["xtrace", "range_profile", "pc_sampling"])
        self.assertEqual(plan["range_profile"], {"metrics": ["sm__cycles_elapsed.avg"], "max_invocations": 4})
        self.assertEqual(plan["pc_sampling"], {"repeats": 3})
        self.assertEqual(self.plan(self.load())["stages"], ["xtrace"])
        profile_only = self.plan(self.load(tables=BUFFERS + LAUNCH + '[trace]\nstages = ["pc_sampling"]\n[trace.pc_sampling]\nrepeats = 1\n'))
        self.assertEqual(profile_only["sites"], [])
        with self.assertRaisesRegex(ManifestError, r"\[trace.pc_sampling\] is given exactly when"):
            self.load(tables=BUFFERS + LAUNCH + TRACE.replace("[trace]\n", '[trace]\nstages = ["xtrace", "pc_sampling"]\n'))
        with self.assertRaisesRegex(ManifestError, "max_invocations must be an integer in \\[2, 128\\]"):
            self.load(tables=BUFFERS + LAUNCH + trace.replace("max_invocations = 4", "max_invocations = 1"))

    def test_planner_is_platform_planned(self) -> None:
        planner = '[planner]\nmodel = "planner-small"\nmax_calls = 3\nmax_input_tokens = 20000\nmax_output_tokens = 4000\n'
        request = self.request(self.load(tables=BUFFERS + LAUNCH + planner))
        self.assertEqual(
            request["analysis"],
            {
                "mode": "platform_planned", "model": "planner-small", "max_calls": 3,
                "max_input_tokens_total": 20000, "max_output_tokens_total": 4000, "capture_values": False,
            },
        )
        with self.assertRaisesRegex(ManifestError, "either \\[trace\\]"):
            self.load(tables=BUFFERS + LAUNCH + TRACE + planner)
        with self.assertRaisesRegex(ManifestError, "max_calls must be an integer in \\[1, 20\\]"):
            self.load(tables=BUFFERS + LAUNCH + planner.replace("max_calls = 3", "max_calls = 30"))

    def test_limits_default_to_the_capabilities(self) -> None:
        self.assertEqual(self.request(self.load())["limits"], CAPABILITIES["limits"])
        limits = "[limits]\nmax_cost_usd = 1.25\nmax_gpu_seconds = 60\nmax_wall_seconds = 600\nmax_attempts = 1\nmax_trace_bytes = 4096\n"
        self.assertEqual(
            self.request(self.load(tables=BUFFERS + LAUNCH + limits + TRACE))["limits"],
            {
                "max_total_cost_micro_usd": "1250000", "max_active_gpu_seconds": 60, "max_wall_seconds": 600,
                "max_attempts": 1, "max_trace_bytes": 4096,
            },
        )
        with self.assertRaisesRegex(ManifestError, "exceeds max_wall_seconds"):
            self.request(self.load(tables=BUFFERS + LAUNCH + "[limits]\nmax_gpu_seconds = 4000\n" + TRACE))
        with self.assertRaisesRegex(ManifestError, "at least \\$0.000001"):
            self.load(tables=BUFFERS + LAUNCH + "[limits]\nmax_cost_usd = 0\n" + TRACE)

    def test_removed_keys_are_refused(self) -> None:
        for top in ("timeout_seconds = 120\n", 'kind = "inspect"\n'):
            with self.subTest(top=top), self.assertRaisesRegex(ManifestError, "unknown key"):
                self.load(top=top)

    def test_paths_stay_inside(self) -> None:
        with self.assertRaisesRegex(ManifestError, "inside"):
            JobManifest.load(self.directory(TOP + KERNEL.replace('["matmul.cu"]', '["../secret.cu"]') + BUFFERS + LAUNCH + TRACE))

    def test_launch_manifest_bytes_are_canonical(self) -> None:
        manifest = self.load()
        self.assertEqual(manifest.launch_manifest_bytes("sm_90a", None), manifest.launch_manifest_bytes("sm_90a", None))
        self.assertEqual(json.loads(manifest.launch_manifest_bytes("sm_90a", None))["reset"], {"policy": "restore_pristine_before_every_invocation"})


if __name__ == "__main__":
    unittest.main()
