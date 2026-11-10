import io
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import unittest
from fake_service import EXAMPLE_DIRECTORY, HARDWARE, RequestSchema, ServiceError
from marstrace.binary import KernelBinary
from marstrace.error import ManifestError, MarsTraceError
from marstrace.job import JobManifest
from marstrace.tensors import TensorFile

UPLOAD_IDS = {
    "cubin": ["upl_" + "c" * 26],
    "source_archive": ["upl_" + "s" * 26],
    "launch_manifest": ["upl_" + "m" * 26],
    "tensor": ["upl_" + "t" * 26],
    "file": ["upl_" + "f" * 26],
}


def write_safetensors(path: str, map_header: dict) -> None:
    """Write a safetensors file with zeroed data behind a header."""
    header = json.dumps(map_header).encode("utf-8")
    data_bytes = max(entry["data_offsets"][1] for entry in map_header.values())
    with open(path, "wb") as tensor_file:
        tensor_file.write(struct.pack("<Q", len(header)))
        tensor_file.write(header)
        tensor_file.write(bytes(data_bytes))


AGENT_TOP = 'hardware = "h200"\nintent = "find the longest bubble of the mainloop"\n'
WORKSPACE = (
    '[workspace]\ncubins = ["matmul.cubin"]\nsource = "src"\nmanifest = "launch.json"\n'
    'tensors = ["inputs.safetensors"]\nfiles = ["run.py"]\n'
)
AGENT = (
    "[agent]\nmax_turns = 200\nmax_input_tokens = 20000000\n"
    "max_output_tokens = 200000\nmax_wall_seconds = 2400\n"
)
LIMITS = "[limits]\nmax_attempts = 3\n"


class TranslationTest(unittest.TestCase):
    """The mapping of marstrace.toml onto a MarsTrace job request."""

    def directory(self, manifest: str) -> str:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        shutil.copyfile(os.path.join(EXAMPLE_DIRECTORY, "matmul.cubin"), os.path.join(directory, "matmul.cubin"))
        for relative_path, text in (
            ("src/matmul.cu", "// kernel\n"),
            ("src/include/common.h", "#pragma once\n"),
            ("src/build.sh", "nvcc -cubin matmul.cu\n"),
            ("src/.git/HEAD", "ref: refs/heads/main\n"),
            ("src/sub/.git", "gitdir: ../../.git/modules/sub\n"),
            ("run.py", "print('drive')\n"),
        ):
            os.makedirs(os.path.dirname(os.path.join(directory, relative_path)), exist_ok=True)
            with open(os.path.join(directory, relative_path), "w") as source_file:
                source_file.write(text)
        os.chmod(os.path.join(directory, "src", "build.sh"), 0o775)
        with open(os.path.join(directory, "launch.json"), "w") as launch_file:
            json.dump({"schema_version": "marscompute.kernel-launch"}, launch_file)
        write_safetensors(os.path.join(directory, "inputs.safetensors"), {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}})
        with open(os.path.join(directory, "marstrace.toml"), "w") as manifest_file:
            manifest_file.write(manifest)
        return directory

    def load(self, manifest: str = AGENT_TOP + WORKSPACE + AGENT + LIMITS) -> JobManifest:
        return JobManifest.load(self.directory(manifest))

    def request(self, manifest: JobManifest) -> dict:
        """The job request, checked against the service's request schema."""
        request = manifest.request(
            map_hardware=HARDWARE,
            map_upload_ids=dict(UPLOAD_IDS),
        )
        try:
            RequestSchema.parse(request, RequestSchema.JOB)
        except ServiceError as err:
            self.fail(f"the service refuses the request: {err}")
        return request

    def test_example_translates(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        self.assertEqual([job_input.role for job_input in manifest.list_input()], ["cubin", "source_archive"])
        request = self.request(manifest)
        self.assertEqual(request["hardware"], "h200")
        self.assertIn("square_matrix_mul", request["prompt"])
        self.assertEqual(request["limits"], {"max_attempts": 2})

    def test_workspace_and_agent_are_the_inputs_and_agent(self) -> None:
        manifest = self.load()
        self.assertEqual(
            [(job_input.role, os.path.basename(job_input.path)) for job_input in manifest.list_input()],
            [
                ("cubin", "matmul.cubin"), ("source_archive", "source.tar.gz"), ("launch_manifest", "launch.json"),
                ("tensor", "inputs.safetensors"), ("file", "run.py"),
            ],
        )
        request = self.request(manifest)
        self.assertEqual(
            request["inputs"],
            {
                "cubins": UPLOAD_IDS["cubin"], "source_archive": UPLOAD_IDS["source_archive"][0],
                "launch_manifest": UPLOAD_IDS["launch_manifest"][0], "tensors": UPLOAD_IDS["tensor"], "files": UPLOAD_IDS["file"],
            },
        )
        self.assertEqual(
            request["agent"],
            {"max_turns": 200, "max_input_tokens": 20000000, "max_output_tokens": 200000, "max_wall_seconds": 2400},
        )
        self.assertEqual((request["prompt"], request["limits"]), ("find the longest bubble of the mainloop", {"max_attempts": 3}))

    def test_optional_parts_stay_out_of_the_inputs(self) -> None:
        manifest = self.load(AGENT_TOP + '[workspace]\ncubins = ["matmul.cubin"]\n' + LIMITS)
        self.assertEqual([job_input.role for job_input in manifest.list_input()], ["cubin"])
        self.assertEqual(JobManifest.inputs({"cubin": UPLOAD_IDS["cubin"]}), {"cubins": UPLOAD_IDS["cubin"], "tensors": [], "files": []})
        self.assertEqual(self.request(manifest)["agent"], HARDWARE["agent"]["defaults"])

    def test_source_is_archived_without_git_and_reproducibly(self) -> None:
        manifest = self.load()
        data = manifest.workspace.archive_source(manifest.directory)
        self.assertEqual(data, manifest.workspace.archive_source(manifest.directory))
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            map_member = {member.name: member for member in tar.getmembers()}
        self.assertEqual(sorted(map_member), ["build.sh", "include", "include/common.h", "matmul.cu", "sub"])
        self.assertEqual((map_member["build.sh"].mode, map_member["matmul.cu"].mode), (0o755, 0o644))
        self.assertEqual({(member.uid, member.mtime) for member in map_member.values()}, {(0, 0)})

    def test_hardware_must_be_offered_and_run_every_cubin(self) -> None:
        manifest = self.load()
        self.assertEqual(manifest.select_hardware(HARDWARE)["architecture"], "sm_90a")
        with self.assertRaisesRegex(ManifestError, "no hardware 'a100'; .* h200, b200, b300"):
            self.load(AGENT_TOP.replace("h200", "a100") + WORKSPACE + AGENT + LIMITS).select_hardware(HARDWARE)
        with self.assertRaisesRegex(ManifestError, "b300 is not offered now"):
            self.load(AGENT_TOP.replace("h200", "b300") + WORKSPACE + AGENT + LIMITS).select_hardware(HARDWARE)
        with self.assertRaisesRegex(ManifestError, "matmul.cubin targets sm_90, but b200 is sm_100a"):
            self.load(AGENT_TOP.replace("h200", "b200") + WORKSPACE + AGENT + LIMITS).select_hardware(HARDWARE)

    def test_agent_budgets_are_bounded(self) -> None:
        for key, value, message in (
            ("max_turns", 501, r"max_turns must be an integer in \[1, 500\]"),
            ("max_input_tokens", 50000001, r"max_input_tokens must be an integer in \[1, 50000000\]"),
            ("max_output_tokens", 0, r"max_output_tokens must be an integer in \[1, 1000000\]"),
            ("max_wall_seconds", 2700, r"max_wall_seconds must be an integer in \[60, 2580\]"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ManifestError, message):
                self.load(AGENT_TOP + WORKSPACE + re.sub(f"{key} = \\d+", f"{key} = {value}", AGENT) + LIMITS)
        with self.assertRaisesRegex(ManifestError, r"max_turns 400 exceeds the service's maximum 300"):
            self.load(AGENT_TOP + WORKSPACE + AGENT.replace("200\n", "400\n", 1) + LIMITS).request(
                map_hardware={**HARDWARE, "agent": {**HARDWARE["agent"], "maximums": {**HARDWARE["agent"]["maximums"], "max_turns": 300}}},
                map_upload_ids=dict(UPLOAD_IDS),
            )
        with self.assertRaisesRegex(ManifestError, r"unknown key\(s\) model"):
            self.load(AGENT_TOP + WORKSPACE + AGENT + 'model = "claude-opus-5-5"\n' + LIMITS)

    def test_limits_hold_only_the_attempts(self) -> None:
        manifest = self.load(AGENT_TOP + WORKSPACE + AGENT)
        self.assertNotIn("limits", self.request(manifest))
        self.assertEqual(manifest.limits.attempts(), 2)
        self.assertNotIn("limits", self.request(self.load(AGENT_TOP + WORKSPACE + AGENT + "[limits]\n")))
        self.assertEqual(self.request(self.load(AGENT_TOP + WORKSPACE + AGENT + "[limits]\nmax_attempts = 1\n"))["limits"], {"max_attempts": 1})
        for limits, message in (
            ("[limits]\nmax_attempts = 4\n", r"max_attempts must be an integer in \[1, 3\]"),
            ("[limits]\nmax_attempts = 0\n", r"max_attempts must be an integer in \[1, 3\]"),
            ("[limits]\nmax_usd = 20.0\n", r"unknown key\(s\) max_usd"),
            ("[limits]\nmax_gpu_seconds = 60\n", r"unknown key\(s\) max_gpu_seconds"),
        ):
            with self.subTest(limits=limits), self.assertRaisesRegex(ManifestError, message):
                self.load(AGENT_TOP + WORKSPACE + AGENT + limits)

    def test_inputs_are_held_to_the_size_limits(self) -> None:
        manifest = self.load()
        with open(os.path.join(manifest.directory, "run.py"), "wb") as script_file:
            script_file.truncate(16 * (1 << 20) + 1)
        with self.assertRaisesRegex(ManifestError, "a file holds at most 16 MiB"):
            manifest.list_input()

    def test_ptx_is_refused(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as ptx_file:
            ptx_file.write(".version 8.5\n.target sm_90a\n.visible .entry k(\n)\n{}\n")
        self.addCleanup(os.unlink, ptx_file.name)
        with self.assertRaisesRegex(MarsTraceError, "PTX"):
            KernelBinary.load(ptx_file.name)

    def test_tensors_are_read_by_their_header(self) -> None:
        manifest = self.load()
        tensor_file = TensorFile.load(os.path.join(manifest.directory, "inputs.safetensors"))
        self.assertEqual((tensor_file.map_tensor["a"].dtype, tensor_file.map_tensor["a"].list_shape), ("f32", [4]))
        write_safetensors(os.path.join(manifest.directory, "inputs.safetensors"), {"a": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}})
        with self.assertRaisesRegex(ManifestError, "unsupported dtype C64"):
            JobManifest.load(manifest.directory)

    def test_workspace_errors_are_precise(self) -> None:
        for manifest, message in (
            (AGENT_TOP + AGENT + LIMITS, r"\[workspace\] is required"),
            (AGENT_TOP + WORKSPACE + AGENT + LIMITS + '[kernel]\nbinary = "matmul.cubin"\n', r"unknown key\(s\) kernel"),
            ('timeout_seconds = 120\n' + AGENT_TOP + WORKSPACE + AGENT + LIMITS, r"unknown key\(s\) timeout_seconds"),
            (AGENT_TOP + WORKSPACE + AGENT + LIMITS + "[compute]\ngpu = 1\n", r"unknown key\(s\) compute"),
            (AGENT_TOP.replace("intent", "purpose") + WORKSPACE + AGENT + LIMITS, "unknown key"),
            ('hardware = "h200"\n' + WORKSPACE + AGENT + LIMITS, "'intent' is required"),
            ('hardware = "h200"\nintent = "' + "x" * 32001 + '"\n' + WORKSPACE + AGENT + LIMITS, "at most 32000 characters"),
            ('hardware = "h200"\nintent = "a\\u0000b"\n' + WORKSPACE + AGENT + LIMITS, "NUL"),
            (AGENT_TOP + WORKSPACE.replace('cubins = ["matmul.cubin"]', "cubins = []") + AGENT + LIMITS, "cubins lists 1 to 8 paths"),
            (AGENT_TOP + WORKSPACE.replace('files = ["run.py"]', 'files = ["run.py", "./run.py"]') + AGENT + LIMITS, "names ./run.py twice"),
            (AGENT_TOP + WORKSPACE.replace('source = "src"', 'source = "../src"') + AGENT + LIMITS, "must lie inside the job directory"),
            (AGENT_TOP + WORKSPACE.replace('source = "src"', 'source = "run.py"') + AGENT + LIMITS, "run.py is not a directory"),
            (AGENT_TOP + WORKSPACE.replace('manifest = "launch.json"', 'manifest = "src/matmul.cu"') + AGENT + LIMITS, "is not JSON"),
            (AGENT_TOP + WORKSPACE.replace('tensors = ["inputs.safetensors"]', 'tensors = ["src/build.sh"]') + AGENT + LIMITS, "is not a safetensors file"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ManifestError, message):
                self.load(manifest)
        directory = self.directory(AGENT_TOP + WORKSPACE + AGENT + LIMITS)
        with open(os.path.join(directory, "launch.json"), "w") as launch_file:
            json.dump({"schema_version": "marscompute.kernel-launch.v1"}, launch_file)
        with self.assertRaisesRegex(ManifestError, "launch.json is not a marscompute.kernel-launch manifest"):
            JobManifest.load(directory)


if __name__ == "__main__":
    unittest.main()
