import io
import json
import os
import re
import shutil
import struct
import tarfile
import tempfile
import unittest
from fake_control_plane import CAPABILITIES, EXAMPLE_DIRECTORY
from schema_check import SchemaCheck
from xtrace_harness.binary import KernelBinary
from xtrace_harness.error import HarnessError, ManifestError
from xtrace_harness.job import JobManifest
from xtrace_harness.tensors import TensorFile

ARTIFACT_IDS = {
    "cubin": ["cubin-1"],
    "source": ["source-1"],
    "launch_manifest": ["manifest-1"],
    "safetensors": ["tensors-1"],
    "workspace_file": ["file-1"],
}


def write_safetensors(path: str, map_header: dict) -> None:
    """Write a safetensors file with zeroed data behind a header."""
    header = json.dumps(map_header).encode("utf-8")
    data_bytes = max(entry["data_offsets"][1] for entry in map_header.values())
    with open(path, "wb") as tensor_file:
        tensor_file.write(struct.pack("<Q", len(header)))
        tensor_file.write(header)
        tensor_file.write(bytes(data_bytes))


AGENT_TOP = 'hardware = "h200-cuda12.9"\nintent = "find the longest bubble of the mainloop"\n'
WORKSPACE = (
    '[workspace]\ncubins = ["matmul.cubin"]\nsource = "src"\nmanifest = "launch.json"\n'
    'tensors = ["inputs.safetensors"]\nfiles = ["run.py"]\n'
)
AGENT = (
    '[agent]\nmodel = "claude-opus-5-5"\nmax_turns = 200\nmax_input_tokens = 20000000\n'
    "max_output_tokens = 200000\nmax_wall_seconds = 2400\n"
)


class TranslationTest(unittest.TestCase):
    """The mapping of xtrace.toml onto an agent_workspace bundle and an agent analysis."""

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
        with open(os.path.join(directory, "xtrace.toml"), "w") as manifest_file:
            manifest_file.write(manifest)
        return directory

    def load(self, manifest: str = AGENT_TOP + WORKSPACE + AGENT) -> JobManifest:
        return JobManifest.load(self.directory(manifest))

    def request(self, manifest: JobManifest) -> dict:
        """The estimate input, checked against the vendored schema."""
        request = manifest.estimate_input(
            map_capabilities=CAPABILITIES,
            map_artifact_ids=dict(ARTIFACT_IDS),
        )
        SchemaCheck.validate("estimate-input", request)
        return request

    def test_example_translates(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        self.assertEqual([artifact.kind for artifact in manifest.list_artifact()], ["cubin", "source"])
        request = self.request(manifest)
        self.assertEqual(request["compute"]["gpu_type"], "H200")
        self.assertIn("square_matrix_mul", request["prompt"])
        self.assertEqual(request["analysis"]["mode"], "agent")

    def test_workspace_and_agent_are_the_bundle_and_analysis(self) -> None:
        manifest = self.load()
        list_artifact = manifest.list_artifact()
        self.assertEqual(
            [(artifact.kind, os.path.basename(artifact.path)) for artifact in list_artifact],
            [
                ("cubin", "matmul.cubin"), ("source", "source.tar.gz"), ("launch_manifest", "launch.json"),
                ("safetensors", "inputs.safetensors"), ("workspace_file", "run.py"),
            ],
        )
        request = self.request(manifest)
        self.assertEqual(
            request["bundle"],
            {
                "kind": "agent_workspace", "cubin_artifact_ids": ["cubin-1"], "source_archive_artifact_id": "source-1",
                "launch_manifest_artifact_id": "manifest-1", "tensor_artifact_ids": ["tensors-1"], "file_artifact_ids": ["file-1"],
            },
        )
        self.assertEqual(
            request["analysis"],
            {
                "mode": "agent", "model": "claude-opus-5-5", "max_turns": 200, "max_input_tokens_total": 20000000,
                "max_output_tokens_total": 200000, "max_wall_seconds": 2400,
            },
        )
        self.assertEqual((request["prompt"], request["compute"]["gpu_type"]), ("find the longest bubble of the mainloop", "H200"))

    def test_optional_parts_stay_out_of_the_bundle(self) -> None:
        manifest = self.load(AGENT_TOP + '[workspace]\ncubins = ["matmul.cubin"]\n' + AGENT)
        self.assertEqual([artifact.kind for artifact in manifest.list_artifact()], ["cubin"])
        self.assertEqual(
            manifest.bundle({"cubin": ["cubin-1"]}),
            {"kind": "agent_workspace", "cubin_artifact_ids": ["cubin-1"], "tensor_artifact_ids": [], "file_artifact_ids": []},
        )

    def test_source_is_archived_without_git_and_reproducibly(self) -> None:
        manifest = self.load()
        data = manifest.workspace.archive_source(manifest.directory)
        self.assertEqual(data, manifest.workspace.archive_source(manifest.directory))
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            map_member = {member.name: member for member in tar.getmembers()}
        self.assertEqual(sorted(map_member), ["build.sh", "include", "include/common.h", "matmul.cu", "sub"])
        self.assertEqual((map_member["build.sh"].mode, map_member["matmul.cu"].mode), (0o755, 0o644))
        self.assertEqual({(member.uid, member.mtime) for member in map_member.values()}, {(0, 0)})

    def test_hardware_must_run_every_cubin_and_offer_the_model(self) -> None:
        manifest = self.load()
        self.assertEqual(manifest.select_compute(CAPABILITIES)["environment_id"], "h200-cuda12.9")
        with self.assertRaisesRegex(ManifestError, "no hardware 'a100'; .* h200-cuda12.9, b200-cuda12.9, b300-cuda13.0"):
            self.load(AGENT_TOP.replace("h200-cuda12.9", "a100") + WORKSPACE + AGENT).select_compute(CAPABILITIES)
        with self.assertRaisesRegex(ManifestError, "not enabled"):
            manifest.select_compute(dict(CAPABILITIES, enabled=False))
        with self.assertRaisesRegex(ManifestError, "matmul.cubin targets sm_90, but b300-cuda13.0 is sm_103a"):
            self.load(AGENT_TOP.replace("h200-cuda12.9", "b300-cuda13.0") + WORKSPACE + AGENT).select_compute(CAPABILITIES)
        with self.assertRaisesRegex(ManifestError, "agent model 'other-model' is not offered; .* claude-opus-5-5"):
            self.load(AGENT_TOP + WORKSPACE + AGENT.replace("claude-opus-5-5", "other-model")).select_compute(CAPABILITIES)

    def test_agent_budgets_are_bounded(self) -> None:
        for key, value, message in (
            ("max_turns", 501, r"max_turns must be an integer in \[1, 500\]"),
            ("max_input_tokens", 50000001, r"max_input_tokens must be an integer in \[1, 50000000\]"),
            ("max_output_tokens", 0, r"max_output_tokens must be an integer in \[1, 1000000\]"),
            ("max_wall_seconds", 2700, r"max_wall_seconds must be an integer in \[60, 2580\]"),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ManifestError, message):
                self.load(AGENT_TOP + WORKSPACE + re.sub(f"{key} = \\d+", f"{key} = {value}", AGENT))

    def test_limits_default_to_the_capabilities(self) -> None:
        self.assertEqual(self.request(self.load())["limits"], CAPABILITIES["limits"])
        limits = "[limits]\nmax_cost_usd = 1.25\nmax_gpu_seconds = 60\nmax_wall_seconds = 600\nmax_attempts = 1\nmax_trace_bytes = 4096\n"
        self.assertEqual(
            self.request(self.load(AGENT_TOP + WORKSPACE + AGENT + limits))["limits"],
            {
                "max_total_cost_micro_usd": "1250000", "max_active_gpu_seconds": 60, "max_wall_seconds": 600,
                "max_attempts": 1, "max_trace_bytes": 4096,
            },
        )
        with self.assertRaisesRegex(ManifestError, "exceeds max_wall_seconds"):
            self.request(self.load(AGENT_TOP + WORKSPACE + AGENT + "[limits]\nmax_gpu_seconds = 4000\n"))
        with self.assertRaisesRegex(ManifestError, "at least \\$0.000001"):
            self.load(AGENT_TOP + WORKSPACE + AGENT + "[limits]\nmax_cost_usd = 0\n")

    def test_ptx_is_refused(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as ptx_file:
            ptx_file.write(".version 8.5\n.target sm_90a\n.visible .entry k(\n)\n{}\n")
        self.addCleanup(os.unlink, ptx_file.name)
        with self.assertRaisesRegex(HarnessError, "PTX"):
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
            (AGENT_TOP + WORKSPACE, r"\[agent\] is required"),
            (AGENT_TOP + AGENT, r"\[workspace\] is required"),
            (AGENT_TOP + WORKSPACE + AGENT + '[kernel]\nbinary = "matmul.cubin"\n', r"unknown key\(s\) kernel"),
            ('timeout_seconds = 120\n' + AGENT_TOP + WORKSPACE + AGENT, r"unknown key\(s\) timeout_seconds"),
            (AGENT_TOP.replace("intent", "purpose") + WORKSPACE + AGENT, "unknown key"),
            ('hardware = "h200-cuda12.9"\n' + WORKSPACE + AGENT, "'intent' is required"),
            ('hardware = "h200-cuda12.9"\nintent = "' + "x" * 32001 + '"\n' + WORKSPACE + AGENT, "at most 32000 characters"),
            (AGENT_TOP + WORKSPACE.replace('cubins = ["matmul.cubin"]', "cubins = []") + AGENT, "cubins lists 1 to 8 paths"),
            (AGENT_TOP + WORKSPACE.replace('files = ["run.py"]', 'files = ["run.py", "./run.py"]') + AGENT, "names ./run.py twice"),
            (AGENT_TOP + WORKSPACE.replace('source = "src"', 'source = "../src"') + AGENT, "must lie inside the job directory"),
            (AGENT_TOP + WORKSPACE.replace('source = "src"', 'source = "run.py"') + AGENT, "run.py is not a directory"),
            (AGENT_TOP + WORKSPACE.replace('manifest = "launch.json"', 'manifest = "src/matmul.cu"') + AGENT, "is not JSON"),
            (AGENT_TOP + WORKSPACE.replace('tensors = ["inputs.safetensors"]', 'tensors = ["src/build.sh"]') + AGENT, "is not a safetensors file"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ManifestError, message):
                self.load(manifest)
        directory = self.directory(AGENT_TOP + WORKSPACE + AGENT)
        with open(os.path.join(directory, "launch.json"), "w") as launch_file:
            json.dump({"schema_version": "marscompute.kernel-launch.v1"}, launch_file)
        with self.assertRaisesRegex(ManifestError, "launch.json is not a marscompute.kernel-launch manifest"):
            JobManifest.load(directory)


if __name__ == "__main__":
    unittest.main()
