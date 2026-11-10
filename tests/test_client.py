import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from fake_control_plane import EXAMPLE_DIRECTORY, FIXTURE_DIRECTORY, TOP_UP_URL, FakeServer, agent_job
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import PlatformError
from xtrace_harness.workflow import XtraceWorkflow


def copy_example() -> str:
    """A copy of the example job directory, so results fetched into it stay local."""
    directory = tempfile.mkdtemp()
    shutil.copytree(EXAMPLE_DIRECTORY, directory, dirs_exist_ok=True)
    return directory


class ControlPlaneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeServer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.fake.reset()
        self.client = MarsComputeClient(Credentials(api_key="mc_live_v1_test", api_url=self.fake.origin))
        self.workflow = XtraceWorkflow(client=self.client, poll_seconds=0)
        self.directory = copy_example()
        self.addCleanup(shutil.rmtree, self.directory)

    def tearDown(self) -> None:
        self.assertEqual(self.fake.state.list_schema_error, [])

    def test_capabilities_and_credit(self) -> None:
        self.assertEqual(self.workflow.hardware()["compute"][0]["environment_id"], "h200-cuda12.9")
        self.assertEqual(self.workflow.credit()["top_up_url"], TOP_UP_URL)
        for request in self.fake.state.list_request:
            self.assertEqual(request["authorization"], "Bearer mc_live_v1_test")

    def test_single_upload(self) -> None:
        artifact = self.client.upload(data=b"abc", kind="cubin", name="k.cubin")
        self.assertEqual(artifact["sha256"], hashlib.sha256(b"abc").hexdigest())
        put = self.fake.state.requests_to("PUT", r"^/upload/")[0]
        self.assertIsNone(put["authorization"])
        self.assertEqual(put["body"], b"abc")
        complete = self.fake.state.requests_to("POST", r"^/v1/xtrace/artifacts/.*/complete$")[0]
        self.assertNotIn("parts", json.loads(complete["body"]))

    def test_multipart_upload_sends_etags(self) -> None:
        self.fake.state.multipart_threshold = 9
        self.fake.state.part_bytes = 4
        data = bytes(range(10))
        artifact = self.client.upload(data=data, kind="safetensors", name="t.safetensors")
        self.assertEqual(artifact["size_bytes"], 10)
        list_put = self.fake.state.requests_to("PUT", r"^/upload/.*/part/")
        self.assertEqual([put["body"] for put in list_put], [data[0:4], data[4:8], data[8:10]])
        complete = json.loads(self.fake.state.requests_to("POST", r".*/complete$")[0]["body"])
        self.assertEqual(
            complete["parts"],
            [{"part_number": number, "etag": f'"etag-{number}"'} for number in (1, 2, 3)],
        )

    def test_estimate_uploads_the_example_and_quotes_it(self) -> None:
        quoted = self.workflow.quote(self.directory)
        request = quoted.map_request
        self.assertEqual(request["schema_version"], "marscompute.xtrace.submit.v1")
        self.assertEqual(request["compute"]["gpu_type"], "H200")
        self.assertEqual(set(request["bundle"]), {"kind", "cubin_artifact_ids", "source_archive_artifact_id", "tensor_artifact_ids", "file_artifact_ids"})
        with tarfile.open(fileobj=io.BytesIO(self.fake.state.map_blob[request["bundle"]["source_archive_artifact_id"]]), mode="r:gz") as tar:
            self.assertEqual(tar.getnames(), ["matmul.cu"])
        self.assertEqual(quoted.summary()["max_cost_usd"], "$20.000000")
        self.assertEqual(quoted.summary()["gpu_usd_per_hour"], "$4.000000")

    def test_workspace_uploads_by_kind_and_quotes_the_fees(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        agent_job(directory)
        quoted = self.workflow.quote(directory)
        bundle = quoted.map_request["bundle"]
        self.assertEqual(bundle["kind"], "agent_workspace")
        self.assertEqual(
            [self.fake.state.map_artifact[artifact_id]["kind"] for artifact_id in [*bundle["cubin_artifact_ids"], bundle["source_archive_artifact_id"], *bundle["file_artifact_ids"]]],
            ["cubin", "source", "workspace_file"],
        )
        with tarfile.open(fileobj=io.BytesIO(self.fake.state.map_blob[bundle["source_archive_artifact_id"]]), mode="r:gz") as tar:
            self.assertEqual(tar.getnames(), ["matmul.cu"])
        self.assertEqual(quoted.map_request["analysis"]["mode"], "agent")
        self.assertEqual(quoted.map_request["compute"]["gpu_type"], "H200")
        summary = quoted.summary()
        self.assertEqual(summary["job_fee_usd"], "$0.500000")
        self.assertEqual(
            summary["agent_usd_per_million_tokens"],
            {
                "model": "claude-opus-5-5",
                "input": "$18.000000",
                "cached_input": "$1.800000",
                "output": "$90.000000",
                "map_cache_write": {"cache_write_300_explicit_tokens": "$22.500000"},
            },
        )
        job = self.workflow.submit(quoted=quoted, idempotency_key="agent-1")
        self.assertEqual(self.workflow.wait(job["job_id"])["job"]["state"], "succeeded")

    def test_submit_is_idempotent(self) -> None:
        quoted = self.workflow.quote(self.directory)
        job = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        again = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        self.assertEqual(job["job_id"], again["job_id"])
        for request in self.fake.state.requests_to("POST", r"^/v1/xtrace/jobs$"):
            self.assertEqual(request["idempotency"], "key-1")
            self.assertEqual(json.loads(request["body"])["idempotency_key"], "key-1")
        self.assertEqual(len(self.fake.state.map_job), 1)

    def test_insufficient_credit_names_the_top_up_page(self) -> None:
        self.fake.state.available_micro_usd = 1
        quoted = self.workflow.quote(self.directory)
        with self.assertRaises(PlatformError) as context:
            self.workflow.submit(quoted)
        self.assertEqual(context.exception.status, 402)
        self.assertEqual(context.exception.code, "insufficient_credit")
        self.assertEqual(context.exception.top_up_url, TOP_UP_URL)
        self.assertIn(TOP_UP_URL, str(context.exception))
        self.assertIn("corr-1", str(context.exception))

    def test_wait_then_fetch(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        view = self.workflow.wait(job["job_id"])
        self.assertEqual(view["job"]["state"], "succeeded")
        output = os.path.join(self.directory, "out")
        list_path = self.workflow.fetch(job_id=job["job_id"], output_directory=output)
        self.assertEqual(
            sorted(os.path.basename(path) for path in list_path),
            ["result-manifest.json", "xtrace.json"],
        )
        with open(os.path.join(output, "xtrace.json"), "rb") as trace_file, open(os.path.join(FIXTURE_DIRECTORY, "trace_sass_regions.json"), "rb") as fixture_file:
            self.assertEqual(trace_file.read(), fixture_file.read())
        with open(os.path.join(output, "result-manifest.json")) as manifest_file:
            self.assertEqual(json.load(manifest_file)["quality"], "usable")

    def test_wait_returns_at_the_timeout(self) -> None:
        self.fake.state.job_polls = 1000
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(self.workflow.wait(job_id=job["job_id"], timeout_seconds=0)["job"]["state"], "running")

    def test_list_pages_by_cursor(self) -> None:
        quoted = self.workflow.quote(self.directory)
        list_id = [self.workflow.submit(quoted)["job_id"] for _ in range(3)]
        first = self.workflow.jobs(limit=2)
        second = self.workflow.jobs(limit=2, cursor=first["next_cursor"])
        self.assertEqual(
            [job["job_id"] for job in first["items"] + second["items"]],
            list(reversed(list_id)),
        )
        self.assertIsNone(second["next_cursor"])

    def test_cancel_sends_an_empty_object(self) -> None:
        self.fake.state.job_polls = 1000
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(self.workflow.cancel(job["job_id"])["state"], "cancelled")
        cancel = self.fake.state.requests_to("POST", r".*/cancel$")[0]
        self.assertEqual(json.loads(cancel["body"]), {})
        self.assertEqual(cancel["content_type"], "application/json")

    def test_unknown_job_is_a_platform_error(self) -> None:
        with self.assertRaises(PlatformError) as context:
            self.workflow.job("missing")
        self.assertEqual(context.exception.code, "not_found")

    def test_download_checks_the_digest(self) -> None:
        artifact_id = self.client.upload(data=b"abc", kind="cubin", name="k.cubin")["artifact_id"]
        grant = {
            "download_url": f"{self.fake.origin}/download/{artifact_id}",
            "sha256": "0" * 64,
            "size_bytes": 3,
            "name": "k.cubin",
        }
        with self.assertRaisesRegex(PlatformError, "different bytes"):
            self.client.download(grant)


if __name__ == "__main__":
    unittest.main()
