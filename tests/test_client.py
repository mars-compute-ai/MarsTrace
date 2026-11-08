import hashlib
import json
import os
import shutil
import tempfile
import unittest
from fake_control_plane import EXAMPLE_DIRECTORY, TOP_UP_URL, FakeControlPlane, FakeServer
from xtrace_harness.binary import KernelBinary
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import HarnessError, PlatformError
from xtrace_harness.listing import InspectionCache, KernelListing
from xtrace_harness.tensors import TensorFile
from xtrace_harness.workflow import XtraceWorkflow


def copy_example() -> str:
    """A copy of the example job directory, so caches written beside the cubin stay local."""
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
        self.assertEqual(self.workflow.hardware()["compute"][0]["environment_id"], "h100-sxm")
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

    def inspect(self, kernel_symbol: str | None = "square_matrix_mul", idempotency_key: str | None = None) -> str:
        """Run a whole inspection of the example cubin and return the cached result's path."""
        binary = os.path.join(self.directory, "matmul.cubin")
        quoted = self.workflow.quote_inspection(binary_path=binary, kernel_symbol=kernel_symbol)
        status = self.workflow.submit_inspection(quoted=quoted, idempotency_key=idempotency_key)
        self.workflow.wait_inspection(status["inspection_id"])
        return self.workflow.fetch_inspection(
            inspection_id=status["inspection_id"],
            binary_path=binary,
            kernel_symbol=kernel_symbol,
        )

    def test_inspection_is_quoted_accepted_and_cached_beside_the_cubin(self) -> None:
        binary = os.path.join(self.directory, "matmul.cubin")
        quoted = self.workflow.quote_inspection(binary_path=binary, kernel_symbol="square_matrix_mul")
        self.assertEqual(quoted.summary()["fee_usd"], "$0.020000")
        self.assertEqual(quoted.summary()["profile_id"], "static-cubin-v1")
        request = json.loads(self.fake.state.requests_to("POST", r"^/v1/xtrace/inspections/quotes$")[0]["body"])
        self.assertEqual(request["kernel_selector"], {"kind": "symbol", "symbol": "square_matrix_mul"})
        self.assertEqual(request["binary"]["sha256"], KernelBinary.load(binary).sha256)
        self.assertEqual(self.fake.state.map_inspection, {})
        status = self.workflow.submit_inspection(quoted=quoted, idempotency_key="inspect-1")
        accept = self.fake.state.requests_to("POST", r"^/v1/xtrace/inspections$")[0]
        self.assertEqual(accept["idempotency"], "inspect-1")
        self.assertEqual(json.loads(accept["body"])["quote_sha256"], FakeControlPlane.digest(quoted.map_quote))
        self.assertEqual(self.workflow.wait_inspection(status["inspection_id"])["execution"]["outcome"], "completed")
        path = self.workflow.fetch_inspection(
            inspection_id=status["inspection_id"],
            binary_path=binary,
            kernel_symbol="square_matrix_mul",
        )
        self.assertTrue(path.startswith(os.path.join(self.directory, InspectionCache.DIRECTORY_NAME)))
        listing = KernelListing.load(path)
        self.assertEqual(listing.inspected_kernel().list_instruction[199].opcode, "STG.E")
        self.assertEqual(
            InspectionCache.load(binary_path=binary, sha256=listing.binary_sha256, kernel_symbol="square_matrix_mul").inspection_id,
            status["inspection_id"],
        )

    def test_inspection_acceptance_is_idempotent(self) -> None:
        binary = os.path.join(self.directory, "matmul.cubin")
        quoted = self.workflow.quote_inspection(binary_path=binary, kernel_symbol=None)
        first = self.workflow.submit_inspection(quoted=quoted, idempotency_key="inspect-1")
        again = self.workflow.submit_inspection(quoted=quoted, idempotency_key="inspect-1")
        self.assertEqual(first["inspection_id"], again["inspection_id"])
        self.assertEqual(len(self.fake.state.map_inspection), 1)

    def test_kernel_catalog_downloads_no_listing(self) -> None:
        path = self.inspect(kernel_symbol=None)
        self.assertFalse(os.path.isfile(InspectionCache.instructions_path_of(path)))
        self.assertEqual(self.fake.state.requests_to("POST", r".*/artifacts/downloads$"), [])
        self.assertEqual(list(KernelListing.load(path).map_kernel), ["square_matrix_mul"])

    def test_failed_inspection_names_its_outcome(self) -> None:
        self.fake.state.inspection_outcome = "rejected_input"
        with self.assertRaisesRegex(HarnessError, "ended rejected_input"):
            self.inspect()

    def test_unavailable_inspections_are_refused_before_upload(self) -> None:
        self.fake.state.is_inspection_available = False
        with self.assertRaisesRegex(HarnessError, "inspections are unavailable: not_configured"):
            self.inspect()
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/xtrace/artifacts/uploads$"), [])

    def test_inspection_list_cancel_and_downloads(self) -> None:
        self.fake.state.inspection_polls = 1000
        binary = os.path.join(self.directory, "matmul.cubin")
        list_id = [
            self.workflow.submit_inspection(self.workflow.quote_inspection(binary_path=binary, kernel_symbol=None))["inspection_id"]
            for _ in range(3)
        ]
        first = self.workflow.inspections(limit=2)
        second = self.workflow.inspections(limit=2, cursor=first["next_cursor"])
        self.assertEqual([status["inspection_id"] for status in first["items"] + second["items"]], list_id)
        self.assertIsNone(second["next_cursor"])
        self.assertEqual(self.workflow.cancel_inspection(list_id[0])["execution"]["state"], "cancel_requested")
        with self.assertRaisesRegex(HarnessError, "is still cancel_requested"):
            self.workflow.fetch_inspection(inspection_id=list_id[0], binary_path=binary, kernel_symbol=None)
        self.fake.state.inspection_polls = 0
        self.workflow.wait_inspection(list_id[1])
        result = self.client.inspection_result(list_id[1])
        artifact_id = result["kernel_catalog"]["kernels"][0]["instructions"]["artifact"]["artifact_id"]
        (path,) = self.workflow.download_inspection(
            inspection_id=list_id[1],
            list_artifact_id=[artifact_id],
            output_directory=os.path.join(self.directory, "inspection"),
        )
        with open(path) as listing_file:
            self.assertEqual(json.load(listing_file)["symbol"], "square_matrix_mul")

    def test_estimate_uploads_resolves_and_quotes(self) -> None:
        quoted = self.workflow.quote(self.directory)
        request = quoted.map_request
        self.assertEqual(request["schema_version"], "marscompute.xtrace.submit.v1")
        self.assertEqual(request["compute"]["gpu_type"], "H100")
        self.assertEqual(
            [site["instruction_index"] for site in request["analysis"]["trace_plan"]["sites"]],
            [13, 199],
        )
        self.assertEqual(
            set(request["bundle"]),
            {"kind", "cubin_artifact_id", "manifest_artifact_id", "tensors_artifact_id"},
        )
        manifest = json.loads(self.fake.state.map_blob[request["bundle"]["manifest_artifact_id"]])
        self.assertEqual(manifest["target"]["architecture"], "sm_90a")
        self.assertIsNone(manifest["parameter_layout"])
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tensors_file:
            tensors_file.write(self.fake.state.map_blob[request["bundle"]["tensors_artifact_id"]])
            tensors_file.flush()
            self.assertEqual(sorted(TensorFile.load(tensors_file.name).map_tensor), ["a", "b"])
        self.assertEqual(quoted.summary()["max_cost_usd"], "$2.000000")
        self.assertEqual(quoted.summary()["gpu_usd_per_hour"], "$3.200000")

    def test_estimate_carries_the_cached_parameter_layout(self) -> None:
        self.inspect()
        request = self.workflow.quote(self.directory).map_request
        manifest = json.loads(self.fake.state.map_blob[request["bundle"]["manifest_artifact_id"]])
        self.assertEqual(
            manifest["parameter_layout"],
            [{"index": index, "offset": 8 * index, "size_bytes": 8 if index < 3 else 4} for index in range(4)],
        )

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
            ["result-manifest.json", "summary.json", "xtrace.json"],
        )
        with open(os.path.join(output, "summary.json")) as summary_file:
            self.assertEqual(json.load(summary_file)["quality"], "usable")

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
