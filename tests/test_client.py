import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock
from fake_service import ADD_CREDIT_URL, API_KEY, EXAMPLE_DIRECTORY, FIXTURE_DIRECTORY, FakeServer, agent_job
from marstrace.client import Credentials, MarsTraceClient
from marstrace.error import MarsTraceError, ServiceError
from marstrace.schema import ResponseSchema
from marstrace.workflow import MarsTraceWorkflow


def copy_example() -> str:
    """A copy of the example job directory, so results fetched into it stay local."""
    directory = tempfile.mkdtemp()
    shutil.copytree(EXAMPLE_DIRECTORY, directory, dirs_exist_ok=True)
    return directory


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeServer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.fake.reset()
        self.client = MarsTraceClient(Credentials(api_key=API_KEY, api_url=self.fake.origin))
        self.workflow = MarsTraceWorkflow(client=self.client)
        self.directory = copy_example()
        self.addCleanup(shutil.rmtree, self.directory)

    def tearDown(self) -> None:
        self.assertEqual(self.fake.state.list_contract_error, [])

    def test_hardware_and_credit(self) -> None:
        self.assertEqual([entry["id"] for entry in self.workflow.hardware()["hardware"]], ["h200", "b200", "b300"])
        self.assertEqual(self.workflow.credit()["add_credit_url"], ADD_CREDIT_URL)
        for request in self.fake.state.list_request:
            self.assertEqual(request["authorization"], f"Bearer {API_KEY}")
            self.assertEqual(request["headers"]["user-agent"], "mars-trace")

    def test_environment_needs_a_marstrace_key(self) -> None:
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": "not-a-marstrace-key"}), self.assertRaisesRegex(ServiceError, "mt_live_"):
            Credentials.from_environment()
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": API_KEY}, clear=True):
            self.assertEqual(Credentials.from_environment().api_url, "https://trace.marscompute.ai")

    def test_single_upload_is_checksummed(self) -> None:
        upload = self.client.upload(data=b"abc", role="cubin", name="k.cubin")
        self.assertEqual(upload["status"], "verified")
        put = self.fake.state.requests_to("PUT", r"^/r2/")[0]
        self.assertIsNone(put["authorization"])
        self.assertEqual(put["body"], b"abc")
        self.assertIn("x-amz-checksum-sha256", put["headers"])
        complete = self.fake.state.requests_to("POST", r"^/v1/uploads/.*/complete$")[0]
        self.assertEqual(json.loads(complete["body"]), {"sha256": hashlib.sha256(b"abc").hexdigest(), "size_bytes": 3})

    def test_multipart_upload_sends_etags(self) -> None:
        self.fake.state.multipart_threshold = 9
        self.fake.state.part_bytes = 4
        data = bytes(range(10))
        self.client.upload(data=data, role="tensor", name="t.safetensors")
        list_put = self.fake.state.requests_to("PUT", r"^/r2/.*/part/")
        self.assertEqual([put["body"] for put in list_put], [data[0:4], data[4:8], data[8:10]])
        complete = json.loads(self.fake.state.requests_to("POST", r".*/complete$")[0]["body"])
        self.assertEqual(complete["parts"], [{"part_number": number, "etag": f'"etag-{number}"'} for number in (1, 2, 3)])

    def test_estimate_uploads_the_example_and_quotes_it(self) -> None:
        quoted = self.workflow.quote(self.directory)
        request = quoted.map_request
        self.assertEqual(set(request), {"hardware", "prompt", "inputs", "agent", "limits"})
        self.assertEqual(request["hardware"], "h200")
        self.assertEqual(set(request["inputs"]), {"cubins", "source_archive", "tensors", "files"})
        self.assertEqual(request["limits"], {"max_cost_micro_usd": "20000000", "max_attempts": 2})
        with tarfile.open(fileobj=io.BytesIO(self.fake.state.map_upload[request["inputs"]["source_archive"]]["data"]), mode="r:gz") as tar:
            self.assertEqual(tar.getnames(), ["matmul.cu"])
        summary = quoted.summary()
        self.assertEqual(
            {key: summary[key] for key in ("max_cost_usd", "job_fee_usd", "gpu_usd_per_hour", "gpu_ceiling_seconds", "gpu_ceiling_usd", "agent_budget_usd", "agent_model")},
            {
                "max_cost_usd": "$20.000000", "job_fee_usd": "$0.500000", "gpu_usd_per_hour": "$4.320000", "gpu_ceiling_seconds": 1320,
                "gpu_ceiling_usd": "$1.584000", "agent_budget_usd": "$17.916000", "agent_model": "claude-opus-5-5",
            },
        )
        self.assertEqual(summary["agent_usd_per_million_tokens"]["cache_write_1h"], "$36.000000")
        self.assertTrue(summary["admissible"])

    def test_workspace_uploads_by_role(self) -> None:
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        agent_job(directory)
        quoted = self.workflow.quote(directory)
        map_inputs = quoted.map_request["inputs"]
        self.assertEqual(
            [self.fake.state.map_upload[upload_id]["role"] for upload_id in [*map_inputs["cubins"], map_inputs["source_archive"], *map_inputs["files"]]],
            ["cubin", "source_archive", "file"],
        )
        self.assertEqual(quoted.map_request["agent"], {"max_turns": 50, "max_input_tokens": 2000000, "max_output_tokens": 100000, "max_wall_seconds": 1800})
        job = self.workflow.submit(quoted=quoted, idempotency_key="agent-1")
        self.assertEqual(self.workflow.wait(job["job_id"])["status"], "succeeded")

    def test_inadmissible_quote_is_not_submitted(self) -> None:
        with open(os.path.join(self.directory, "marstrace.toml")) as manifest_file:
            text = manifest_file.read().replace("max_cost_usd = 20.0", "max_cost_usd = 2.5")
        with open(os.path.join(self.directory, "marstrace.toml"), "w") as manifest_file:
            manifest_file.write(text)
        quoted = self.workflow.quote(self.directory)
        self.assertFalse(quoted.summary()["admissible"])
        with self.assertRaisesRegex(MarsTraceError, "does not admit"):
            self.workflow.submit(quoted)
        self.assertEqual(self.fake.state.map_job, {})

    def test_submit_is_idempotent(self) -> None:
        quoted = self.workflow.quote(self.directory)
        job = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        again = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        self.assertEqual(job["job_id"], again["job_id"])
        list_submission = self.fake.state.requests_to("POST", r"^/v1/jobs$")
        for request in list_submission:
            self.assertEqual(request["idempotency"], "key-1")
            self.assertEqual(json.loads(request["body"])["quote_id"], quoted.map_quote["quote_id"])
        self.assertEqual(len(self.fake.state.map_job), 1)

    def test_a_lost_answer_is_retried_under_the_same_key(self) -> None:
        self.fake.state.drop_submissions = 1
        quoted = self.workflow.quote(self.directory)
        with mock.patch.object(MarsTraceClient, "RETRY_SECONDS", 0):
            job = self.workflow.submit(quoted=quoted, idempotency_key="lost-1")
        self.assertEqual(len(self.fake.state.requests_to("POST", r"^/v1/jobs$")), 2)
        self.assertEqual(list(self.fake.state.map_job), [job["job_id"]])

    def test_a_recorded_submission_is_resubmitted_without_uploading(self) -> None:
        job = self.workflow.submit(quoted=self.workflow.quote(self.directory), idempotency_key="rec-1")
        upload_count = len(self.fake.state.requests_to("POST", r"^/v1/uploads$"))
        again = self.workflow.resubmit(directory=self.directory, idempotency_key="rec-1")
        self.assertEqual(again["job_id"], job["job_id"])
        self.assertEqual(len(self.fake.state.requests_to("POST", r"^/v1/uploads$")), upload_count)
        self.assertIsNone(self.workflow.resubmit(directory=self.directory, idempotency_key="other"))

    def test_a_key_reused_with_another_body_conflicts(self) -> None:
        self.workflow.submit(quoted=self.workflow.quote(self.directory), idempotency_key="key-2")
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted=self.workflow.quote(self.directory), idempotency_key="key-2")
        self.assertEqual((context.exception.status, context.exception.code), (409, "idempotency_conflict"))

    def test_a_malformed_key_is_refused_locally(self) -> None:
        quoted = self.workflow.quote(self.directory)
        with self.assertRaisesRegex(ServiceError, "idempotency key"):
            self.workflow.submit(quoted=quoted, idempotency_key="has space")

    def test_insufficient_credit_names_the_credit_page(self) -> None:
        self.fake.state.available_micro_usd = 1
        quoted = self.workflow.quote(self.directory)
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted)
        self.assertEqual((context.exception.status, context.exception.code), (402, "insufficient_credit"))
        self.assertFalse(context.exception.retryable)
        self.assertEqual(context.exception.add_credit_url, ADD_CREDIT_URL)
        self.assertIn(ADD_CREDIT_URL, str(context.exception))
        self.assertIn("(request req_", str(context.exception))

    def test_wait_then_fetch(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(job["status"], "queued")
        finished = self.workflow.wait(job["job_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["cost"]["charged_micro_usd"], "14128100")
        list_wait = self.fake.state.requests_to("GET", r"^/v1/jobs/[^/]+/wait$")
        self.assertEqual(len(list_wait), 4)
        self.assertEqual(list_wait[0]["query"]["timeout_seconds"], ["60"])
        output = os.path.join(self.directory, "out")
        list_path = self.workflow.fetch(job_id=job["job_id"], output_directory=output)
        self.assertEqual(sorted(os.path.basename(path) for path in list_path), ["marstrace.json", "result-manifest.json"])
        with open(os.path.join(output, "marstrace.json"), "rb") as trace_file, open(os.path.join(FIXTURE_DIRECTORY, "trace_sass_regions.json"), "rb") as fixture_file:
            self.assertEqual(trace_file.read(), fixture_file.read())
        with open(os.path.join(output, "result-manifest.json")) as manifest_file:
            self.assertEqual(json.load(manifest_file)["quality"], "usable")

    def test_wait_returns_at_the_timeout(self) -> None:
        self.fake.state.hold_status = "running"
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(self.workflow.wait(job_id=job["job_id"], timeout_seconds=0)["status"], "queued")
        self.assertEqual(self.fake.state.requests_to("GET", r".*/wait$"), [])

    def test_result_of_an_unfinished_job_is_unavailable(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        with self.assertRaises(ServiceError) as context:
            self.workflow.fetch(job_id=job["job_id"], output_directory=os.path.join(self.directory, "out"))
        self.assertEqual(context.exception.code, "result_unavailable")

    def test_list_pages_by_cursor(self) -> None:
        list_id = [self.workflow.submit(self.workflow.quote(self.directory))["job_id"] for _ in range(3)]
        first = self.workflow.jobs(limit=2)
        second = self.workflow.jobs(limit=2, cursor=first["next_cursor"])
        self.assertEqual([job["job_id"] for job in first["items"] + second["items"]], list(reversed(list_id)))
        self.assertIsNone(second["next_cursor"])

    def test_cancel_sends_an_empty_object(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(self.workflow.cancel(job["job_id"])["status"], "cancelled")
        cancel = self.fake.state.requests_to("POST", r".*/cancel$")[0]
        self.assertEqual((json.loads(cancel["body"]), cancel["content_type"]), ({}, "application/json"))
        with self.assertRaises(ServiceError) as context:
            self.workflow.cancel(job["job_id"])
        self.assertEqual(context.exception.code, "job_not_cancellable")

    def test_unknown_job_is_a_service_error(self) -> None:
        with self.assertRaises(ServiceError) as context:
            self.workflow.job("job_" + "0" * 26)
        self.assertEqual((context.exception.status, context.exception.code), (404, "not_found"))

    def test_unavailable_hardware_is_refused_before_uploading(self) -> None:
        with self.assertRaisesRegex(MarsTraceError, "b300 is not offered now"):
            self.workflow.quote(directory=self.directory, hardware="b300")
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/uploads$"), [])

    def test_download_checks_the_digest(self) -> None:
        self.fake.state.map_object["x"] = b"abc"
        with self.assertRaisesRegex(ServiceError, "different bytes"):
            self.client.download(url=f"{self.fake.origin}/r2/results/x", sha256="0" * 64, size_bytes=3)

    def test_a_malformed_answer_is_refused(self) -> None:
        self.fake.state.map_hardware["hardware"][0]["rates"]["job_fee_micro_usd"] = 500000
        with self.assertRaisesRegex(ServiceError, r"GET /v1/hardware answered with an unexpected body: the answer.hardware\[0\].rates.job_fee_micro_usd"):
            self.workflow.hardware()
        self.fake.state.map_hardware["hardware"][0]["rates"]["job_fee_micro_usd"] = "500000"
        del self.fake.state.map_hardware["agent"]
        with self.assertRaisesRegex(ServiceError, "the answer has no 'agent'"):
            self.workflow.hardware()

    def test_shapes_accept_added_fields_and_check_ids(self) -> None:
        self.assertIsNone(ResponseSchema.matches({"upload_id": "upl_" + "a" * 26, "status": "verified", "extra": 1}, ResponseSchema.UPLOAD))
        self.assertIn("is not a string", ResponseSchema.matches({"upload_id": "upl_" + "i" * 26, "status": "verified"}, ResponseSchema.UPLOAD))
        self.assertIn("is not one of", ResponseSchema.matches({"upload_id": "upl_" + "a" * 26, "status": "pending"}, ResponseSchema.UPLOAD))


if __name__ == "__main__":
    unittest.main()
