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
        self.client = MarsTraceClient(Credentials(api_key=API_KEY, api_url=self.fake.api_url))
        self.workflow = MarsTraceWorkflow(client=self.client)
        self.directory = copy_example()
        self.addCleanup(shutil.rmtree, self.directory)

    def tearDown(self) -> None:
        self.assertEqual(self.fake.state.list_contract_error, [])

    def test_hardware_and_credit(self) -> None:
        self.assertEqual([entry["id"] for entry in self.workflow.hardware()["hardware"]], ["h200", "b200", "b300"])
        credit = self.workflow.credit()
        self.assertEqual(
            {key: credit[key] for key in ("available_micro_usd", "held_micro_usd", "minimum_micro_usd", "admissible", "add_credit_url")},
            {"available_micro_usd": "118250000", "held_micro_usd": "0", "minimum_micro_usd": "100000000", "admissible": True, "add_credit_url": ADD_CREDIT_URL},
        )
        self.fake.state.available_micro_usd = -2_500_000
        self.assertEqual(self.workflow.credit()["available_micro_usd"], "-2500000")
        for request in self.fake.state.list_request:
            self.assertEqual(request["authorization"], f"Bearer {API_KEY}")
            self.assertEqual(request["headers"]["user-agent"], "mars-trace")

    def test_environment_needs_a_marstrace_key(self) -> None:
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": "not-a-marstrace-key"}), self.assertRaisesRegex(ServiceError, "mt_live_"):
            Credentials.from_environment()
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": API_KEY}, clear=True):
            self.assertEqual(Credentials.from_environment().api_url, "https://console.marscompute.ai/trace")
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": API_KEY, "MARSTRACE_API_URL": "https://console.marscompute.test/trace/"}, clear=True):
            self.assertEqual(Credentials.from_environment().api_url, "https://console.marscompute.test/trace")

    def test_requests_keep_the_base_path(self) -> None:
        self.assertEqual(len(self.client.hardware()["hardware"]), 3)
        bare_client = MarsTraceClient(Credentials(api_key=API_KEY, api_url=self.fake.origin))
        with self.assertRaises(ServiceError):
            bare_client.hardware()

    def test_single_upload_is_checksummed(self) -> None:
        upload = self.client.upload(data=b"abc", role="cubin", name="k.cubin")
        self.assertEqual(upload["status"], "verified")
        put = self.fake.state.requests_to("PUT", r"^/storage/")[0]
        self.assertIsNone(put["authorization"])
        self.assertEqual(put["body"], b"abc")
        self.assertIn("x-amz-checksum-sha256", put["headers"])
        complete = self.fake.state.requests_to("POST", r"^/v1/uploads/.*/complete$")[0]
        self.assertEqual(json.loads(complete["body"]), {"sha256": hashlib.sha256(b"abc").hexdigest(), "size_bytes": 3})

    def test_multipart_upload_sends_etags(self) -> None:
        self.fake.state.multipart_threshold = 9
        self.fake.state.part_bytes = 4
        data = bytes(range(10))
        self.assertEqual(self.client.upload(data=data, role="tensor", name="t.safetensors")["size_bytes"], 10)
        list_put = self.fake.state.requests_to("PUT", r"^/storage/.*/part/")
        self.assertEqual([put["body"] for put in list_put], [data[0:4], data[4:8], data[8:10]])
        complete = json.loads(self.fake.state.requests_to("POST", r".*/complete$")[0]["body"])
        self.assertEqual(complete["parts"], [{"part_number": number, "etag": f'"etag-{number}"'} for number in (1, 2, 3)])

    def test_estimate_uploads_the_example_and_prices_it(self) -> None:
        quoted = self.workflow.quote(self.directory)
        request = quoted.map_request
        self.assertEqual(set(request), {"hardware", "prompt", "mode", "inputs", "agent", "limits"})
        self.assertEqual(request["mode"], "fast")
        self.assertEqual(request["hardware"], "h200")
        self.assertEqual(set(request["inputs"]), {"cubins", "source_archive", "tensors", "files"})
        self.assertEqual(request["limits"], {"max_attempts": 2})
        with tarfile.open(fileobj=io.BytesIO(self.fake.state.map_upload[request["inputs"]["source_archive"]]["data"]), mode="r:gz") as tar:
            self.assertEqual(tar.getnames(), ["matmul.cu"])
        summary = quoted.summary()
        self.assertEqual(set(summary["rates"]), {"agent_model", "agent_usd_per_million_tokens"})
        self.assertEqual(summary["rates"]["agent_usd_per_million_tokens"]["cache_write_1h"], "$14.000000")
        self.assertEqual(summary["cuda_version"], "13.0")
        self.assertEqual(summary["bounds_per_attempt"], {"agent_input_tokens": 4000000, "agent_output_tokens": 60000})
        self.assertEqual(
            summary["cost_at_bounds"],
            {"agent_tokens_per_attempt_usd": "$58.100000", "total_usd": "$116.200000"},
        )
        self.assertEqual((summary["max_attempts"], summary["minimum_usd"], summary["available_usd"]), (2, "$100.000000", "$118.250000"))
        self.assertTrue(summary["admissible"])
        self.assertEqual(self.fake.state.map_job, {})

    def test_estimate_without_limits(self) -> None:
        with open(os.path.join(self.directory, "marstrace.toml")) as manifest_file:
            text = manifest_file.read().replace("[limits]\nmax_attempts = 2\n", "")
        with open(os.path.join(self.directory, "marstrace.toml"), "w") as manifest_file:
            manifest_file.write(text)
        quoted = self.workflow.quote(self.directory)
        self.assertNotIn("limits", quoted.map_request)
        summary = quoted.summary()
        self.assertEqual((summary["max_attempts"], summary["cost_at_bounds"]["total_usd"]), (2, "$116.200000"))
        job = self.workflow.submit(quoted)
        self.assertEqual(job["limits"], {"max_attempts": 2})

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

    def test_a_balance_below_the_minimum_is_refused(self) -> None:
        self.fake.state.available_micro_usd = 99_999_999
        quoted = self.workflow.quote(self.directory)
        summary = quoted.summary()
        self.assertFalse(summary["admissible"])
        self.assertIn("at least 100000000 micro-USD", summary["message"])
        with self.assertRaisesRegex(MarsTraceError, rf"does not admit this job now \(\$99.999999 available, \$100.000000 needed to start\): .*; add credit at {ADD_CREDIT_URL}"):
            self.workflow.check_admissible(quoted)
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted)
        self.assertEqual(context.exception.code, "insufficient_credit")
        self.assertEqual(self.fake.state.map_job, {})

    def test_submit_is_idempotent(self) -> None:
        quoted = self.workflow.quote(self.directory)
        job = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        again = self.workflow.submit(quoted=quoted, idempotency_key="key-1")
        self.assertEqual(job["job_id"], again["job_id"])
        list_submission = self.fake.state.requests_to("POST", r"^/v1/jobs$")
        for request in list_submission:
            self.assertEqual(request["idempotency"], "key-1")
            self.assertEqual(json.loads(request["body"]), quoted.map_request)
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
        quoted = self.workflow.quote(self.directory)
        self.fake.state.available_micro_usd = 1
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted)
        self.assertEqual((context.exception.status, context.exception.code), (402, "insufficient_credit"))
        self.assertFalse(context.exception.retryable)
        self.assertEqual(context.exception.add_credit_url, ADD_CREDIT_URL)
        self.assertIn(f"minimum balance a job needs; add credit ({ADD_CREDIT_URL})", str(context.exception))
        self.assertIn("(request ", str(context.exception))
        self.assertEqual(self.fake.state.requests_to("GET", r"^/v1/credit$"), [])
        self.assertEqual({upload["status"] for upload in self.fake.state.map_upload.values()}, {"verified"})

    def test_a_postpaid_limit_and_suspended_billing_are_refusals_for_credit(self) -> None:
        quoted = self.workflow.quote(self.directory)
        self.fake.state.billing_mode = "postpaid"
        self.fake.state.available_micro_usd = 0
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted=quoted, idempotency_key="limit-1")
        self.assertEqual((context.exception.code, context.exception.add_credit_url), ("spending_limit_exceeded", ADD_CREDIT_URL))
        self.fake.state.available_micro_usd = None
        self.fake.state.billing_status = "suspended"
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted=quoted, idempotency_key="limit-2")
        self.assertEqual((context.exception.status, context.exception.code), (402, "billing_suspended"))

    def test_a_rate_limited_request_is_sent_again_after_the_wait(self) -> None:
        self.fake.state.rate_limited_requests = 2
        with mock.patch("marstrace.client.time.sleep") as sleep:
            quoted = self.workflow.quote(self.directory)
        self.assertEqual([call.args for call in sleep.call_args_list], [(11,), (11,)])
        self.assertEqual({upload["status"] for upload in self.fake.state.map_upload.values()}, {"verified"})
        self.fake.state.rate_limited_requests = MarsTraceClient.RATE_LIMITED_ATTEMPTS
        with mock.patch("marstrace.client.time.sleep"), self.assertRaises(ServiceError) as context:
            self.workflow.submit(quoted=quoted, idempotency_key="rate-1")
        self.assertEqual((context.exception.status, context.exception.code), (429, "rate_limited"))
        self.assertEqual(len(self.fake.state.requests_to("POST", r"^/v1/jobs$")), MarsTraceClient.RATE_LIMITED_ATTEMPTS)
        self.assertEqual(self.fake.state.map_job, {})

    def test_too_many_jobs_names_the_wait(self) -> None:
        self.fake.state.available_micro_usd = None
        self.fake.state.max_queued = 1
        self.workflow.submit(self.workflow.quote(self.directory))
        with self.assertRaises(ServiceError) as context:
            self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual((context.exception.status, context.exception.code), (429, "too_many_jobs"))
        self.assertTrue(context.exception.retryable)
        self.assertEqual(context.exception.retry_after_seconds, 30)
        self.assertIn("retry after 30 s", str(context.exception))

    def test_a_job_is_charged_every_attempt(self) -> None:
        self.fake.state.do_preempt_first_attempt = True
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(job["cost"], {"held_micro_usd": "100000000", "charged_micro_usd": None, "lines": []})
        finished = self.workflow.wait(job["job_id"])
        self.assertEqual([(attempt["status"], attempt["failure"]) for attempt in finished["attempts"]], [("failed", "supplier"), ("succeeded", None)])
        self.assertEqual(finished["attempts"][0]["error"]["code"], "vm_preempted")
        self.assertEqual(finished["cost"]["charged_micro_usd"], "2145150")
        self.assertEqual(sum(int(line["amount_micro_usd"]) for line in finished["cost"]["lines"]), 2145150)
        self.assertTrue(all(line["dimension"].startswith("service_agent_") for line in finished["cost"]["lines"]))
        self.assertEqual(finished["result"]["name"], "marstrace.json")

    def test_wait_then_fetch(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(job["status"], "queued")
        finished = self.workflow.wait(job["job_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["cost"]["charged_micro_usd"], "1926750")
        self.assertEqual(finished["progress"]["usage_micro_usd"], "1958000")
        self.assertEqual(finished["cuda_version"], "13.0")
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

    def test_wait_asks_again_when_the_gateway_is_unavailable(self) -> None:
        self.fake.state.available_micro_usd = None
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.fake.state.unavailable_waits = 2
        with mock.patch("marstrace.workflow.time.sleep") as sleep:
            self.assertEqual(self.workflow.wait(job["job_id"])["status"], "succeeded")
        self.assertEqual([call.args for call in sleep.call_args_list], [(MarsTraceWorkflow.WAIT_RETRY_SECONDS,)] * 2)
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.fake.state.unavailable_waits = MarsTraceWorkflow.WAIT_FAILURES_MAX
        with mock.patch("marstrace.workflow.time.sleep"), self.assertRaises(ServiceError) as context:
            self.workflow.wait(job["job_id"])
        self.assertEqual(context.exception.status, 503)

    def test_wait_says_why_a_queued_job_waits(self) -> None:
        self.fake.state.queue_view = {"reason": "workspace_limit", "position": 1, "workspace_dispatched_limit": 2}
        list_line: list[str] = []
        workflow = MarsTraceWorkflow(client=self.workflow._client, progress=list_line.append)
        job = workflow.submit(workflow.quote(self.directory))
        list_line.clear()
        self.assertEqual(workflow.wait(job["job_id"])["status"], "succeeded")
        self.assertIn("queued: your workspace already runs its 2 jobs; number 1 in its queue", list_line[0])
        self.assertTrue(list_line[1].endswith(" starting"))

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
        self.fake.state.available_micro_usd = None
        list_id = [self.workflow.submit(self.workflow.quote(self.directory))["job_id"] for _ in range(3)]
        first = self.workflow.jobs(limit=2)
        second = self.workflow.jobs(limit=2, cursor=first["next_cursor"])
        self.assertEqual([job["job_id"] for job in first["items"] + second["items"]], list(reversed(list_id)))
        self.assertIsNone(second["next_cursor"])

    def test_cancel_sends_an_empty_object(self) -> None:
        job = self.workflow.submit(self.workflow.quote(self.directory))
        cancelled = self.workflow.cancel(job["job_id"])
        self.assertEqual((cancelled["status"], cancelled["error"]["code"]), ("cancelled", "cancelled"))
        cancel = self.fake.state.requests_to("POST", r".*/cancel$")[0]
        self.assertEqual((json.loads(cancel["body"]), cancel["content_type"]), ({}, "application/json"))
        with self.assertRaises(ServiceError) as context:
            self.workflow.cancel(job["job_id"])
        self.assertEqual(context.exception.code, "job_not_cancellable")

    def test_a_running_job_ends_cancelled_once_stopped(self) -> None:
        self.fake.state.hold_status = "running"
        job = self.workflow.submit(self.workflow.quote(self.directory))
        self.assertEqual(self.client.wait_job(job_id=job["job_id"], timeout_seconds=1)["status"], "starting")
        self.assertEqual(self.client.wait_job(job_id=job["job_id"], timeout_seconds=1)["status"], "running")
        self.assertEqual(self.workflow.cancel(job["job_id"])["status"], "running")
        self.assertEqual(self.workflow.wait(job["job_id"])["status"], "cancelled")

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
            self.client.download(url=f"{self.fake.origin}/storage/results/x", sha256="0" * 64, size_bytes=3)

    def test_a_malformed_answer_is_refused(self) -> None:
        self.assertNotIn("rates", self.workflow.hardware()["hardware"][2])
        self.fake.state.map_hardware["rates"]["agent_output_micro_usd_per_million"] = 90000000
        with self.assertRaisesRegex(ServiceError, r"GET /v1/hardware answered with an unexpected body: the answer.rates.agent_output_micro_usd_per_million"):
            self.workflow.hardware()
        self.fake.state.map_hardware["rates"]["agent_output_micro_usd_per_million"] = "90000000"
        self.fake.state.map_hardware["hardware"][0]["cuda_versions"] = ["13"]
        with self.assertRaisesRegex(ServiceError, r"the answer.hardware\[0\].cuda_versions\[0\] is not a string"):
            self.workflow.hardware()
        self.fake.state.map_hardware["hardware"][0]["cuda_versions"] = ["13.0"]
        del self.fake.state.map_hardware["agent"]
        with self.assertRaisesRegex(ServiceError, "the answer has no 'agent'"):
            self.workflow.hardware()

    def test_shapes_accept_added_fields_and_check_ids(self) -> None:
        upload = {
            "upload_id": "upl_" + "a" * 26, "role": "cubin", "name": "k.cubin", "size_bytes": 3, "sha256": "0" * 64,
            "content_type": "application/octet-stream", "status": "verified", "expires_at": "2026-11-10T12:00:00.000Z",
        }
        self.assertIsNone(ResponseSchema.matches(dict(upload, extra=1), ResponseSchema.UPLOAD))
        self.assertIn("is not a string", ResponseSchema.matches(dict(upload, upload_id="upl_" + "i" * 26), ResponseSchema.UPLOAD))
        self.assertIn("is not one of", ResponseSchema.matches(dict(upload, status="granted"), ResponseSchema.UPLOAD))
        self.assertIn("is not a string of the form", ResponseSchema.matches(dict(upload, expires_at="today"), ResponseSchema.UPLOAD))


if __name__ == "__main__":
    unittest.main()
