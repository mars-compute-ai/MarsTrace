import contextlib
import json
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock
from fake_service import ADD_CREDIT_URL, API_KEY, EXAMPLE_DIRECTORY, FakeServer, agent_job
from marstrace.cli import CommandLine


class CommandLineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fake = FakeServer()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fake.stop()

    def setUp(self) -> None:
        self.fake.reset()
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        shutil.copytree(EXAMPLE_DIRECTORY, self.directory, dirs_exist_ok=True)

    def run_cli(self, *list_argument: str, is_tty: bool = False, answer: str = "") -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": API_KEY, "MARSTRACE_API_URL": self.fake.api_url}),
            mock.patch("sys.stdin.isatty", return_value=is_tty),
            mock.patch("builtins.input", return_value=answer),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = CommandLine.run(list(list_argument))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_hardware_lists_the_service_rates_once_and_each_architecture(self) -> None:
        status, output, _ = self.run_cli("hardware")
        self.assertEqual(status, 0)
        self.assertEqual(output.count("$18.000000 input, $90.000000 output per million, on every hardware"), 1)
        self.assertRegex(output, r"h200\s+H200\s+sm_90a\s+available")
        self.assertIn("CUDA 12.8, 12.9, 13.0 (default)", output)
        self.assertIn("CUDA 12.9, 13.0 (default)", output)
        self.assertNotIn("GPU-hour", output)
        self.assertRegex(output, r"b300\s+B300\s+sm_103a\s+not available now")

    def test_hardware_says_when_the_service_has_no_price(self) -> None:
        self.fake.state.map_hardware["rates"] = None
        status, output, _ = self.run_cli("hardware")
        self.assertEqual(status, 0)
        self.assertIn("tokens    no published price", output)
        self.assertIn("agent model: claude-opus-5-5", output)
        self.assertIn("max_wall_seconds 1200", output)

    def test_credit(self) -> None:
        status, output, _ = self.run_cli("credit")
        self.assertEqual(status, 0)
        self.assertIn("available $118.250000", output)
        self.assertIn("minimum   $100.000000 to start a job", output)
        self.assertIn("jobs      can start", output)
        self.assertIn(ADD_CREDIT_URL, output)
        self.fake.state.available_micro_usd = -500000
        output = self.run_cli("credit")[1]
        self.assertIn("available -$0.500000", output)
        self.assertIn("cannot start; add credit", output)
        self.fake.state.available_micro_usd = None
        self.assertIn("available no limit", self.run_cli("credit")[1])

    def test_estimate_prints_the_cost_at_the_bounds_and_the_balance(self) -> None:
        status, output, _ = self.run_cli("estimate", self.directory)
        self.assertEqual(status, 0)
        self.assertIn("hardware  h200, H200 (sm_90a)", output)
        self.assertIn("cuda      13.0", output)
        self.assertIn("bounds    per attempt 4000000 input and 60000 output tokens; 2 attempt(s)", output)
        self.assertIn("at most   $298.800000: $149.400000 of tokens per attempt", output)
        self.assertNotIn("GPU", output)
        self.assertIn("failed ones included", output)
        self.assertIn("balance   $118.250000 available, $100.000000 needed to start", output)
        self.assertNotIn("refused", output)
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/jobs$"), [])

    def test_estimate_prints_the_token_rates_and_the_chosen_cuda_version(self) -> None:
        directory = os.path.join(self.directory, "agent")
        os.makedirs(directory)
        agent_job(directory)
        status, output, error = self.run_cli("estimate", directory, "--hardware", "h200", "--cuda-version", "12.9")
        self.assertEqual(status, 0, error)
        self.assertIn(
            "tokens    claude-opus-5-5: $18.000000 input, $1.800000 cache read, $22.500000 5-minute cache write, "
            "$36.000000 1-hour cache write, $90.000000 output per million",
            output,
        )
        self.assertIn("per attempt 2000000 input and 100000 output tokens; 1 attempt(s)", output)
        self.assertIn("cuda      12.9", output)
        self.assertEqual(json.loads(self.fake.state.requests_to("POST", r"^/v1/quotes$")[-1]["body"])["cuda_version"], "12.9")
        self.assertIn("upload    source_archive", error)
        self.assertIn("upload    file", error)
        nb_upload = len(self.fake.state.requests_to("POST", r"^/v1/uploads$"))
        status, _, error = self.run_cli("estimate", directory, "--cuda-version", "12.7")
        self.assertEqual(status, 1)
        self.assertIn("h200 does not offer CUDA 12.7; it offers 12.8, 12.9, 13.0", error)
        self.assertEqual(len(self.fake.state.requests_to("POST", r"^/v1/uploads$")), nb_upload)
        status, _, error = self.run_cli("estimate", directory, "--cuda-version", "13")
        self.assertEqual(status, 1)
        self.assertIn("must be a CUDA version such as", error)

    def test_submit_refuses_below_the_minimum_before_asking(self) -> None:
        self.fake.state.available_micro_usd = 50_000_000
        status, output, error = self.run_cli("submit", self.directory, is_tty=True, answer="y")
        self.assertEqual(status, 1)
        self.assertIn("refused   A job starts only while", output)
        self.assertIn("($50.000000 available, $100.000000 needed to start)", error)
        self.assertIn(ADD_CREDIT_URL, error)
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/jobs$"), [])

    def test_submit_asks_before_spending(self) -> None:
        status, _, error = self.run_cli("submit", self.directory)
        self.assertEqual(status, 1)
        self.assertIn("--yes", error)
        status, _, error = self.run_cli("submit", self.directory, is_tty=True, answer="n")
        self.assertEqual(status, 1)
        self.assertIn("not submitted", error)
        self.assertEqual(self.fake.state.map_job, {})
        status, output, _ = self.run_cli("submit", self.directory, is_tty=True, answer="y")
        self.assertEqual(status, 0)
        self.assertEqual(len(self.fake.state.map_job), 1)

    def test_submit_wait_fetches_the_results(self) -> None:
        status, output, _ = self.run_cli("submit", self.directory, "--yes", "--wait")
        self.assertEqual(status, 0, output)
        self.assertIn("quality   usable", output)
        self.assertIn("held      $100.000000", output)
        self.assertIn("charged   $4.954500", output)
        self.assertIn("cuda      13.0", output)
        self.assertIn("          service_agent_output_tokens 38810: $3.492900", output)
        self.assertNotIn("service_agent_input_tokens", output)
        self.assertIn("attempt   1 succeeded", output)
        job_id = next(iter(self.fake.state.map_job))
        results = os.path.join(self.directory, "marstrace-results", job_id)
        self.assertTrue(os.path.isfile(os.path.join(results, "marstrace.json")))
        status, output, _ = self.run_cli("show", os.path.join(results, "marstrace.json"), "--stats")
        self.assertEqual(status, 0)
        self.assertIn("mainloop", output)

    def test_every_attempt_is_charged_and_the_balance_gates_the_next_job(self) -> None:
        self.fake.state.do_preempt_first_attempt = True
        status, output, error = self.run_cli("submit", self.directory, "--yes", "--wait")
        self.assertEqual(status, 0, error)
        self.assertIn("attempt   1 failed (supplier): vm_preempted: The GPU machine was reclaimed by the supplier.", output)
        self.assertIn("charged   $5.516100", output)
        status, output, error = self.run_cli("submit", self.directory, "--yes")
        self.assertEqual(status, 1)
        self.assertIn("balance   $12.733900 available, $100.000000 needed to start", output)
        self.assertIn(ADD_CREDIT_URL, error)

    def test_a_recorded_key_is_resubmitted(self) -> None:
        status, _, error = self.run_cli("submit", self.directory, "--yes", "--idempotency-key", "cli-1")
        self.assertEqual(status, 0, error)
        status, output, error = self.run_cli("submit", self.directory, "--idempotency-key", "cli-1")
        self.assertEqual(status, 0, error)
        self.assertIn("resubmit", error)
        self.assertEqual(len(self.fake.state.map_job), 1)

    def test_jobs_job_cancel_fetch(self) -> None:
        self.fake.state.hold_status = "running"
        self.run_cli("submit", self.directory, "--yes")
        job_id = next(iter(self.fake.state.map_job))
        status, output, _ = self.run_cli("jobs")
        self.assertEqual(status, 0)
        self.assertIn(job_id, output)
        status, output, _ = self.run_cli("job", job_id)
        self.assertIn("status    queued", output)
        status, output, _ = self.run_cli("fetch", job_id)
        self.assertEqual(status, 1)
        status, output, _ = self.run_cli("cancel", job_id)
        self.assertIn("status    cancelled", output)
        self.fake.state.hold_status = None
        self.fake.state.map_job[job_id]["status"] = "queued"
        self.run_cli("job", job_id, "--wait")
        output_directory = os.path.join(self.directory, "fetched")
        status, output, _ = self.run_cli("fetch", job_id, "-o", output_directory)
        self.assertEqual(status, 0)
        self.assertEqual(sorted(os.listdir(output_directory)), ["marstrace.json", "result-manifest.json"])

    def test_cubin_find_copies_into_the_workspace(self) -> None:
        cache = os.path.join(self.directory, "cache")
        os.makedirs(cache)
        shutil.copyfile(os.path.join(self.directory, "matmul.cubin"), os.path.join(cache, "matmul.cubin"))
        job = os.path.join(self.directory, "job")
        with mock.patch("marstrace.binary.DslCache.default_roots", return_value=[]):
            status, output, error = self.run_cli("cubin", "find", "square", "--dir", cache, "--copy-to", job)
        self.assertEqual(status, 0, error)
        self.assertIn("[0] square_matrix_mul", output)
        self.assertIn("add it to [workspace] cubins", output)
        self.assertTrue(os.path.isfile(os.path.join(job, "matmul.cubin")))

    def test_cubin_show(self) -> None:
        status, output, _ = self.run_cli("cubin", "show", os.path.join(self.directory, "matmul.cubin"))
        self.assertEqual(status, 0)
        self.assertIn("cubin, sm_90", output)
        self.assertIn("square_matrix_mul", output)

    def test_missing_key_is_reported(self) -> None:
        with mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": ""}):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(CommandLine.run(["credit"]), 1)
        self.assertIn("marstrace: set MARSTRACE_API_KEY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
