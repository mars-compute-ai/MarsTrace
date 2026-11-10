import contextlib
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
            mock.patch.dict(os.environ, {"MARSTRACE_API_KEY": API_KEY, "MARSTRACE_API_URL": self.fake.origin}),
            mock.patch("sys.stdin.isatty", return_value=is_tty),
            mock.patch("builtins.input", return_value=answer),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = CommandLine.run(list(list_argument))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_hardware_lists_rates_and_architectures(self) -> None:
        status, output, _ = self.run_cli("hardware")
        self.assertEqual(status, 0)
        self.assertRegex(output, r"h200\s+H200\s+sm_90a\s+\$4.320000/GPU-hour  fee \$0.500000  available")
        self.assertRegex(output, r"b300\s+B300\s+sm_103a .* not available now")
        self.assertIn("agent model: claude-opus-5-5", output)
        self.assertIn("max_wall_seconds 1200", output)

    def test_credit(self) -> None:
        status, output, _ = self.run_cli("credit")
        self.assertEqual(status, 0)
        self.assertIn("available $100.000000", output)
        self.assertIn(ADD_CREDIT_URL, output)
        self.fake.state.available_micro_usd = None
        self.assertIn("available no limit", self.run_cli("credit")[1])

    def test_estimate_prints_the_cap_and_rates(self) -> None:
        status, output, _ = self.run_cli("estimate", self.directory)
        self.assertEqual(status, 0)
        self.assertIn("hardware  h200, H200 (sm_90a)", output)
        self.assertIn("cost cap  $20.000000 at most", output)
        self.assertIn("GPU       $4.320000 per GPU-hour, at most 1320 s: $1.584000", output)
        self.assertIn("agent     at most $17.916000 of tokens", output)
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/jobs$"), [])

    def test_estimate_prints_the_job_fee_and_token_rates(self) -> None:
        directory = os.path.join(self.directory, "agent")
        os.makedirs(directory)
        agent_job(directory)
        status, output, error = self.run_cli("estimate", directory, "--hardware", "h200")
        self.assertEqual(status, 0, error)
        self.assertIn("job fee   $0.500000", output)
        self.assertIn(
            "tokens    claude-opus-5-5: $18.000000 input, $1.800000 cache read, $22.500000 5-minute cache write, "
            "$36.000000 1-hour cache write, $90.000000 output per million",
            output,
        )
        self.assertIn("cost cap  $30.000000 at most", output)
        self.assertIn("upload    source_archive", error)
        self.assertIn("upload    file", error)

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
        self.assertIn("charged   $14.128100", output)
        job_id = next(iter(self.fake.state.map_job))
        results = os.path.join(self.directory, "marstrace-results", job_id)
        self.assertTrue(os.path.isfile(os.path.join(results, "marstrace.json")))
        status, output, _ = self.run_cli("show", os.path.join(results, "marstrace.json"), "--stats")
        self.assertEqual(status, 0)
        self.assertIn("mainloop", output)

    def test_insufficient_credit_exits_with_the_credit_page(self) -> None:
        self.fake.state.available_micro_usd = 0
        status, _, error = self.run_cli("submit", self.directory, "--yes")
        self.assertEqual(status, 1)
        self.assertIn("insufficient_credit", error)
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
