import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock
from fake_control_plane import EXAMPLE_DIRECTORY, TOP_UP_URL, FakeServer
from xtrace_harness.cli import CommandLine
from xtrace_harness.workflow import XtraceWorkflow


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
            mock.patch.dict(os.environ, {"MARSCOMPUTE_API_KEY": "mc_live_v1_test", "MARSCOMPUTE_API_URL": self.fake.origin}),
            mock.patch("sys.stdin.isatty", return_value=is_tty),
            mock.patch("builtins.input", return_value=answer),
            mock.patch.object(XtraceWorkflow.__init__, "__defaults__", (None, 0)),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = CommandLine.run(list(list_argument))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_hardware_lists_rates_and_architectures(self) -> None:
        status, output, _ = self.run_cli("hardware")
        self.assertEqual(status, 0)
        self.assertRegex(output, r"h100-sxm\s+H100\s+x1 sm_90a\s+\$3.200000/GPU-hour")
        self.assertIn("planner models: planner-small", output)

    def test_credit(self) -> None:
        status, output, _ = self.run_cli("credit")
        self.assertEqual(status, 0)
        self.assertIn("available $100.000000", output)
        self.assertIn(TOP_UP_URL, output)

    def test_inspect_caches_then_sass_reads_the_cache(self) -> None:
        binary = os.path.join(self.directory, "matmul.cubin")
        status, output, _ = self.run_cli("inspect", binary, "--kernel", "square_matrix_mul")
        self.assertEqual(status, 0)
        self.assertIn("param 3", output)
        count = len(self.fake.state.list_request)
        status, output, _ = self.run_cli("sass", binary, "--kernel", "square_matrix_mul", "--line", "matmul.cu:11")
        self.assertEqual(status, 0)
        self.assertIn("STG.E", output)
        self.assertEqual(len(self.fake.state.list_request), count)

    def test_estimate_prints_the_cap_and_rate(self) -> None:
        status, output, _ = self.run_cli("estimate", self.directory)
        self.assertEqual(status, 0)
        self.assertIn("cost cap  $2.000000 at most", output)
        self.assertIn("rate      $3.200000 per GPU-hour", output)
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/xtrace/jobs$"), [])

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
        job_id = next(iter(self.fake.state.map_job))
        results = os.path.join(self.directory, "xtrace-results", job_id)
        self.assertTrue(os.path.isfile(os.path.join(results, "xtrace.json")))
        status, output, _ = self.run_cli("show", os.path.join(results, "xtrace.json"), "--stats")
        self.assertEqual(status, 0)
        self.assertIn("mainloop", output)

    def test_insufficient_credit_exits_with_the_top_up_page(self) -> None:
        self.fake.state.available_micro_usd = 0
        status, _, error = self.run_cli("submit", self.directory, "--yes")
        self.assertEqual(status, 1)
        self.assertIn("insufficient_credit", error)
        self.assertIn(TOP_UP_URL, error)

    def test_jobs_job_cancel_fetch(self) -> None:
        self.fake.state.job_polls = 1000
        self.run_cli("submit", self.directory, "--yes")
        job_id = next(iter(self.fake.state.map_job))
        status, output, _ = self.run_cli("jobs")
        self.assertEqual(status, 0)
        self.assertIn(job_id, output)
        status, output, _ = self.run_cli("job", job_id)
        self.assertIn("state     running", output)
        status, output, _ = self.run_cli("cancel", job_id)
        self.assertIn("state     cancelled", output)
        self.fake.state.job_polls = 0
        self.fake.state.map_job[job_id]["state"] = "queued"
        self.run_cli("job", job_id)
        output_directory = os.path.join(self.directory, "fetched")
        status, output, _ = self.run_cli("fetch", job_id, "-o", output_directory)
        self.assertEqual(status, 0)
        self.assertEqual(
            sorted(os.listdir(output_directory)),
            ["result-manifest.json", "summary.json", "xtrace.json"],
        )

    def test_cubin_show(self) -> None:
        status, output, _ = self.run_cli("cubin", "show", os.path.join(self.directory, "matmul.cubin"))
        self.assertEqual(status, 0)
        self.assertIn("cubin, sm_90", output)
        self.assertIn("square_matrix_mul", output)

    def test_missing_key_is_reported(self) -> None:
        with mock.patch.dict(os.environ, {"MARSCOMPUTE_API_KEY": ""}):
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(CommandLine.run(["credit"]), 1)
        self.assertIn("MARSCOMPUTE_API_KEY", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
