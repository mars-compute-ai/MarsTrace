import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from fake_service import ADD_CREDIT_URL, API_KEY, EXAMPLE_DIRECTORY, FakeServer, agent_job
from marstrace.viewer.common.trace.format.panel import SelectionToken

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class McpStdioTest(unittest.TestCase):
    """The MCP server as a client sees it: a child process speaking JSON-RPC over stdio."""

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
        environment = dict(
            os.environ,
            MARSTRACE_API_KEY=API_KEY,
            MARSTRACE_API_URL=self.fake.api_url,
            PYTHONPATH=REPOSITORY,
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "marstrace.cli", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
            cwd=self.directory,
        )
        self.addCleanup(self.stop)
        self.next_id = 0

    def stop(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()

    def send(self, method: str, params: dict[str, Any] | None = None, is_notification: bool = False) -> dict[str, Any] | None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not is_notification:
            self.next_id += 1
            message["id"] = self.next_id
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        if is_notification:
            return None
        response = json.loads(self.process.stdout.readline())
        self.assertEqual(response["id"], self.next_id)
        return response

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        return self.send("tools/call", {"name": name, "arguments": arguments})["result"]

    def initialize(self) -> dict[str, Any]:
        response = self.send(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
        )
        self.send("notifications/initialized", is_notification=True)
        return response

    def test_handshake_and_tool_list(self) -> None:
        result = self.initialize()["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["serverInfo"]["name"], "marstrace")
        self.assertEqual(self.send("ping")["result"], {})
        list_tool = self.send("tools/list")["result"]["tools"]
        self.assertEqual(
            [tool["name"] for tool in list_tool],
            [
                "marstrace_guide", "marstrace_hardware", "marstrace_credit", "marstrace_cubin_find",
                "marstrace_cubin_show",
                "marstrace_estimate", "marstrace_submit", "marstrace_jobs", "marstrace_job", "marstrace_wait",
                "marstrace_cancel", "marstrace_result", "marstrace_fetch", "marstrace_show",
            ],
        )
        for tool in list_tool:
            self.assertEqual(tool["inputSchema"]["type"], "object")
        submit = next(tool for tool in list_tool if tool["name"] == "marstrace_submit")
        self.assertEqual(submit["inputSchema"]["required"], ["directory", "accept_cost_at_bounds_usd"])
        self.assertIn("failed ones included", submit["description"])
        self.assertIn("minimum", result["instructions"])
        self.assertIn("marstrace_guide", result["instructions"])
        self.assertIn("intent", result["instructions"])

    def test_guide_and_show_need_no_service(self) -> None:
        self.initialize()
        remote = self.call("marstrace_guide", topic="remote")
        self.assertFalse(remote.get("isError", False))
        text = remote["content"][0]["text"]
        self.assertIn("# Tracing a kernel with MarsTrace", text)
        self.assertNotIn("name: marstrace-remote", text)
        read = self.call("marstrace_guide", topic="read")["content"][0]["text"]
        self.assertIn("Gate the trace first", read)
        self.assertEqual(self.send("tools/call", {"name": "marstrace_guide", "arguments": {}})["error"]["code"], -32602)

    def test_show_with_a_selection_and_json_answers_the_json_alone(self) -> None:
        self.initialize()
        trace = os.path.join(REPOSITORY, "tests", "fixtures", "trace_sass_regions.json")
        kernel = json.loads(self.call("marstrace_show", trace=trace, options=["--json"])["content"][0]["text"])[
            "kernel_prototype"
        ]
        token = f"mtrace/1 kernel={SelectionToken.kernel_tag(kernel)} t=0:1000000000ns lane=block:0 order=thread"
        shown = self.call("marstrace_show", trace=trace, options=["--select", token, "--json"])
        self.assertFalse(shown["isError"])
        self.assertIn("scope_stats", json.loads(shown["content"][0]["text"]))
        stats = self.call("marstrace_show", trace=trace, options=["--select", token, "--stats"])["content"][0]["text"]
        self.assertTrue(stats.startswith("\x1b[1mSelection:") or stats.startswith("Selection:"))

    def test_protocol_errors(self) -> None:
        self.initialize()
        self.assertEqual(self.send("resources/list")["error"]["code"], -32601)
        self.assertEqual(self.send("tools/call", {"name": "missing"})["error"]["code"], -32602)
        error = self.send("tools/call", {"name": "marstrace_job", "arguments": {"job": "x"}})["error"]
        self.assertEqual(error["code"], -32602)
        self.assertIn("needs 'job_id'", error["message"])
        self.process.stdin.write("not json\n")
        self.process.stdin.flush()
        self.assertEqual(json.loads(self.process.stdout.readline())["error"]["code"], -32700)

    def test_trace_through_mcp_alone(self) -> None:
        self.initialize()
        self.assertEqual(self.call("marstrace_credit")["structuredContent"]["add_credit_url"], ADD_CREDIT_URL)
        binary = os.path.join(self.directory, "matmul.cubin")
        self.assertEqual(self.call("marstrace_cubin_show", binary=binary)["structuredContent"]["kernels"], ["square_matrix_mul"])
        quote = self.call("marstrace_estimate", directory=self.directory)["structuredContent"]
        self.assertEqual((quote["cost_at_bounds"]["total_usd"], quote["cuda_version"]), ("$298.800000", "13.0"))
        self.assertEqual(self.call("marstrace_estimate", directory=self.directory, cuda_version="12.9")["structuredContent"]["cuda_version"], "12.9")
        self.assertTrue(self.call("marstrace_estimate", directory=self.directory, cuda_version="12.7")["isError"])
        self.assertEqual((quote["minimum_usd"], quote["available_usd"], quote["admissible"]), ("$100.000000", "$118.250000", True))
        refused = self.call("marstrace_submit", directory=self.directory, accept_cost_at_bounds_usd=298)
        self.assertTrue(refused["isError"])
        self.assertIn("is $298.800000, above the accepted $298", refused["structuredContent"]["error"]["message"])
        self.assertEqual(self.fake.state.map_job, {})
        submitted = self.call("marstrace_submit", directory=self.directory, accept_cost_at_bounds_usd=298.8, cuda_version="12.9", idempotency_key="k-1")
        self.assertEqual(submitted["structuredContent"]["estimate"]["cost_at_bounds"]["total_usd"], "$298.800000")
        self.assertEqual(submitted["structuredContent"]["job"]["cuda_version"], "12.9")
        job_id = submitted["structuredContent"]["job"]["job_id"]
        again = self.call("marstrace_submit", directory=self.directory, accept_cost_at_bounds_usd=298.8, cuda_version="12.9", idempotency_key="k-1")
        self.assertEqual(again["structuredContent"]["job"]["job_id"], job_id)
        self.assertEqual(len(self.fake.state.map_job), 1)
        self.assertEqual(self.call("marstrace_wait", job_id=job_id, timeout_seconds=30)["structuredContent"]["status"], "succeeded")
        self.assertEqual(self.call("marstrace_result", job_id=job_id)["structuredContent"]["manifest"]["file"]["name"], "marstrace.json")
        files = self.call("marstrace_fetch", job_id=job_id)["structuredContent"]["files"]
        trace = next(path for path in files if path.endswith("marstrace.json"))
        self.assertTrue(os.path.isfile(os.path.join(self.directory, trace)))
        shown = self.call("marstrace_show", trace=trace, options=["--stats"])
        self.assertFalse(shown["isError"])
        self.assertIn("mainloop", shown["content"][0]["text"])
        self.assertEqual(self.call("marstrace_jobs")["structuredContent"]["items"][0]["job_id"], job_id)

    def test_workspace_with_files_through_mcp(self) -> None:
        self.initialize()
        directory = os.path.join(self.directory, "agent")
        os.makedirs(directory)
        agent_job(directory)
        self.assertEqual(self.call("marstrace_hardware")["structuredContent"]["agent"]["model"], "claude-opus-5-5")
        quote = self.call("marstrace_estimate", directory=directory)["structuredContent"]
        self.assertEqual((quote["hardware"], quote["max_attempts"]), ("h200", 1))
        submitted = self.call("marstrace_submit", directory=directory, accept_cost_at_bounds_usd=float(quote["cost_at_bounds"]["total_usd"].lstrip("$")))
        self.assertFalse(submitted["isError"])
        self.assertEqual(submitted["structuredContent"]["job"]["status"], "queued")
        self.assertEqual(self.call("marstrace_cancel", job_id=submitted["structuredContent"]["job"]["job_id"])["structuredContent"]["status"], "cancelled")

    def test_service_refusal_is_a_tool_error(self) -> None:
        self.initialize()
        self.fake.state.available_micro_usd = 0
        result = self.call("marstrace_submit", directory=self.directory, accept_cost_at_bounds_usd=400)
        self.assertTrue(result["isError"])
        self.assertIn("needed to start", result["structuredContent"]["error"]["message"])
        self.assertIn(ADD_CREDIT_URL, result["structuredContent"]["error"]["message"])
        self.assertEqual(self.fake.state.requests_to("POST", r"^/v1/jobs$"), [])
        self.fake.state.available_micro_usd = 118_250_000
        self.fake.state.do_refuse_admission = True
        result = self.call("marstrace_submit", directory=self.directory, accept_cost_at_bounds_usd=400)
        self.assertTrue(result["isError"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "insufficient_credit")
        self.assertEqual(result["structuredContent"]["error"]["add_credit_url"], ADD_CREDIT_URL)
        missing = self.call("marstrace_cubin_show", binary=os.path.join(self.directory, "absent.cubin"))
        self.assertTrue(missing["isError"])


class McpWithoutKeyTest(unittest.TestCase):
    """The MCP server started before the user has a key: it answers, and says how to get one."""

    def setUp(self) -> None:
        environment = {key: value for key, value in os.environ.items() if key != "MARSTRACE_API_KEY"}
        environment["PYTHONPATH"] = REPOSITORY
        self.process = subprocess.Popen(
            [sys.executable, "-m", "marstrace.cli", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
        )
        self.addCleanup(self.stop)

    def stop(self) -> None:
        self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()

    def send(self, request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n")
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())

    def test_a_service_tool_answers_how_to_get_a_key(self) -> None:
        initialized = self.send(
            1,
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
        )
        self.assertIn("marstrace_guide", initialized["result"]["instructions"])
        credit = self.send(2, "tools/call", {"name": "marstrace_credit", "arguments": {}})["result"]
        self.assertTrue(credit["isError"])
        error = json.loads(credit["content"][0]["text"])["error"]
        self.assertEqual(error["code"], "api_key_missing")
        for step in ("/register", "/tenant/billing/credits", "/tenant/trace/keys", "export MARSTRACE_API_KEY"):
            self.assertIn(step, error["message"])
        guide = self.send(3, "tools/call", {"name": "marstrace_guide", "arguments": {"topic": "read"}})["result"]
        self.assertFalse(guide.get("isError", False))


if __name__ == "__main__":
    unittest.main()
