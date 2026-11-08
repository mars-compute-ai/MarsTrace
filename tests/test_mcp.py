import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from fake_control_plane import EXAMPLE_DIRECTORY, TOP_UP_URL, FakeServer, agent_job

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
        self.fake.state.job_polls = 0
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        shutil.copytree(EXAMPLE_DIRECTORY, self.directory, dirs_exist_ok=True)
        environment = dict(
            os.environ,
            MARSCOMPUTE_API_KEY="mc_live_v1_test",
            MARSCOMPUTE_API_URL=self.fake.origin,
            PYTHONPATH=REPOSITORY,
        )
        self.process = subprocess.Popen(
            [sys.executable, "-m", "xtrace_harness.cli", "mcp"],
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
        self.assertEqual(result["serverInfo"]["name"], "xtrace-harness")
        self.assertEqual(self.send("ping")["result"], {})
        list_tool = self.send("tools/list")["result"]["tools"]
        self.assertEqual(
            [tool["name"] for tool in list_tool],
            [
                "xtrace_hardware", "xtrace_credit", "xtrace_cubin_find", "xtrace_cubin_show",
                "xtrace_inspection_capabilities", "xtrace_inspection_estimate", "xtrace_inspection_submit",
                "xtrace_inspection_get", "xtrace_inspection_wait", "xtrace_inspection_query",
                "xtrace_inspection_cancel", "xtrace_inspection_result", "xtrace_inspection_downloads",
                "xtrace_sass", "xtrace_estimate", "xtrace_submit", "xtrace_jobs", "xtrace_job", "xtrace_wait",
                "xtrace_cancel", "xtrace_fetch", "xtrace_show",
            ],
        )
        for tool in list_tool:
            self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_protocol_errors(self) -> None:
        self.initialize()
        self.assertEqual(self.send("resources/list")["error"]["code"], -32601)
        self.assertEqual(self.send("tools/call", {"name": "missing"})["error"]["code"], -32602)
        error = self.send("tools/call", {"name": "xtrace_job", "arguments": {"job": "x"}})["error"]
        self.assertEqual(error["code"], -32602)
        self.assertIn("needs 'job_id'", error["message"])
        self.process.stdin.write("not json\n")
        self.process.stdin.flush()
        self.assertEqual(json.loads(self.process.stdout.readline())["error"]["code"], -32700)

    def test_trace_through_mcp_alone(self) -> None:
        self.initialize()
        self.assertEqual(self.call("xtrace_credit")["structuredContent"]["top_up_url"], TOP_UP_URL)
        binary = os.path.join(self.directory, "matmul.cubin")
        profiles = self.call("xtrace_inspection_capabilities")["structuredContent"]
        self.assertEqual(profiles["availability"]["state"], "available")
        fee = self.call("xtrace_inspection_estimate", binary=binary, kernel="square_matrix_mul")["structuredContent"]
        self.assertEqual(fee["fee_usd"], "$0.020000")
        refused = self.call("xtrace_inspection_submit", binary=binary, kernel="square_matrix_mul", accept_max_fee_usd=0.01)
        self.assertTrue(refused["isError"])
        self.assertIn("fee is $0.020000, above the accepted $0.01", refused["structuredContent"]["error"]["message"])
        self.assertEqual(self.fake.state.map_inspection, {})
        accepted = self.call("xtrace_inspection_submit", binary=binary, kernel="square_matrix_mul", accept_max_fee_usd=0.02)
        inspection_id = accepted["structuredContent"]["inspection"]["inspection_id"]
        waited = self.call("xtrace_inspection_wait", inspection_id=inspection_id, timeout_seconds=30)["structuredContent"]
        self.assertEqual(waited["execution"]["outcome"], "completed")
        self.assertEqual(self.call("xtrace_inspection_query")["structuredContent"]["items"][0]["inspection_id"], inspection_id)
        inspected = self.call("xtrace_inspection_result", inspection_id=inspection_id, binary=binary, kernel="square_matrix_mul")["structuredContent"]
        self.assertEqual(inspected["listed_kernel"], "square_matrix_mul")
        self.assertEqual(inspected["kernels"]["square_matrix_mul"]["instructions"], 216)
        self.assertEqual(len(inspected["kernels"]["square_matrix_mul"]["parameters"]), 4)
        listing = self.call("xtrace_sass", listing=binary, kernel="square_matrix_mul", first=13, last=13)
        self.assertIn("ISETP.GE.AND", listing["content"][0]["text"])
        quote = self.call("xtrace_estimate", directory=self.directory)["structuredContent"]
        self.assertEqual(quote["max_cost_usd"], "$2.000000")
        refused = self.call("xtrace_submit", directory=self.directory, accept_max_cost_usd=1.5)
        self.assertTrue(refused["isError"])
        self.assertIn("above the accepted $1.5", refused["structuredContent"]["error"]["message"])
        self.assertEqual(self.fake.state.map_job, {})
        submitted = self.call("xtrace_submit", directory=self.directory, accept_max_cost_usd=2, idempotency_key="k-1")
        job_id = submitted["structuredContent"]["job"]["job_id"]
        self.assertEqual(self.call("xtrace_wait", job_id=job_id, timeout_seconds=30)["structuredContent"]["job"]["state"], "succeeded")
        files = self.call("xtrace_fetch", job_id=job_id)["structuredContent"]["files"]
        trace = next(path for path in files if path.endswith("xtrace.json"))
        self.assertTrue(os.path.isfile(os.path.join(self.directory, trace)))
        shown = self.call("xtrace_show", trace=trace, options=["--stats"])
        self.assertFalse(shown["isError"])
        self.assertIn("mainloop", shown["content"][0]["text"])
        self.assertEqual(self.call("xtrace_jobs")["structuredContent"]["items"][0]["job_id"], job_id)

    def test_agent_mode_through_mcp(self) -> None:
        self.initialize()
        directory = os.path.join(self.directory, "agent")
        os.makedirs(directory)
        agent_job(directory)
        self.assertEqual(self.call("xtrace_hardware")["structuredContent"]["agent_models"], ["claude-opus-5-5"])
        quote = self.call("xtrace_estimate", directory=directory)["structuredContent"]
        self.assertEqual((quote["hardware"], quote["job_fee_usd"]), ("h200-cuda12.9", "$0.500000"))
        submitted = self.call("xtrace_submit", directory=directory, accept_max_cost_usd=30)
        self.assertFalse(submitted["isError"])
        self.assertEqual(self.fake.state.map_inspection, {})

    def test_platform_refusal_is_a_tool_error(self) -> None:
        self.initialize()
        self.fake.state.available_micro_usd = 0
        result = self.call("xtrace_submit", directory=self.directory, accept_max_cost_usd=5)
        self.assertTrue(result["isError"])
        self.assertEqual(result["structuredContent"]["error"]["code"], "insufficient_credit")
        self.assertEqual(result["structuredContent"]["error"]["top_up_url"], TOP_UP_URL)
        missing = self.call("xtrace_cubin_show", binary=os.path.join(self.directory, "absent.cubin"))
        self.assertTrue(missing["isError"])


if __name__ == "__main__":
    unittest.main()
