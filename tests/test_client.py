import http.server
import json
import os
import threading
import unittest
from typing import Any
from unittest import mock
from xtrace_harness.cli import CommandLine
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import PlatformError
from xtrace_harness.job import JobBundle, JobManifest

EXAMPLE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "matmul",
)


class FakePlatform(http.server.BaseHTTPRequestHandler):
    """A stand-in for the job API that records what the client sent."""

    list_request: list[dict[str, Any]] = []
    origin = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _answer(self, status: int, map_body: dict[str, Any]) -> None:
        data = json.dumps(map_body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("x-request-id", "req-1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        FakePlatform.list_request.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "idempotency": self.headers.get("Idempotency-Key"),
                "body": body,
            }
        )
        return body

    def do_POST(self) -> None:
        body = self._record()
        if self.path == "/v1/xtrace/uploads":
            self._answer(201, {"id": "xtu_1", "upload_url": f"{FakePlatform.origin}/signed/xtu_1"})
        elif self.path == "/v1/xtrace/uploads/xtu_1/complete":
            self._answer(200, {"id": "xtu_1", "status": "validated"})
        elif self.path == "/v1/xtrace/jobs":
            if json.loads(body)["hardware"] == "missing":
                self._answer(400, {"error": {"code": "unknown_hardware", "message": "no such class"}})
            else:
                self._answer(201, {"id": "xtj_1", "state": "queued"})
        else:
            self._answer(404, {"error": {"code": "not_found", "message": "no route"}})

    def do_GET(self) -> None:
        self._record()
        if self.path == "/v1/xtrace/hardware":
            self._answer(
                200,
                {
                    "data": [
                        {"id": "h100-sxm", "gpu": "H100 SXM", "arch": "sm_90a", "available": True},
                        {"id": "b200", "gpu": "B200", "arch": "sm_100a", "available": True},
                    ]
                },
            )
        else:
            self._answer(404, {"error": {"code": "not_found", "message": "no route"}})

    def do_PUT(self) -> None:
        self._record()
        self._answer(200, {})


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakePlatform)
        FakePlatform.origin = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.client = MarsComputeClient(Credentials(api_key="mc_test", api_url=FakePlatform.origin))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        FakePlatform.list_request.clear()

    def test_upload_then_submit(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        bundle = JobBundle.pack(manifest)
        upload_id = self.client.upload(bundle)
        map_job = self.client.submit(upload_id=upload_id, hardware="h100-sxm", spec=manifest.spec)
        self.assertEqual(map_job["id"], "xtj_1")
        list_path = [request["path"] for request in FakePlatform.list_request]
        self.assertEqual(
            list_path,
            ["/v1/xtrace/uploads", "/signed/xtu_1", "/v1/xtrace/uploads/xtu_1/complete", "/v1/xtrace/jobs"],
        )
        grant, put, _, submit = FakePlatform.list_request
        self.assertEqual(json.loads(grant["body"])["sha256"], bundle.sha256)
        self.assertEqual(put["body"], bundle.data)
        self.assertIsNone(put["authorization"])
        self.assertEqual(submit["authorization"], "Bearer mc_test")
        self.assertTrue(submit["idempotency"])
        self.assertEqual(json.loads(submit["body"])["spec"]["schema"], "xtrace.job/1")

    def test_refusal_carries_code_and_request(self) -> None:
        manifest = JobManifest.load(EXAMPLE_DIRECTORY)
        with self.assertRaises(PlatformError) as context:
            self.client.submit(upload_id="xtu_1", hardware="missing", spec=manifest.spec)
        self.assertEqual(context.exception.code, "unknown_hardware")
        self.assertEqual(context.exception.status, 400)
        self.assertIn("req-1", str(context.exception))


    def run_cli(self, *list_argument: str) -> int:
        with mock.patch.dict(
            os.environ,
            {"MARSCOMPUTE_API_KEY": "mc_test", "MARSCOMPUTE_API_URL": FakePlatform.origin},
        ):
            return CommandLine.run(list(list_argument))

    def test_submit_checks_the_binary_target(self) -> None:
        self.assertEqual(self.run_cli("submit", EXAMPLE_DIRECTORY, "--hardware", "b200"), 1)
        self.assertEqual([request["path"] for request in FakePlatform.list_request], ["/v1/xtrace/hardware"])

    def test_submit_runs_on_a_matching_class(self) -> None:
        self.assertEqual(self.run_cli("submit", EXAMPLE_DIRECTORY), 0)
        self.assertEqual(FakePlatform.list_request[-1]["path"], "/v1/xtrace/jobs")


if __name__ == "__main__":
    unittest.main()
