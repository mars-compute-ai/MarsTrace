import gzip
import hashlib
import http.server
import json
import os
import re
import threading
import urllib.parse
import uuid
from typing import Any
import jsonschema
from schema_check import SchemaCheck

FIXTURE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
)
EXAMPLE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "matmul",
)
TIMESTAMP = "2026-11-08T12:00:00.000Z"
TOP_UP_URL = "https://console.marscompute.ai/billing/credit"
COMPUTE_H100 = {
    "provider": "nebius",
    "region": "eu-north1",
    "gpu_type": "H100",
    "gpu_count": 1,
    "purchase_option": "on_demand",
    "fallback": "none",
}
COMPUTE_B200 = dict(COMPUTE_H100, gpu_type="B200")
CAPABILITIES = {
    "schema_version": "marscompute.xtrace.capabilities.v1",
    "enabled": True,
    "compute": [
        {
            "compute": COMPUTE_H100,
            "environment_id": "h100-sxm",
            "image_digest": "sha256:" + "a" * 64,
            "architecture": "sm_90a",
            "qualified": True,
            "bundle_kinds": ["cubin_launch"],
            "build_templates": [],
            "stages": ["xtrace", "range_profile", "pc_sampling"],
            "gpu_micro_usd_per_hour": "3200000",
            "architectures": {
                "xtrace": ["sm_90a"],
                "range_profile": ["sm_90a"],
                "pc_sampling": ["sm_90a"],
            },
        },
        {
            "compute": COMPUTE_B200,
            "environment_id": "b200",
            "image_digest": "sha256:" + "b" * 64,
            "architecture": "sm_100a",
            "qualified": True,
            "bundle_kinds": ["cubin_launch"],
            "build_templates": [],
            "stages": ["xtrace"],
            "gpu_micro_usd_per_hour": "6400000",
            "architectures": {"xtrace": ["sm_100a"]},
        },
    ],
    "models": ["planner-small"],
    "max_upload_bytes": 1 << 30,
    "max_query_records": 1000,
    "retention_seconds": {"inputs": 86400, "results": 86400, "staging": 3600},
    "limits": {
        "max_total_cost_micro_usd": "5000000",
        "max_active_gpu_seconds": 300,
        "max_wall_seconds": 3600,
        "max_attempts": 2,
        "max_trace_bytes": 1 << 30,
    },
}


class FakeState:
    """What the fake Control Plane holds between requests."""

    def __init__(self) -> None:
        self.origin = ""
        self.list_request: list[dict[str, Any]] = []
        self.list_schema_error: list[str] = []
        self.map_artifact: dict[str, dict[str, Any]] = {}
        self.map_blob: dict[str, bytes] = {}
        self.map_quote: dict[str, dict[str, Any]] = {}
        self.map_job: dict[str, dict[str, Any]] = {}
        self.map_job_of_key: dict[str, str] = {}
        self.map_inspection: dict[str, dict[str, Any]] = {}
        self.list_job_order: list[str] = []
        self.available_micro_usd = 100_000_000
        # files larger than this are granted as multipart uploads of PART_BYTES parts
        self.multipart_threshold = 1 << 20
        self.part_bytes = 4096
        # polls a job and an inspection stay unfinished for
        self.job_polls = 1
        self.inspection_polls = 1
        self.inspection_fixture = os.path.join(FIXTURE_DIRECTORY, "matmul_inspection.json")

    def requests_to(self, method: str, pattern: str) -> list[dict[str, Any]]:
        regex = re.compile(pattern)
        return [
            request
            for request in self.list_request
            if request["method"] == method and regex.match(request["path"])
        ]


class FakeControlPlane(http.server.BaseHTTPRequestHandler):
    """A stand-in for the Control Plane's Xtrace API that checks and records what it is sent."""

    state = FakeState()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send(self, status: int, data: bytes, map_header: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (map_header or {}).items():
            self.send_header(name, value)
        self.send_header("x-request-id", "req-1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _answer(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value).encode(), {"Content-Type": "application/json"})

    def _fail(self, status: int, code: str, message: str, **extra: Any) -> None:
        self._answer(status, {"error": {"code": code, "message": message, "correlation_id": "corr-1", **extra}})

    def _record(self) -> bytes:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = urllib.parse.urlsplit(self.path).path
        FakeControlPlane.state.list_request.append(
            {
                "method": self.command,
                "path": path,
                "query": urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query),
                "authorization": self.headers.get("Authorization"),
                "idempotency": self.headers.get("Idempotency-Key"),
                "content_type": self.headers.get("Content-Type"),
                "body": body,
            }
        )
        if self.command == "POST" and path.startswith("/v1/"):
            try:
                SchemaCheck.validate_request(self.command, path, body)
            except jsonschema.ValidationError as err:
                FakeControlPlane.state.list_schema_error.append(f"{path}: {err.message}")
        return body

    def _download_grant(self, artifact_id: str, name: str, encoding: str = "identity") -> dict[str, Any]:
        data = FakeControlPlane.state.map_blob[artifact_id]
        return {
            "artifact_id": artifact_id,
            "download_url": f"{FakeControlPlane.state.origin}/download/{artifact_id}",
            "expires_at": TIMESTAMP,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "name": name,
            "content_type": "application/json",
            "encoding": encoding,
        }

    def _store(self, data: bytes, kind: str, name: str) -> str:
        artifact_id = str(uuid.uuid4())
        FakeControlPlane.state.map_blob[artifact_id] = data
        FakeControlPlane.state.map_artifact[artifact_id] = {"kind": kind, "name": name, "status": "verified"}
        return artifact_id

    def do_PUT(self) -> None:
        body = self._record()
        state = FakeControlPlane.state
        match = re.match(r"^/upload/([^/]+)(?:/part/(\d+))?$", urllib.parse.urlsplit(self.path).path)
        if match is None or self.headers.get("Authorization") is not None:
            self._fail(403, "forbidden", "bad upload")
            return
        artifact = state.map_artifact[match.group(1)]
        if match.group(2) is None:
            artifact["data"] = body
            self._send(200, b"")
            return
        artifact["parts"][int(match.group(2))] = body
        self._send(200, b"", {"ETag": f'"etag-{match.group(2)}"'})

    def do_GET(self) -> None:
        self._record()
        state = FakeControlPlane.state
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/download/"):
            self._send(200, state.map_blob[path.rsplit("/", 1)[1]])
            return
        if self.headers.get("Authorization") != "Bearer mc_live_v1_test":
            self._fail(401, "unauthorized", "no key")
            return
        if path == "/v1/xtrace/capabilities":
            self._answer(200, CAPABILITIES)
        elif path == "/v1/xtrace/credit":
            self._answer(
                200,
                {
                    "available_micro_usd": str(state.available_micro_usd),
                    "reserved_micro_usd": "0",
                    "top_up_url": TOP_UP_URL,
                },
            )
        elif match := re.match(r"^/v1/xtrace/inspections/([^/]+)$", path):
            self._answer(200, self._poll_inspection(match.group(1)))
        elif path == "/v1/xtrace/jobs":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            limit = int(query["limit"][0])
            start = int(query.get("cursor", ["0"])[0])
            list_id = list(reversed(state.list_job_order))[start:start + limit]
            self._answer(
                200,
                {
                    "items": [state.map_job[job_id] for job_id in list_id],
                    "next_cursor": str(start + limit) if start + limit < len(state.list_job_order) else None,
                },
            )
        elif match := re.match(r"^/v1/xtrace/jobs/([^/]+)/result$", path):
            self._answer(200, self._result(match.group(1)))
        elif match := re.match(r"^/v1/xtrace/jobs/([^/]+)$", path):
            job = state.map_job.get(match.group(1))
            if job is None:
                self._fail(404, "not_found", "no such job")
                return
            if job["state"] in ("queued", "running"):
                job["_polls"] += 1
                if job["_polls"] > state.job_polls:
                    self._finish(job)
                else:
                    job["state"] = "running"
            self._answer(
                200,
                {"job": self._public(job), "attempts": [], "poll_after_milliseconds": 1},
            )
        else:
            self._fail(404, "not_found", "no route")

    @staticmethod
    def _public(job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if not key.startswith("_")}

    def _poll_inspection(self, inspection_id: str) -> dict[str, Any]:
        state = FakeControlPlane.state
        inspection = state.map_inspection[inspection_id]
        inspection["_polls"] += 1
        if inspection["state"] in ("queued", "running") and inspection["_polls"] > state.inspection_polls:
            artifact = state.map_artifact[inspection["cubin_artifact_id"]]
            if not artifact["data"].startswith(b"\x7fELF"):
                inspection["state"] = "failed"
                inspection["failure"] = {"code": "binary_invalid", "message": "not a cubin"}
            else:
                with open(state.inspection_fixture, "rb") as fixture_file:
                    map_listing = json.load(fixture_file)
                if inspection["kernel_symbol"] is None:
                    map_listing["kernel"] = None
                result_id = self._store(json.dumps(map_listing).encode(), "kernel_metadata", "inspection.json")
                grant = self._download_grant(result_id, "inspection.json")
                inspection["state"] = "succeeded"
                inspection["finished_at"] = TIMESTAMP
                inspection["result"] = {
                    key: grant[key] for key in ("artifact_id", "download_url", "expires_at", "sha256", "size_bytes")
                }
        return self._public(inspection)

    def _finish(self, job: dict[str, Any]) -> None:
        state = FakeControlPlane.state
        with open(os.path.join(FIXTURE_DIRECTORY, "trace_sass_regions.json"), "rb") as trace_file:
            trace = trace_file.read()
        summary = json.dumps({"quality": "usable", "reason_codes": [], "correctness": "passed"}).encode()
        list_file = []
        for name, kind, data, encoding in (
            ("xtrace.json", "xtrace", trace, "identity"),
            ("summary.json", "summary", gzip.compress(summary), "gzip"),
        ):
            artifact_id = self._store(data, kind, name)
            entry = {
                "artifact_id": artifact_id,
                "kind": kind,
                "name": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "content_type": "application/json",
                "encoding": encoding,
            }
            if encoding == "gzip":
                entry["uncompressed_sha256"] = hashlib.sha256(summary).hexdigest()
                entry["uncompressed_size_bytes"] = len(summary)
            list_file.append(entry)
        job["state"] = "succeeded"
        job["result_manifest"] = {
            "schema_version": "marscompute.xtrace.result.v1",
            "kind": "trace",
            "job_id": job["job_id"],
            "attempt_id": "attempt-1",
            "epoch": 1,
            "publication_version": 1,
            "published_at": TIMESTAMP,
            "status": "complete",
            "evidence": "gpu",
            "quality": "usable",
            "reason_codes": [],
            "correctness": "passed",
            "files": list_file,
            "environment": {},
        }
        state.map_job[job["job_id"]] = job

    def _result(self, job_id: str) -> dict[str, Any]:
        job = FakeControlPlane.state.map_job[job_id]
        return {
            "manifest": job["result_manifest"],
            "downloads": [
                self._download_grant(entry["artifact_id"], entry["name"], entry["encoding"])
                for entry in job["result_manifest"]["files"]
            ],
        }

    def do_POST(self) -> None:
        body = self._record()
        if self.headers.get("Authorization") != "Bearer mc_live_v1_test":
            self._fail(401, "unauthorized", "no key")
            return
        state = FakeControlPlane.state
        path = urllib.parse.urlsplit(self.path).path
        if self.headers.get("Content-Type") != "application/json":
            self._fail(400, "json_content_type_required", "json only")
            return
        map_body = json.loads(body)
        if path == "/v1/xtrace/artifacts/uploads":
            artifact_id = str(uuid.uuid4())
            artifact = {**map_body, "status": "staging", "data": b"", "parts": {}}
            state.map_artifact[artifact_id] = artifact
            grant: dict[str, Any] = {
                "artifact_id": artifact_id,
                "upload_url": f"{state.origin}/upload/{artifact_id}",
                "expires_at": TIMESTAMP,
                "method": "PUT",
                "headers": {"x-amz-meta-kind": map_body["kind"]},
            }
            if map_body["size_bytes"] > state.multipart_threshold:
                grant["parts"] = [
                    {
                        "part_number": number + 1,
                        "upload_url": f"{state.origin}/upload/{artifact_id}/part/{number + 1}",
                        "offset": offset,
                        "size_bytes": min(state.part_bytes, map_body["size_bytes"] - offset),
                    }
                    for number, offset in enumerate(range(0, map_body["size_bytes"], state.part_bytes))
                ]
            artifact["grant"] = grant
            self._answer(201, grant)
        elif match := re.match(r"^/v1/xtrace/artifacts/([^/]+)/complete$", path):
            artifact = state.map_artifact[match.group(1)]
            if "parts" in artifact["grant"]:
                if [part["etag"] for part in map_body.get("parts", [])] != [
                    f'"etag-{number}"' for number in sorted(artifact["parts"])
                ]:
                    self._fail(422, "multipart_mismatch", "receipts differ")
                    return
                artifact["data"] = b"".join(artifact["parts"][number] for number in sorted(artifact["parts"]))
            data = artifact["data"]
            if hashlib.sha256(data).hexdigest() != map_body["sha256"] or len(data) != map_body["size_bytes"]:
                self._fail(422, "artifact_verification_failed", "bytes differ")
                return
            artifact["status"] = "verified"
            state.map_blob[match.group(1)] = data
            if artifact["kind"] == "launch_manifest":
                try:
                    SchemaCheck.validate(SchemaCheck.LAUNCH_MANIFEST, json.loads(data))
                except jsonschema.ValidationError as err:
                    state.list_schema_error.append(f"launch manifest: {err.message}")
            self._answer(
                200,
                {
                    "artifact_id": match.group(1),
                    "kind": artifact["kind"],
                    "name": artifact.get("name"),
                    "sha256": map_body["sha256"],
                    "size_bytes": map_body["size_bytes"],
                    "content_type": artifact["content_type"],
                    "status": "verified",
                    "created_at": TIMESTAMP,
                    "expires_at": TIMESTAMP,
                },
            )
        elif path == "/v1/xtrace/inspections":
            artifact = state.map_artifact.get(map_body["cubin_artifact_id"])
            if artifact is None or artifact["kind"] != "cubin" or artifact["status"] != "verified":
                self._fail(422, "artifact_not_cubin", "not a verified cubin")
                return
            inspection_id = str(uuid.uuid4())
            state.map_inspection[inspection_id] = {
                "inspection_id": inspection_id,
                "state": "queued",
                "cubin_artifact_id": map_body["cubin_artifact_id"],
                "kernel_symbol": map_body.get("kernel_symbol"),
                "created_at": TIMESTAMP,
                "finished_at": None,
                "failure": None,
                "result": None,
                "_polls": 0,
            }
            self._answer(202, self._public(state.map_inspection[inspection_id]))
        elif path == "/v1/xtrace/quotes":
            for key, artifact_id in map_body["bundle"].items():
                if key.endswith("_artifact_id") and state.map_artifact[artifact_id]["status"] != "verified":
                    self._fail(422, "artifact_not_verified", key)
                    return
            quote_id = str(uuid.uuid4())
            state.map_quote[quote_id] = {"request": map_body}
            self._answer(
                201,
                {
                    "quote_id": quote_id,
                    "request_hash": "c" * 64,
                    "input_digests": [],
                    "compute": map_body["compute"],
                    "permitted_fallbacks": [],
                    "limits": map_body["limits"],
                    "rate_snapshot": {
                        "rate_version": "rate-1",
                        "sku_id": "sku-1",
                        "image_digest": "sha256:" + "a" * 64,
                        "gpu_micro_usd_per_hour": "3200000",
                        "failure_policy_version": "policy-1",
                        "gpu_time_policy": "authorized_execution_only",
                        "model_rates": [],
                    },
                    "created_at": TIMESTAMP,
                    "expires_at": TIMESTAMP,
                },
            )
        elif path == "/v1/xtrace/jobs":
            key = self.headers.get("Idempotency-Key")
            if not key or map_body.get("idempotency_key") not in (None, key):
                self._fail(400, "idempotency_key_mismatch", "header and body differ")
                return
            if key in state.map_job_of_key:
                self._answer(201, self._public(state.map_job[state.map_job_of_key[key]]))
                return
            quote = state.map_quote.get(map_body["quote_id"])
            request = {k: v for k, v in map_body.items() if k not in ("quote_id", "idempotency_key")}
            if quote is None or quote["request"] != request:
                self._fail(409, "quote_mismatch", "the request differs from the quote")
                return
            if int(map_body["limits"]["max_total_cost_micro_usd"]) > state.available_micro_usd:
                self._fail(402, "insufficient_credit", "credit does not cover the quote", top_up_url=TOP_UP_URL)
                return
            job_id = str(uuid.uuid4())
            state.map_job[job_id] = {
                "job_id": job_id,
                "tenant_id": "tenant-1",
                "account_id": "account-1",
                "credential_id": "credential-1",
                "quote_id": map_body["quote_id"],
                "state": "queued",
                "version": 1,
                "created_at": TIMESTAMP,
                "updated_at": TIMESTAMP,
                "deadline_at": TIMESTAMP,
                "cancel_requested_at": None,
                "compute": map_body["compute"],
                "limits": map_body["limits"],
                "cleanup": {"state": "not_required", "provider_confirmed_at": None, "resources": []},
                "accounting": {
                    "state": "held",
                    "hold_id": "hold-1",
                    "reserved_micro_usd": map_body["limits"]["max_total_cost_micro_usd"],
                    "settled_micro_usd": "0",
                    "producer_closed": False,
                },
                "result_manifest": None,
                "error": None,
                "_polls": 0,
            }
            state.map_job_of_key[key] = job_id
            state.list_job_order.append(job_id)
            self._answer(201, self._public(state.map_job[job_id]))
        elif match := re.match(r"^/v1/xtrace/jobs/([^/]+)/cancel$", path):
            if map_body != {}:
                self._fail(400, "invalid_payload", "cancel takes {}")
                return
            job = state.map_job[match.group(1)]
            job["state"] = "cancelled"
            job["cancel_requested_at"] = TIMESTAMP
            self._answer(200, self._public(job))
        else:
            self._fail(404, "not_found", "no route")


class FakeServer:
    """The fake Control Plane on a local port, started once per test class."""

    def __init__(self) -> None:
        FakeControlPlane.state = FakeState()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeControlPlane)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        FakeControlPlane.state.origin = self.origin
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def state(self) -> FakeState:
        return FakeControlPlane.state

    def reset(self) -> None:
        FakeControlPlane.state = FakeState()
        FakeControlPlane.state.origin = self.origin

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
