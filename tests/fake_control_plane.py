import gzip
import hashlib
import http.server
import json
import os
import re
import shutil
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
LATER = "2026-11-08T13:00:00.000Z"
OWNER = {"tenant_id": "6f1c3b8e-0d4a-4c3e-9a51-0e3f4b2a7c10", "account_id": "0b8e6f52-3c1d-4b7a-8e2f-5a9c1d3e7b20"}
DIGEST = "d" * 64
IMAGE = "sha256:" + "e" * 64
INSPECTION_PROFILE = {
    "profile": {
        "profile_id": "static-cubin-v1",
        "revision": "r1",
        "image_digest": IMAGE,
        "parser_revision": "p1",
        "capability_evidence_sha256": DIGEST,
    },
    "limits": {
        "max_input_bytes": 1 << 27,
        "max_output_bytes": 1 << 24,
        "max_kernels": 256,
        "max_instructions_per_kernel": 1 << 20,
        "max_parameters_per_kernel": 256,
        "cpus": 1,
        "memory_bytes": 1 << 30,
        "pids": 64,
        "scratch_bytes": 1 << 28,
        "max_queue_seconds": 600,
        "max_wall_seconds": 120,
        "max_tool_seconds": 60,
        "max_upload_seconds": 60,
        "max_cleanup_seconds": 60,
    },
    "price": {
        "currency": "USD",
        "fixed_fee_micro_usd": "20000",
        "price_revision": "price-1",
        "charge_policy_revision": "policy-1",
        "charge_policy_sha256": DIGEST,
        "tax_behavior": "exclusive",
        "billing_mapping_sha256": DIGEST,
    },
}
# the parameter layout of square_matrix_mul(const int*, const int*, int*, int)
MATMUL_PARAMETERS = [
    {"index": 0, "offset": 0, "size": 8},
    {"index": 1, "offset": 8, "size": 8},
    {"index": 2, "offset": 16, "size": 8},
    {"index": 3, "offset": 24, "size": 4},
]
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
COMPUTE_H200 = dict(COMPUTE_H100, gpu_type="H200", purchase_option="spot")
COMPUTE_B300 = dict(COMPUTE_H100, gpu_type="B300", purchase_option="spot")
AGENT_MODEL = "claude-opus-5-5"
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
                "comparison": ["sm_90a"],
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
            "architectures": {"xtrace": ["sm_100a"], "range_profile": [], "pc_sampling": [], "comparison": []},
        },
        {
            "compute": COMPUTE_H200,
            "environment_id": "h200-cuda12.9",
            "image_digest": "sha256:" + "c" * 64,
            "architecture": "sm_90a",
            "qualified": True,
            "bundle_kinds": ["cubin_launch", "agent_workspace"],
            "build_templates": [],
            "stages": ["xtrace"],
            "gpu_micro_usd_per_hour": "4000000",
            "architectures": {"xtrace": ["sm_90a"], "range_profile": [], "pc_sampling": [], "comparison": []},
        },
        {
            "compute": COMPUTE_B300,
            "environment_id": "b300-cuda13.0",
            "image_digest": "sha256:" + "f" * 64,
            "architecture": "sm_103a",
            "qualified": True,
            "bundle_kinds": ["agent_workspace"],
            "build_templates": [],
            "stages": ["xtrace"],
            "gpu_micro_usd_per_hour": "9600000",
            "architectures": {"xtrace": ["sm_103a"], "range_profile": [], "pc_sampling": [], "comparison": []},
        },
    ],
    "models": ["planner-small"],
    "agent_models": [AGENT_MODEL],
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


def agent_job(directory: str) -> None:
    """Write an agent-mode job of the example cubin, its source and a driver script into a directory."""
    shutil.copyfile(os.path.join(EXAMPLE_DIRECTORY, "matmul.cubin"), os.path.join(directory, "matmul.cubin"))
    for relative_path, text in (
        ("src/matmul.cu", "// kernel\n"),
        ("src/.git/HEAD", "ref: refs/heads/main\n"),
        ("run.py", "print('drive')\n"),
    ):
        os.makedirs(os.path.dirname(os.path.join(directory, relative_path)), exist_ok=True)
        with open(os.path.join(directory, relative_path), "w") as source_file:
            source_file.write(text)
    with open(os.path.join(directory, "xtrace.toml"), "w") as manifest_file:
        manifest_file.write(
            'hardware = "h200-cuda12.9"\nintent = "find the longest bubble of the mainloop"\n'
            '[workspace]\ncubins = ["matmul.cubin"]\nsource = "src"\nfiles = ["run.py"]\n'
            f'[agent]\nmodel = "{AGENT_MODEL}"\nmax_turns = 50\nmax_input_tokens = 2000000\n'
            "max_output_tokens = 100000\nmax_wall_seconds = 1800\n"
            "[limits]\nmax_cost_usd = 30.0\nmax_gpu_seconds = 2400\n"
        )


def candidates_of(symbol: str) -> bytes:
    """The site candidates document of a kernel, which names no qualified site."""
    return json.dumps(
        {
            "schema_version": "marscompute.xtrace.static-candidates.v1",
            "symbol": symbol,
            "policy_version": "no-qualified-native-sites-v1",
            "candidates": [],
        }
    ).encode()


def inspection_result(
    inspection_id: str,
    quote_id: str,
    quote_sha256: str,
    request: dict[str, Any],
    listing: bytes,
    listing_id: str,
    candidates: bytes,
    candidates_id: str,
) -> dict[str, Any]:
    """An inspection result of one kernel whose listing and candidates are the given artifacts."""
    map_listing = json.loads(listing)
    count = len(map_listing["instructions"])
    attempt = {"attempt_id": "attempt-1", "epoch": 1, "supervisor_id": "supervisor-1", "started_at": TIMESTAMP}
    return {
        "schema_version": "marscompute.xtrace.inspection-result.v1",
        "inspection_id": inspection_id,
        "quote_id": quote_id,
        "quote_sha256": quote_sha256,
        "request_sha256": FakeControlPlane.digest(request),
        "binary": request["binary"],
        "profile_id": "static-cubin-v1",
        "profile_revision": "r1",
        "image_digest": IMAGE,
        "parser_revision": "p1",
        "static_only": True,
        "gpu_executed": False,
        "kernel_catalog": {
            "coverage": {"returned_count": 1, "total_count": 1, "complete": True, "truncated": False},
            "kernels": [
                {
                    "symbol": map_listing["symbol"],
                    "architecture": "sm_90a",
                    "parameter_layout": {"state": "extracted", "evidence_sha256": DIGEST, "parameters": MATMUL_PARAMETERS},
                    "resources": {"registers": None, "static_shared_memory_bytes": None},
                    "instructions": {
                        "artifact": {
                            "artifact_id": listing_id,
                            "sha256": hashlib.sha256(listing).hexdigest(),
                            "size_bytes": len(listing),
                            "media_type": "application/json",
                        },
                        "coverage": {"returned_count": count, "total_count": count, "complete": True, "truncated": False},
                        "index_space_id": map_listing["index_space_id"],
                    },
                    "site_policy": {
                        "policy_version": "no-qualified-native-sites-v1",
                        "candidates": {
                            "artifact": {
                                "artifact_id": candidates_id,
                                "sha256": hashlib.sha256(candidates).hexdigest(),
                                "size_bytes": len(candidates),
                                "media_type": "application/json",
                            },
                            "coverage": {"returned_count": 0, "total_count": None, "complete": False, "truncated": True},
                        },
                        "native_equivalence": {"state": "not_qualified", "reason": "native_equivalence_not_established"},
                    },
                }
            ],
        },
        "provenance": dict(attempt, toolchain_sha256=DIGEST, completed_at=TIMESTAMP),
    }


def matmul_documents() -> tuple[dict, dict, bytes]:
    """An inspection result of the example kernel, its listing, and the listing's bytes."""
    with open(os.path.join(FIXTURE_DIRECTORY, "matmul_instructions.json"), "rb") as fixture_file:
        listing = fixture_file.read()
    result = inspection_result(
        inspection_id="0c9d2a4e-1b3f-4e5a-8c6d-7e8f9a0b1c2d",
        quote_id="quote-1",
        quote_sha256="a" * 64,
        request={"binary": {"artifact_id": "cubin-1", "sha256": "b" * 64, "size_bytes": 10664, "format": "cubin"}},
        listing=listing,
        listing_id="listing-1",
        candidates=candidates_of("square_matrix_mul"),
        candidates_id="candidates-1",
    )
    return result, json.loads(listing), listing


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
        self.map_inspection_quote: dict[str, dict[str, Any]] = {}
        self.map_inspection_of_key: dict[str, str] = {}
        self.list_inspection_order: list[str] = []
        self.is_inspection_available = True
        # the outcome an inspection ends with, or None to complete every cubin
        self.inspection_outcome: str | None = None
        self.list_job_order: list[str] = []
        self.available_micro_usd = 100_000_000
        # files larger than this are granted as multipart uploads of PART_BYTES parts
        self.multipart_threshold = 1 << 20
        self.part_bytes = 4096
        # polls a job and an inspection stay unfinished for
        self.job_polls = 1
        self.inspection_polls = 1
        self.instructions_fixture = os.path.join(FIXTURE_DIRECTORY, "matmul_instructions.json")

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
        path = urllib.parse.urlsplit(self.path).path
        if status < 300:
            try:
                SchemaCheck.validate_response(self.command, path, value)
            except jsonschema.ValidationError as err:
                FakeControlPlane.state.list_schema_error.append(f"response {path}: {err.message}")
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
        elif path == "/v1/xtrace/inspections/capabilities":
            availability: dict[str, Any] = {"state": "unavailable", "reason": "not_configured", "profiles": []}
            if state.is_inspection_available:
                availability = {"state": "available", "profiles": [dict(INSPECTION_PROFILE, valid_until=LATER)]}
            self._answer(
                200,
                {
                    "schema_version": "marscompute.xtrace.inspection-capabilities.v1",
                    "environment": "test",
                    "availability": availability,
                },
            )
        elif path == "/v1/xtrace/inspections":
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
            limit = int(query["limit"][0])
            list_id = state.list_inspection_order
            if "cursor" in query:
                list_id = list_id[list_id.index(query["cursor"][0]) + 1:]
            self._answer(
                200,
                {
                    "items": [self._public(state.map_inspection[inspection_id]) for inspection_id in list_id[:limit]],
                    "next_cursor": list_id[limit - 1] if len(list_id) > limit else None,
                },
            )
        elif match := re.match(r"^/v1/xtrace/inspections/([^/]+)/result$", path):
            inspection = state.map_inspection.get(match.group(1))
            if inspection is None:
                self._fail(404, "inspection_not_found", "no such inspection")
            elif inspection["_result"] is None:
                self._fail(409, "inspection_result_not_ready", "not completed")
            else:
                self._answer(200, inspection["_result"])
        elif match := re.match(r"^/v1/xtrace/inspections/([^/]+)$", path):
            if match.group(1) not in state.map_inspection:
                self._fail(404, "inspection_not_found", "no such inspection")
                return
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

    @staticmethod
    def digest(value: Any) -> str:
        """The canonical digest the platform binds quotes with: sorted keys, no whitespace."""
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()

    def _inspection_result(self, inspection: dict[str, Any]) -> dict[str, Any]:
        """The result of a completed inspection, with the listing of its kernel stored as an artifact."""
        with open(FakeControlPlane.state.instructions_fixture, "rb") as fixture_file:
            listing = fixture_file.read()
        candidates = candidates_of(json.loads(listing)["symbol"])
        return inspection_result(
            inspection_id=inspection["inspection_id"],
            quote_id=inspection["quote_id"],
            quote_sha256=inspection["quote_sha256"],
            request=inspection["_request"],
            listing=listing,
            listing_id=self._store(listing, "kernel_metadata", "kernel-000-instructions.json"),
            candidates=candidates,
            candidates_id=self._store(candidates, "kernel_metadata", "kernel-000-candidates.json"),
        )

    def _poll_inspection(self, inspection_id: str) -> dict[str, Any]:
        state = FakeControlPlane.state
        inspection = state.map_inspection[inspection_id]
        inspection["_polls"] += 1
        attempt = {"attempt_id": "attempt-1", "epoch": 1, "supervisor_id": "supervisor-1", "started_at": TIMESTAMP}
        if inspection["execution"]["state"] == "queued":
            inspection["execution"] = {"state": "running", "attempt": attempt}
        elif inspection["execution"]["state"] == "running" and inspection["_polls"] > state.inspection_polls:
            artifact = state.map_artifact[inspection["_request"]["binary"]["artifact_id"]]
            outcome = state.inspection_outcome or ("completed" if artifact["data"].startswith(b"\x7fELF") else "rejected_input")
            if outcome == "completed":
                inspection["_result"] = self._inspection_result(inspection)
            inspection["execution"] = {
                "state": "terminal",
                "outcome": outcome,
                "attempt": attempt,
                "completed_at": TIMESTAMP,
                "result_sha256": FakeControlPlane.digest(inspection["_result"]) if outcome == "completed" else None,
            }
            inspection["cleanup"] = {"state": "confirmed", "evidence_sha256": DIGEST, "confirmed_at": TIMESTAMP}
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
            artifact["sha256"] = map_body["sha256"]
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
        elif path == "/v1/xtrace/inspections/quotes":
            artifact = state.map_artifact.get(map_body["binary"]["artifact_id"])
            if (
                artifact is None
                or artifact["kind"] != "cubin"
                or artifact["status"] != "verified"
                or artifact["sha256"] != map_body["binary"]["sha256"]
            ):
                self._fail(422, "inspection_input_unavailable", "not a verified cubin")
                return
            quote = {
                "schema_version": "marscompute.xtrace.inspection-quote.v1",
                "quote_id": str(uuid.uuid4()),
                "owner": OWNER,
                "environment": "test",
                "request_sha256": FakeControlPlane.digest(map_body),
                "profile": INSPECTION_PROFILE["profile"],
                "limits": INSPECTION_PROFILE["limits"],
                "price": INSPECTION_PROFILE["price"],
                "issued_at": TIMESTAMP,
                "expires_at": LATER,
            }
            state.map_inspection_quote[quote["quote_id"]] = quote
            self._answer(201, quote)
        elif path == "/v1/xtrace/inspections":
            key = self.headers.get("Idempotency-Key")
            if key is not None and key != map_body["idempotency_key"]:
                self._fail(409, "idempotency_conflict", "header and body differ")
                return
            if map_body["idempotency_key"] in state.map_inspection_of_key:
                self._answer(201, self._public(state.map_inspection[state.map_inspection_of_key[map_body["idempotency_key"]]]))
                return
            quote = state.map_inspection_quote.get(map_body["quote_id"])
            if (
                quote is None
                or map_body["quote_sha256"] != FakeControlPlane.digest(quote)
                or quote["request_sha256"] != FakeControlPlane.digest(map_body["request"])
            ):
                self._fail(409, "inspection_acceptance_binding_mismatch", "the acceptance differs from the quote")
                return
            if int(quote["price"]["fixed_fee_micro_usd"]) > state.available_micro_usd:
                self._fail(409, "insufficient_credit", "credit does not cover the fee", top_up_url=TOP_UP_URL)
                return
            inspection_id = str(uuid.uuid4())
            state.map_inspection[inspection_id] = {
                "schema_version": "marscompute.xtrace.inspection-status.v1",
                "inspection_id": inspection_id,
                "owner": OWNER,
                "environment": "test",
                "quote_id": quote["quote_id"],
                "quote_sha256": map_body["quote_sha256"],
                "created_at": TIMESTAMP,
                "execution": {"state": "queued"},
                "cleanup": {"state": "pending"},
                "accounting": {"state": "held", "reserved_micro_usd": quote["price"]["fixed_fee_micro_usd"]},
                "_request": map_body["request"],
                "_result": None,
                "_polls": 0,
            }
            state.map_inspection_of_key[map_body["idempotency_key"]] = inspection_id
            state.list_inspection_order.append(inspection_id)
            self._answer(201, self._public(state.map_inspection[inspection_id]))
        elif match := re.match(r"^/v1/xtrace/inspections/([^/]+)/cancel$", path):
            inspection = state.map_inspection[match.group(1)]
            if map_body != {}:
                self._fail(400, "invalid_payload", "cancel takes {}")
                return
            if inspection["execution"]["state"] != "terminal":
                inspection["execution"] = {"state": "cancel_requested", "attempt": inspection["execution"].get("attempt")}
            self._answer(200, self._public(inspection))
        elif match := re.match(r"^/v1/xtrace/inspections/([^/]+)/artifacts/downloads$", path):
            inspection = state.map_inspection[match.group(1)]
            result = inspection["_result"]
            set_named = {
                kernel[part]["artifact"]["artifact_id"] if part == "instructions" else kernel[part]["candidates"]["artifact"]["artifact_id"]
                for kernel in (result or {"kernel_catalog": {"kernels": []}})["kernel_catalog"]["kernels"]
                for part in ("instructions", "site_policy")
            }
            if not set(map_body["artifact_ids"]) <= set_named:
                self._fail(404, "inspection_result_artifact_unavailable", "not named by the result")
                return
            self._answer(
                200,
                {"artifacts": [self._download_grant(artifact_id, state.map_artifact[artifact_id]["name"]) for artifact_id in map_body["artifact_ids"]]},
            )
        elif path == "/v1/xtrace/quotes":
            for key, value in map_body["bundle"].items():
                list_id = value if key.endswith("_artifact_ids") else [value] if key.endswith("_artifact_id") else []
                if any(state.map_artifact[artifact_id]["status"] != "verified" for artifact_id in list_id):
                    self._fail(422, "artifact_not_verified", key)
                    return
            quote_id = str(uuid.uuid4())
            state.map_quote[quote_id] = {"request": map_body}
            quote = {
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
            }
            if map_body["analysis"]["mode"] == "agent":
                quote["rate_snapshot"]["model_rates"] = [
                    {
                        "model": map_body["analysis"]["model"],
                        "price_snapshot_id": "snapshot-1",
                        "input_micro_usd_per_million": "18000000",
                        "cached_input_micro_usd_per_million": "1800000",
                        "output_micro_usd_per_million": "90000000",
                    },
                ]
                quote["rate_snapshot"]["price_book"] = {
                    "rate_card_version_id": "card-1",
                    "tool_fee_micro_usd": "500000",
                    "cache_write_rates": [
                        {
                            "dimension": "cache_write_300_explicit_tokens",
                            "micro_usd_per_million": "22500000",
                        },
                    ],
                }
            self._answer(201, quote)
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
