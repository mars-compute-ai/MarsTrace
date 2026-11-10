import base64
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

FIXTURE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
)
EXAMPLE_DIRECTORY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples",
    "matmul",
)
API_KEY = "mt_live_test"
TIMESTAMP = "2026-11-10T12:00:00Z"
ADD_CREDIT_URL = "https://console.marscompute.ai/tenant/billing/credits"
AGENT_MODEL = "claude-opus-5-5"
CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"
RATES_H200 = {
    "gpu_micro_usd_per_hour": "4320000",
    "job_fee_micro_usd": "500000",
    "agent_input_micro_usd_per_million": "18000000",
    "agent_cache_read_micro_usd_per_million": "1800000",
    "agent_cache_write_5m_micro_usd_per_million": "22500000",
    "agent_cache_write_1h_micro_usd_per_million": "36000000",
    "agent_output_micro_usd_per_million": "90000000",
}
AGENT_DEFAULTS = {"max_turns": 60, "max_input_tokens": 4000000, "max_output_tokens": 60000, "max_wall_seconds": 1200}
AGENT_MAXIMUMS = {"max_turns": 500, "max_input_tokens": 50000000, "max_output_tokens": 1000000, "max_wall_seconds": 2580}
HARDWARE = {
    "hardware": [
        {"id": "h200", "gpu": "H200", "architecture": "sm_90a", "available": True, "service_model": "marstrace-h200", "rates": RATES_H200},
        {
            "id": "b200", "gpu": "B200", "architecture": "sm_100a", "available": True, "service_model": "marstrace-b200",
            "rates": dict(RATES_H200, gpu_micro_usd_per_hour="6480000"),
        },
        {
            "id": "b300", "gpu": "B300", "architecture": "sm_103a", "available": False, "service_model": "marstrace-b300",
            "rates": dict(RATES_H200, gpu_micro_usd_per_hour="8640000"),
        },
    ],
    "agent": {"model": AGENT_MODEL, "defaults": AGENT_DEFAULTS, "maximums": AGENT_MAXIMUMS},
}
# the order a job's status moves in, one step per long poll
LIST_JOB_STEP = ["queued", "starting", "running", "finishing", "succeeded"]
SET_FINAL = {"succeeded", "failed", "cancelled"}
SET_ROLE = {"cubin", "source_archive", "launch_manifest", "tensor", "file"}
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
FINALIZE_MARGIN_SECONDS = 120
MILLISECONDS_PER_HOUR = 3_600_000
AGENT_BUDGET_MIN = 1_000_000


def new_id(prefix: str) -> str:
    """A prefixed id of 26 Crockford base32 characters."""
    number = uuid.uuid4().int
    return prefix + "".join(CROCKFORD[(number >> (5 * index)) & 31] for index in range(26))


def agent_job(directory: str) -> None:
    """Write a job of the example cubin, its source and a driver script into a directory."""
    shutil.copyfile(os.path.join(EXAMPLE_DIRECTORY, "matmul.cubin"), os.path.join(directory, "matmul.cubin"))
    for relative_path, text in (
        ("src/matmul.cu", "// kernel\n"),
        ("src/.git/HEAD", "ref: refs/heads/main\n"),
        ("run.py", "print('drive')\n"),
    ):
        os.makedirs(os.path.dirname(os.path.join(directory, relative_path)), exist_ok=True)
        with open(os.path.join(directory, relative_path), "w") as source_file:
            source_file.write(text)
    with open(os.path.join(directory, "marstrace.toml"), "w") as manifest_file:
        manifest_file.write(
            'hardware = "h200"\nintent = "find the longest bubble of the mainloop"\n'
            '[workspace]\ncubins = ["matmul.cubin"]\nsource = "src"\nfiles = ["run.py"]\n'
            "[agent]\nmax_turns = 50\nmax_input_tokens = 2000000\nmax_output_tokens = 100000\nmax_wall_seconds = 1800\n"
            "[limits]\nmax_cost_usd = 30.0\nmax_attempts = 1\n"
        )


class RequestShape:
    """The request bodies of section 3 of the MarsTrace contract; a key not named is refused."""

    UPLOAD = {"role": str, "name": str, "size_bytes": int, "sha256": str, "content_type": str}
    COMPLETE = {"sha256": str, "size_bytes": int, "parts?": list}
    AGENT = {"max_turns": int, "max_input_tokens": int, "max_output_tokens": int, "max_wall_seconds": int}
    INPUTS = {"cubins": list, "source_archive?": str, "launch_manifest?": str, "tensors?": list, "files?": list}
    LIMITS = {"max_attempts?": int, "max_cost_micro_usd": str}
    JOB = {"hardware": str, "prompt": str, "inputs": INPUTS, "agent?": AGENT, "limits": LIMITS}
    SUBMIT = dict(JOB, **{"quote_id?": str})

    @staticmethod
    def departure(value: Any, shape: Any, where: str = "body") -> str | None:
        """The first way a request departs from its shape, or None."""
        if isinstance(shape, dict):
            if not isinstance(value, dict):
                return f"{where} is not an object"
            set_known = {key.rstrip("?") for key in shape}
            set_unknown = set(value) - set_known
            if set_unknown:
                return f"{where} has unknown field(s) {', '.join(sorted(set_unknown))}"
            for key, item_shape in shape.items():
                name = key.rstrip("?")
                if name not in value:
                    if key.endswith("?"):
                        continue
                    return f"{where}.{name} is required"
                departure = RequestShape.departure(value[name], item_shape, f"{where}.{name}")
                if departure:
                    return departure
            return None
        if shape is int and isinstance(value, bool):
            return f"{where} is not an integer"
        return None if isinstance(value, shape) else f"{where} is not of type {shape.__name__}"


class FakeState:
    """What the fake MarsTrace service holds between requests."""

    def __init__(self) -> None:
        self.origin = ""
        self.list_request: list[dict[str, Any]] = []
        self.list_contract_error: list[str] = []
        self.map_upload: dict[str, dict[str, Any]] = {}
        self.map_object: dict[str, bytes] = {}
        self.map_quote: dict[str, dict[str, Any]] = {}
        self.map_job: dict[str, dict[str, Any]] = {}
        # idempotency key -> (digest of the canonical body, job id)
        self.map_idempotency: dict[str, tuple[str, str]] = {}
        self.list_job_order: list[str] = []
        self.available_micro_usd: int | None = 100_000_000
        self.map_hardware = json.loads(json.dumps(HARDWARE))
        # files larger than this are granted as multipart uploads of part_bytes parts
        self.multipart_threshold = 1 << 20
        self.part_bytes = 4096
        # whether a long poll moves a job a step; False keeps it where it is
        self.do_advance = True
        # the job status a job stops at, before succeeding
        self.hold_status: str | None = None
        # submissions to drop without an answer, to exercise the retry under one key
        self.drop_submissions = 0

    def requests_to(self, method: str, pattern: str) -> list[dict[str, Any]]:
        regex = re.compile(pattern)
        return [request for request in self.list_request if request["method"] == method and regex.match(request["path"])]


class ServiceError(Exception):
    """A refusal the fake answers with an error envelope."""

    def __init__(self, status: int, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable


class FakeService(http.server.BaseHTTPRequestHandler):
    """A stand-in for the MarsTrace service's customer API and for R2's presigned URLs."""

    state = FakeState()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send(self, status: int, data: bytes, map_header: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (map_header or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _envelope(self, status: int, data: Any, error: dict[str, Any] | None) -> None:
        request_id = "req_" + uuid.uuid4().hex[:12]
        body = json.dumps({"request_id": request_id, "data": data, "error": error}).encode()
        self._send(status, body, {"Content-Type": "application/json", "x-request-id": request_id})

    def _record(self) -> bytes:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        split = urllib.parse.urlsplit(self.path)
        FakeService.state.list_request.append(
            {
                "method": self.command,
                "path": split.path,
                "query": urllib.parse.parse_qs(split.query),
                "authorization": self.headers.get("Authorization"),
                "idempotency": self.headers.get("Idempotency-Key"),
                "content_type": self.headers.get("Content-Type"),
                "headers": {name.lower(): value for name, value in self.headers.items()},
                "body": body,
            }
        )
        return body

    def _dispatch(self, handler: Any, *args: Any) -> None:
        try:
            status, data = handler(*args)
        except ServiceError as err:
            if err.status == 400:
                FakeService.state.list_contract_error.append(f"{self.command} {self.path}: {err}")
            self._envelope(err.status, None, {"code": err.code, "message": str(err), "retryable": err.retryable})
            return
        self._envelope(status, data, None)

    def _authorize(self) -> None:
        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            raise ServiceError(401, "authentication_required", "Missing or unknown MarsTrace API key")

    def _body(self, body: bytes, shape: dict[str, Any]) -> dict[str, Any]:
        if self.headers.get("Content-Type") != "application/json":
            raise ServiceError(400, "invalid_request", "the body must be JSON")
        try:
            map_body = json.loads(body)
        except ValueError as err:
            raise ServiceError(400, "invalid_request", "the body is not JSON") from err
        departure = RequestShape.departure(map_body, shape)
        if departure:
            raise ServiceError(400, "invalid_request", departure)
        return map_body

    # presigned storage

    def do_PUT(self) -> None:
        body = self._record()
        state = FakeService.state
        match = re.match(r"^/r2/(upl_[a-z0-9]+)(?:/part/(\d+))?$", urllib.parse.urlsplit(self.path).path)
        if match is None or self.headers.get("Authorization") is not None:
            self._send(403, b"<Error>forbidden</Error>")
            return
        upload = state.map_upload[match.group(1)]
        if match.group(2) is None:
            digest = base64.b64encode(hashlib.sha256(body).digest()).decode()
            if self.headers.get("x-amz-checksum-sha256") != digest:
                self._send(400, b"<Error>BadDigest</Error>")
                return
            upload["data"] = body
            self._send(200, b"")
            return
        upload["parts"][int(match.group(2))] = body
        self._send(200, b"", {"ETag": f'"etag-{match.group(2)}"'})

    def do_GET(self) -> None:
        self._record()
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/r2/results/"):
            self._send(200, FakeService.state.map_object[path.rsplit("/", 1)[1]])
            return
        self._dispatch(self._get, path)

    def do_POST(self) -> None:
        body = self._record()
        state = FakeService.state
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/jobs" and state.drop_submissions > 0:
            state.drop_submissions -= 1
            self.close_connection = True
            self.connection.close()
            return
        self._dispatch(self._post, path, body)

    def _get(self, path: str) -> tuple[int, Any]:
        self._authorize()
        state = FakeService.state
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        if path == "/v1/hardware":
            return 200, state.map_hardware
        if path == "/v1/credit":
            return 200, {
                "billing_mode": "prepaid",
                "billing_status": "billable_active",
                "available_micro_usd": None if state.available_micro_usd is None else str(state.available_micro_usd),
                "reserved_micro_usd": str(sum(int(job["cost"]["reserved_micro_usd"] or 0) for job in state.map_job.values() if job["status"] not in SET_FINAL)),
                "add_credit_url": ADD_CREDIT_URL,
            }
        if path == "/v1/jobs":
            limit = int(query.get("limit", ["20"])[0])
            if not 1 <= limit <= 100:
                raise ServiceError(400, "invalid_request", "limit is 1 to 100")
            start = int(query.get("cursor", ["0"])[0])
            list_id = list(reversed(state.list_job_order))[start:start + limit]
            return 200, {
                "items": [state.map_job[job_id] for job_id in list_id],
                "next_cursor": str(start + limit) if start + limit < len(state.list_job_order) else None,
            }
        if match := re.match(r"^/v1/jobs/([^/]+)/wait$", path):
            timeout_seconds = int(query.get("timeout_seconds", ["30"])[0])
            if not 1 <= timeout_seconds <= 60:
                raise ServiceError(400, "invalid_request", "timeout_seconds is 1 to 60")
            job = self._job(match.group(1))
            self._advance(job)
            return 200, job
        if match := re.match(r"^/v1/jobs/([^/]+)/result$", path):
            job = self._job(match.group(1))
            if job["status"] != "succeeded":
                raise ServiceError(409, "result_unavailable", "The job has no result")
            return 200, self._result(job)
        if match := re.match(r"^/v1/jobs/([^/]+)$", path):
            return 200, self._job(match.group(1))
        raise ServiceError(404, "not_found", "No such route")

    def _job(self, job_id: str) -> dict[str, Any]:
        job = FakeService.state.map_job.get(job_id)
        if job is None:
            raise ServiceError(404, "not_found", "No such job for this account")
        return job

    def _advance(self, job: dict[str, Any]) -> None:
        state = FakeService.state
        if not state.do_advance or job["status"] in SET_FINAL or job["status"] == state.hold_status:
            return
        job["status"] = LIST_JOB_STEP[LIST_JOB_STEP.index(job["status"]) + 1]
        if job["status"] == "starting":
            job["attempts"].append({"number": 1, "status": "running", "started_at": TIMESTAMP})
        if job["status"] == "running":
            job["progress"] = {"agent_turn": 14, "gpu_seconds": 312}
        if job["status"] == "succeeded":
            with open(os.path.join(FIXTURE_DIRECTORY, "trace_sass_regions.json"), "rb") as trace_file:
                trace = trace_file.read()
            state.map_object[job["job_id"]] = trace
            job["attempts"][-1].update(status="succeeded", ended_at=TIMESTAMP)
            job["cost"] = {
                "reserved_micro_usd": job["cost"]["reserved_micro_usd"],
                "charged_micro_usd": "14128100",
                "lines": [
                    {"dimension": "service_job_fee", "quantity": 1, "amount": "500000"},
                    {"dimension": "service_gpu_milliseconds", "quantity": 905000, "amount": "1086000"},
                    {"dimension": "service_agent_output_tokens", "quantity": 38810, "amount": "12542100"},
                ],
            }

    def _result(self, job: dict[str, Any]) -> dict[str, Any]:
        data = FakeService.state.map_object[job["job_id"]]
        return {
            "manifest": {
                "quality": "usable",
                "reason_codes": [],
                "file": {"name": "marstrace.json", "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
                "device": {"gpu_name": "H200", "compute_capability": "90a", "cuda_driver_version": "575.57.08", "cuda_toolkit_version": "12.9"},
                "clock_type": "gpu",
                "granularity": "warp",
            },
            "download_url": f"{FakeService.state.origin}/r2/results/{job['job_id']}",
            "download_expires_at": TIMESTAMP,
            "result_expires_at": TIMESTAMP,
        }

    def _post(self, path: str, body: bytes) -> tuple[int, Any]:
        self._authorize()
        if path == "/v1/uploads":
            return self._grant(self._body(body, RequestShape.UPLOAD))
        if match := re.match(r"^/v1/uploads/([^/]+)/complete$", path):
            return self._complete(match.group(1), self._body(body, RequestShape.COMPLETE))
        if path == "/v1/quotes":
            return 200, self._quote(self._body(body, RequestShape.JOB))
        if path == "/v1/jobs":
            return self._submit(self._body(body, RequestShape.SUBMIT), body)
        if match := re.match(r"^/v1/jobs/([^/]+)/cancel$", path):
            if self._body(body, {}) != {}:
                raise ServiceError(400, "invalid_request", "cancel takes no fields")
            job = self._job(match.group(1))
            if job["status"] in SET_FINAL:
                raise ServiceError(409, "job_not_cancellable", "The job has ended")
            job["status"] = "cancelled"
            return 200, job
        raise ServiceError(404, "not_found", "No such route")

    def _grant(self, map_body: dict[str, Any]) -> tuple[int, Any]:
        state = FakeService.state
        if map_body["role"] not in SET_ROLE:
            raise ServiceError(400, "invalid_request", f"role must be one of {', '.join(sorted(SET_ROLE))}")
        if not re.match(r"^[0-9a-f]{64}$", map_body["sha256"]):
            raise ServiceError(400, "invalid_request", "sha256 must be 64 lowercase hex characters")
        upload_id = new_id("upl_")
        upload = {**map_body, "status": "pending", "data": b"", "parts": {}, "job_id": None}
        state.map_upload[upload_id] = upload
        grant: dict[str, Any] = {"upload_id": upload_id, "expires_at": TIMESTAMP}
        if map_body["size_bytes"] > state.multipart_threshold:
            grant["parts"] = [
                {
                    "part_number": number + 1,
                    "offset": offset,
                    "size_bytes": min(state.part_bytes, map_body["size_bytes"] - offset),
                    "upload_url": f"{state.origin}/r2/{upload_id}/part/{number + 1}",
                }
                for number, offset in enumerate(range(0, map_body["size_bytes"], state.part_bytes))
            ]
            grant["headers"] = {"content-type": map_body["content_type"]}
        else:
            grant["upload_url"] = f"{state.origin}/r2/{upload_id}"
            grant["headers"] = {
                "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(map_body["sha256"])).decode(),
                "content-type": map_body["content_type"],
            }
        upload["multipart"] = "parts" in grant
        return 200, grant

    def _complete(self, upload_id: str, map_body: dict[str, Any]) -> tuple[int, Any]:
        upload = FakeService.state.map_upload.get(upload_id)
        if upload is None:
            raise ServiceError(404, "not_found", "No such upload for this account")
        if upload["multipart"]:
            list_receipt = map_body.get("parts", [])
            if list_receipt != [{"part_number": number, "etag": f'"etag-{number}"'} for number in sorted(upload["parts"])]:
                raise ServiceError(400, "invalid_request", "parts must list every part's number and ETag in order")
            upload["data"] = b"".join(upload["parts"][number] for number in sorted(upload["parts"]))
        elif "parts" in map_body:
            raise ServiceError(400, "invalid_request", "parts is for a multipart upload")
        data = upload["data"]
        if len(data) != map_body["size_bytes"] or len(data) != upload["size_bytes"] or map_body["sha256"] != upload["sha256"]:
            raise ServiceError(400, "invalid_request", "size_bytes or sha256 differs from the stored object")
        upload["status"] = "verified"
        return 200, {"upload_id": upload_id, "status": "verified", "role": upload["role"], "sha256": upload["sha256"], "size_bytes": upload["size_bytes"]}

    def _quote(self, map_request: dict[str, Any]) -> dict[str, Any]:
        state = FakeService.state
        entry = next((entry for entry in state.map_hardware["hardware"] if entry["id"] == map_request["hardware"]), None)
        if entry is None or not entry["available"]:
            raise ServiceError(422, "hardware_unavailable", f"Hardware {map_request['hardware']} is not offered now")
        self._check_inputs(map_request["inputs"])
        map_agent = dict(AGENT_DEFAULTS, **map_request.get("agent", {}))
        rates = entry["rates"]
        gpu_milliseconds = (map_agent["max_wall_seconds"] + FINALIZE_MARGIN_SECONDS) * 1000
        gpu_micro = gpu_milliseconds * int(rates["gpu_micro_usd_per_hour"]) // MILLISECONDS_PER_HOUR
        input_rate = max(int(rates[key]) for key in rates if key.startswith("agent_") and key != "agent_output_micro_usd_per_million")
        tokens_micro = (map_agent["max_input_tokens"] * input_rate + map_agent["max_output_tokens"] * int(rates["agent_output_micro_usd_per_million"])) // 1_000_000
        fee = int(rates["job_fee_micro_usd"])
        max_cost = int(map_request["limits"]["max_cost_micro_usd"])
        budget = max_cost - fee - gpu_micro
        quote_id = new_id("quo_")
        quote = {
            "quote_id": quote_id,
            "expires_at": TIMESTAMP,
            "service_model": entry["service_model"],
            "rate_card_version_id": "rcv_1",
            "rates": rates,
            "ceilings": {
                "job_fee_micro_usd": str(fee),
                "gpu_milliseconds": gpu_milliseconds,
                "gpu_micro_usd": str(gpu_micro),
                "agent_tokens_micro_usd": str(tokens_micro),
            },
            "max_cost_micro_usd": str(max_cost),
            "agent_budget_micro_usd": str(budget),
            "admissible": budget >= AGENT_BUDGET_MIN,
        }
        if budget < AGENT_BUDGET_MIN:
            quote["message"] = "max_cost_micro_usd leaves less than $1 of agent budget after the fee and the GPU ceiling"
        state.map_quote[quote_id] = quote
        return quote

    def _check_inputs(self, map_inputs: dict[str, Any], job_id: str | None = None) -> None:
        state = FakeService.state
        for key, role in (("cubins", "cubin"), ("source_archive", "source_archive"), ("launch_manifest", "launch_manifest"), ("tensors", "tensor"), ("files", "file")):
            value = map_inputs.get(key)
            list_id = value if isinstance(value, list) else [] if value is None else [value]
            for upload_id in list_id:
                upload = state.map_upload.get(upload_id)
                if upload is None or upload["status"] != "verified" or upload["role"] != role or upload["job_id"] not in (None, job_id):
                    raise ServiceError(409, "upload_not_ready", f"inputs.{key}: {upload_id} is not a verified, unused {role} upload")
        if not 1 <= len(map_inputs["cubins"]) <= 8:
            raise ServiceError(413, "limit_exceeded", "inputs.cubins holds 1 to 8 cubins")

    def _submit(self, map_body: dict[str, Any], body: bytes) -> tuple[int, Any]:
        state = FakeService.state
        key = self.headers.get("Idempotency-Key") or ""
        if not IDEMPOTENCY_KEY_PATTERN.match(key):
            raise ServiceError(400, "invalid_request", "Idempotency-Key is required: 1 to 255 characters of [A-Za-z0-9._:-]")
        digest = hashlib.sha256(json.dumps(map_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if key in state.map_idempotency:
            known_digest, job_id = state.map_idempotency[key]
            if known_digest != digest:
                raise ServiceError(409, "idempotency_conflict", "The Idempotency-Key was used with a different body")
            return 200, state.map_job[job_id]
        quote_id = map_body.get("quote_id")
        request = {name: value for name, value in map_body.items() if name != "quote_id"}
        if quote_id is not None and quote_id not in state.map_quote:
            raise ServiceError(404, "not_found", "No such quote for this account")
        quote = self._quote(request)
        if not quote["admissible"]:
            raise ServiceError(400, "invalid_request", quote["message"])
        job_id = new_id("job_")
        self._check_inputs(request["inputs"])
        reservation = min(int(request["limits"]["max_cost_micro_usd"]), int(quote["ceilings"]["job_fee_micro_usd"]) + int(quote["ceilings"]["gpu_micro_usd"]) + int(quote["ceilings"]["agent_tokens_micro_usd"]))
        if state.available_micro_usd is not None and reservation > state.available_micro_usd:
            raise ServiceError(402, "insufficient_credit", "Prepaid credit does not cover the job's reservation; add credit in the console")
        for value in request["inputs"].values():
            for upload_id in value if isinstance(value, list) else [value]:
                state.map_upload[upload_id]["job_id"] = job_id
        state.map_job[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": TIMESTAMP,
            "hardware": request["hardware"],
            "agent": dict(AGENT_DEFAULTS, **request.get("agent", {})),
            "limits": dict({"max_attempts": 2}, **request["limits"]),
            "progress": None,
            "attempts": [],
            "cost": {"reserved_micro_usd": str(reservation), "charged_micro_usd": None, "lines": []},
            "result": None,
            "error": None,
        }
        state.map_idempotency[key] = (digest, job_id)
        state.list_job_order.append(job_id)
        return 200, state.map_job[job_id]


class FakeServer:
    """The fake MarsTrace service on a local port, started once per test class."""

    def __init__(self) -> None:
        FakeService.state = FakeState()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        FakeService.state.origin = self.origin
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def state(self) -> FakeState:
        return FakeService.state

    def reset(self) -> None:
        FakeService.state = FakeState()
        FakeService.state.origin = self.origin

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
