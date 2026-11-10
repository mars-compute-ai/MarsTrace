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
ACCOUNT_ID = "acc_test"
TIMESTAMP = "2026-11-10T12:00:00.000Z"
ADD_CREDIT_URL = "https://console.marscompute.ai/tenant/billing/credits"
AGENT_MODEL = "claude-opus-5-5"
CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"
MIB = 1 << 20
MILLION = 1_000_000
RATES_H200 = {
    "agent_input_micro_usd_per_million": "18000000",
    "agent_cache_read_micro_usd_per_million": "1800000",
    "agent_cache_write_5m_micro_usd_per_million": "22500000",
    "agent_cache_write_1h_micro_usd_per_million": "36000000",
    "agent_output_micro_usd_per_million": "90000000",
}
LIST_INPUT_RATE = [
    "agent_input_micro_usd_per_million",
    "agent_cache_read_micro_usd_per_million",
    "agent_cache_write_5m_micro_usd_per_million",
    "agent_cache_write_1h_micro_usd_per_million",
]
AGENT_DEFAULTS = {"max_turns": 60, "max_input_tokens": 4000000, "max_output_tokens": 60000, "max_wall_seconds": 1200}
AGENT_MAXIMUMS = {"max_turns": 500, "max_input_tokens": 50000000, "max_output_tokens": 1000000, "max_wall_seconds": 2580}
LIST_CUDA_VERSION = ["12.8", "12.9", "13.0"]
LIST_CUDA_VERSION_B300 = ["12.9", "13.0"]
CUDA_VERSION_DEFAULT = "13.0"
HARDWARE = {
    "hardware": [
        {
            "id": "h200", "gpu": "H200", "architecture": "sm_90a", "available": True, "service_model": "marstrace-h200",
            "cuda_versions": LIST_CUDA_VERSION, "default_cuda_version": CUDA_VERSION_DEFAULT, "rates": RATES_H200,
        },
        {
            "id": "b200", "gpu": "B200", "architecture": "sm_100a", "available": True, "service_model": "marstrace-b200",
            "cuda_versions": LIST_CUDA_VERSION, "default_cuda_version": CUDA_VERSION_DEFAULT,
            "rates": dict(RATES_H200, agent_output_micro_usd_per_million="95000000"),
        },
        {
            "id": "b300", "gpu": "B300", "architecture": "sm_103a", "available": False, "service_model": "marstrace-b300",
            "cuda_versions": LIST_CUDA_VERSION_B300, "default_cuda_version": CUDA_VERSION_DEFAULT, "rates": None,
        },
    ],
    "agent": {"model": AGENT_MODEL, "defaults": AGENT_DEFAULTS, "maximums": AGENT_MAXIMUMS},
}
# the order a job's status moves in, one step per long poll
LIST_JOB_STEP = ["queued", "starting", "running", "finishing", "succeeded"]
SET_FINAL = {"succeeded", "failed", "cancelled"}
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
MAX_ATTEMPTS_DEFAULT = 2
# HTTP status and retryability of each error code, as errors.ts maps them
MAP_ERROR_STATUS = {
    "invalid_request": 400,
    "authentication_required": 401,
    "insufficient_credit": 402,
    "spending_limit_exceeded": 402,
    "billing_suspended": 402,
    "forbidden": 403,
    "not_found": 404,
    "idempotency_conflict": 409,
    "upload_not_ready": 409,
    "job_not_cancellable": 409,
    "result_unavailable": 409,
    "limit_exceeded": 413,
    "hardware_unavailable": 422,
    "rate_limited": 429,
    "too_many_jobs": 429,
    "service_unavailable": 503,
}
SET_RETRYABLE = {"rate_limited", "too_many_jobs", "service_unavailable"}
MAP_ROLE_MAX_BYTES = {"cubin": 64 * MIB, "source_archive": 256 * MIB, "launch_manifest": MIB, "tensor": 128 * MIB, "file": 16 * MIB}
MAX_TOTAL_BYTES = 256 * MIB
MAP_FAILURE_MESSAGE = {"vm_preempted": "The GPU machine was reclaimed by the supplier.", "cancelled": "The job was cancelled."}


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
            "[limits]\nmax_attempts = 1\n"
        )


def ceil_rate(quantity: int, unit_micro_usd: int, denominator: int) -> int:
    """A quantity at a price, rounded up to whole micro-USD."""
    return -(-quantity * unit_micro_usd // denominator)


class ServiceError(Exception):
    """A refusal the fake answers with an error envelope, as ApiError of errors.ts."""

    def __init__(self, code: str, message: str, details: dict[str, str] | None = None, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = MAP_ERROR_STATUS[code]
        self.retryable = code in SET_RETRYABLE
        self.details = details
        self.retry_after_seconds = retry_after_seconds

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": str(self), "retryable": self.retryable}
        if self.details is not None:
            body["details"] = self.details
        return body


class Field:
    """One field of a strict request schema: its type, bounds, and whether it may be absent."""

    def __init__(self, kind: Any, low: int | None = None, high: int | None = None, pattern: str | None = None, is_optional: bool = False, default: Any = None) -> None:
        self.kind = kind
        self.low = low
        self.high = high
        self.pattern = re.compile(pattern) if pattern else None
        self.is_optional = is_optional
        self.default = default


class RequestSchema:
    """The request schemas of job-request.ts and customer-api.ts: unknown fields are refused,
    defaults are applied, and a refusal names the field as CustomerApi.parse does."""

    UPLOAD_ID = r"^upl_[0-9a-z]{26}$"
    AGENT = {
        name: Field(int, low=60 if name == "max_wall_seconds" else 1, high=AGENT_MAXIMUMS[name], is_optional=True, default=AGENT_DEFAULTS[name])
        for name in AGENT_DEFAULTS
    }
    INPUTS = {
        "cubins": Field([Field(str, pattern=UPLOAD_ID)], low=1, high=8),
        "source_archive": Field(str, pattern=UPLOAD_ID, is_optional=True),
        "launch_manifest": Field(str, pattern=UPLOAD_ID, is_optional=True),
        "tensors": Field([Field(str, pattern=UPLOAD_ID)], high=8, is_optional=True, default=[]),
        "files": Field([Field(str, pattern=UPLOAD_ID)], high=8, is_optional=True, default=[]),
    }
    LIMITS = {"max_attempts": Field(int, low=1, high=3, is_optional=True, default=MAX_ATTEMPTS_DEFAULT)}
    JOB = {
        "hardware": Field(str, low=1, high=32),
        "cuda_version": Field(str, pattern=r"^[1-9][0-9]*\.[0-9]+$", is_optional=True),
        "prompt": Field(str, low=1, high=32000, pattern=r"^[^\x00]*$"),
        "inputs": Field(INPUTS),
        "agent": Field(AGENT, is_optional=True, default={}),
        "limits": Field(LIMITS, is_optional=True, default={}),
    }
    UPLOAD = {
        "role": Field({"cubin", "source_archive", "launch_manifest", "tensor", "file"}),
        "name": Field(str, pattern=r"^[A-Za-z0-9._-]{1,255}$"),
        "size_bytes": Field(int, low=1),
        "sha256": Field(str, pattern=r"^[a-f0-9]{64}$"),
        "content_type": Field(str, low=1, high=255, is_optional=True, default="application/octet-stream"),
    }
    COMPLETE = {
        "sha256": Field(str, pattern=r"^[a-f0-9]{64}$"),
        "size_bytes": Field(int, low=1),
        "parts": Field([Field({"part_number": Field(int, low=1, high=10000), "etag": Field(str, low=1, high=256)})], high=10000, is_optional=True),
    }
    CANCEL: dict[str, Field] = {}
    WAIT = {"timeout_seconds": Field("int_text", low=1, high=60, is_optional=True, default=30)}
    LIST = {
        "limit": Field("int_text", low=1, high=100, is_optional=True, default=20),
        "cursor": Field(str, low=1, high=200, is_optional=True),
        "status": Field({"queued", "starting", "running", "finishing", "succeeded", "failed", "cancelled"}, is_optional=True),
        "account_id": Field(str, low=1, high=64, is_optional=True),
    }

    @staticmethod
    def parse(value: Any, schema: dict[str, Field], path: str = "") -> dict[str, Any]:
        """The value with defaults applied; raises invalid_request naming the first bad field."""
        where = path or "body"
        if not isinstance(value, dict):
            raise ServiceError("invalid_request", f"Invalid {where}: expected object")
        set_unknown = set(value) - set(schema)
        if set_unknown:
            raise ServiceError("invalid_request", f"Invalid {where}: unknown field {', '.join(sorted(set_unknown))}")
        map_parsed: dict[str, Any] = {}
        for name, field in schema.items():
            field_path = f"{path}.{name}" if path else name
            if name not in value:
                if not field.is_optional:
                    raise ServiceError("invalid_request", f"Invalid {field_path}: required")
                if field.default is not None:
                    map_parsed[name] = RequestSchema.parse(field.default, field.kind, field_path) if isinstance(field.kind, dict) else json.loads(json.dumps(field.default))
                continue
            map_parsed[name] = RequestSchema.value(value[name], field, field_path)
        return map_parsed

    @staticmethod
    def value(value: Any, field: Field, path: str) -> Any:
        kind = field.kind
        if isinstance(kind, dict):
            return RequestSchema.parse(value, kind, path)
        if isinstance(kind, set):
            if value not in kind:
                raise ServiceError("invalid_request", f"Invalid {path}: expected one of {', '.join(sorted(kind))}")
            return value
        if isinstance(kind, list):
            if not isinstance(value, list):
                raise ServiceError("invalid_request", f"Invalid {path}: expected array")
            RequestSchema.bounds(len(value), field, path, "items")
            return [RequestSchema.value(item, kind[0], f"{path}.{index}") for index, item in enumerate(value)]
        if kind == "int_text":
            if not re.match(r"^-?\d+$", str(value)):
                raise ServiceError("invalid_request", f"Invalid {path}: expected number")
            value = int(value)
            kind = int
        if kind is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ServiceError("invalid_request", f"Invalid {path}: expected int")
            RequestSchema.bounds(value, field, path, "")
            return value
        if not isinstance(value, str):
            raise ServiceError("invalid_request", f"Invalid {path}: expected string")
        RequestSchema.bounds(len(value), field, path, "characters")
        if field.pattern is not None and not field.pattern.match(value):
            raise ServiceError("invalid_request", f"Invalid {path}: invalid string")
        return value

    @staticmethod
    def bounds(size: int, field: Field, path: str, unit: str) -> None:
        if field.low is not None and size < field.low:
            raise ServiceError("invalid_request", f"Invalid {path}: too small, expected at least {field.low} {unit}".rstrip())
        if field.high is not None and size > field.high:
            raise ServiceError("invalid_request", f"Invalid {path}: too big, expected at most {field.high} {unit}".rstrip())


class FakeState:
    """What the fake MarsTrace service holds between requests."""

    def __init__(self) -> None:
        self.origin = ""
        self.list_request: list[dict[str, Any]] = []
        self.list_contract_error: list[str] = []
        self.map_upload: dict[str, dict[str, Any]] = {}
        self.map_object: dict[str, bytes] = {}
        self.map_job: dict[str, dict[str, Any]] = {}
        # idempotency key -> (digest of the canonical raw body, job id)
        self.map_idempotency: dict[str, tuple[str, str]] = {}
        self.list_job_order: list[str] = []
        # running jobs asked to stop, which end cancelled at their next step
        self.set_cancel_requested: set[str] = set()
        # the workspace's headroom: available balance (None without a limit) and the admission minimum
        self.billing_mode = "prepaid"
        self.billing_status = "billable_active"
        self.available_micro_usd: int | None = 118_250_000
        self.minimum_micro_usd = 100_000_000
        self.max_queued = 10
        self.map_hardware = json.loads(json.dumps(HARDWARE))
        # files larger than this are granted as multipart uploads of part_bytes parts
        self.multipart_threshold = 64 * MIB
        self.part_bytes = 32 * MIB
        # whether a long poll moves a job a step; False keeps it where it is
        self.do_advance = True
        # the job status a job stops at, before succeeding
        self.hold_status: str | None = None
        # whether a job's first attempt is preempted while it runs, and retried
        self.do_preempt_first_attempt = False
        # whether admission refuses although the estimate said admissible, as when another job took the balance
        self.do_refuse_admission = False
        # submissions to drop without an answer, to exercise the retry under one key
        self.drop_submissions = 0

    def requests_to(self, method: str, pattern: str) -> list[dict[str, Any]]:
        regex = re.compile(pattern)
        return [request for request in self.list_request if request["method"] == method and regex.match(request["path"])]

    def is_admissible(self) -> bool:
        return self.billing_status == "billable_active" and (self.available_micro_usd is None or self.available_micro_usd >= self.minimum_micro_usd)

    def held_micro_usd(self) -> int:
        return sum(int(job["cost"]["held_micro_usd"] or 0) for job in self.map_job.values() if job["cost"]["charged_micro_usd"] is None and job["status"] not in SET_FINAL)


class FakeService(http.server.BaseHTTPRequestHandler):
    """A stand-in for the routes of http/app.ts behind nginx's /trace/ location, and for R2's presigned URLs."""

    # the base path nginx strips before the service sees a request, as at console.marscompute.ai/trace
    BASE_PATH = "/trace"
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

    def _envelope(self, status: int, data: Any, error: dict[str, Any] | None, map_header: dict[str, str] | None = None) -> None:
        request_id = str(uuid.uuid4())
        body = json.dumps({"request_id": request_id, "data": data, "error": error}).encode()
        self._send(status, body, {"Content-Type": "application/json", "x-request-id": request_id, **(map_header or {})})

    def _service_path(self) -> str | None:
        """The path the service sees, or None for a path outside the base path."""
        path = urllib.parse.urlsplit(self.path).path
        if not path.startswith(f"{FakeService.BASE_PATH}/"):
            return None
        return path.removeprefix(FakeService.BASE_PATH)

    def _record(self) -> bytes:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        split = urllib.parse.urlsplit(self.path)
        FakeService.state.list_request.append(
            {
                "method": self.command,
                "path": self._service_path() or split.path,
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
            if self.headers.get("Origin") is not None:
                raise ServiceError("forbidden", "Browser requests are not accepted here; use the console")
            data = handler(*args)
        except ServiceError as err:
            if err.code == "invalid_request":
                FakeService.state.list_contract_error.append(f"{self.command} {self.path}: {err}")
            map_header = {} if err.retry_after_seconds is None else {"retry-after": str(err.retry_after_seconds)}
            self._envelope(err.status, None, err.to_body(), map_header)
            return
        self._envelope(200, data, None)

    def _authorize(self) -> None:
        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            raise ServiceError("authentication_required", "A MarsTrace API key is required")

    def _json(self, body: bytes, is_empty_allowed: bool = False) -> Any:
        text = body.decode("utf-8")
        if not text.strip():
            if is_empty_allowed:
                return {}
            raise ServiceError("invalid_request", "Invalid body: a JSON object is required")
        try:
            return json.loads(text)
        except ValueError as err:
            raise ServiceError("invalid_request", "Invalid body: not JSON") from err

    def _query(self) -> dict[str, str]:
        return {name: list_value[-1] for name, list_value in urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items()}

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
        service_path = self._service_path()
        if service_path is None:
            self._send(404, b"not found")
            return
        self._dispatch(self._get, service_path)

    def do_POST(self) -> None:
        body = self._record()
        state = FakeService.state
        path = self._service_path()
        if path is None:
            self._send(404, b"not found")
            return
        if path == "/v1/jobs" and state.drop_submissions > 0:
            state.drop_submissions -= 1
            self.close_connection = True
            self.connection.close()
            return
        self._dispatch(self._post, path, body)

    def _get(self, path: str) -> Any:
        if path == "/healthz":
            return {"status": "ok", "release": "test"}
        self._authorize()
        state = FakeService.state
        if path == "/v1/hardware":
            return state.map_hardware
        if path == "/v1/credit":
            return {
                "billing_mode": state.billing_mode,
                "billing_status": state.billing_status,
                "available_micro_usd": None if state.available_micro_usd is None else str(state.available_micro_usd),
                "held_micro_usd": str(state.held_micro_usd()),
                "minimum_micro_usd": str(state.minimum_micro_usd),
                "admissible": state.is_admissible(),
                "add_credit_url": ADD_CREDIT_URL,
            }
        if path == "/v1/jobs":
            parameters = RequestSchema.parse(self._query(), RequestSchema.LIST)
            list_id = [job_id for job_id in reversed(state.list_job_order) if parameters.get("status") in (None, state.map_job[job_id]["status"])]
            start = int(parameters.get("cursor", "0"))
            limit = parameters["limit"]
            return {
                "items": [state.map_job[job_id] for job_id in list_id[start:start + limit]],
                "next_cursor": str(start + limit) if start + limit < len(list_id) else None,
            }
        if match := re.match(r"^/v1/jobs/([^/]+)/wait$", path):
            RequestSchema.parse(self._query(), RequestSchema.WAIT)
            job = self._job(match.group(1))
            self._advance(job)
            return job
        if match := re.match(r"^/v1/jobs/([^/]+)/result$", path):
            job = self._job(match.group(1))
            if job["status"] != "succeeded":
                raise ServiceError("result_unavailable", "This job has no result, or its result has expired")
            return self._result(job)
        if match := re.match(r"^/v1/jobs/([^/]+)$", path):
            return self._job(match.group(1))
        raise ServiceError("not_found", "No such route")

    def _job(self, job_id: str) -> dict[str, Any]:
        job = FakeService.state.map_job.get(job_id)
        if job is None:
            raise ServiceError("not_found", "No such job")
        return job

    def _attempt(self, number: int) -> dict[str, Any]:
        return {"number": number, "status": "starting", "failure": None, "error": None, "started_at": TIMESTAMP, "ended_at": None}

    def _advance(self, job: dict[str, Any]) -> None:
        state = FakeService.state
        if job["job_id"] in state.set_cancel_requested:
            FakeService._end_cancelled(job)
            return
        if not state.do_advance or job["status"] in SET_FINAL or job["status"] == state.hold_status:
            return
        if job["status"] == "running" and state.do_preempt_first_attempt and len(job["attempts"]) == 1 and job["limits"]["max_attempts"] > 1:
            job["attempts"][0].update(
                status="failed", failure="supplier", ended_at=TIMESTAMP,
                error={"code": "vm_preempted", "message": MAP_FAILURE_MESSAGE["vm_preempted"]},
            )
            job["attempts"].append(self._attempt(2))
            job["status"] = "starting"
            return
        job["status"] = LIST_JOB_STEP[LIST_JOB_STEP.index(job["status"]) + 1]
        if job["status"] == "starting":
            job["started_at"] = TIMESTAMP
            job["attempts"].append(self._attempt(1))
        if job["status"] in ("running", "finishing"):
            job["attempts"][-1]["status"] = job["status"]
            job["progress"] = {"agent_turn": 14, "usage_micro_usd": "5034000"}
        if job["status"] == "succeeded":
            with open(os.path.join(FIXTURE_DIRECTORY, "trace_sass_regions.json"), "rb") as trace_file:
                trace = trace_file.read()
            state.map_object[job["job_id"]] = trace
            job["attempts"][-1].update(status="succeeded", ended_at=TIMESTAMP)
            job["ended_at"] = TIMESTAMP
            list_line = [
                {"dimension": "service_agent_cache_read_tokens", "quantity": 812000, "amount_micro_usd": "1461600"},
                *(
                    {"dimension": dimension, "quantity": 0, "amount_micro_usd": "0"}
                    for dimension in ("service_agent_input_tokens", "service_agent_cache_write_5m_tokens", "service_agent_cache_write_1h_tokens")
                ),
                {"dimension": "service_agent_output_tokens", "quantity": 38810, "amount_micro_usd": "3492900"},
            ]
            if len(job["attempts"]) > 1:
                list_line[0] = {"dimension": "service_agent_cache_read_tokens", "quantity": 1124000, "amount_micro_usd": "2023200"}
            job["cost"]["charged_micro_usd"] = str(sum(int(line["amount_micro_usd"]) for line in list_line))
            job["cost"]["lines"] = list_line
            job["result"] = self._result_summary(job)
            if state.available_micro_usd is not None:
                state.available_micro_usd -= int(job["cost"]["charged_micro_usd"])

    def _result_summary(self, job: dict[str, Any]) -> dict[str, Any]:
        data = FakeService.state.map_object[job["job_id"]]
        return {
            "name": "marstrace.json",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "download_url": f"{FakeService.state.origin}/r2/results/{job['job_id']}",
            "expires_at": TIMESTAMP,
        }

    def _result(self, job: dict[str, Any]) -> dict[str, Any]:
        summary = self._result_summary(job)
        return {
            "manifest": {
                "quality": "usable",
                "reason_codes": [],
                "file": {"name": "marstrace.json", "size_bytes": summary["size_bytes"], "sha256": summary["sha256"]},
                "device": {"gpu_name": "H200", "compute_capability": "90a", "cuda_driver_version": "575.57.08", "cuda_toolkit_version": job["cuda_version"]},
                "clock_type": "gpu",
                "granularity": "warp",
            },
            "download_url": summary["download_url"],
            "download_expires_at": TIMESTAMP,
            "result_expires_at": TIMESTAMP,
        }

    def _post(self, path: str, body: bytes) -> Any:
        self._authorize()
        if path == "/v1/uploads":
            return self._grant(RequestSchema.parse(self._json(body), RequestSchema.UPLOAD))
        if match := re.match(r"^/v1/uploads/([^/]+)/complete$", path):
            return self._complete(match.group(1), RequestSchema.parse(self._json(body), RequestSchema.COMPLETE))
        if path == "/v1/quotes":
            return self._quote(RequestSchema.parse(self._json(body), RequestSchema.JOB))
        if path == "/v1/jobs":
            return self._submit(self._json(body))
        if match := re.match(r"^/v1/jobs/([^/]+)/cancel$", path):
            RequestSchema.parse(self._json(body, is_empty_allowed=True), RequestSchema.CANCEL)
            return self._cancel(self._job(match.group(1)))
        raise ServiceError("not_found", "No such route")

    def _grant(self, request: dict[str, Any]) -> dict[str, Any]:
        state = FakeService.state
        max_bytes = MAP_ROLE_MAX_BYTES[request["role"]]
        if request["size_bytes"] > max_bytes:
            raise ServiceError("limit_exceeded", f"A {request['role']} file may be at most {max_bytes} bytes")
        upload_id = new_id("upl_")
        upload = {**request, "upload_id": upload_id, "status": "granted", "data": b"", "parts": {}, "job_id": None}
        state.map_upload[upload_id] = upload
        grant: dict[str, Any] = {"upload_id": upload_id, "expires_at": TIMESTAMP}
        if request["size_bytes"] > state.multipart_threshold:
            grant["parts"] = [
                {
                    "part_number": number + 1,
                    "offset": offset,
                    "size_bytes": min(state.part_bytes, request["size_bytes"] - offset),
                    "upload_url": f"{state.origin}/r2/{upload_id}/part/{number + 1}",
                }
                for number, offset in enumerate(range(0, request["size_bytes"], state.part_bytes))
            ]
        else:
            grant["upload_url"] = f"{state.origin}/r2/{upload_id}"
            grant["headers"] = {
                "x-amz-checksum-sha256": base64.b64encode(bytes.fromhex(request["sha256"])).decode(),
                "content-type": request["content_type"],
            }
        upload["multipart"] = "parts" in grant
        return grant

    @staticmethod
    def _upload_view(upload: dict[str, Any]) -> dict[str, Any]:
        return {
            "upload_id": upload["upload_id"],
            "role": upload["role"],
            "name": upload["name"],
            "size_bytes": upload["size_bytes"],
            "sha256": upload["sha256"],
            "content_type": upload["content_type"],
            "status": upload["status"],
            "expires_at": TIMESTAMP,
        }

    def _complete(self, upload_id: str, request: dict[str, Any]) -> dict[str, Any]:
        upload = FakeService.state.map_upload.get(upload_id)
        if upload is None:
            raise ServiceError("not_found", "No such upload")
        if upload["sha256"] != request["sha256"] or upload["size_bytes"] != request["size_bytes"]:
            raise ServiceError("invalid_request", "Invalid sha256 or size_bytes: they differ from the upload grant")
        if upload["status"] != "granted":
            return FakeService._upload_view(upload)
        if upload["multipart"]:
            expected_parts = -(-upload["size_bytes"] // FakeService.state.part_bytes)
            list_part = sorted(request.get("parts") or [], key=lambda part: part["part_number"])
            if [part["part_number"] for part in list_part] != list(range(1, expected_parts + 1)) or any(
                part["etag"] != f'"etag-{part["part_number"]}"' for part in list_part
            ):
                raise ServiceError("invalid_request", f"Invalid parts: expected parts 1 to {expected_parts} with their ETags")
            upload["data"] = b"".join(upload["parts"][number] for number in sorted(upload["parts"]))
        if not upload["data"]:
            raise ServiceError("upload_not_ready", "The file has not been uploaded yet")
        if len(upload["data"]) != upload["size_bytes"]:
            raise ServiceError("upload_not_ready", "The uploaded file's size differs from size_bytes")
        upload["status"] = "verified"
        return FakeService._upload_view(upload)

    def _hardware(self, request: dict[str, Any]) -> dict[str, Any]:
        entry = next((entry for entry in FakeService.state.map_hardware["hardware"] if entry["id"] == request["hardware"]), None)
        if entry is None:
            raise ServiceError("hardware_unavailable", f"Unknown hardware {request['hardware']}; see GET /v1/hardware")
        if entry["rates"] is None:
            raise ServiceError("hardware_unavailable", "This hardware has no published price now")
        cuda_version = request.get("cuda_version", entry["default_cuda_version"])
        if cuda_version not in entry["cuda_versions"]:
            raise ServiceError(
                "invalid_request",
                f"Invalid cuda_version: {entry['id']} offers CUDA {', '.join(entry['cuda_versions'])}",
            )
        return entry

    def _quote(self, request: dict[str, Any]) -> dict[str, Any]:
        state = FakeService.state
        entry = self._hardware(request)
        rates = entry["rates"]
        map_agent = request["agent"]
        attempts = request["limits"]["max_attempts"]
        input_per_attempt = max(ceil_rate(map_agent["max_input_tokens"], int(rates[key]), MILLION) for key in LIST_INPUT_RATE)
        tokens_per_attempt = input_per_attempt + ceil_rate(map_agent["max_output_tokens"], int(rates["agent_output_micro_usd_per_million"]), MILLION)
        map_cost = {
            "agent_tokens_per_attempt": str(tokens_per_attempt),
            "total": str(attempts * tokens_per_attempt),
        }
        quote = {
            "service_model": entry["service_model"],
            "rate_card_version_id": "rcv_1",
            "cuda_version": request.get("cuda_version", entry["default_cuda_version"]),
            "rates": rates,
            "bounds_per_attempt": {
                "agent_input_tokens": map_agent["max_input_tokens"],
                "agent_output_tokens": map_agent["max_output_tokens"],
            },
            "cost_at_bounds_micro_usd": map_cost,
            "minimum_micro_usd": str(state.minimum_micro_usd),
            "available_micro_usd": None if state.available_micro_usd is None else str(state.available_micro_usd),
            "admissible": state.is_admissible(),
        }
        if not quote["admissible"]:
            quote["message"] = (
                f"A job starts only while the workspace's available balance is at least {state.minimum_micro_usd} micro-USD; add credit"
            )
        return quote

    def _claim_uploads(self, inputs: dict[str, Any], job_id: str) -> None:
        state = FakeService.state
        list_slot = [
            *((upload_id, "cubin", f"inputs.cubins.{index}") for index, upload_id in enumerate(inputs["cubins"])),
            *(((inputs["source_archive"], "source_archive", "inputs.source_archive"),) if "source_archive" in inputs else ()),
            *(((inputs["launch_manifest"], "launch_manifest", "inputs.launch_manifest"),) if "launch_manifest" in inputs else ()),
            *((upload_id, "tensor", f"inputs.tensors.{index}") for index, upload_id in enumerate(inputs["tensors"])),
            *((upload_id, "file", f"inputs.files.{index}") for index, upload_id in enumerate(inputs["files"])),
        ]
        set_seen: set[str] = set()
        total_bytes = 0
        for upload_id, role, field in list_slot:
            if upload_id in set_seen:
                raise ServiceError("invalid_request", f"Invalid {field}: an upload is named twice")
            set_seen.add(upload_id)
            upload = state.map_upload.get(upload_id)
            if upload is None:
                raise ServiceError("not_found", f"No such upload in {field}")
            if upload["status"] != "verified" or upload["job_id"] is not None:
                raise ServiceError("upload_not_ready", f"The upload in {field} is not verified or belongs to another job")
            if upload["role"] != role:
                raise ServiceError("invalid_request", f"Invalid {field}: the upload's role is {upload['role']}")
            total_bytes += upload["size_bytes"]
        if total_bytes > MAX_TOTAL_BYTES:
            raise ServiceError("limit_exceeded", f"All inputs of a job may be at most {MAX_TOTAL_BYTES} bytes")
        for upload_id in set_seen:
            state.map_upload[upload_id].update(status="used", job_id=job_id)

    def _submit(self, raw_body: Any) -> dict[str, Any]:
        state = FakeService.state
        key = self.headers.get("Idempotency-Key")
        if key is None or not IDEMPOTENCY_KEY_PATTERN.match(key):
            raise ServiceError("invalid_request", "Invalid Idempotency-Key header: 1 to 255 characters of A-Z a-z 0-9 . _ : -")
        request = RequestSchema.parse(raw_body, RequestSchema.JOB)
        digest = hashlib.sha256(json.dumps(raw_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if key in state.map_idempotency:
            known_digest, job_id = state.map_idempotency[key]
            if known_digest != digest:
                raise ServiceError("idempotency_conflict", "This Idempotency-Key was used with a different request")
            return state.map_job[job_id]
        entry = self._hardware(request)
        if sum(1 for job in state.map_job.values() if job["status"] == "queued") >= state.max_queued:
            raise ServiceError("too_many_jobs", f"The workspace has {state.max_queued} queued jobs; wait for one to start", retry_after_seconds=30)
        job_id = new_id("job_")
        self._claim_uploads(request["inputs"], job_id)
        if state.do_refuse_admission or not state.is_admissible():
            for upload in state.map_upload.values():
                if upload["job_id"] == job_id:
                    upload.update(status="verified", job_id=None)
            if state.billing_status != "billable_active":
                raise ServiceError("billing_suspended", "Billing is not active for this workspace", details={"add_credit_url": ADD_CREDIT_URL})
            code = "insufficient_credit" if state.billing_mode == "prepaid" else "spending_limit_exceeded"
            message = (
                "The workspace's available credit is below the minimum balance a job needs; add credit"
                if code == "insufficient_credit"
                else "The workspace's remaining spending limit is below the minimum balance a job needs"
            )
            raise ServiceError(code, message, details={"add_credit_url": ADD_CREDIT_URL})
        if state.available_micro_usd is not None:
            state.available_micro_usd -= state.minimum_micro_usd
        state.map_job[job_id] = {
            "job_id": job_id,
            "account_id": ACCOUNT_ID,
            "status": "queued",
            "created_at": TIMESTAMP,
            "started_at": None,
            "ended_at": None,
            "hardware": entry["id"],
            "cuda_version": request.get("cuda_version", entry["default_cuda_version"]),
            "agent": request["agent"],
            "limits": request["limits"],
            "progress": None,
            "attempts": [],
            "cost": {"held_micro_usd": str(state.minimum_micro_usd), "charged_micro_usd": None, "lines": []},
            "result": None,
            "error": None,
        }
        state.map_idempotency[key] = (digest, job_id)
        state.list_job_order.append(job_id)
        return state.map_job[job_id]

    def _cancel(self, job: dict[str, Any]) -> dict[str, Any]:
        if job["status"] in SET_FINAL:
            raise ServiceError("job_not_cancellable", "The job has already ended")
        if job["status"] != "queued":
            FakeService.state.set_cancel_requested.add(job["job_id"])
            return job
        FakeService._end_cancelled(job)
        return job

    @staticmethod
    def _end_cancelled(job: dict[str, Any]) -> None:
        FakeService.state.set_cancel_requested.discard(job["job_id"])
        job["status"] = "cancelled"
        job["ended_at"] = TIMESTAMP
        job["error"] = {"code": "cancelled", "message": MAP_FAILURE_MESSAGE["cancelled"]}
        if job["attempts"]:
            job["attempts"][-1].update(status="cancelled", ended_at=TIMESTAMP)


class FakeServer:
    """The fake MarsTrace service on a local port, started once per test class."""

    def __init__(self) -> None:
        FakeService.state = FakeState()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeService)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.api_url = f"{self.origin}{FakeService.BASE_PATH}"
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
