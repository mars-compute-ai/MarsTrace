import base64
import hashlib
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from marstrace.error import ServiceError
from marstrace.schema import ResponseSchema


@dataclass
class Credentials:
    """The API key and the service URL the client calls MarsTrace with."""

    ENV_API_KEY = "MARSTRACE_API_KEY"
    ENV_API_URL = "MARSTRACE_API_URL"
    API_URL_DEFAULT = "https://console.marscompute.ai/trace"
    API_KEY_PREFIX = "mt_live_"

    api_key: str
    api_url: str

    @classmethod
    def from_environment(cls) -> "Credentials":
        """Read the key and service URL from the environment.

        Returns:
            the credentials

        Raises:
            ServiceError: no API key is set, or it is not a MarsTrace key
        """
        api_key = os.environ.get(
            key=Credentials.ENV_API_KEY,
            default="",
        ).strip()
        if not api_key.startswith(Credentials.API_KEY_PREFIX):
            raise ServiceError(
                f"set {Credentials.ENV_API_KEY} to a MarsTrace API key, {Credentials.API_KEY_PREFIX}... "
                "(console.marscompute.ai, MarsTrace keys)"
            )
        return cls(
            api_key=api_key,
            api_url=os.environ.get(
                key=Credentials.ENV_API_URL,
                default=Credentials.API_URL_DEFAULT,
            ).rstrip("/"),
        )


class MarsTraceClient:
    """The customer REST API of the MarsTrace service.

    Every answer is the envelope ``{request_id, data, error}``; the client returns ``data``
    after checking it against the shape of its route, and raises the ``error``.

    Args:
        credentials: the API key and service URL
    """

    # seconds a request may take before it is abandoned, and the one that moves a file
    TIMEOUT_SECONDS = 60
    TRANSFER_TIMEOUT_SECONDS = 600
    # seconds a long poll may take beyond the wait it asked the service for
    WAIT_MARGIN_SECONDS = 15
    WAIT_SECONDS_MAX = 60
    # attempts of a submission whose answer never arrived, under the same idempotency key
    SUBMIT_ATTEMPTS = 3
    RETRY_SECONDS = 2
    IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
    UPLOAD_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
    UPLOAD_NAME_CHARACTERS_MAX = 255
    USER_AGENT = "mars-trace"
    CHECKSUM_HEADER = "x-amz-checksum-sha256"
    # headers a grant may not set on an upload
    SET_FORBIDDEN_HEADER = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "host",
        "connection",
        "transfer-encoding",
    }
    RETRY_AFTER_HEADER = "retry-after"
    ADD_CREDIT_DETAIL = "add_credit_url"

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        map_header: dict[str, str] | None = None,
        timeout: float = TIMEOUT_SECONDS,
        do_authorize: bool = True,
        map_response_header: dict[str, str] | None = None,
    ) -> bytes:
        """Send one request and return the response body, raising on any refusal.

        A presigned upload or download URL carries its own authorization, so it is sent
        without the API key. ``map_response_header``, when given, receives the response's
        headers by lower-case name.
        """
        request = urllib.request.Request(
            url=url,
            data=body,
            method=method,
        )
        if do_authorize:
            request.add_header(
                key="Authorization",
                val=f"Bearer {self._credentials.api_key}",
            )
        request.add_header(
            key="User-Agent",
            val=MarsTraceClient.USER_AGENT,
        )
        for name, value in (map_header or {}).items():
            request.add_header(
                key=name,
                val=value,
            )
        try:
            with urllib.request.urlopen(
                url=request,
                timeout=timeout,
            ) as response:
                if map_response_header is not None:
                    map_response_header.update(
                        {name.lower(): value for name, value in response.headers.items()}
                    )
                return response.read()
        except urllib.error.HTTPError as err:
            raise MarsTraceClient._error_of(
                err=err,
                is_service=do_authorize,
            ) from err
        except (
            OSError,
            http.client.HTTPException,
        ) as err:
            reason = getattr(
                err,       # object
                "reason",  # name
                err,       # default
            )
            raise ServiceError(f"cannot reach {urllib.parse.urlsplit(url).netloc}: {reason}") from err

    @staticmethod
    def _error_of(
        err: urllib.error.HTTPError,
        is_service: bool,
    ) -> ServiceError:
        """The envelope's error read into an exception, or the bare status when it is unreadable."""
        map_error: dict[str, Any] = {}
        request_id = err.headers.get("x-request-id") if err.headers else None
        retry_after = err.headers.get(MarsTraceClient.RETRY_AFTER_HEADER) if err.headers else None
        if is_service:
            try:
                map_envelope = json.loads(err.read())
            except ValueError:
                map_envelope = None
            if ResponseSchema.matches(
                value=map_envelope,
                shape=ResponseSchema.ENVELOPE,
            ) is None:
                map_error = map_envelope["error"] or {}
                request_id = map_envelope["request_id"]
        code = map_error.get("code")
        add_credit_url = (map_error.get("details") or {}).get(MarsTraceClient.ADD_CREDIT_DETAIL)
        retry_after_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
        message = f"HTTP {err.code} {code or err.reason}"
        if map_error.get("message"):
            message += f": {map_error['message']}"
        if add_credit_url:
            message += f" ({add_credit_url})"
        if retry_after_seconds is not None:
            message += f"; retry after {retry_after_seconds} s"
        if request_id:
            message += f" (request {request_id})"
        return ServiceError(
            message=message,
            status=err.code,
            code=code,
            retryable=bool(map_error.get("retryable")),
            request_id=request_id,
            add_credit_url=add_credit_url,
            retry_after_seconds=retry_after_seconds,
        )

    def _call(
        self,
        method: str,
        path: str,
        shape: Any,
        map_body: dict[str, Any] | None = None,
        map_header: dict[str, str] | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> Any:
        """Send a JSON request to the API and return the envelope's ``data``, checked."""
        map_request_header = dict(map_header or {})
        body = None
        if map_body is not None:
            body = json.dumps(map_body).encode("utf-8")
            map_request_header["Content-Type"] = "application/json"
        where = f"{method} {path.split('?')[0]}"
        data = self._request(
            method=method,
            url=f"{self._credentials.api_url}{path}",
            body=body,
            map_header=map_request_header,
            timeout=timeout,
        )
        try:
            map_envelope = json.loads(data)
        except ValueError as err:
            raise ServiceError(f"{where} answered with a body that is not JSON") from err
        ResponseSchema.check(
            value=map_envelope,
            shape=ResponseSchema.ENVELOPE,
            where=where,
        )
        if map_envelope["error"] is not None:
            map_error = map_envelope["error"]
            raise ServiceError(
                message=f"{map_error['code']}: {map_error['message']} (request {map_envelope['request_id']})",
                code=map_error["code"],
                retryable=map_error["retryable"],
                request_id=map_envelope["request_id"],
            )
        return ResponseSchema.check(
            value=map_envelope.get("data"),
            shape=shape,
            where=where,
        )

    @staticmethod
    def _job_path(
        job_id: str,
        suffix: str = "",
    ) -> str:
        """The path of a job's route."""
        return f"/v1/jobs/{urllib.parse.quote(job_id, safe='')}{suffix}"

    def hardware(self) -> dict[str, Any]:
        """The hardware, rates, agent model, and default and maximum limits.

        Returns:
            ``hardware`` and ``agent``
        """
        return self._call(
            method="GET",
            path="/v1/hardware",
            shape=ResponseSchema.HARDWARE,
        )

    def credit(self) -> dict[str, Any]:
        """The workspace's available balance, against the minimum a job needs to start.

        Returns:
            ``billing_mode``, ``billing_status``, ``available_micro_usd`` (null without a
            spending limit), ``held_micro_usd``, ``minimum_micro_usd``, ``admissible`` and
            ``add_credit_url``
        """
        return self._call(
            method="GET",
            path="/v1/credit",
            shape=ResponseSchema.CREDIT,
        )

    @staticmethod
    def upload_name(path: str) -> str:
        """The name an uploaded file carries: its base name, in safe characters.

        Args:
            path: the file

        Returns:
            the name
        """
        name = MarsTraceClient.UPLOAD_NAME_PATTERN.sub(
            "_",                     # repl
            os.path.basename(path),  # string
        ).lstrip("._-")
        return (name or "input")[:MarsTraceClient.UPLOAD_NAME_CHARACTERS_MAX]

    def upload(
        self,
        data: bytes,
        role: str,
        name: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload one input file: a grant, a single or multipart PUT, then the verification.

        Args:
            data: the bytes
            role: the input's role: ``cubin``, ``source_archive``, ``launch_manifest``,
                ``tensor`` or ``file``
            name: the file's name
            content_type: the media type

        Returns:
            the verified upload

        Raises:
            ServiceError: the service refused a step, or granted something unusable
        """
        sha256 = hashlib.sha256(data).hexdigest()
        map_grant = self._call(
            method="POST",
            path="/v1/uploads",
            shape=ResponseSchema.UPLOAD_GRANT,
            map_body={
                "role": role,
                "name": name,
                "size_bytes": len(data),
                "sha256": sha256,
                "content_type": content_type,
            },
        )
        map_header = {"Content-Type": content_type}
        for header_name, value in (map_grant.get("headers") or {}).items():
            if header_name.lower() in MarsTraceClient.SET_FORBIDDEN_HEADER:
                raise ServiceError(f"the upload grant sets the forbidden header {header_name}")
            if (
                header_name.lower() == MarsTraceClient.CHECKSUM_HEADER
                and value != base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
            ):
                raise ServiceError(f"the upload grant of {name} is signed for different bytes")
            map_header[header_name] = value
        map_complete: dict[str, Any] = {
            "sha256": sha256,
            "size_bytes": len(data),
        }
        list_part = map_grant.get("parts")
        if list_part:
            map_complete["parts"] = self._upload_parts(
                data=data,
                list_part=list_part,
                map_header=map_header,
            )
        elif map_grant.get("upload_url"):
            self._request(
                method="PUT",
                url=map_grant["upload_url"],
                body=data,
                map_header=map_header,
                timeout=MarsTraceClient.TRANSFER_TIMEOUT_SECONDS,
                do_authorize=False,
            )
        else:
            raise ServiceError(f"the upload grant of {name} has neither upload_url nor parts")
        map_upload = self._call(
            method="POST",
            path=f"/v1/uploads/{urllib.parse.quote(map_grant['upload_id'], safe='')}/complete",
            shape=ResponseSchema.UPLOAD,
            map_body=map_complete,
        )
        if (
            map_upload["upload_id"] != map_grant["upload_id"]
            or map_upload.get("sha256", sha256) != sha256
            or map_upload.get("size_bytes", len(data)) != len(data)
        ):
            raise ServiceError(f"the service verified different bytes for {name}")
        return map_upload

    def _upload_parts(
        self,
        data: bytes,
        list_part: list[dict[str, Any]],
        map_header: dict[str, str],
    ) -> list[dict[str, Any]]:
        """PUT each part of a multipart grant and collect the receipts its completion names."""
        offset = 0
        list_receipt = []
        for part_position, map_part in enumerate(list_part):
            if (
                map_part["part_number"] != part_position + 1
                or map_part["offset"] != offset
                or not 0 < map_part["size_bytes"] <= len(data) - offset
            ):
                raise ServiceError("the multipart grant does not cover the file in ordered parts")
            map_response_header: dict[str, str] = {}
            self._request(
                method="PUT",
                url=map_part["upload_url"],
                body=data[offset:offset + map_part["size_bytes"]],
                map_header=map_header,
                timeout=MarsTraceClient.TRANSFER_TIMEOUT_SECONDS,
                do_authorize=False,
                map_response_header=map_response_header,
            )
            etag = map_response_header.get("etag")
            if not etag:
                raise ServiceError(f"part {map_part['part_number']} was stored without an ETag")
            list_receipt.append(
                {
                    "part_number": map_part["part_number"],
                    "etag": etag,
                }
            )
            offset += map_part["size_bytes"]
        if offset != len(data):
            raise ServiceError("the multipart grant does not cover the whole file")
        return list_receipt

    def quote(self, map_request: dict[str, Any]) -> dict[str, Any]:
        """Price a job request; the service holds and stores nothing.

        Args:
            map_request: the job request

        Returns:
            the rates, the bounds of one attempt, the cost at the bounds over every attempt,
            the minimum balance a job needs, the available balance and ``admissible``
        """
        return self._call(
            method="POST",
            path="/v1/quotes",
            shape=ResponseSchema.QUOTE,
            map_body=map_request,
        )

    def submit(
        self,
        map_request: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Submit a job; the same key with the same body never starts a second job.

        A submission whose answer never arrived, or that the service could not take for
        now, is sent again with the same key.

        Args:
            map_request: the job request
            idempotency_key: the key that makes the submission happen once; under one key
                the service takes only the same body

        Returns:
            the job

        Raises:
            ServiceError: the key is malformed, or the service refused the job, e.g.
                ``insufficient_credit`` when the available balance is below the minimum
        """
        if not MarsTraceClient.IDEMPOTENCY_KEY_PATTERN.match(idempotency_key):
            raise ServiceError("an idempotency key is 1 to 255 characters of A-Z, a-z, 0-9, '.', '_', ':' and '-'")
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._call(
                    method="POST",
                    path="/v1/jobs",
                    shape=ResponseSchema.JOB,
                    map_body=map_request,
                    map_header={"Idempotency-Key": idempotency_key},
                )
            except ServiceError as err:
                is_unanswered = err.status is None and err.code is None
                if (
                    not (is_unanswered or err.code == "service_unavailable")
                    or attempt >= MarsTraceClient.SUBMIT_ATTEMPTS
                ):
                    raise
                time.sleep(MarsTraceClient.RETRY_SECONDS)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """One job: its status, attempts, progress, cost and error.

        Args:
            job_id: the job

        Returns:
            the job
        """
        return self._call(
            method="GET",
            path=MarsTraceClient._job_path(job_id),
            shape=ResponseSchema.JOB,
        )

    def wait_job(
        self,
        job_id: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """The job as soon as its status changes, or at the timeout.

        Args:
            job_id: the job
            timeout_seconds: the longest the service holds the call, 1 to 60

        Returns:
            the job
        """
        timeout_seconds = max(
            1,
            min(
                timeout_seconds,
                MarsTraceClient.WAIT_SECONDS_MAX,
            ),
        )
        return self._call(
            method="GET",
            path=MarsTraceClient._job_path(
                job_id=job_id,
                suffix=f"/wait?{urllib.parse.urlencode({'timeout_seconds': timeout_seconds})}",
            ),
            shape=ResponseSchema.JOB,
            timeout=timeout_seconds + MarsTraceClient.WAIT_MARGIN_SECONDS,
        )

    def list_jobs(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """A page of the account's jobs, newest first.

        Args:
            limit: how many to list, 1 to 100
            cursor: the ``next_cursor`` of the previous page, or None for the first

        Returns:
            ``items`` and ``next_cursor``
        """
        map_query: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            map_query["cursor"] = cursor
        return self._call(
            method="GET",
            path=f"/v1/jobs?{urllib.parse.urlencode(map_query)}",
            shape=ResponseSchema.JOB_PAGE,
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop.

        Args:
            job_id: the job

        Returns:
            the job after the request was recorded
        """
        return self._call(
            method="POST",
            path=MarsTraceClient._job_path(
                job_id=job_id,
                suffix="/cancel",
            ),
            shape=ResponseSchema.JOB,
            map_body={},
        )

    def result(self, job_id: str) -> dict[str, Any]:
        """A succeeded job's result manifest and a presigned download URL of ``marstrace.json``.

        Args:
            job_id: the job

        Returns:
            ``manifest``, ``download_url``, ``download_expires_at`` and ``result_expires_at``
        """
        return self._call(
            method="GET",
            path=MarsTraceClient._job_path(
                job_id=job_id,
                suffix="/result",
            ),
            shape=ResponseSchema.RESULT,
        )

    def download(
        self,
        url: str,
        sha256: str,
        size_bytes: int,
    ) -> bytes:
        """The bytes behind a presigned download URL, checked against their digest and size.

        Args:
            url: the presigned URL
            sha256: the digest the bytes must have
            size_bytes: the size they must have

        Returns:
            the bytes

        Raises:
            ServiceError: the bytes differ
        """
        data = self._request(
            method="GET",
            url=url,
            map_header={"Accept-Encoding": "identity"},
            timeout=MarsTraceClient.TRANSFER_TIMEOUT_SECONDS,
            do_authorize=False,
        )
        if (
            len(data) != size_bytes
            or hashlib.sha256(data).hexdigest() != sha256
        ):
            raise ServiceError("the result arrived with different bytes than its manifest names")
        return data
