import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from xtrace_harness.error import PlatformError


@dataclass
class Credentials:
    """The API key and the Control Plane origin the harness calls MarsCompute with."""

    ENV_API_KEY = "MARSCOMPUTE_API_KEY"
    ENV_API_URL = "MARSCOMPUTE_API_URL"
    API_URL_DEFAULT = "https://control.marscompute.ai"

    api_key: str
    api_url: str

    @classmethod
    def from_environment(cls) -> "Credentials":
        """Read the key and origin from the environment.

        Returns:
            the credentials

        Raises:
            PlatformError: no API key is set
        """
        api_key = os.environ.get(
            key=Credentials.ENV_API_KEY,
            default="",
        ).strip()
        if not api_key:
            raise PlatformError(
                f"set {Credentials.ENV_API_KEY} to a MarsCompute API key, mc_live_v1_... "
                "(console.marscompute.ai, API keys)"
            )
        return cls(
            api_key=api_key,
            api_url=os.environ.get(
                key=Credentials.ENV_API_URL,
                default=Credentials.API_URL_DEFAULT,
            ).rstrip("/"),
        )


class MarsComputeClient:
    """The Xtrace API of the MarsCompute Control Plane, as the harness's commands use it.

    Args:
        credentials: the API key and origin
    """

    # seconds a request may take before it is abandoned, and the one that moves a file
    TIMEOUT_SECONDS = 60
    TRANSFER_TIMEOUT_SECONDS = 600
    # attempts of a submission whose answer never arrived, under the same idempotency key
    SUBMIT_ATTEMPTS = 3
    RETRY_SECONDS = 2
    INSPECTION_SCHEMA = "marscompute.xtrace.inspection-request.v1"
    ARTIFACT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]")
    ARTIFACT_NAME_CHARACTERS_MAX = 256
    # headers a grant may not set on an upload
    SET_FORBIDDEN_HEADER = {
        "authorization",
        "proxy-authorization",
        "cookie",
        "host",
        "connection",
        "transfer-encoding",
    }

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        map_header: dict[str, str] | None = None,
        timeout: int = TIMEOUT_SECONDS,
        do_authorize: bool = True,
        map_response_header: dict[str, str] | None = None,
    ) -> bytes:
        """Send one request and return the response body, raising on any refusal.

        A signed upload or download URL carries its own authorization, so it is sent
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
            val="xtrace-harness",
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
            raise MarsComputeClient._error_of(err) from err
        except urllib.error.URLError as err:
            raise PlatformError(f"cannot reach {urllib.parse.urlsplit(url).netloc}: {err.reason}") from err

    @staticmethod
    def _error_of(err: urllib.error.HTTPError) -> PlatformError:
        """The platform's error body read into an error, or the bare status when it is unreadable."""
        try:
            map_error = json.loads(err.read()).get("error") or {}
        except (
            ValueError,
            AttributeError,
        ):
            map_error = {}
        correlation_id = map_error.get("correlation_id") or (
            err.headers.get("x-request-id") if err.headers else None
        )
        code = map_error.get("code")
        message = f"HTTP {err.code} {code or err.reason}"
        if map_error.get("message"):
            message += f": {map_error['message']}"
        if map_error.get("top_up_url"):
            message += f"; add credit at {map_error['top_up_url']}"
        if correlation_id:
            message += f" (request {correlation_id})"
        return PlatformError(
            message=message,
            status=err.code,
            code=code,
            correlation_id=correlation_id,
            top_up_url=map_error.get("top_up_url"),
        )

    def _json(
        self,
        method: str,
        path: str,
        map_body: dict[str, Any] | None = None,
        map_header: dict[str, str] | None = None,
    ) -> Any:
        """Send a JSON request to the API and parse the JSON response."""
        map_request_header = dict(map_header or {})
        body = None
        if map_body is not None:
            body = json.dumps(map_body).encode("utf-8")
            map_request_header["Content-Type"] = "application/json"
        data = self._request(
            method=method,
            url=f"{self._credentials.api_url}{path}",
            body=body,
            map_header=map_request_header,
        )
        try:
            return json.loads(data)
        except ValueError as err:
            raise PlatformError(f"{method} {path} answered with a body that is not JSON") from err

    def capabilities(self) -> dict[str, Any]:
        """The hardware, stages, models and limits open to the account.

        Returns:
            the capabilities, ``marscompute.xtrace.capabilities.v1``
        """
        return self._json(
            method="GET",
            path="/v1/xtrace/capabilities",
        )

    def credit(self) -> dict[str, Any]:
        """The tenant's credit: available, held by running jobs, and where to add more.

        Returns:
            ``available_micro_usd``, ``reserved_micro_usd`` and ``top_up_url``
        """
        return self._json(
            method="GET",
            path="/v1/xtrace/credit",
        )

    @staticmethod
    def artifact_name(path: str) -> str:
        """The name an uploaded file carries: its base name, in the characters artifacts allow.

        Args:
            path: the file

        Returns:
            the name
        """
        name = MarsComputeClient.ARTIFACT_NAME_PATTERN.sub(
            "_",                     # repl
            os.path.basename(path),  # string
        ).lstrip("._-")
        return (name or "artifact")[:MarsComputeClient.ARTIFACT_NAME_CHARACTERS_MAX]

    def upload(
        self,
        data: bytes,
        kind: str,
        name: str,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload bytes as an artifact: a grant, a single or multipart PUT, then the completion.

        Args:
            data: the bytes
            kind: the artifact kind, e.g. ``cubin``
            name: the artifact's name
            content_type: the media type

        Returns:
            the verified artifact

        Raises:
            PlatformError: the platform refused a step or verified different bytes
        """
        sha256 = hashlib.sha256(data).hexdigest()
        map_grant = self._json(
            method="POST",
            path="/v1/xtrace/artifacts/uploads",
            map_body={
                "kind": kind,
                "name": name,
                "size_bytes": len(data),
                "sha256": sha256,
                "content_type": content_type,
            },
        )
        map_header = {"Content-Type": content_type}
        for header_name, value in (map_grant.get("headers") or {}).items():
            if header_name.lower() in MarsComputeClient.SET_FORBIDDEN_HEADER:
                raise PlatformError(f"the upload grant sets the forbidden header {header_name}")
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
        else:
            self._request(
                method="PUT",
                url=map_grant["upload_url"],
                body=data,
                map_header=map_header,
                timeout=MarsComputeClient.TRANSFER_TIMEOUT_SECONDS,
                do_authorize=False,
            )
        map_artifact = self._json(
            method="POST",
            path=f"/v1/xtrace/artifacts/{urllib.parse.quote(map_grant['artifact_id'])}/complete",
            map_body=map_complete,
        )
        if (
            map_artifact.get("artifact_id") != map_grant["artifact_id"]
            or map_artifact.get("sha256") != sha256
            or map_artifact.get("size_bytes") != len(data)
        ):
            raise PlatformError(f"the platform verified different bytes for {name}")
        return map_artifact

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
                map_part.get("part_number") != part_position + 1
                or map_part.get("offset") != offset
                or not 0 < map_part.get("size_bytes", 0) <= len(data) - offset
            ):
                raise PlatformError("the multipart grant does not cover the file in ordered parts")
            map_response_header: dict[str, str] = {}
            self._request(
                method="PUT",
                url=map_part["upload_url"],
                body=data[offset:offset + map_part["size_bytes"]],
                map_header=map_header,
                timeout=MarsComputeClient.TRANSFER_TIMEOUT_SECONDS,
                do_authorize=False,
                map_response_header=map_response_header,
            )
            etag = map_response_header.get("etag")
            if not etag:
                raise PlatformError(f"part {map_part['part_number']} was stored without an ETag")
            list_receipt.append(
                {
                    "part_number": map_part["part_number"],
                    "etag": etag,
                }
            )
            offset += map_part["size_bytes"]
        if offset != len(data):
            raise PlatformError("the multipart grant does not cover the whole file")
        return list_receipt

    def estimate(self, map_request: dict[str, Any]) -> dict[str, Any]:
        """Validate a job and freeze a quote: the cost cap, the rates and the limits.

        Args:
            map_request: the estimate input

        Returns:
            the quote
        """
        return self._json(
            method="POST",
            path="/v1/xtrace/quotes",
            map_body=map_request,
        )

    def submit(
        self,
        map_request: dict[str, Any],
        quote_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Submit a quoted job; a retry with the same key never starts a second job.

        A submission whose answer never arrived is sent again with the same key.

        Args:
            map_request: the estimate input the quote was made for
            quote_id: the accepted quote
            idempotency_key: the key that makes the submission happen once

        Returns:
            the admitted job

        Raises:
            PlatformError: the platform refused the job, e.g. ``insufficient_credit``
        """
        map_body = {
            **map_request,
            "quote_id": quote_id,
            "idempotency_key": idempotency_key,
        }
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._json(
                    method="POST",
                    path="/v1/xtrace/jobs",
                    map_body=map_body,
                    map_header={"Idempotency-Key": idempotency_key},
                )
            except PlatformError as err:
                if (
                    err.status is not None
                    or attempt >= MarsComputeClient.SUBMIT_ATTEMPTS
                ):
                    raise
                time.sleep(MarsComputeClient.RETRY_SECONDS)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """One job's state, attempts, accounting and result manifest.

        Args:
            job_id: the job

        Returns:
            the job view: ``job``, ``attempts`` and ``poll_after_milliseconds``
        """
        return self._json(
            method="GET",
            path=f"/v1/xtrace/jobs/{urllib.parse.quote(job_id)}",
        )

    def list_jobs(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """A page of the account's jobs, newest first.

        Args:
            limit: how many to list, at most 100
            cursor: the ``next_cursor`` of the previous page, or None for the first

        Returns:
            ``items`` and ``next_cursor``
        """
        map_query: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            map_query["cursor"] = cursor
        return self._json(
            method="GET",
            path=f"/v1/xtrace/jobs?{urllib.parse.urlencode(map_query)}",
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop and release its resources.

        Args:
            job_id: the job

        Returns:
            the job after the request was recorded
        """
        return self._json(
            method="POST",
            path=f"/v1/xtrace/jobs/{urllib.parse.quote(job_id)}/cancel",
            map_body={},
        )

    def result(self, job_id: str) -> dict[str, Any]:
        """A finished job's result manifest and a download grant for each of its files.

        Args:
            job_id: the job

        Returns:
            ``manifest`` and ``downloads``
        """
        return self._json(
            method="GET",
            path=f"/v1/xtrace/jobs/{urllib.parse.quote(job_id)}/result",
        )

    def download(self, map_grant: dict[str, Any]) -> bytes:
        """The bytes of a download grant, checked against its digest and size.

        Args:
            map_grant: the download grant

        Returns:
            the file's bytes, decompressed when the grant serves it gzip-encoded

        Raises:
            PlatformError: the bytes differ from the grant
        """
        data = self._request(
            method="GET",
            url=map_grant["download_url"],
            map_header={"Accept-Encoding": "identity"},
            timeout=MarsComputeClient.TRANSFER_TIMEOUT_SECONDS,
            do_authorize=False,
        )
        if (
            len(data) != map_grant["size_bytes"]
            or hashlib.sha256(data).hexdigest() != map_grant["sha256"]
        ):
            raise PlatformError(f"{map_grant.get('name')} arrived with different bytes than granted")
        if map_grant.get("encoding") == "gzip":
            return gzip.decompress(data)
        return data

    def create_inspection(
        self,
        cubin_artifact_id: str,
        kernel_symbol: str | None,
    ) -> dict[str, Any]:
        """Ask for a cubin's SASS listing; inspections are free and rate-limited.

        Args:
            cubin_artifact_id: the verified cubin artifact
            kernel_symbol: the kernel to list, or None for the cubin's kernels only

        Returns:
            the inspection, queued
        """
        map_body = {
            "schema_version": MarsComputeClient.INSPECTION_SCHEMA,
            "cubin_artifact_id": cubin_artifact_id,
        }
        if kernel_symbol is not None:
            map_body["kernel_symbol"] = kernel_symbol
        return self._json(
            method="POST",
            path="/v1/xtrace/inspections",
            map_body=map_body,
        )

    def get_inspection(self, inspection_id: str) -> dict[str, Any]:
        """One inspection's state and, once it succeeded, its result's download grant.

        Args:
            inspection_id: the inspection

        Returns:
            the inspection
        """
        return self._json(
            method="GET",
            path=f"/v1/xtrace/inspections/{urllib.parse.quote(inspection_id)}",
        )
