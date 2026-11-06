import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any
from xtrace_harness.error import PlatformError
from xtrace_harness.job import JobBundle, JobSpec


@dataclass
class Credentials:
    """The API key and the API origin the harness calls MarsCompute with."""

    ENV_API_KEY = "MARSCOMPUTE_API_KEY"
    ENV_API_URL = "MARSCOMPUTE_API_URL"
    API_URL_DEFAULT = "https://api.marscompute.ai"

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
                f"set {Credentials.ENV_API_KEY} to a MarsCompute API key "
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
    """The Xtrace job API of MarsCompute, as the harness's commands use it.

    The MCP server at ``<api_url>/mcp`` exposes the same operations to an agent; this client
    covers what moves bytes, which an MCP tool call cannot carry: the bundle upload and the
    result download.

    Args:
        credentials: the API key and origin
    """

    # seconds a request may take before it is abandoned, and the one that moves a bundle
    TIMEOUT_SECONDS = 60
    TRANSFER_TIMEOUT_SECONDS = 600
    BUNDLE_CONTENT_TYPE = "application/gzip"

    def __init__(self, credentials: Credentials) -> None:
        self._credentials = credentials

    def _request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
        content_type: str | None = None,
        map_header: dict[str, str] | None = None,
        timeout: int = TIMEOUT_SECONDS,
        do_authorize: bool = True,
    ) -> bytes:
        """Send one request and return the response body, raising on any refusal.

        A signed upload or download URL carries its own authorization, so it is sent
        without the API key.
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
        if content_type is not None:
            request.add_header(
                key="Content-Type",
                val=content_type,
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
                return response.read()
        except urllib.error.HTTPError as err:
            raise MarsComputeClient._error_of(err) from err
        except urllib.error.URLError as err:
            raise PlatformError(f"cannot reach {self._credentials.api_url}: {err.reason}") from err

    @staticmethod
    def _error_of(err: urllib.error.HTTPError) -> PlatformError:
        """The platform's error body read into an error, or the bare status when it is unreadable."""
        request_id = err.headers.get("x-request-id") if err.headers else None
        suffix = f" (request {request_id})" if request_id else ""
        try:
            map_error = json.loads(err.read()).get("error") or {}
        except (
            ValueError,
            AttributeError,
        ):
            map_error = {}
        return PlatformError(
            message=f"HTTP {err.code}: {map_error.get('message') or err.reason}{suffix}",
            status=err.code,
            code=map_error.get("code"),
        )

    def _json(
        self,
        method: str,
        path: str,
        map_body: dict[str, Any] | None = None,
        map_header: dict[str, str] | None = None,
    ) -> Any:
        """Send a JSON request and parse the JSON response."""
        body = json.dumps(map_body).encode() if map_body is not None else None
        data = self._request(
            method=method,
            url=f"{self._credentials.api_url}{path}",
            body=body,
            content_type="application/json" if body is not None else None,
            map_header=map_header,
        )
        try:
            return json.loads(data)
        except ValueError as err:
            raise PlatformError(f"{method} {path} answered with a body that is not JSON") from err

    def list_hardware(self) -> list[dict[str, Any]]:
        """The GPU classes a job can run on, with their prices and availability.

        Returns:
            one entry per hardware class
        """
        return self._json(
            method="GET",
            path="/v1/xtrace/hardware",
        )["data"]

    def get_balance(self) -> dict[str, Any]:
        """The tenant's credit available to new jobs.

        Returns:
            the balance, ``available_micro_usd``
        """
        return self._json(
            method="GET",
            path="/v1/xtrace/balance",
        )

    def upload(self, bundle: JobBundle) -> str:
        """Upload a job bundle to the signed URL the platform grants, then have it checked.

        Args:
            bundle: the packed job directory

        Returns:
            the upload's id, which a job submission names
        """
        map_upload = self._json(
            method="POST",
            path="/v1/xtrace/uploads",
            map_body={
                "size_bytes": len(bundle.data),
                "sha256": bundle.sha256,
            },
        )
        self._request(
            method="PUT",
            url=map_upload["upload_url"],
            body=bundle.data,
            content_type=MarsComputeClient.BUNDLE_CONTENT_TYPE,
            timeout=MarsComputeClient.TRANSFER_TIMEOUT_SECONDS,
            do_authorize=False,
        )
        self._json(
            method="POST",
            path=f"/v1/xtrace/uploads/{urllib.parse.quote(map_upload['id'])}/complete",
        )
        return map_upload["id"]

    def submit(
        self,
        upload_id: str,
        hardware: str,
        spec: JobSpec,
    ) -> dict[str, Any]:
        """Submit a job; a retry of the same call never starts a second job.

        Args:
            upload_id: the bundle's upload
            hardware: the hardware class to run on
            spec: what the job runs

        Returns:
            the job as the platform admitted it
        """
        return self._json(
            method="POST",
            path="/v1/xtrace/jobs",
            map_body={
                "upload_id": upload_id,
                "hardware": hardware,
                "spec": spec.to_json(),
            },
            map_header={
                "Idempotency-Key": str(uuid.uuid4()),
            },
        )

    def get_job(self, job_id: str) -> dict[str, Any]:
        """One job's state, cost and result files.

        Args:
            job_id: the job

        Returns:
            the job
        """
        return self._json(
            method="GET",
            path=f"/v1/xtrace/jobs/{urllib.parse.quote(job_id)}",
        )

    def list_jobs(self, limit: int) -> list[dict[str, Any]]:
        """The account's latest jobs, newest first.

        Args:
            limit: how many to list

        Returns:
            the jobs
        """
        return self._json(
            method="GET",
            path=f"/v1/xtrace/jobs?limit={limit}",
        )["data"]

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a job that has not finished; the time it ran is still charged.

        Args:
            job_id: the job

        Returns:
            the job after the cancellation was recorded
        """
        return self._json(
            method="POST",
            path=f"/v1/xtrace/jobs/{urllib.parse.quote(job_id)}/cancel",
        )

    def download(self, map_file: dict[str, Any]) -> bytes:
        """One result file of a finished job, from the signed URL the job lists it with.

        Args:
            map_file: the entry of the job's ``files``

        Returns:
            the file's bytes
        """
        return self._request(
            method="GET",
            url=map_file["url"],
            timeout=MarsComputeClient.TRANSFER_TIMEOUT_SECONDS,
            do_authorize=False,
        )
