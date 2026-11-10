import decimal
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import HarnessError, PlatformError
from xtrace_harness.job import JobArtifact, JobManifest


@dataclass
class QuotedJob:
    """A job directory translated, uploaded and quoted, ready to submit.

    Args:
        manifest: the job directory's manifest
        map_request: the estimate input the quote was made for
        map_quote: the quote: the cost cap, the rates and the limits
        architecture: the architecture of the chosen hardware
    """

    manifest: JobManifest
    map_request: dict[str, Any]
    map_quote: dict[str, Any]
    architecture: str

    def summary(self) -> dict[str, Any]:
        """What the quote commits to, in dollars.

        Returns:
            the quote id, the hardware, the cost cap, the GPU rate, the job fee, the agent
            model's token rates, the limits and the expiry
        """
        map_rate = self.map_quote["rate_snapshot"]
        map_summary = {
            "quote_id": self.map_quote["quote_id"],
            "hardware": self.manifest.hardware,
            "architecture": self.architecture,
            "max_cost_usd": XtraceWorkflow.usd(self.map_quote["limits"]["max_total_cost_micro_usd"]),
            "gpu_usd_per_hour": XtraceWorkflow.usd(map_rate["gpu_micro_usd_per_hour"]),
            "limits": self.map_quote["limits"],
            "expires_at": self.map_quote.get("expires_at"),
        }
        map_price_book = map_rate.get("price_book")
        if map_price_book is not None:
            map_summary["job_fee_usd"] = XtraceWorkflow.usd(map_price_book["tool_fee_micro_usd"])
            map_agent_rate = next(
                (
                    map_model_rate
                    for map_model_rate in map_rate["model_rates"]
                    if map_model_rate["model"] == self.map_request["analysis"].get("model")
                ),
                None,
            )
            if map_agent_rate is not None:
                map_summary["agent_usd_per_million_tokens"] = {
                    "model": map_agent_rate["model"],
                    "input": XtraceWorkflow.usd(map_agent_rate["input_micro_usd_per_million"]),
                    "cached_input": XtraceWorkflow.usd(map_agent_rate["cached_input_micro_usd_per_million"]),
                    "output": XtraceWorkflow.usd(map_agent_rate["output_micro_usd_per_million"]),
                    "map_cache_write": {
                        map_cache_write["dimension"]: XtraceWorkflow.usd(map_cache_write["micro_usd_per_million"])
                        for map_cache_write in map_price_book["cache_write_rates"]
                    },
                }
        return map_summary


class XtraceWorkflow:
    """The operations the ``xtrace`` command and its MCP server offer, on local paths.

    Args:
        client: the Control Plane client
        progress: called with a line of progress, or None to stay silent
        poll_seconds: the floor between polls of a job
    """

    SET_JOB_FINAL_STATE = {
        "succeeded",
        "failed",
        "cancelled",
    }
    POLL_SECONDS_DEFAULT = 2.0
    POLL_SECONDS_MAX = 60.0
    MILLISECONDS_PER_SECOND = 1000
    MICRO_PER_UNIT = 1_000_000
    RESULTS_DIRECTORY = "xtrace-results"
    RESULT_MANIFEST_FILE = "result-manifest.json"
    JSON_INDENT = 2

    def __init__(
        self,
        client: MarsComputeClient,
        progress: Callable[[str], None] | None = None,
        poll_seconds: float = POLL_SECONDS_DEFAULT,
    ) -> None:
        self._client = client
        self._progress = progress
        self._poll_seconds = poll_seconds

    @classmethod
    def from_environment(
        cls,
        progress: Callable[[str], None] | None = None,
    ) -> "XtraceWorkflow":
        """A workflow calling the Control Plane with the environment's credentials.

        Args:
            progress: called with a line of progress, or None to stay silent

        Returns:
            the workflow
        """
        return cls(
            client=MarsComputeClient(Credentials.from_environment()),
            progress=progress,
        )

    @staticmethod
    def usd(micro_usd: Any) -> str:
        """A money string of micro-USD as dollars.

        Args:
            micro_usd: the amount, a decimal string or an integer

        Returns:
            e.g. ``$1.250000``, or ``-`` when the amount is absent
        """
        if micro_usd is None:
            return "-"
        try:
            amount = decimal.Decimal(str(micro_usd)) / XtraceWorkflow.MICRO_PER_UNIT
        except decimal.InvalidOperation:
            return "-"
        return f"${amount:.6f}"

    def _say(self, line: str) -> None:
        """Report a step of progress."""
        if self._progress is not None:
            self._progress(line)

    def hardware(self) -> dict[str, Any]:
        """The hardware, rates, architectures, agent models and limits open to the account.

        Returns:
            the capabilities
        """
        return self._client.capabilities()

    def credit(self) -> dict[str, Any]:
        """The tenant's credit.

        Returns:
            ``available_micro_usd``, ``reserved_micro_usd`` and ``top_up_url``
        """
        return self._client.credit()

    def _upload_file(
        self,
        artifact: JobArtifact,
        upload_bytes_max: int | None,
    ) -> dict[str, Any]:
        """Upload one file of a job as an artifact."""
        data = artifact.read()
        if (
            upload_bytes_max is not None
            and len(data) > upload_bytes_max
        ):
            raise HarnessError(
                f"{artifact.path} is {len(data)} bytes; the platform accepts at most {upload_bytes_max}"
            )
        self._say(f"upload    {artifact.kind} {artifact.path} ({len(data)} bytes)")
        return self._client.upload(
            data=data,
            kind=artifact.kind,
            name=MarsComputeClient.artifact_name(artifact.path),
            content_type=artifact.content_type,
        )

    def quote(
        self,
        directory: str,
        hardware: str | None = None,
    ) -> QuotedJob:
        """Translate a job directory, upload its workspace and quote it.

        Args:
            directory: the job directory, holding ``xtrace.toml``
            hardware: an environment id overriding the manifest's ``hardware``

        Returns:
            the quoted job

        Raises:
            HarnessError: the manifest is wrong, or the platform refused
        """
        manifest = JobManifest.load(directory)
        if hardware is not None:
            manifest.hardware = hardware
        map_capabilities = self._client.capabilities()
        map_entry = manifest.select_compute(map_capabilities)
        map_artifact_ids: dict[str, list[str]] = {}
        for artifact in manifest.list_artifact():
            map_artifact_ids.setdefault(artifact.kind, []).append(
                self._upload_file(
                    artifact=artifact,
                    upload_bytes_max=map_capabilities.get("max_upload_bytes"),
                )["artifact_id"]
            )
        map_request = manifest.estimate_input(
            map_capabilities=map_capabilities,
            map_artifact_ids=map_artifact_ids,
        )
        return QuotedJob(
            manifest=manifest,
            map_request=map_request,
            map_quote=self._client.estimate(map_request),
            architecture=map_entry["architecture"],
        )

    def submit(
        self,
        quoted: QuotedJob,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a quoted job.

        Args:
            quoted: the quoted job
            idempotency_key: the key that makes the submission happen once, or None for a
                fresh one

        Returns:
            the admitted job
        """
        return self._client.submit(
            map_request=quoted.map_request,
            quote_id=quoted.map_quote["quote_id"],
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )

    def jobs(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """A page of the account's jobs.

        Args:
            limit: how many to list
            cursor: the previous page's ``next_cursor``, or None

        Returns:
            ``items`` and ``next_cursor``
        """
        return self._client.list_jobs(
            limit=limit,
            cursor=cursor,
        )

    def job(self, job_id: str) -> dict[str, Any]:
        """One job's view.

        Args:
            job_id: the job

        Returns:
            ``job``, ``attempts`` and ``poll_after_milliseconds``
        """
        return self._client.get_job(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop.

        Args:
            job_id: the job

        Returns:
            the job
        """
        return self._client.cancel(job_id)

    def wait(
        self,
        job_id: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll a job at the pace the platform asks for until it ends or the time is up.

        Args:
            job_id: the job
            timeout_seconds: how long to wait, or None to wait until it ends

        Returns:
            the last view of the job
        """
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        last_state = None
        while True:
            map_view = self._client.get_job(job_id)
            state = map_view["job"]["state"]
            if state != last_state:
                self._say(f"{time.strftime('%H:%M:%S')} {job_id} {state}")
                last_state = state
            if state in XtraceWorkflow.SET_JOB_FINAL_STATE:
                return map_view
            poll_seconds = min(
                max(
                    self._poll_seconds,
                    map_view.get(
                        "poll_after_milliseconds",
                        0,
                    ) / XtraceWorkflow.MILLISECONDS_PER_SECOND,
                ),
                XtraceWorkflow.POLL_SECONDS_MAX,
            )
            if (
                deadline is not None
                and time.monotonic() + poll_seconds > deadline
            ):
                return map_view
            time.sleep(poll_seconds)

    def fetch(
        self,
        job_id: str,
        output_directory: str,
    ) -> list[str]:
        """Download a finished job's result files and its result manifest.

        Args:
            job_id: the job
            output_directory: the directory the files land in

        Returns:
            the paths written

        Raises:
            PlatformError: a file's bytes differ from the result manifest
        """
        map_result = self._client.result(job_id)
        map_manifest = map_result["manifest"]
        map_file = {map_entry["artifact_id"]: map_entry for map_entry in map_manifest.get("files", [])}
        os.makedirs(
            name=output_directory,
            exist_ok=True,
        )
        list_path = []
        for map_grant in map_result.get("downloads", []):
            data = self._client.download(map_grant)
            map_entry = map_file.get(map_grant["artifact_id"], {})
            uncompressed_sha256 = map_entry.get("uncompressed_sha256")
            if (
                map_grant.get("encoding") == "gzip"
                and uncompressed_sha256 is not None
                and hashlib.sha256(data).hexdigest() != uncompressed_sha256
            ):
                raise PlatformError(f"{map_grant['name']} decompressed to different bytes than its manifest")
            path = os.path.join(
                output_directory,
                MarsComputeClient.artifact_name(map_entry.get("name") or map_grant["name"]),
            )
            with open(
                file=path,
                mode="wb",
            ) as result_file:
                result_file.write(data)
            list_path.append(path)
        path = os.path.join(
            output_directory,
            XtraceWorkflow.RESULT_MANIFEST_FILE,
        )
        with open(
            file=path,
            mode="w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                map_manifest,
                manifest_file,
                indent=XtraceWorkflow.JSON_INDENT,
            )
        list_path.append(path)
        return list_path
