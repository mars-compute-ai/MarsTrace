import decimal
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from marstrace.client import Credentials, MarsTraceClient
from marstrace.error import MarsTraceError, ServiceError
from marstrace.job import JobInput, JobManifest


@dataclass
class QuotedJob:
    """A job directory translated, uploaded and quoted, ready to submit.

    Args:
        manifest: the job directory's manifest
        map_request: the job request the quote was made for
        map_quote: the quote: the ceilings, the agent budget and the rates
        map_hardware_entry: the hardware the job runs on
        agent_model: the model every job's agent runs
    """

    MILLISECONDS_PER_SECOND = 1000
    # rate key of each agent token kind
    MAP_TOKEN_RATE = {
        "input": "agent_input_micro_usd_per_million",
        "cache_read": "agent_cache_read_micro_usd_per_million",
        "cache_write_5m": "agent_cache_write_5m_micro_usd_per_million",
        "cache_write_1h": "agent_cache_write_1h_micro_usd_per_million",
        "output": "agent_output_micro_usd_per_million",
    }

    manifest: JobManifest
    map_request: dict[str, Any]
    map_quote: dict[str, Any]
    map_hardware_entry: dict[str, Any]
    agent_model: str

    def summary(self) -> dict[str, Any]:
        """What the quote commits to, in dollars.

        Returns:
            the quote id and expiry, the hardware, the cost cap, the job fee, the GPU rate and
            ceiling, the agent's model, budget and token rates, and whether the service admits
            the request
        """
        map_rates = self.map_quote["rates"]
        map_ceilings = self.map_quote["ceilings"]
        return {
            "quote_id": self.map_quote["quote_id"],
            "expires_at": self.map_quote["expires_at"],
            "hardware": self.map_request["hardware"],
            "gpu": self.map_hardware_entry["gpu"],
            "architecture": self.map_hardware_entry["architecture"],
            "max_cost_usd": MarsTraceWorkflow.usd(self.map_quote["max_cost_micro_usd"]),
            "job_fee_usd": MarsTraceWorkflow.usd(map_ceilings["job_fee_micro_usd"]),
            "gpu_usd_per_hour": MarsTraceWorkflow.usd(map_rates["gpu_micro_usd_per_hour"]),
            "gpu_ceiling_seconds": map_ceilings["gpu_milliseconds"] // QuotedJob.MILLISECONDS_PER_SECOND,
            "gpu_ceiling_usd": MarsTraceWorkflow.usd(map_ceilings["gpu_micro_usd"]),
            "agent_model": self.agent_model,
            "agent_budget_usd": MarsTraceWorkflow.usd(self.map_quote["agent_budget_micro_usd"]),
            "agent_tokens_worst_case_usd": MarsTraceWorkflow.usd(map_ceilings["agent_tokens_micro_usd"]),
            "agent_usd_per_million_tokens": {
                kind: MarsTraceWorkflow.usd(map_rates[rate_key])
                for kind, rate_key in QuotedJob.MAP_TOKEN_RATE.items()
            },
            "admissible": self.map_quote["admissible"],
            "message": self.map_quote.get("message"),
        }


class MarsTraceWorkflow:
    """The operations the ``marstrace`` command and its MCP server offer, on local paths.

    Args:
        client: the MarsTrace service client
        progress: called with a line of progress, or None to stay silent
    """

    SET_JOB_FINAL_STATUS = {
        "succeeded",
        "failed",
        "cancelled",
    }
    MICRO_PER_UNIT = 1_000_000
    RESULTS_DIRECTORY = "marstrace-results"
    SUBMISSIONS_DIRECTORY = "submissions"
    RESULT_FILE = "marstrace.json"
    RESULT_MANIFEST_FILE = "result-manifest.json"
    JSON_INDENT = 2

    def __init__(
        self,
        client: MarsTraceClient,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._progress = progress

    @classmethod
    def from_environment(
        cls,
        progress: Callable[[str], None] | None = None,
    ) -> "MarsTraceWorkflow":
        """A workflow calling the service with the environment's credentials.

        Args:
            progress: called with a line of progress, or None to stay silent

        Returns:
            the workflow
        """
        return cls(
            client=MarsTraceClient(Credentials.from_environment()),
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
            amount = decimal.Decimal(str(micro_usd)) / MarsTraceWorkflow.MICRO_PER_UNIT
        except decimal.InvalidOperation:
            return "-"
        return f"${amount:.6f}"

    def _say(self, line: str) -> None:
        """Report a step of progress."""
        if self._progress is not None:
            self._progress(line)

    def hardware(self) -> dict[str, Any]:
        """The hardware, rates, agent model and limits.

        Returns:
            ``hardware`` and ``agent``
        """
        return self._client.hardware()

    def credit(self) -> dict[str, Any]:
        """The workspace's credit.

        Returns:
            the available and reserved credit and ``add_credit_url``
        """
        return self._client.credit()

    def _upload_file(self, job_input: JobInput) -> dict[str, Any]:
        """Upload one input of a job."""
        data = job_input.read()
        self._say(f"upload    {job_input.role} {job_input.path} ({len(data)} bytes)")
        return self._client.upload(
            data=data,
            role=job_input.role,
            name=MarsTraceClient.upload_name(job_input.path),
            content_type=job_input.content_type,
        )

    def quote(
        self,
        directory: str,
        hardware: str | None = None,
    ) -> QuotedJob:
        """Translate a job directory, upload its workspace and quote it.

        Args:
            directory: the job directory, holding ``marstrace.toml``
            hardware: a hardware id overriding the manifest's ``hardware``

        Returns:
            the quoted job

        Raises:
            MarsTraceError: the manifest is wrong, or the service refused
        """
        manifest = JobManifest.load(directory)
        if hardware is not None:
            manifest.hardware = hardware
        map_hardware = self._client.hardware()
        map_entry = manifest.select_hardware(map_hardware)
        list_input = manifest.list_input()
        map_upload_ids: dict[str, list[str]] = {}
        for job_input in list_input:
            map_upload_ids.setdefault(job_input.role, []).append(
                self._upload_file(job_input)["upload_id"]
            )
        map_request = manifest.request(
            map_hardware=map_hardware,
            map_upload_ids=map_upload_ids,
        )
        return QuotedJob(
            manifest=manifest,
            map_request=map_request,
            map_quote=self._client.quote(map_request),
            map_hardware_entry=map_entry,
            agent_model=map_hardware["agent"]["model"],
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
            the job

        Raises:
            MarsTraceError: the quote is not admissible
            ServiceError: the service refused the job; a refusal for credit names the page
                where credit is added
        """
        if not quoted.map_quote["admissible"]:
            raise MarsTraceError(
                f"the service does not admit this job: {quoted.map_quote.get('message') or 'see the quote'}; "
                "raise [limits] max_cost_usd or lower [agent] max_wall_seconds"
            )
        idempotency_key = idempotency_key or uuid.uuid4().hex
        map_submission = {
            "request": quoted.map_request,
            "quote_id": quoted.map_quote["quote_id"],
        }
        submission_path = MarsTraceWorkflow._submission_path(
            directory=quoted.manifest.directory,
            idempotency_key=idempotency_key,
        )
        os.makedirs(
            name=os.path.dirname(submission_path),
            exist_ok=True,
        )
        with open(
            file=submission_path,
            mode="w",
            encoding="utf-8",
        ) as submission_file:
            json.dump(
                map_submission,
                submission_file,
                indent=MarsTraceWorkflow.JSON_INDENT,
            )
        return self._submit_recorded(
            map_submission=map_submission,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _submission_path(
        directory: str,
        idempotency_key: str,
    ) -> str:
        """Where a job directory records the request it submitted under a key."""
        return os.path.join(
            directory,
            MarsTraceWorkflow.RESULTS_DIRECTORY,
            MarsTraceWorkflow.SUBMISSIONS_DIRECTORY,
            f"{idempotency_key}.json",
        )

    def resubmit(
        self,
        directory: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Submit again the request a job directory recorded under a key, without uploading again.

        The service answers a repeated key and body with the job it started, so a submission
        whose answer was lost is recovered without a second job.

        Args:
            directory: the job directory
            idempotency_key: the key of the earlier submission

        Returns:
            the job, or None when the directory recorded nothing under the key
        """
        submission_path = MarsTraceWorkflow._submission_path(
            directory=directory,
            idempotency_key=idempotency_key,
        )
        if not os.path.isfile(submission_path):
            return None
        with open(
            file=submission_path,
            mode="r",
            encoding="utf-8",
        ) as submission_file:
            map_submission = json.load(submission_file)
        self._say(f"resubmit  the request recorded under {idempotency_key}")
        return self._submit_recorded(
            map_submission=map_submission,
            idempotency_key=idempotency_key,
        )

    def _submit_recorded(
        self,
        map_submission: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Submit a recorded request; a refusal for credit names the page where credit is added."""
        try:
            return self._client.submit(
                map_request=map_submission["request"],
                idempotency_key=idempotency_key,
                quote_id=map_submission["quote_id"],
            )
        except ServiceError as err:
            if err.code not in MarsTraceClient.SET_CREDIT_CODE:
                raise
            add_credit_url = self._credit_page()
            if add_credit_url is None:
                raise
            raise ServiceError(
                message=f"{err}; add credit at {add_credit_url}",
                status=err.status,
                code=err.code,
                retryable=err.retryable,
                request_id=err.request_id,
                add_credit_url=add_credit_url,
            ) from err

    def _credit_page(self) -> str | None:
        """The console page where credit is added, or None when the service cannot say."""
        try:
            return self._client.credit()["add_credit_url"]
        except ServiceError:
            return None

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
        """One job.

        Args:
            job_id: the job

        Returns:
            the job
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

    def result(self, job_id: str) -> dict[str, Any]:
        """A succeeded job's result manifest and download URL.

        Args:
            job_id: the job

        Returns:
            the result
        """
        return self._client.result(job_id)

    def wait(
        self,
        job_id: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Long-poll a job until it ends or the time is up.

        Args:
            job_id: the job
            timeout_seconds: how long to wait, or None to wait until it ends

        Returns:
            the last view of the job
        """
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        map_job = self._client.get_job(job_id)
        self._say(f"{time.strftime('%H:%M:%S')} {job_id} {map_job['status']}")
        while map_job["status"] not in MarsTraceWorkflow.SET_JOB_FINAL_STATUS:
            wait_seconds = MarsTraceClient.WAIT_SECONDS_MAX
            if deadline is not None:
                wait_seconds = min(
                    wait_seconds,
                    int(deadline - time.monotonic()),
                )
                if wait_seconds < 1:
                    return map_job
            last_status = map_job["status"]
            map_job = self._client.wait_job(
                job_id=job_id,
                timeout_seconds=wait_seconds,
            )
            if map_job["status"] != last_status:
                self._say(f"{time.strftime('%H:%M:%S')} {job_id} {map_job['status']}")
        return map_job

    def fetch(
        self,
        job_id: str,
        output_directory: str,
    ) -> list[str]:
        """Download a succeeded job's ``marstrace.json`` and write its result manifest beside it.

        Args:
            job_id: the job
            output_directory: the directory the files land in

        Returns:
            the paths written

        Raises:
            ServiceError: the job has no result, or the file's bytes differ from its manifest
        """
        map_result = self._client.result(job_id)
        map_manifest = map_result["manifest"]
        data = self._client.download(
            url=map_result["download_url"],
            sha256=map_manifest["file"]["sha256"],
            size_bytes=map_manifest["file"]["size_bytes"],
        )
        os.makedirs(
            name=output_directory,
            exist_ok=True,
        )
        trace_path = os.path.join(
            output_directory,
            MarsTraceWorkflow.RESULT_FILE,
        )
        with open(
            file=trace_path,
            mode="wb",
        ) as result_file:
            result_file.write(data)
        manifest_path = os.path.join(
            output_directory,
            MarsTraceWorkflow.RESULT_MANIFEST_FILE,
        )
        with open(
            file=manifest_path,
            mode="w",
            encoding="utf-8",
        ) as manifest_file:
            json.dump(
                map_manifest,
                manifest_file,
                indent=MarsTraceWorkflow.JSON_INDENT,
            )
        return [
            trace_path,
            manifest_path,
        ]
