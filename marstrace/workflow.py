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
    """A job directory translated, uploaded and priced, ready to submit.

    Args:
        manifest: the job directory's manifest
        map_request: the job request the estimate was made for
        map_quote: the service's estimate: rates, bounds, cost at the bounds and admission
        map_hardware_entry: the hardware the job runs on
        agent_model: the model every job's agent runs
    """

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
        """What the job would cost, in dollars.

        A job is charged the agent tokens it actually used over every attempt, failed ones
        included, and nothing when its trace is delivered unusable or it fails without a result; the cost at the bounds is the most it can reach, and nothing is held against
        it.

        Returns:
            the hardware, the CUDA version, the mode, the token rates, the token bounds of one
            attempt, the attempts, the cost at the bounds, the minimum balance a job of the mode
            needs, the available balance, whether the service admits the job now, and the
            service's notes on the job
        """
        map_rates = self.map_quote["rates"]
        map_bounds = self.map_quote["bounds_per_attempt"]
        map_cost = self.map_quote["cost_at_bounds_micro_usd"]
        available = self.map_quote["available_micro_usd"]
        return {
            "hardware": self.map_request["hardware"],
            "gpu": self.map_hardware_entry["gpu"],
            "architecture": self.map_hardware_entry["architecture"],
            "cuda_version": self.map_quote["cuda_version"],
            "mode": self.manifest.mode,
            "rates": {
                "agent_model": self.agent_model,
                "agent_usd_per_million_tokens": {
                    kind: MarsTraceWorkflow.usd(map_rates[rate_key])
                    for kind, rate_key in QuotedJob.MAP_TOKEN_RATE.items()
                },
            },
            "bounds_per_attempt": {
                "agent_input_tokens": map_bounds["agent_input_tokens"],
                "agent_output_tokens": map_bounds["agent_output_tokens"],
            },
            "max_attempts": self.manifest.limits.attempts(self.manifest.mode),
            "cost_at_bounds": {
                "agent_tokens_per_attempt_usd": MarsTraceWorkflow.usd(map_cost["agent_tokens_per_attempt"]),
                "total_usd": MarsTraceWorkflow.usd(map_cost["total"]),
            },
            "minimum_usd": MarsTraceWorkflow.usd(self.map_quote["minimum_micro_usd"]),
            "available_usd": "no limit" if available is None else MarsTraceWorkflow.usd(available),
            "admissible": self.map_quote["admissible"],
            "message": self.map_quote.get("message"),
            "notes": list(self.map_quote.get("notes") or []),
        }


class MarsTraceWorkflow:
    """The operations the ``marstrace`` command and its MCP server offer, on local paths.

    Args:
        client: the MarsTrace service client
        progress: called with a line of progress, or None to stay silent
    """

    UNUSABLE_REASON = "unusable_result"  # why a job whose trace is unusable is not charged
    FAILED_NO_RESULT_REASON = "failed_no_result"  # why a job that failed without a result is not charged
    MAP_NOT_CHARGED_NOTE = {
        UNUSABLE_REASON: "the trace is unusable, so the job is not charged",
        FAILED_NO_RESULT_REASON: "the job failed without a result, so it is not charged",
    }
    SET_JOB_FINAL_STATUS = {
        "succeeded",
        "failed",
        "cancelled",
    }
    # a long poll that fails in transit, or that a gateway answers 502, 503 or 504, is sent
    # again after WAIT_RETRY_SECONDS, up to WAIT_FAILURES_MAX failures in a row; the job runs on
    SET_TRANSIENT_STATUS = {502, 503, 504}
    WAIT_RETRY_SECONDS = 15
    WAIT_FAILURES_MAX = 10
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
            e.g. ``$1.250000`` or ``-$0.500000``, or ``-`` when the amount is absent
        """
        if micro_usd is None:
            return "-"
        try:
            amount = decimal.Decimal(str(micro_usd)) / MarsTraceWorkflow.MICRO_PER_UNIT
        except decimal.InvalidOperation:
            return "-"
        if amount < 0:
            return f"-${-amount:.6f}"
        return f"${amount:.6f}"

    @staticmethod
    def speedup_text(speedup: Any) -> str:
        """A speedup as a reader quotes it.

        Args:
            speedup: baseline time over optimized time, or None

        Returns:
            e.g. ``1.37x``, or ``none yet`` when there is none
        """
        if (
            isinstance(
                speedup,  # obj
                bool,     # class_or_tuple
            )
            or not isinstance(
                speedup,       # obj
                (int, float),  # class_or_tuple
            )
        ):
            return "none yet"
        return f"{speedup:.2f}x"

    @staticmethod
    def optimize_progress_text(map_optimize: dict[str, Any]) -> str:
        """The progress of a thinking job's optimization stage, on one line.

        Args:
            map_optimize: ``progress.optimize`` of the job

        Returns:
            the stage, the iteration, the changes kept and reverted, the best speedup, and the
            time spent against the stage's budget
        """
        return (
            f"{map_optimize['stage']}, iteration {map_optimize['iteration']}: "
            f"{map_optimize['kept']} kept, {map_optimize['reverted']} reverted, "
            f"best speedup {MarsTraceWorkflow.speedup_text(map_optimize['best_speedup'])}, "
            f"{map_optimize['elapsed_seconds']:.0f} s of {map_optimize['budget_seconds']:.0f} s"
        )

    @staticmethod
    def optimization_result_text(map_optimization: dict[str, Any]) -> str:
        """The outcome of a thinking job's optimization, on one line.

        Args:
            map_optimization: ``result.optimization`` of the job

        Returns:
            the status, the best speedup over the configurations, the correctness, and the
            iterations kept and reverted
        """
        return (
            f"{map_optimization['status']}, best speedup "
            f"{MarsTraceWorkflow.speedup_text(map_optimization['best_speedup'])} over "
            f"{map_optimization['configs']} configuration(s), correctness "
            f"{map_optimization['correctness']}, {map_optimization['iterations']} iteration(s): "
            f"{map_optimization['kept']} kept, {map_optimization['reverted']} reverted"
        )

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
        """The workspace's balance against the minimum a job needs.

        Returns:
            the available, held and minimum balance, ``admissible`` and ``add_credit_url``
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
        cuda_version: str | None = None,
    ) -> QuotedJob:
        """Translate a job directory, upload its workspace and price it.

        Args:
            directory: the job directory, holding ``marstrace.toml``
            hardware: a hardware id overriding the manifest's ``hardware``
            cuda_version: a CUDA version overriding the manifest's ``cuda_version``

        Returns:
            the priced job

        Raises:
            MarsTraceError: the manifest is wrong, or the service refused
        """
        manifest = JobManifest.load(directory)
        if hardware is not None:
            manifest.hardware = hardware
        if cuda_version is not None:
            JobManifest.check_cuda_version(
                value=cuda_version,
                where="the CUDA version",
            )
            manifest.cuda_version = cuda_version
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

    def check_admissible(self, quoted: QuotedJob) -> None:
        """Refuse, before asking the user, a job the estimate says the service would not admit.

        Args:
            quoted: the priced job

        Raises:
            MarsTraceError: the available balance is below the minimum a job needs, or the
                workspace's billing is not active; the message names the page where credit
                is added
        """
        if quoted.map_quote["admissible"]:
            return
        raise MarsTraceError(
            f"the service does not admit this job now ({quoted.summary()['available_usd']} available, "
            f"{MarsTraceWorkflow.usd(quoted.map_quote['minimum_micro_usd'])} needed to start): "
            f"{quoted.map_quote.get('message') or 'billing is not active'}; "
            f"add credit at {self._client.credit()['add_credit_url']}"
        )

    def submit(
        self,
        quoted: QuotedJob,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a priced job.

        Args:
            quoted: the priced job
            idempotency_key: the key that makes the submission happen once, or None for a
                fresh one

        Returns:
            the job

        Raises:
            ServiceError: the service refused the job; a refusal for credit names the page
                where credit is added
        """
        idempotency_key = idempotency_key or uuid.uuid4().hex
        map_submission = {"request": quoted.map_request}
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
        """Submit a recorded request under its key."""
        return self._client.submit(
            map_request=map_submission["request"],
            idempotency_key=idempotency_key,
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

    @staticmethod
    def is_transient(err: ServiceError) -> bool:
        """Whether a refusal is one the same read may get past later: no answer, a gateway, or
        ``service_unavailable``."""
        return (
            err.status is None
            or err.status in MarsTraceWorkflow.SET_TRANSIENT_STATUS
            or err.code == "service_unavailable"
        )

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
        failures = 0
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
            last_optimize = MarsTraceWorkflow._optimize_progress(map_job)
            try:
                map_job = self._client.wait_job(
                    job_id=job_id,
                    timeout_seconds=wait_seconds,
                )
            except ServiceError as err:
                failures += 1
                if not MarsTraceWorkflow.is_transient(err) or failures >= MarsTraceWorkflow.WAIT_FAILURES_MAX:
                    raise
                self._say(
                    f"{time.strftime('%H:%M:%S')} {job_id} not reachable ({err}); "
                    f"asking again in {MarsTraceWorkflow.WAIT_RETRY_SECONDS} s"
                )
                time.sleep(MarsTraceWorkflow.WAIT_RETRY_SECONDS)
                continue
            failures = 0
            if map_job["status"] != last_status:
                self._say(f"{time.strftime('%H:%M:%S')} {job_id} {map_job['status']}")
            map_optimize = MarsTraceWorkflow._optimize_progress(map_job)
            if (
                map_optimize is not None
                and map_optimize != last_optimize
            ):
                self._say(
                    f"{time.strftime('%H:%M:%S')} {job_id} optimize "
                    f"{MarsTraceWorkflow.optimize_progress_text(map_optimize)}"
                )
        return map_job

    @staticmethod
    def _optimize_progress(map_job: dict[str, Any]) -> dict[str, Any] | None:
        """The progress of a job's optimization stage, or None before its first checkpoint."""
        map_progress = map_job["progress"] or {}
        return map_progress.get("optimize")

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
