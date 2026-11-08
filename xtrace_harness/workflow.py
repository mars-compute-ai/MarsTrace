import decimal
import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from xtrace_harness.binary import KernelBinary
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import HarnessError, PlatformError
from xtrace_harness.job import JobArtifact, JobManifest
from xtrace_harness.listing import InspectionCache, KernelListing


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
            the quote id, the hardware, the cost cap, the GPU rate, the limits and the expiry
        """
        return {
            "quote_id": self.map_quote["quote_id"],
            "hardware": self.manifest.hardware,
            "architecture": self.architecture,
            "max_cost_usd": XtraceWorkflow.usd(self.map_quote["limits"]["max_total_cost_micro_usd"]),
            "gpu_usd_per_hour": XtraceWorkflow.usd(
                self.map_quote["rate_snapshot"]["gpu_micro_usd_per_hour"]
            ),
            "limits": self.map_quote["limits"],
            "expires_at": self.map_quote.get("expires_at"),
        }


@dataclass
class QuotedInspection:
    """A cubin uploaded and its inspection quoted, ready to accept.

    Args:
        binary_path: the cubin
        kernel_symbol: the kernel to list, or None for the kernel catalog only
        map_request: the inspection request the quote was made for
        map_quote: the quote: the profile, the limits and the fixed fee
    """

    binary_path: str
    kernel_symbol: str | None
    map_request: dict[str, Any]
    map_quote: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """What the quote commits to, in dollars.

        Returns:
            the quote id, the profile, the fee and the expiry
        """
        return {
            "quote_id": self.map_quote["quote_id"],
            "kernel": self.kernel_symbol,
            "profile_id": self.map_quote["profile"]["profile_id"],
            "fee_usd": XtraceWorkflow.usd(self.map_quote["price"]["fixed_fee_micro_usd"]),
            "expires_at": self.map_quote["expires_at"],
        }


class XtraceWorkflow:
    """The operations the ``xtrace`` command and its MCP server offer, on local paths.

    Args:
        client: the Control Plane client
        progress: called with a line of progress, or None to stay silent
        poll_seconds: seconds between polls of an inspection, and the floor between polls of a job
    """

    SET_JOB_FINAL_STATE = {
        "succeeded",
        "failed",
        "cancelled",
    }
    INSPECTION_FINAL_STATE = "terminal"
    INSPECTION_COMPLETED = "completed"
    INSPECTION_REQUEST_SCHEMA = "marscompute.xtrace.inspection-request.v1"
    INSPECTION_AVAILABLE = "available"
    INSPECTIONS_DIRECTORY = "xtrace-inspections"
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
        """The hardware, stages, rates, architectures, models and limits open to the account.

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

    def inspection_capabilities(self) -> dict[str, Any]:
        """The inspection profiles open to the account.

        Returns:
            the capabilities
        """
        return self._client.inspection_capabilities()

    def quote_inspection(
        self,
        binary_path: str,
        kernel_symbol: str | None,
        profile_id: str | None = None,
    ) -> QuotedInspection:
        """Upload a cubin and quote its inspection.

        Args:
            binary_path: the cubin
            kernel_symbol: the kernel to list, or None for the kernel catalog only
            profile_id: the inspection profile, or None for the first one offered

        Returns:
            the quoted inspection

        Raises:
            HarnessError: the cubin lacks the kernel, no profile fits, or the platform refused
        """
        binary = KernelBinary.load(binary_path)
        if (
            kernel_symbol is not None
            and kernel_symbol not in binary.list_kernel
        ):
            raise HarnessError(
                f"{binary_path} defines no kernel '{kernel_symbol}'; it defines: "
                f"{', '.join(binary.list_kernel) or 'none'}"
            )
        map_availability = self._client.inspection_capabilities()["availability"]
        if map_availability["state"] != XtraceWorkflow.INSPECTION_AVAILABLE:
            raise HarnessError(f"inspections are unavailable: {map_availability.get('reason')}")
        list_profile = [
            map_profile
            for map_profile in map_availability["profiles"]
            if profile_id in (None, map_profile["profile"]["profile_id"])
        ]
        if not list_profile:
            raise HarnessError(
                f"no inspection profile '{profile_id}'; offered: "
                f"{', '.join(map_profile['profile']['profile_id'] for map_profile in map_availability['profiles'])}"
            )
        map_artifact = self._upload_file(
            artifact=JobArtifact(
                kind="cubin",
                path=binary_path,
            ),
            upload_bytes_max=list_profile[0]["limits"]["max_input_bytes"],
        )
        map_request = {
            "schema_version": XtraceWorkflow.INSPECTION_REQUEST_SCHEMA,
            "binary": {
                "artifact_id": map_artifact["artifact_id"],
                "sha256": map_artifact["sha256"],
                "size_bytes": map_artifact["size_bytes"],
                "format": "cubin",
            },
            "kernel_selector": (
                {"kind": "all"}
                if kernel_symbol is None
                else {
                    "kind": "symbol",
                    "symbol": kernel_symbol,
                }
            ),
            "inspection_profile_id": list_profile[0]["profile"]["profile_id"],
        }
        return QuotedInspection(
            binary_path=binary_path,
            kernel_symbol=kernel_symbol,
            map_request=map_request,
            map_quote=self._client.inspection_estimate(map_request),
        )

    def submit_inspection(
        self,
        quoted: QuotedInspection,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Accept a quoted inspection.

        Args:
            quoted: the quoted inspection
            idempotency_key: the key that makes the acceptance happen once, or None for a fresh one

        Returns:
            the inspection's status
        """
        map_status = self._client.inspection_submit(
            map_request=quoted.map_request,
            map_quote=quoted.map_quote,
            idempotency_key=idempotency_key or str(uuid.uuid4()),
        )
        self._say(f"inspect   {map_status['inspection_id']} {map_status['execution']['state']}")
        return map_status

    def inspections(
        self,
        limit: int,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """A page of the account's inspections.

        Args:
            limit: how many to list
            cursor: the previous page's ``next_cursor``, or None

        Returns:
            ``items`` and ``next_cursor``
        """
        return self._client.list_inspections(
            limit=limit,
            cursor=cursor,
        )

    def inspection(self, inspection_id: str) -> dict[str, Any]:
        """One inspection's status.

        Args:
            inspection_id: the inspection

        Returns:
            its execution, cleanup and accounting
        """
        return self._client.get_inspection(inspection_id)

    def cancel_inspection(self, inspection_id: str) -> dict[str, Any]:
        """Ask an inspection to stop.

        Args:
            inspection_id: the inspection

        Returns:
            its status
        """
        return self._client.cancel_inspection(inspection_id)

    def wait_inspection(
        self,
        inspection_id: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Poll an inspection until it ends or the time is up.

        Args:
            inspection_id: the inspection
            timeout_seconds: how long to wait, or None to wait until it ends

        Returns:
            the last status
        """
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        last_state = None
        while True:
            map_status = self._client.get_inspection(inspection_id)
            state = map_status["execution"]["state"]
            if state != last_state:
                self._say(f"{time.strftime('%H:%M:%S')} {inspection_id} {state}")
                last_state = state
            if state == XtraceWorkflow.INSPECTION_FINAL_STATE:
                return map_status
            if (
                deadline is not None
                and time.monotonic() + self._poll_seconds > deadline
            ):
                return map_status
            time.sleep(self._poll_seconds)

    def download_inspection(
        self,
        inspection_id: str,
        list_artifact_id: list[str],
        output_directory: str,
    ) -> list[str]:
        """Download artifacts an inspection result names.

        Args:
            inspection_id: the inspection
            list_artifact_id: the artifacts
            output_directory: the directory the files land in, by artifact id

        Returns:
            the paths written
        """
        os.makedirs(
            name=output_directory,
            exist_ok=True,
        )
        list_path = []
        for map_grant in self._client.inspection_downloads(
            inspection_id=inspection_id,
            list_artifact_id=list_artifact_id,
        ):
            path = os.path.join(
                output_directory,
                MarsComputeClient.artifact_name(f"{map_grant['artifact_id']}.json"),
            )
            with open(
                file=path,
                mode="wb",
            ) as artifact_file:
                artifact_file.write(self._client.download(map_grant))
            list_path.append(path)
        return list_path

    def fetch_inspection(
        self,
        inspection_id: str,
        binary_path: str,
        kernel_symbol: str | None,
    ) -> str:
        """Fetch a completed inspection of a cubin and cache it beside the cubin, with the
        instruction listing of its kernel.

        Args:
            inspection_id: the inspection
            binary_path: the cubin it inspected
            kernel_symbol: the kernel whose listing to download, or None for the catalog only

        Returns:
            the path of the cached result

        Raises:
            HarnessError: the inspection did not complete, describes another cubin or lacks the kernel
        """
        binary = KernelBinary.load(binary_path)
        map_status = self._client.get_inspection(inspection_id)
        map_execution = map_status["execution"]
        if map_execution["state"] != XtraceWorkflow.INSPECTION_FINAL_STATE:
            raise HarnessError(f"inspection {inspection_id} is still {map_execution['state']}; wait for it")
        if map_execution["outcome"] != XtraceWorkflow.INSPECTION_COMPLETED:
            raise HarnessError(f"inspection {inspection_id} ended {map_execution['outcome']}")
        map_result = self._client.inspection_result(inspection_id)
        listing = KernelListing.from_documents(
            map_result=map_result,
            map_instructions=None,
            where=f"inspection {inspection_id}",
        )
        if listing.binary_sha256 != binary.sha256:
            raise HarnessError(f"inspection {inspection_id} describes another cubin than {binary_path}")
        instructions_data = None
        if kernel_symbol is not None:
            kernel = listing.map_kernel.get(kernel_symbol)
            if kernel is None:
                raise HarnessError(f"inspection {inspection_id} lists no kernel '{kernel_symbol}'")
            map_artifact = kernel.map_instructions_artifact
            (map_grant,) = self._client.inspection_downloads(
                inspection_id=inspection_id,
                list_artifact_id=[map_artifact["artifact_id"]],
            )
            instructions_data = self._client.download(map_grant)
            if (
                len(instructions_data) != map_artifact["size_bytes"]
                or hashlib.sha256(instructions_data).hexdigest() != map_artifact["sha256"]
            ):
                raise PlatformError(f"the listing of {kernel_symbol} differs from its inspection result")
            try:
                map_instructions = json.loads(instructions_data)
            except ValueError as err:
                raise HarnessError(f"inspection {inspection_id} returned a listing that is not JSON") from err
            KernelListing.from_documents(
                map_result=map_result,
                map_instructions=map_instructions,
                where=f"inspection {inspection_id}",
            )
        return InspectionCache.store(
            binary_path=binary_path,
            sha256=binary.sha256,
            kernel_symbol=kernel_symbol,
            result_data=json.dumps(
                map_result,
                indent=XtraceWorkflow.JSON_INDENT,
            ).encode("utf-8"),
            instructions_data=instructions_data,
        )

    def quote(
        self,
        directory: str,
        hardware: str | None = None,
    ) -> QuotedJob:
        """Translate a job directory, check its sites against a cached listing, upload it and quote it.

        Args:
            directory: the job directory, holding ``xtrace.toml``
            hardware: an environment id overriding the manifest's ``hardware``

        Returns:
            the quoted job

        Raises:
            HarnessError: the manifest is wrong, a site is not in the listing, or the platform refused
        """
        manifest = JobManifest.load(directory)
        if hardware is not None:
            manifest.hardware = hardware
        map_capabilities = self._client.capabilities()
        map_entry = manifest.select_compute(map_capabilities)
        upload_bytes_max = map_capabilities.get("max_upload_bytes")
        listing = InspectionCache.load(
            binary_path=os.path.join(
                directory,
                manifest.kernel.binary,
            ),
            sha256=manifest.binary.sha256,
            kernel_symbol=manifest.kernel.name,
        )
        if listing is not None:
            manifest.check_parameters(listing)
        manifest.analysis(listing)
        launch_manifest = manifest.launch_manifest_bytes(
            architecture=map_entry["architecture"],
            listing=listing,
        )
        map_artifact_id = {}
        for artifact in manifest.list_artifact():
            map_artifact_id[artifact.kind] = self._upload_file(
                artifact=artifact,
                upload_bytes_max=upload_bytes_max,
            )["artifact_id"]
        self._say(f"upload    launch_manifest ({len(launch_manifest)} bytes)")
        map_artifact_id["launch_manifest"] = self._client.upload(
            data=launch_manifest,
            kind="launch_manifest",
            name=JobManifest.MANIFEST_NAME,
            content_type="application/json",
        )["artifact_id"]
        map_request = manifest.estimate_input(
            map_capabilities=map_capabilities,
            map_artifact_id=map_artifact_id,
            listing=listing,
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
