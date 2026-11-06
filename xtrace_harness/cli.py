import argparse
import json
import os
import shutil
import sys
import time
from typing import Any
from xtrace_harness.binary import DslCache, KernelBinary
from xtrace_harness.client import Credentials, MarsComputeClient
from xtrace_harness.error import HarnessError
from xtrace_harness.job import JobBundle, JobManifest
from xtrace_harness.listing import KernelListing
from xtrace_harness.viewer.cli.show import ShowCommand


class JobCommands:
    """The ``xtrace`` commands that call the platform: hardware, upload, submit, wait and fetch."""

    # the states after which a job no longer changes
    SET_FINAL_STATE = {
        "succeeded",
        "failed",
        "cancelled",
        "expired",
    }
    # seconds between two polls of a running job
    POLL_SECONDS = 5
    # the directory, under the job directory, that fetched results land in
    RESULTS_DIRECTORY = "xtrace-results"
    # micro-USD per USD
    MICRO_PER_UNIT = 1_000_000
    JSON_INDENT = 2
    JOBS_LIMIT_DEFAULT = 20

    @staticmethod
    def _client() -> MarsComputeClient:
        """A client with the credentials of the environment."""
        return MarsComputeClient(Credentials.from_environment())

    @staticmethod
    def _usd(micro_usd: Any) -> str:
        """An amount in micro-USD as dollars, or '-' when the platform gave none."""
        if not isinstance(
            micro_usd,  # obj
            int,        # class_or_tuple
        ):
            return "-"
        return f"${micro_usd / JobCommands.MICRO_PER_UNIT:.4f}"

    @staticmethod
    def _print_job(map_job: dict[str, Any]) -> None:
        """One job's state, timing, cost and files."""
        print(f"job       {map_job['id']}")
        print(f"state     {map_job['state']}")
        print(f"kind      {map_job.get('kind')}")
        print(f"hardware  {map_job.get('hardware')}")
        if map_job.get("gpu_seconds") is not None:
            print(f"gpu time  {map_job['gpu_seconds']} s")
        print(f"reserved  {JobCommands._usd(map_job.get('reserved_micro_usd'))}")
        print(f"charged   {JobCommands._usd(map_job.get('charge_micro_usd'))}")
        map_error = map_job.get("error")
        if map_error:
            print(f"error     {map_error.get('code')}: {map_error.get('message')}")
        for map_file in map_job.get("files") or []:
            print(f"file      {map_file['name']} ({map_file.get('size_bytes')} bytes)")

    @staticmethod
    def hardware(args: argparse.Namespace) -> None:
        """List the hardware classes and their prices."""
        list_hardware = JobCommands._client().list_hardware()
        if args.json:
            print(
                json.dumps(
                    list_hardware,
                    indent=JobCommands.JSON_INDENT,
                )
            )
            return
        for map_hardware in list_hardware:
            available = "available" if map_hardware.get("available") else "unavailable"
            print(
                f"{map_hardware['id']:<16} {map_hardware.get('gpu', ''):<24} "
                f"{map_hardware.get('arch', ''):<8} "
                f"{JobCommands._usd(map_hardware.get('gpu_second_micro_usd'))}/GPU-s "
                f"+ {JobCommands._usd(map_hardware.get('job_micro_usd'))}/job  {available}"
            )

    @staticmethod
    def balance(args: argparse.Namespace) -> None:
        """Print the credit available to new jobs."""
        map_balance = JobCommands._client().get_balance()
        print(f"available {JobCommands._usd(map_balance.get('available_micro_usd'))}")

    @staticmethod
    def _manifest(args: argparse.Namespace) -> JobManifest:
        """The job directory's manifest, with the command line's kind applied."""
        return JobManifest.load(
            directory=args.directory,
            kind=args.kind,
        )

    @staticmethod
    def _pack(manifest: JobManifest) -> JobBundle:
        """Pack the files a manifest sends and report the bundle's size."""
        bundle = JobBundle.pack(manifest)
        print(
            f"bundle    {len(bundle.list_file)} file(s), {len(bundle.data)} bytes, "
            f"sha256 {bundle.sha256[:12]}",
            file=sys.stderr,
        )
        return bundle

    @staticmethod
    def _hardware(
        client: MarsComputeClient,
        manifest: JobManifest,
        hardware: str | None,
    ) -> str:
        """The job's hardware class, checked to exist and to run the job's binary."""
        if not hardware:
            raise HarnessError("name the hardware with --hardware or 'hardware' in xtrace.toml")
        for map_hardware in client.list_hardware():
            if map_hardware["id"] != hardware:
                continue
            arch = map_hardware.get("arch", "")
            if not manifest.binary.runs_on(arch):
                raise HarnessError(
                    f"{manifest.spec.kernel.binary} targets {manifest.binary.target()}, but "
                    f"{hardware} is {arch}; rebuild it for {arch}"
                )
            return hardware
        raise HarnessError(f"no hardware class '{hardware}'; run 'xtrace hardware' to list them")

    @staticmethod
    def upload(args: argparse.Namespace) -> None:
        """Pack and upload a job directory; print the upload id and the specification to submit."""
        manifest = JobCommands._manifest(args)
        bundle = JobCommands._pack(manifest)
        client = JobCommands._client()
        hardware = JobCommands._hardware(
            client=client,
            manifest=manifest,
            hardware=args.hardware or manifest.hardware,
        )
        upload_id = client.upload(bundle)
        print(
            json.dumps(
                {
                    "upload_id": upload_id,
                    "hardware": hardware,
                    "spec": manifest.spec.to_json(),
                },
                indent=JobCommands.JSON_INDENT,
            )
        )

    @staticmethod
    def submit(args: argparse.Namespace) -> None:
        """Pack, upload and submit a job directory, then optionally wait and fetch its results."""
        manifest = JobCommands._manifest(args)
        bundle = JobCommands._pack(manifest)
        client = JobCommands._client()
        hardware = JobCommands._hardware(
            client=client,
            manifest=manifest,
            hardware=args.hardware or manifest.hardware,
        )
        upload_id = client.upload(bundle)
        map_job = client.submit(
            upload_id=upload_id,
            hardware=hardware,
            spec=manifest.spec,
        )
        JobCommands._print_job(map_job)
        if args.wait:
            JobCommands._wait_and_fetch(
                client=client,
                job_id=map_job["id"],
                output_directory=os.path.join(
                    args.directory,
                    JobCommands.RESULTS_DIRECTORY,
                    map_job["id"],
                ),
            )

    @staticmethod
    def _wait_and_fetch(
        client: MarsComputeClient,
        job_id: str,
        output_directory: str,
    ) -> None:
        """Poll a job until it ends, then fetch its files."""
        last_state = None
        while True:
            map_job = client.get_job(job_id)
            if map_job["state"] != last_state:
                last_state = map_job["state"]
                print(
                    f"{time.strftime('%H:%M:%S')} {job_id} {last_state}",
                    file=sys.stderr,
                )
            if map_job["state"] in JobCommands.SET_FINAL_STATE:
                break
            time.sleep(JobCommands.POLL_SECONDS)
        JobCommands._print_job(map_job)
        JobCommands._fetch_files(
            client=client,
            map_job=map_job,
            output_directory=output_directory,
        )

    @staticmethod
    def _fetch_files(
        client: MarsComputeClient,
        map_job: dict[str, Any],
        output_directory: str,
    ) -> None:
        """Download every result file of a job into a directory."""
        list_file = map_job.get("files") or []
        if not list_file:
            return
        os.makedirs(
            name=output_directory,
            exist_ok=True,
        )
        for map_file in list_file:
            name = os.path.basename(map_file["name"])
            path = os.path.join(
                output_directory,
                name,
            )
            with open(
                file=path,
                mode="wb",
            ) as result_file:
                result_file.write(
                    client.download(map_file)
                )
            print(f"saved     {path}")

    @staticmethod
    def status(args: argparse.Namespace) -> None:
        """Print one job, or wait for it to end and fetch its files."""
        client = JobCommands._client()
        if args.wait:
            JobCommands._wait_and_fetch(
                client=client,
                job_id=args.job,
                output_directory=os.path.join(
                    args.output,
                    args.job,
                ),
            )
            return
        map_job = client.get_job(args.job)
        if args.json:
            print(
                json.dumps(
                    map_job,
                    indent=JobCommands.JSON_INDENT,
                )
            )
            return
        JobCommands._print_job(map_job)

    @staticmethod
    def jobs(args: argparse.Namespace) -> None:
        """List the account's latest jobs."""
        for map_job in JobCommands._client().list_jobs(args.limit):
            print(
                f"{map_job['id']:<28} {map_job['state']:<11} {map_job.get('kind', ''):<8} "
                f"{map_job.get('hardware', ''):<14} {map_job.get('created_at', '')}"
            )

    @staticmethod
    def cancel(args: argparse.Namespace) -> None:
        """Cancel a job."""
        JobCommands._print_job(JobCommands._client().cancel(args.job))

    @staticmethod
    def fetch(args: argparse.Namespace) -> None:
        """Download a finished job's files."""
        client = JobCommands._client()
        JobCommands._fetch_files(
            client=client,
            map_job=client.get_job(args.job),
            output_directory=os.path.join(
                args.output,
                args.job,
            ),
        )


class ListingCommand:
    """``xtrace sass``: browse the SASS listing an ``inspect`` job returned, to pick trace sites."""

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        """Print the binary's kernels, or the named kernel's narrowed listing."""
        listing = KernelListing.load(args.listing)
        if args.kernels:
            print(listing.render_kernels())
            return
        print(
            listing.render_listing(
                source=args.line,
                opcode_pattern=args.grep,
                first_index=args.first,
                last_index=args.last,
            )
        )


class BinaryCommands:
    """``xtrace cubin``: find a kernel's cubin in compiler caches, and read what a binary holds."""

    @staticmethod
    def find(args: argparse.Namespace) -> None:
        """List the cached kernels whose name matches, newest first, and copy one into a job."""
        list_root = DslCache.default_roots() + [
            (
                "directory",
                directory,
            )
            for directory in args.dir
        ]
        if not list_root:
            raise HarnessError("no compiler cache found; name a directory of cubins with --dir")
        list_found = DslCache.find(
            pattern=args.pattern,
            list_root=list_root,
        )
        if not list_found:
            raise HarnessError(
                f"no cached kernel matches '{args.pattern}' under "
                f"{', '.join(directory for _, directory in list_root)}"
            )
        for found_id, cached in enumerate(list_found):
            print(f"[{found_id}] {cached.name}")
            print(f"    {cached.origin}, {cached.binary.target()}, {cached.binary.path}")
            for key, value in cached.map_hint.items():
                print(f"    {key}: {value}")
        if args.copy_to is None:
            return
        cached = list_found[args.pick]
        os.makedirs(
            name=args.copy_to,
            exist_ok=True,
        )
        target = os.path.join(
            args.copy_to,
            os.path.basename(cached.binary.path),
        )
        shutil.copyfile(
            src=cached.binary.path,
            dst=target,
        )
        print(f"copied [{args.pick}] to {target}; set [kernel] binary and name = \"{cached.name}\"")

    @staticmethod
    def show(args: argparse.Namespace) -> None:
        """Print a binary's format, target and kernels."""
        binary = KernelBinary.load(args.binary)
        print(f"{binary.format}, {binary.target()}")
        for kernel_name in binary.list_kernel:
            print(f"  {kernel_name}")


class CommandLine:
    """The ``xtrace`` command line."""

    EXIT_FAILURE = 1

    @staticmethod
    def _add_job_directory(parser: argparse.ArgumentParser) -> None:
        """The options of a command that reads a job directory."""
        parser.add_argument(
            "directory",
            help="the job directory, holding xtrace.toml and the files it names",
        )
        parser.add_argument(
            "--kind",
            choices=[
                "inspect",
                "trace",
            ],
            default=None,
            help="override the manifest's kind",
        )
        parser.add_argument(
            "--hardware",
            default=None,
            help="override the manifest's hardware class",
        )

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """The parser of every ``xtrace`` command.

        Returns:
            the parser
        """
        parser = argparse.ArgumentParser(
            prog="xtrace",
            description="Trace CUDA kernels on MarsCompute GPUs and read the traces.",
        )
        subparsers = parser.add_subparsers()

        hardware = subparsers.add_parser(
            name="hardware",
            help="list the GPU classes a job can run on, with their prices",
        )
        hardware.add_argument(
            "--json",
            action="store_true",
        )
        hardware.set_defaults(func=JobCommands.hardware)

        cubin = subparsers.add_parser(
            name="cubin",
            help="find a kernel's cubin in compiler caches, or show what a binary holds",
        )
        cubin_commands = cubin.add_subparsers()
        cubin_find = cubin_commands.add_parser(
            name="find",
            help="search the Triton, torch.compile and TileLang caches and --dir for a kernel",
        )
        cubin_find.add_argument(
            "pattern",
            help="a regex searched in the kernel names",
        )
        cubin_find.add_argument(
            "--dir",
            action="append",
            default=[],
            help="another directory of cubins to search, e.g. a CuTeDSL dump directory",
        )
        cubin_find.add_argument(
            "--copy-to",
            default=None,
            help="copy the picked cubin into this job directory",
        )
        cubin_find.add_argument(
            "--pick",
            type=int,
            default=0,
            help="the match --copy-to copies, by its number in the list",
        )
        cubin_find.set_defaults(func=BinaryCommands.find)
        cubin_show = cubin_commands.add_parser(
            name="show",
            help="print a cubin's or PTX file's target and kernels",
        )
        cubin_show.add_argument("binary")
        cubin_show.set_defaults(func=BinaryCommands.show)

        balance = subparsers.add_parser(
            name="balance",
            help="show the credit available to new jobs",
        )
        balance.set_defaults(func=JobCommands.balance)

        upload = subparsers.add_parser(
            name="upload",
            help="pack and upload a job directory; print what to pass to xtrace_submit_job",
        )
        CommandLine._add_job_directory(upload)
        upload.set_defaults(func=JobCommands.upload)

        submit = subparsers.add_parser(
            name="submit",
            help="pack, upload and submit a job directory",
        )
        CommandLine._add_job_directory(submit)
        submit.add_argument(
            "--wait",
            action="store_true",
            help="wait for the job to end and fetch its files into <directory>/xtrace-results",
        )
        submit.set_defaults(func=JobCommands.submit)

        status = subparsers.add_parser(
            name="status",
            help="show a job's state, cost and files",
        )
        status.add_argument("job")
        status.add_argument(
            "--wait",
            action="store_true",
            help="wait for the job to end and fetch its files",
        )
        status.add_argument(
            "--output",
            default=JobCommands.RESULTS_DIRECTORY,
            help="the directory fetched files land in, under <output>/<job>",
        )
        status.add_argument(
            "--json",
            action="store_true",
        )
        status.set_defaults(func=JobCommands.status)

        jobs = subparsers.add_parser(
            name="jobs",
            help="list the account's latest jobs",
        )
        jobs.add_argument(
            "--limit",
            type=int,
            default=JobCommands.JOBS_LIMIT_DEFAULT,
        )
        jobs.set_defaults(func=JobCommands.jobs)

        cancel = subparsers.add_parser(
            name="cancel",
            help="cancel a job that has not ended",
        )
        cancel.add_argument("job")
        cancel.set_defaults(func=JobCommands.cancel)

        fetch = subparsers.add_parser(
            name="fetch",
            help="download a finished job's files",
        )
        fetch.add_argument("job")
        fetch.add_argument(
            "--output",
            default=JobCommands.RESULTS_DIRECTORY,
            help="the directory files land in, under <output>/<job>",
        )
        fetch.set_defaults(func=JobCommands.fetch)

        sass = subparsers.add_parser(
            name="sass",
            help="browse the SASS listing an inspect job returned",
        )
        sass.add_argument(
            "listing",
            help="the inspect.json of an inspect job",
        )
        sass.add_argument(
            "--kernels",
            action="store_true",
            help="list the binary's kernels and the named one's parameters instead of the listing",
        )
        sass.add_argument(
            "--line",
            default=None,
            help="keep instructions of a source line, <file>:<line> or <line>",
        )
        sass.add_argument(
            "--grep",
            default=None,
            help="keep instructions whose text matches a regex, e.g. 'HGMMA|LDGSTS'",
        )
        sass.add_argument(
            "--first",
            type=int,
            default=None,
            help="keep instructions from this index on",
        )
        sass.add_argument(
            "--last",
            type=int,
            default=None,
            help="keep instructions up to this index",
        )
        sass.set_defaults(func=ListingCommand.run)

        ShowCommand().register(subparsers)
        return parser

    @staticmethod
    def run(list_argument: list[str] | None = None) -> int:
        """Parse the command line and run the command it names.

        Args:
            list_argument: the arguments, or None for the process's own

        Returns:
            the process exit status
        """
        parser = CommandLine.build_parser()
        args = parser.parse_args(list_argument)
        if not hasattr(
            args,    # obj
            "func",  # name
        ):
            parser.print_help()
            return CommandLine.EXIT_FAILURE
        try:
            args.func(args)
        except HarnessError as err:
            print(
                f"xtrace: {err}",
                file=sys.stderr,
            )
            return CommandLine.EXIT_FAILURE
        return 0


def main() -> None:
    """Entry point of the ``xtrace`` command."""
    sys.exit(CommandLine.run())


if __name__ == "__main__":
    main()
