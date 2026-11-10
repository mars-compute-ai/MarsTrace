import argparse
import json
import os
import sys
from typing import Any
from xtrace_harness.binary import DslCache, KernelBinary
from xtrace_harness.error import HarnessError
from xtrace_harness.mcp import McpServer
from xtrace_harness.viewer.cli.show import ShowCommand
from xtrace_harness.workflow import QuotedJob, XtraceWorkflow


class JobCommands:
    """The ``xtrace`` commands that call the Control Plane."""

    JSON_INDENT = 2
    JOBS_LIMIT_DEFAULT = 20
    SET_YES = {
        "y",
        "yes",
    }

    @staticmethod
    def _workflow() -> XtraceWorkflow:
        """A workflow with the environment's credentials, reporting progress on stderr."""
        return XtraceWorkflow.from_environment(
            progress=lambda line: print(
                line,
                file=sys.stderr,
            ),
        )

    @staticmethod
    def _print_json(value: Any) -> None:
        """Print a value as indented JSON."""
        print(
            json.dumps(
                value,
                indent=JobCommands.JSON_INDENT,
            )
        )

    @staticmethod
    def print_job(map_job: dict[str, Any]) -> None:
        """One job's state, hardware, cost and result quality.

        Args:
            map_job: the job
        """
        map_compute = map_job.get("compute") or {}
        map_accounting = map_job.get("accounting") or {}
        print(f"job       {map_job['job_id']}")
        print(f"state     {map_job['state']}")
        print(f"hardware  {map_compute.get('gpu_type')} x{map_compute.get('gpu_count')}")
        print(f"created   {map_job.get('created_at')}")
        print(f"reserved  {XtraceWorkflow.usd(map_accounting.get('reserved_micro_usd'))}")
        print(f"settled   {XtraceWorkflow.usd(map_accounting.get('settled_micro_usd'))}")
        map_error = map_job.get("error")
        if map_error:
            print(f"error     {map_error.get('code')}: {map_error.get('message')}")
        map_manifest = map_job.get("result_manifest")
        if map_manifest:
            print(f"quality   {map_manifest.get('quality')} {', '.join(map_manifest.get('reason_codes', []))}")

    @staticmethod
    def hardware(args: argparse.Namespace) -> None:
        """List the hardware a job can run on, with rates and architectures, and the agent models."""
        map_capabilities = JobCommands._workflow().hardware()
        if args.json:
            JobCommands._print_json(map_capabilities)
            return
        if not map_capabilities.get("enabled"):
            print("Xtrace is not enabled for this account")
        for map_entry in map_capabilities.get("compute", []):
            map_compute = map_entry.get("compute", {})
            print(
                f"{map_entry['environment_id']:<20} {map_compute.get('gpu_type', ''):<12} "
                f"x{map_compute.get('gpu_count', '')} {map_entry.get('architecture', ''):<8} "
                f"{XtraceWorkflow.usd(map_entry.get('gpu_micro_usd_per_hour'))}/GPU-hour  "
                f"{map_compute.get('purchase_option', '')} {map_compute.get('region', '')}"
            )
        if map_capabilities.get("agent_models"):
            print(f"agent models: {', '.join(map_capabilities['agent_models'])}")

    @staticmethod
    def credit(args: argparse.Namespace) -> None:
        """Print the tenant's credit."""
        map_credit = JobCommands._workflow().credit()
        if args.json:
            JobCommands._print_json(map_credit)
            return
        print(f"available {XtraceWorkflow.usd(map_credit.get('available_micro_usd'))}")
        print(f"reserved  {XtraceWorkflow.usd(map_credit.get('reserved_micro_usd'))}")
        print(f"top up    {map_credit.get('top_up_url')}")

    @staticmethod
    def _confirm(
        args: argparse.Namespace,
        prompt: str,
    ) -> None:
        """Go on only when ``--yes`` was given or the user at the terminal agrees."""
        if args.yes:
            return
        if not sys.stdin.isatty():
            raise HarnessError("confirm the cost above: run again with --yes")
        if input(f"{prompt} [y/N] ").strip().lower() not in JobCommands.SET_YES:
            raise HarnessError("not submitted")

    @staticmethod
    def _print_quote(quoted: QuotedJob) -> None:
        """What a quote commits to."""
        map_summary = quoted.summary()
        print(f"quote     {map_summary['quote_id']} (expires {map_summary['expires_at']})")
        print(f"hardware  {map_summary['hardware']} ({map_summary['architecture']})")
        print(f"rate      {map_summary['gpu_usd_per_hour']} per GPU-hour")
        if "job_fee_usd" in map_summary:
            print(f"job fee   {map_summary['job_fee_usd']}")
        map_agent_rate = map_summary.get("agent_usd_per_million_tokens")
        if map_agent_rate is not None:
            print(
                f"tokens    {map_agent_rate['model']}: {map_agent_rate['input']} input, "
                f"{map_agent_rate['cached_input']} cached input, {map_agent_rate['output']} output "
                "per million"
            )
            for dimension, rate in map_agent_rate["map_cache_write"].items():
                print(f"          {rate} per million {dimension}")
        print(f"cost cap  {map_summary['max_cost_usd']} at most")

    @staticmethod
    def estimate(args: argparse.Namespace) -> None:
        """Upload and quote a job directory, then print the cost cap and the rate."""
        quoted = JobCommands._workflow().quote(
            directory=args.directory,
            hardware=args.hardware,
        )
        if args.json:
            JobCommands._print_json(quoted.summary())
            return
        JobCommands._print_quote(quoted)

    @staticmethod
    def submit(args: argparse.Namespace) -> None:
        """Upload, quote, confirm the cost and submit a job directory."""
        workflow = JobCommands._workflow()
        quoted = workflow.quote(
            directory=args.directory,
            hardware=args.hardware,
        )
        JobCommands._print_quote(quoted)
        JobCommands._confirm(
            args=args,
            prompt="submit?",
        )
        map_job = workflow.submit(
            quoted=quoted,
            idempotency_key=args.idempotency_key,
        )
        JobCommands.print_job(map_job)
        if not args.wait:
            return
        map_view = workflow.wait(map_job["job_id"])
        JobCommands.print_job(map_view["job"])
        if map_view["job"].get("result_manifest"):
            for path in workflow.fetch(
                job_id=map_job["job_id"],
                output_directory=os.path.join(
                    args.directory,
                    XtraceWorkflow.RESULTS_DIRECTORY,
                    map_job["job_id"],
                ),
            ):
                print(f"saved     {path}")

    @staticmethod
    def jobs(args: argparse.Namespace) -> None:
        """List a page of the account's jobs."""
        map_page = JobCommands._workflow().jobs(
            limit=args.limit,
            cursor=args.cursor,
        )
        if args.json:
            JobCommands._print_json(map_page)
            return
        for map_job in map_page.get("items", []):
            print(
                f"{map_job['job_id']:<38} {map_job['state']:<16} "
                f"{(map_job.get('compute') or {}).get('gpu_type', ''):<10} {map_job.get('created_at', '')}"
            )
        if map_page.get("next_cursor"):
            print(f"more: --cursor {map_page['next_cursor']}")

    @staticmethod
    def job(args: argparse.Namespace) -> None:
        """Print one job, or wait for it to end."""
        workflow = JobCommands._workflow()
        map_view = workflow.wait(args.job) if args.wait else workflow.job(args.job)
        if args.json:
            JobCommands._print_json(map_view)
            return
        JobCommands.print_job(map_view["job"])

    @staticmethod
    def cancel(args: argparse.Namespace) -> None:
        """Ask a job to stop."""
        JobCommands.print_job(JobCommands._workflow().cancel(args.job))

    @staticmethod
    def fetch(args: argparse.Namespace) -> None:
        """Download a finished job's files."""
        for path in JobCommands._workflow().fetch(
            job_id=args.job,
            output_directory=args.output or os.path.join(
                XtraceWorkflow.RESULTS_DIRECTORY,
                args.job,
            ),
        ):
            print(f"saved     {path}")

    @staticmethod
    def mcp(args: argparse.Namespace) -> None:
        """Serve the operations as MCP tools over stdio."""
        McpServer(XtraceWorkflow.from_environment()).serve(
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )


class BinaryCommands:
    """``xtrace cubin``: find a kernel's cubin in compiler caches, and read what a cubin holds."""

    @staticmethod
    def find(args: argparse.Namespace) -> None:
        """List the cached kernels whose name matches, newest first, and copy one into a job."""
        map_found = DslCache.search(
            pattern=args.pattern,
            list_directory=args.dir,
            copy_to=args.copy_to,
            pick=args.pick,
        )
        for found_id, map_kernel in enumerate(map_found["kernels"]):
            print(f"[{found_id}] {map_kernel['name']}")
            print(f"    {map_kernel['origin']}, {map_kernel['target']}, {map_kernel['path']}")
            for key, value in map_kernel["hints"].items():
                print(f"    {key}: {value}")
        if map_found["copied_to"] is not None:
            print(
                f"copied [{args.pick}] to {map_found['copied_to']}; add it to [workspace] cubins"
            )

    @staticmethod
    def show(args: argparse.Namespace) -> None:
        """Print a cubin's target and kernels."""
        binary = KernelBinary.load(args.binary)
        print(f"cubin, {binary.target()}, sha256 {binary.sha256}")
        for kernel_name in binary.list_kernel:
            print(f"  {kernel_name}")


class CommandLine:
    """The ``xtrace`` command line."""

    EXIT_FAILURE = 1

    @staticmethod
    def _add_json(parser: argparse.ArgumentParser) -> None:
        """The option that prints the platform's answer as JSON."""
        parser.add_argument(
            "--json",
            action="store_true",
            help="print the platform's answer as JSON",
        )

    @staticmethod
    def _add_job_directory(parser: argparse.ArgumentParser) -> None:
        """The options of a command that reads a job directory."""
        parser.add_argument(
            "directory",
            help="the job directory, holding xtrace.toml and the files it names",
        )
        parser.add_argument(
            "--hardware",
            default=None,
            help="override the manifest's hardware, an environment id from 'xtrace hardware'",
        )

    @staticmethod
    def _add_platform_commands(subparsers: argparse._SubParsersAction) -> None:
        """The commands that read the account and run jobs."""
        hardware = subparsers.add_parser(
            name="hardware",
            help="list the hardware a job can run on, with rates and architectures, and the agent models",
        )
        CommandLine._add_json(hardware)
        hardware.set_defaults(func=JobCommands.hardware)

        credit = subparsers.add_parser(
            name="credit",
            help="show the credit available to jobs, and where to add more",
        )
        CommandLine._add_json(credit)
        credit.set_defaults(func=JobCommands.credit)

        estimate = subparsers.add_parser(
            name="estimate",
            help="upload and quote a job directory; print the cost cap, the rates and the job fee",
        )
        CommandLine._add_job_directory(estimate)
        CommandLine._add_json(estimate)
        estimate.set_defaults(func=JobCommands.estimate)

        submit = subparsers.add_parser(
            name="submit",
            help="upload, quote, confirm the cost and submit a job directory",
        )
        CommandLine._add_job_directory(submit)
        submit.add_argument(
            "--yes",
            action="store_true",
            help="accept the quoted cost cap without asking",
        )
        submit.add_argument(
            "--wait",
            action="store_true",
            help="wait for the job to end and fetch its files into <directory>/xtrace-results/<job>",
        )
        submit.add_argument(
            "--idempotency-key",
            default=None,
            help="the key that makes a repeated submission start one job; a fresh one by default",
        )
        submit.set_defaults(func=JobCommands.submit)

        jobs = subparsers.add_parser(
            name="jobs",
            help="list the account's jobs, newest first",
        )
        jobs.add_argument(
            "--limit",
            type=int,
            default=JobCommands.JOBS_LIMIT_DEFAULT,
        )
        jobs.add_argument(
            "--cursor",
            default=None,
            help="the cursor a previous page printed",
        )
        CommandLine._add_json(jobs)
        jobs.set_defaults(func=JobCommands.jobs)

        job = subparsers.add_parser(
            name="job",
            help="show a job's state, cost and result quality",
        )
        job.add_argument("job")
        job.add_argument(
            "--wait",
            action="store_true",
            help="wait for the job to end",
        )
        CommandLine._add_json(job)
        job.set_defaults(func=JobCommands.job)

        cancel = subparsers.add_parser(
            name="cancel",
            help="ask a job to stop",
        )
        cancel.add_argument("job")
        cancel.set_defaults(func=JobCommands.cancel)

        fetch = subparsers.add_parser(
            name="fetch",
            help="download a finished job's files",
        )
        fetch.add_argument("job")
        fetch.add_argument(
            "-o",
            "--output",
            default=None,
            help="the directory the files land in; xtrace-results/<job> by default",
        )
        fetch.set_defaults(func=JobCommands.fetch)

        mcp = subparsers.add_parser(
            name="mcp",
            help="serve these operations as MCP tools over stdio",
        )
        mcp.set_defaults(func=JobCommands.mcp)

    @staticmethod
    def _add_binary_commands(subparsers: argparse._SubParsersAction) -> None:
        """The commands that find cubins and read what they hold."""
        cubin = subparsers.add_parser(
            name="cubin",
            help="find a kernel's cubin in compiler caches, or show what a cubin holds",
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
            help="print a cubin's target, digest and kernels",
        )
        cubin_show.add_argument("binary")
        cubin_show.set_defaults(func=BinaryCommands.show)

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
        CommandLine._add_platform_commands(subparsers)
        CommandLine._add_binary_commands(subparsers)
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
