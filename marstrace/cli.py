import argparse
import json
import os
import sys
from typing import Any
from marstrace.analysis import DeliveredAnalysis
from marstrace.binary import DslCache, KernelBinary
from marstrace.error import MarsTraceError
from marstrace.guide import AgentGuide
from marstrace.mcp import McpServer
from marstrace.viewer.cli.show import ShowCommand
from marstrace.workflow import MarsTraceWorkflow, QuotedJob


class JobCommands:
    """The ``marstrace`` commands that call the MarsTrace service."""

    JSON_INDENT = 2
    JOBS_LIMIT_DEFAULT = 20
    SET_YES = {
        "y",
        "yes",
    }

    @staticmethod
    def _workflow() -> MarsTraceWorkflow:
        """A workflow with the environment's credentials, reporting progress on stderr."""
        return MarsTraceWorkflow.from_environment(
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
        """One job's status, hardware, CUDA version, attempts, progress and cost.

        Args:
            map_job: the job
        """
        map_cost = map_job["cost"]
        map_progress = map_job["progress"]
        print(f"job       {map_job['job_id']}")
        print(f"status    {map_job['status']}")
        print(f"hardware  {map_job['hardware']}")
        print(f"cuda      {map_job['cuda_version']}")
        print(f"created   {map_job['created_at']}")
        if map_job["ended_at"]:
            print(f"ended     {map_job['ended_at']}")
        if map_progress:
            print(
                f"progress  agent turn {map_progress['agent_turn']}, "
                f"{MarsTraceWorkflow.usd(map_progress['usage_micro_usd'])} used so far"
            )
        for map_attempt in map_job["attempts"]:
            line = f"attempt   {map_attempt['number']} {map_attempt['status']}"
            if map_attempt["failure"]:
                line += f" ({map_attempt['failure']})"
            if map_attempt["error"]:
                line += f": {map_attempt['error']['code']}: {map_attempt['error']['message']}"
            print(line)
        print(f"held      {MarsTraceWorkflow.usd(map_cost['held_micro_usd'])}")
        charged = f"charged   {MarsTraceWorkflow.usd(map_cost['charged_micro_usd'])}"
        note = MarsTraceWorkflow.MAP_NOT_CHARGED_NOTE.get(map_cost.get("not_charged_reason") or "")
        if note is not None:
            charged += f" ({note})"
        print(charged)
        for map_line in map_cost["lines"]:
            if map_line["quantity"] == 0:
                continue
            print(
                f"          {map_line['dimension']} {map_line['quantity']}: "
                f"{MarsTraceWorkflow.usd(map_line['amount_micro_usd'])}"
            )
        map_error = map_job["error"]
        if map_error:
            print(f"error     {map_error['code']}: {map_error['message']}")

    @staticmethod
    def hardware(args: argparse.Namespace) -> None:
        """List the hardware a job can run on, with its architecture and CUDA versions, the
        service's token rates, and the agent model."""
        map_hardware = JobCommands._workflow().hardware()
        if args.json:
            JobCommands._print_json(map_hardware)
            return
        map_rates = map_hardware["rates"]
        if map_rates is None:
            print("tokens    no published price")
        else:
            print(
                f"tokens    {MarsTraceWorkflow.usd(map_rates['agent_input_micro_usd_per_million'])} input, "
                f"{MarsTraceWorkflow.usd(map_rates['agent_cache_read_micro_usd_per_million'])} cache read, "
                f"{MarsTraceWorkflow.usd(map_rates['agent_cache_write_5m_micro_usd_per_million'])} 5-minute cache write, "
                f"{MarsTraceWorkflow.usd(map_rates['agent_cache_write_1h_micro_usd_per_million'])} 1-hour cache write, "
                f"{MarsTraceWorkflow.usd(map_rates['agent_output_micro_usd_per_million'])} output "
                "per million, on every hardware"
            )
        for map_entry in map_hardware["hardware"]:
            list_version = [
                f"{version} (default)" if version == map_entry["default_cuda_version"] else version
                for version in map_entry["cuda_versions"]
            ]
            print(
                f"{map_entry['id']:<8} {map_entry['gpu']:<8} {map_entry['architecture']:<8} "
                f"{'available' if map_entry['available'] else 'not available now'}"
            )
            print(f"{'':<8} CUDA {', '.join(list_version)}")
        map_agent = map_hardware["agent"]
        print(f"agent model: {map_agent['model']}")
        print(
            "agent defaults: "
            + ", ".join(f"{key} {value}" for key, value in map_agent["defaults"].items())
        )

    @staticmethod
    def credit(args: argparse.Namespace) -> None:
        """Print the workspace's balance against the minimum a job needs."""
        map_credit = JobCommands._workflow().credit()
        if args.json:
            JobCommands._print_json(map_credit)
            return
        available = map_credit["available_micro_usd"]
        print(f"available {'no limit' if available is None else MarsTraceWorkflow.usd(available)}")
        print(f"held      {MarsTraceWorkflow.usd(map_credit['held_micro_usd'])} by running jobs")
        print(f"minimum   {MarsTraceWorkflow.usd(map_credit['minimum_micro_usd'])} to start a job")
        print(f"jobs      {'can start' if map_credit['admissible'] else 'cannot start; add credit'}")
        print(f"add more  {map_credit['add_credit_url']}")

    @staticmethod
    def _confirm(
        args: argparse.Namespace,
        prompt: str,
    ) -> None:
        """Go on only when ``--yes`` was given or the user at the terminal agrees."""
        if args.yes:
            return
        if not sys.stdin.isatty():
            raise MarsTraceError("confirm the cost above: run again with --yes")
        if input(f"{prompt} [y/N] ").strip().lower() not in JobCommands.SET_YES:
            raise MarsTraceError("not submitted")

    @staticmethod
    def _print_quote(quoted: QuotedJob) -> None:
        """What a job would cost, and whether it can start."""
        map_summary = quoted.summary()
        map_rates = map_summary["rates"]
        map_token = map_rates["agent_usd_per_million_tokens"]
        map_bounds = map_summary["bounds_per_attempt"]
        map_cost = map_summary["cost_at_bounds"]
        print(f"hardware  {map_summary['hardware']}, {map_summary['gpu']} ({map_summary['architecture']})")
        print(f"cuda      {map_summary['cuda_version']}")
        print(
            f"tokens    {map_rates['agent_model']}: {map_token['input']} input, "
            f"{map_token['cache_read']} cache read, {map_token['cache_write_5m']} 5-minute cache write, "
            f"{map_token['cache_write_1h']} 1-hour cache write, {map_token['output']} output per million"
        )
        print(
            f"bounds    per attempt "
            f"{map_bounds['agent_input_tokens']} input and {map_bounds['agent_output_tokens']} output tokens; "
            f"{map_summary['max_attempts']} attempt(s)"
        )
        print(
            f"at most   {map_cost['total_usd']}: {map_cost['agent_tokens_per_attempt_usd']} of tokens per attempt"
        )
        print(
            "charged   the tokens every attempt actually used, failed ones included; nothing if the trace is "
            "unusable or the job fails without a result"
        )
        print(f"balance   {map_summary['available_usd']} available, {map_summary['minimum_usd']} needed to start")
        if not map_summary["admissible"]:
            print(f"refused   {map_summary['message'] or 'the available balance is below the minimum'}")

    @staticmethod
    def estimate(args: argparse.Namespace) -> None:
        """Upload and price a job directory, then print the rates, the cost at the bounds and
        the balance."""
        quoted = JobCommands._workflow().quote(
            directory=args.directory,
            hardware=args.hardware,
            cuda_version=args.cuda_version,
        )
        if args.json:
            JobCommands._print_json(quoted.summary())
            return
        JobCommands._print_quote(quoted)

    @staticmethod
    def submit(args: argparse.Namespace) -> None:
        """Upload, price, confirm the cost and submit a job directory; a key the directory
        already submitted under is submitted again as recorded."""
        workflow = JobCommands._workflow()
        map_job = None
        if args.idempotency_key is not None:
            map_job = workflow.resubmit(
                directory=args.directory,
                idempotency_key=args.idempotency_key,
            )
        if map_job is None:
            quoted = workflow.quote(
                directory=args.directory,
                hardware=args.hardware,
                cuda_version=args.cuda_version,
            )
            JobCommands._print_quote(quoted)
            workflow.check_admissible(quoted)
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
        map_job = workflow.wait(map_job["job_id"])
        JobCommands.print_job(map_job)
        if map_job["status"] == "succeeded":
            JobCommands._print_saved(
                workflow.fetch(
                    job_id=map_job["job_id"],
                    output_directory=os.path.join(
                        args.directory,
                        MarsTraceWorkflow.RESULTS_DIRECTORY,
                        map_job["job_id"],
                    ),
                )
            )

    @staticmethod
    def _print_saved(list_path: list[str]) -> None:
        """The files a fetch wrote, the quality of the trace, and the analysis it carries."""
        for path in list_path:
            print(f"saved     {path}")
        with open(
            file=list_path[-1],
            mode="r",
            encoding="utf-8",
        ) as manifest_file:
            map_manifest = json.load(manifest_file)
        print(f"quality   {map_manifest['quality']} {', '.join(map_manifest['reason_codes'])}".rstrip())
        for path in list_path:
            if os.path.basename(path) == MarsTraceWorkflow.RESULT_FILE:
                analysis = DeliveredAnalysis.from_file(path)
                if analysis is not None:
                    print(analysis.text(), end="")

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
        for map_job in map_page["items"]:
            print(
                f"{map_job['job_id']:<30} {map_job['status']:<10} "
                f"{map_job['hardware']:<6} {map_job['created_at']}"
            )
        if map_page["next_cursor"]:
            print(f"more: --cursor {map_page['next_cursor']}")

    @staticmethod
    def job(args: argparse.Namespace) -> None:
        """Print one job, or wait for it to end."""
        workflow = JobCommands._workflow()
        map_job = workflow.wait(args.job) if args.wait else workflow.job(args.job)
        if args.json:
            JobCommands._print_json(map_job)
            return
        JobCommands.print_job(map_job)

    @staticmethod
    def cancel(args: argparse.Namespace) -> None:
        """Ask a job to stop."""
        JobCommands.print_job(JobCommands._workflow().cancel(args.job))

    @staticmethod
    def fetch(args: argparse.Namespace) -> None:
        """Download a succeeded job's marstrace.json and its result manifest, and print the analysis it carries."""
        list_path = JobCommands._workflow().fetch(
            job_id=args.job,
            output_directory=args.output or os.path.join(
                MarsTraceWorkflow.RESULTS_DIRECTORY,
                args.job,
            ),
        )
        JobCommands._print_saved(list_path)

    @staticmethod
    def guide(args: argparse.Namespace) -> None:
        """Print one guide, or the list of guides."""
        if args.topic is None:
            for topic in AgentGuide.topics():
                print(f"{topic:<8} {AgentGuide.MAP_TOPIC_SUMMARY[topic]}")
            return
        print(AgentGuide.text(args.topic), end="")

    @staticmethod
    def mcp(args: argparse.Namespace) -> None:
        """Serve the operations as MCP tools over stdio."""
        McpServer(workflow_of=MarsTraceWorkflow.from_environment).serve(
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )


class BinaryCommands:
    """``marstrace cubin``: find a kernel's cubin in compiler caches, and read what a cubin holds."""

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
    """The ``marstrace`` command line."""

    EXIT_FAILURE = 1

    @staticmethod
    def _add_json(parser: argparse.ArgumentParser) -> None:
        """The option that prints the service's answer as JSON."""
        parser.add_argument(
            "--json",
            action="store_true",
            help="print the service's answer as JSON",
        )

    @staticmethod
    def _add_job_directory(parser: argparse.ArgumentParser) -> None:
        """The options of a command that reads a job directory."""
        parser.add_argument(
            "directory",
            help="the job directory, holding marstrace.toml and the files it names",
        )
        parser.add_argument(
            "--hardware",
            default=None,
            help="override the manifest's hardware, a hardware id from 'marstrace hardware'",
        )
        parser.add_argument(
            "--cuda-version",
            default=None,
            help="override the manifest's CUDA version, one the hardware offers in 'marstrace hardware'",
        )

    @staticmethod
    def _add_service_commands(subparsers: argparse._SubParsersAction) -> None:
        """The commands that read the account and run jobs."""
        hardware = subparsers.add_parser(
            name="hardware",
            help="list the hardware a job can run on, with its CUDA versions and token rates, and the agent model",
        )
        CommandLine._add_json(hardware)
        hardware.set_defaults(func=JobCommands.hardware)

        credit = subparsers.add_parser(
            name="credit",
            help="show the available balance, the minimum a job needs to start, and where to add credit",
        )
        CommandLine._add_json(credit)
        credit.set_defaults(func=JobCommands.credit)

        estimate = subparsers.add_parser(
            name="estimate",
            help="upload and price a job directory; print the rates, the cost at the bounds and the balance",
        )
        CommandLine._add_job_directory(estimate)
        CommandLine._add_json(estimate)
        estimate.set_defaults(func=JobCommands.estimate)

        submit = subparsers.add_parser(
            name="submit",
            help="upload, price, confirm the cost and submit a job directory",
        )
        CommandLine._add_job_directory(submit)
        submit.add_argument(
            "--yes",
            action="store_true",
            help="accept the estimated cost without asking",
        )
        submit.add_argument(
            "--wait",
            action="store_true",
            help="wait for the job to end and fetch its trace into <directory>/marstrace-results/<job>",
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
            help="show a job's status, attempts, usage so far and charge",
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
            help="ask a job to stop; it is charged what it used",
        )
        cancel.add_argument("job")
        cancel.set_defaults(func=JobCommands.cancel)

        fetch = subparsers.add_parser(
            name="fetch",
            help="download a succeeded job's marstrace.json and its result manifest",
        )
        fetch.add_argument("job")
        fetch.add_argument(
            "-o",
            "--output",
            default=None,
            help="the directory the files land in; marstrace-results/<job> by default",
        )
        fetch.set_defaults(func=JobCommands.fetch)

        mcp = subparsers.add_parser(
            name="mcp",
            help="serve these operations as MCP tools over stdio",
        )
        mcp.set_defaults(func=JobCommands.mcp)

        guide = subparsers.add_parser(
            name="guide",
            help="print the guide an agent follows: remote (set up, submit, fetch) or read (read a trace)",
        )
        guide.add_argument(
            "topic",
            nargs="?",
            choices=AgentGuide.topics(),
            help="the guide to print; the list of guides when absent",
        )
        guide.set_defaults(func=JobCommands.guide)

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
        """The parser of every ``marstrace`` command.

        Returns:
            the parser
        """
        parser = argparse.ArgumentParser(
            prog="marstrace",
            description="Trace CUDA kernels on cloud GPUs with MarsTrace and read the traces.",
        )
        subparsers = parser.add_subparsers()
        CommandLine._add_service_commands(subparsers)
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
        except MarsTraceError as err:
            print(
                f"marstrace: {err}",
                file=sys.stderr,
            )
            return CommandLine.EXIT_FAILURE
        return 0


def main() -> None:
    """Entry point of the ``marstrace`` command."""
    sys.exit(CommandLine.run())


if __name__ == "__main__":
    main()
