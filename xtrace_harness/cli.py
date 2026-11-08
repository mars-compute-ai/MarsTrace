import argparse
import json
import os
import sys
from typing import Any
from xtrace_harness.binary import DslCache, KernelBinary
from xtrace_harness.error import HarnessError
from xtrace_harness.listing import InspectionCache, KernelListing
from xtrace_harness.mcp import McpServer
from xtrace_harness.viewer.cli.show import ShowCommand
from xtrace_harness.workflow import QuotedInspection, QuotedJob, XtraceWorkflow


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
            print(f"correct   {map_manifest.get('correctness')}")

    @staticmethod
    def hardware(args: argparse.Namespace) -> None:
        """List the hardware a job can run on, with rates, architectures and stages."""
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
                f"{map_compute.get('purchase_option', '')} {map_compute.get('region', '')}  "
                f"stages: {', '.join(map_entry.get('stages', []))}"
            )
        if map_capabilities.get("models"):
            print(f"planner models: {', '.join(map_capabilities['models'])}")

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
    def print_inspection(map_status: dict[str, Any]) -> None:
        """One inspection's execution, cleanup and fee.

        Args:
            map_status: the inspection's status
        """
        map_execution = map_status["execution"]
        map_accounting = map_status["accounting"]
        print(f"inspection {map_status['inspection_id']}")
        print(f"state     {map_execution['state']} {map_execution.get('outcome', '')}".rstrip())
        print(f"created   {map_status['created_at']}")
        print(f"cleanup   {map_status['cleanup']['state']}")
        print(f"reserved  {XtraceWorkflow.usd(map_accounting.get('reserved_micro_usd'))}")
        print(f"charged   {XtraceWorkflow.usd(map_accounting.get('actual_micro_usd'))}")

    @staticmethod
    def _print_inspection_quote(quoted: QuotedInspection) -> None:
        """What an inspection quote commits to."""
        map_summary = quoted.summary()
        print(f"quote     {map_summary['quote_id']} (expires {map_summary['expires_at']})")
        print(f"profile   {map_summary['profile_id']}")
        print(f"fee       {map_summary['fee_usd']}")

    @staticmethod
    def _submit_inspection(
        workflow: XtraceWorkflow,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        """Quote an inspection, confirm its fee and accept it."""
        quoted = workflow.quote_inspection(
            binary_path=args.binary,
            kernel_symbol=args.kernel,
            profile_id=args.profile,
        )
        JobCommands._print_inspection_quote(quoted)
        JobCommands._confirm(
            args=args,
            prompt="inspect?",
        )
        return workflow.submit_inspection(
            quoted=quoted,
            idempotency_key=args.idempotency_key,
        )

    @staticmethod
    def inspection_submit(args: argparse.Namespace) -> None:
        """Quote, confirm and accept an inspection without waiting for it."""
        JobCommands.print_inspection(
            JobCommands._submit_inspection(
                workflow=JobCommands._workflow(),
                args=args,
            )
        )

    @staticmethod
    def inspect(args: argparse.Namespace) -> None:
        """Quote, confirm and run an inspection of a cubin's kernel, then cache and print its layout."""
        workflow = JobCommands._workflow()
        binary = KernelBinary.load(args.binary)
        path = InspectionCache.path_of(
            binary_path=args.binary,
            sha256=binary.sha256,
            kernel_symbol=args.kernel,
        )
        if (
            args.refresh
            or not os.path.isfile(path)
        ):
            map_status = JobCommands._submit_inspection(
                workflow=workflow,
                args=args,
            )
            map_status = workflow.wait_inspection(map_status["inspection_id"])
            JobCommands.print_inspection(map_status)
            path = workflow.fetch_inspection(
                inspection_id=map_status["inspection_id"],
                binary_path=args.binary,
                kernel_symbol=args.kernel,
            )
        print(KernelListing.load(path).render_kernels())
        print(f"listing   {path}")

    @staticmethod
    def sass(args: argparse.Namespace) -> None:
        """Print a cached SASS listing, narrowed by opcode or index."""
        path = args.listing
        if not path.endswith(".json"):
            path = InspectionCache.path_of(
                binary_path=args.listing,
                sha256=KernelBinary.load(args.listing).sha256,
                kernel_symbol=args.kernel,
            )
            if not os.path.isfile(path):
                raise HarnessError(
                    f"no inspection of {args.kernel or 'the kernel catalog'} is cached beside "
                    f"{args.listing}; run 'xtrace inspect' first"
                )
        listing = KernelListing.load(path)
        if args.kernels:
            print(listing.render_kernels())
            return
        print(
            listing.render_listing(
                opcode_pattern=args.grep,
                first_index=args.first,
                last_index=args.last,
            )
        )

    @staticmethod
    def inspection_profiles(args: argparse.Namespace) -> None:
        """List the inspection profiles with their fixed fees."""
        map_capabilities = JobCommands._workflow().inspection_capabilities()
        if args.json:
            JobCommands._print_json(map_capabilities)
            return
        map_availability = map_capabilities["availability"]
        if map_availability["state"] != XtraceWorkflow.INSPECTION_AVAILABLE:
            print(f"inspections are unavailable: {map_availability.get('reason')}")
            return
        for map_profile in map_availability["profiles"]:
            print(
                f"{map_profile['profile']['profile_id']:<32} "
                f"{XtraceWorkflow.usd(map_profile['price']['fixed_fee_micro_usd'])} per inspection  "
                f"at most {map_profile['limits']['max_input_bytes']} bytes, "
                f"{map_profile['limits']['max_kernels']} kernels"
            )

    @staticmethod
    def inspection_estimate(args: argparse.Namespace) -> None:
        """Upload a cubin and quote its inspection."""
        quoted = JobCommands._workflow().quote_inspection(
            binary_path=args.binary,
            kernel_symbol=args.kernel,
            profile_id=args.profile,
        )
        if args.json:
            JobCommands._print_json(quoted.summary())
            return
        JobCommands._print_inspection_quote(quoted)

    @staticmethod
    def inspection_list(args: argparse.Namespace) -> None:
        """List a page of the account's inspections."""
        map_page = JobCommands._workflow().inspections(
            limit=args.limit,
            cursor=args.cursor,
        )
        if args.json:
            JobCommands._print_json(map_page)
            return
        for map_status in map_page.get("items", []):
            map_execution = map_status["execution"]
            print(
                f"{map_status['inspection_id']:<38} {map_execution['state']:<16} "
                f"{map_execution.get('outcome', ''):<16} {map_status['created_at']}"
            )
        if map_page.get("next_cursor"):
            print(f"more: --cursor {map_page['next_cursor']}")

    @staticmethod
    def inspection_get(args: argparse.Namespace) -> None:
        """Print one inspection, or wait for it to end."""
        workflow = JobCommands._workflow()
        if args.wait:
            map_status = workflow.wait_inspection(args.inspection)
        else:
            map_status = workflow.inspection(args.inspection)
        if args.json:
            JobCommands._print_json(map_status)
            return
        JobCommands.print_inspection(map_status)

    @staticmethod
    def inspection_cancel(args: argparse.Namespace) -> None:
        """Ask an inspection to stop."""
        JobCommands.print_inspection(JobCommands._workflow().cancel_inspection(args.inspection))

    @staticmethod
    def inspection_result(args: argparse.Namespace) -> None:
        """Fetch a completed inspection of a cubin into the cache beside it."""
        path = JobCommands._workflow().fetch_inspection(
            inspection_id=args.inspection,
            binary_path=args.binary,
            kernel_symbol=args.kernel,
        )
        print(KernelListing.load(path).render_kernels())
        print(f"listing   {path}")

    @staticmethod
    def inspection_downloads(args: argparse.Namespace) -> None:
        """Download artifacts an inspection result names."""
        for path in JobCommands._workflow().download_inspection(
            inspection_id=args.inspection,
            list_artifact_id=args.artifact,
            output_directory=args.output or os.path.join(
                XtraceWorkflow.INSPECTIONS_DIRECTORY,
                args.inspection,
            ),
        ):
            print(f"saved     {path}")

    @staticmethod
    def _print_quote(quoted: QuotedJob) -> None:
        """What a quote commits to."""
        map_summary = quoted.summary()
        print(f"quote     {map_summary['quote_id']} (expires {map_summary['expires_at']})")
        print(f"hardware  {map_summary['hardware']} ({map_summary['architecture']})")
        print(f"rate      {map_summary['gpu_usd_per_hour']} per GPU-hour")
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
        """Upload, resolve, quote, confirm the cost and submit a job directory."""
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
                f"copied [{args.pick}] to {map_found['copied_to']}; set [kernel] binary and "
                f"name = \"{map_found['kernels'][args.pick]['name']}\""
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
            help="list the hardware a job can run on, with rates, architectures and stages",
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
            help="upload and quote a job directory; print the cost cap and the rate",
        )
        CommandLine._add_job_directory(estimate)
        CommandLine._add_json(estimate)
        estimate.set_defaults(func=JobCommands.estimate)

        submit = subparsers.add_parser(
            name="submit",
            help="upload, resolve sites, quote, confirm the cost and submit a job directory",
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
        """The commands that find and read cubins and their listings."""
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

        inspect = subparsers.add_parser(
            name="inspect",
            help="quote, confirm and run an inspection of a cubin; cache its listing beside it",
        )
        CommandLine._add_inspection_target(inspect)
        CommandLine._add_inspection_acceptance(inspect)
        inspect.add_argument(
            "--refresh",
            action="store_true",
            help="inspect again even when a listing is cached",
        )
        inspect.set_defaults(func=JobCommands.inspect)

        sass = subparsers.add_parser(
            name="sass",
            help="browse a kernel's cached SASS listing",
        )
        sass.add_argument(
            "listing",
            help="a cubin, or a cached .inspection-result.json",
        )
        sass.add_argument(
            "--kernel",
            default=None,
            help="the kernel to list, when the first argument is a cubin",
        )
        sass.add_argument(
            "--kernels",
            action="store_true",
            help="list the inspected kernels and their parameter layouts instead",
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
        sass.set_defaults(func=JobCommands.sass)

    @staticmethod
    def _add_inspection_target(parser: argparse.ArgumentParser) -> None:
        """The options of a command that quotes an inspection of a cubin."""
        parser.add_argument(
            "binary",
            help="the cubin",
        )
        parser.add_argument(
            "--kernel",
            default=None,
            help="the kernel to list, as the cubin names it; without it only the kernel catalog is read",
        )
        parser.add_argument(
            "--profile",
            default=None,
            help="the inspection profile, from 'xtrace inspection profiles'; the first one by default",
        )

    @staticmethod
    def _add_inspection_acceptance(parser: argparse.ArgumentParser) -> None:
        """The options of a command that accepts an inspection quote."""
        parser.add_argument(
            "--yes",
            action="store_true",
            help="accept the quoted fee without asking",
        )
        parser.add_argument(
            "--idempotency-key",
            default=None,
            help="the key that makes a repeated acceptance start one inspection; a fresh one by default",
        )

    @staticmethod
    def _add_inspection_commands(subparsers: argparse._SubParsersAction) -> None:
        """``xtrace inspection``: each step of a quoted inspection on its own."""
        inspection = subparsers.add_parser(
            name="inspection",
            help="quote, list, follow, cancel and fetch inspections step by step",
        )
        inspection_commands = inspection.add_subparsers()
        profiles = inspection_commands.add_parser(
            name="profiles",
            help="list the inspection profiles, their limits and fixed fees",
        )
        CommandLine._add_json(profiles)
        profiles.set_defaults(func=JobCommands.inspection_profiles)

        estimate = inspection_commands.add_parser(
            name="estimate",
            help="upload a cubin and quote its inspection",
        )
        CommandLine._add_inspection_target(estimate)
        CommandLine._add_json(estimate)
        estimate.set_defaults(func=JobCommands.inspection_estimate)

        submit = inspection_commands.add_parser(
            name="submit",
            help="quote, confirm and accept an inspection without waiting for it",
        )
        CommandLine._add_inspection_target(submit)
        CommandLine._add_inspection_acceptance(submit)
        submit.set_defaults(func=JobCommands.inspection_submit)

        listed = inspection_commands.add_parser(
            name="list",
            help="list the account's inspections",
        )
        listed.add_argument(
            "--limit",
            type=int,
            default=JobCommands.JOBS_LIMIT_DEFAULT,
        )
        listed.add_argument(
            "--cursor",
            default=None,
            help="the cursor a previous page printed",
        )
        CommandLine._add_json(listed)
        listed.set_defaults(func=JobCommands.inspection_list)

        get = inspection_commands.add_parser(
            name="get",
            help="show an inspection's execution, cleanup and fee",
        )
        get.add_argument("inspection")
        get.add_argument(
            "--wait",
            action="store_true",
            help="wait for the inspection to end",
        )
        CommandLine._add_json(get)
        get.set_defaults(func=JobCommands.inspection_get)

        cancel = inspection_commands.add_parser(
            name="cancel",
            help="ask an inspection to stop",
        )
        cancel.add_argument("inspection")
        cancel.set_defaults(func=JobCommands.inspection_cancel)

        result = inspection_commands.add_parser(
            name="result",
            help="fetch a completed inspection of a cubin into the cache beside it",
        )
        result.add_argument("inspection")
        result.add_argument(
            "binary",
            help="the cubin it inspected",
        )
        result.add_argument(
            "--kernel",
            default=None,
            help="the kernel whose listing to download",
        )
        result.set_defaults(func=JobCommands.inspection_result)

        downloads = inspection_commands.add_parser(
            name="downloads",
            help="download artifacts an inspection result names",
        )
        downloads.add_argument("inspection")
        downloads.add_argument(
            "artifact",
            nargs="+",
            help="an artifact id from the result",
        )
        downloads.add_argument(
            "-o",
            "--output",
            default=None,
            help="the directory the files land in; xtrace-inspections/<inspection> by default",
        )
        downloads.set_defaults(func=JobCommands.inspection_downloads)

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
        CommandLine._add_inspection_commands(subparsers)
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
