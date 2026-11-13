import argparse
import contextlib
import decimal
import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO
from marstrace.binary import DslCache, KernelBinary
from marstrace.error import MarsTraceError, ServiceError, ToolArgumentError
from marstrace.guide import AgentGuide
from marstrace.viewer.cli.show import ShowCommand
from marstrace.workflow import MarsTraceWorkflow


@dataclass
class McpTool:
    """One tool the MCP server offers: its name, what it does, what it takes and what runs it.

    Args:
        name: the tool's name
        description: what the tool does, for the agent choosing it
        map_input_schema: the JSON Schema of its arguments
        handler: called with the arguments; returns a JSON value or text
    """

    # JSON Schema type -> the Python types that satisfy it
    MAP_JSON_TYPE = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
    }

    name: str
    description: str
    map_input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def check_arguments(self, map_arguments: Any) -> None:
        """Check call arguments against the tool's input schema.

        Args:
            map_arguments: the arguments

        Raises:
            ToolArgumentError: an argument is missing, unknown or of the wrong type
        """
        if not isinstance(
            map_arguments,  # obj
            dict,           # class_or_tuple
        ):
            raise ToolArgumentError(f"{self.name} takes an object of arguments")
        map_property = self.map_input_schema.get("properties", {})
        for key in self.map_input_schema.get("required", []):
            if key not in map_arguments:
                raise ToolArgumentError(f"{self.name} needs '{key}'")
        for key, value in map_arguments.items():
            if key not in map_property:
                raise ToolArgumentError(f"{self.name} takes no argument '{key}'")
            json_type = map_property[key]["type"]
            is_bool = isinstance(
                value,  # obj
                bool,   # class_or_tuple
            )
            if (
                not isinstance(
                    value,                              # obj
                    McpTool.MAP_JSON_TYPE[json_type],  # class_or_tuple
                )
                or (
                    is_bool
                    and json_type != "boolean"
                )
                or (
                    json_type == "array"
                    and not all(
                        isinstance(
                            item,  # obj
                            str,   # class_or_tuple
                        )
                        for item in value
                    )
                )
            ):
                raise ToolArgumentError(f"{self.name}: '{key}' must be of type {json_type}")


class McpServer:
    """A local MCP server over stdio: JSON-RPC 2.0, one message per line, offering the
    ``marstrace`` operations on local paths, so an agent can trace a kernel through MCP alone.
    The tools that call the service carry the names of the service's own MCP tools.

    Args:
        workflow: the operations the tools run
    """

    JSONRPC_VERSION = "2.0"
    LIST_PROTOCOL_VERSION = (
        "2025-03-26",
        "2025-06-18",
        "2025-11-25",
    )
    SERVER_NAME = "marstrace"
    SERVER_VERSION = "0.1.0"
    INSTRUCTIONS = (
        "Trace the timeline inside a CUDA kernel on a cloud GPU with MarsTrace. Before the first job, "
        "read marstrace_guide with topic remote, and before reading a trace, topic read: they hold "
        "the whole workflow. The user needs a MarsCompute account with credit and a MarsTrace key "
        "(mt_live_...) in MARSTRACE_API_KEY; a tool that calls the service without one answers how to "
        "get it. A job is a local directory holding marstrace.toml: hardware (an id from "
        "marstrace_hardware), intent (what the trace should answer, with the problem size), "
        "[workspace] cubins (1 to 8 cubins built for the hardware's architecture) and optionally "
        "source, manifest, tensors and files, [agent] caps, cuda_version and [limits] max_attempts. "
        "A job starts only while the available balance reaches the minimum the estimate names, and "
        "is charged the agent tokens it actually used over every attempt, failed ones included, even "
        "past the balance. Estimate, show the user the token rates and the cost at the bounds and "
        "get their yes, submit with accept_cost_at_bounds_usd, wait, fetch. Then check quality and "
        "reason_codes in result-manifest.json before any number, and read marstrace.json with "
        "marstrace_show, never with a JSON parser of your own."
    )
    # JSON-RPC error codes
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    # the most characters a text result carries before it is cut
    TEXT_CHARACTERS_MAX = 60000
    WAIT_SECONDS_DEFAULT = 600
    WAIT_SECONDS_MAX = 3600
    JOBS_LIMIT_DEFAULT = 20

    def __init__(self, workflow_of: Callable[[], MarsTraceWorkflow]) -> None:
        """A server whose tools reach the service through the workflow ``workflow_of`` makes.

        The workflow is made at the first tool that calls the service, so the server starts and
        answers its guide without a key, and a tool that needs one says how to get it.

        Args:
            workflow_of: makes the workflow; raises when the environment holds no usable key
        """
        self._workflow_of = workflow_of
        self._workflow_made: MarsTraceWorkflow | None = None
        self._map_tool = {tool.name: tool for tool in self._tools()}

    def _workflow(self) -> MarsTraceWorkflow:
        """The workflow, made on first use."""
        if self._workflow_made is None:
            self._workflow_made = self._workflow_of()
        return self._workflow_made

    def serve(
        self,
        input_stream: TextIO,
        output_stream: TextIO,
    ) -> None:
        """Answer every message read from the input until it ends.

        Args:
            input_stream: the client's messages, one JSON-RPC message per line
            output_stream: where the answers go, one per line
        """
        for line in input_stream:
            if not line.strip():
                continue
            map_response = self.handle_line(line)
            if map_response is None:
                continue
            output_stream.write(json.dumps(map_response) + "\n")
            output_stream.flush()

    def handle_line(self, line: str) -> dict[str, Any] | None:
        """The answer to one message.

        Args:
            line: the message

        Returns:
            the response, or None for a notification
        """
        try:
            map_request = json.loads(line)
        except ValueError:
            return McpServer._error(
                request_id=None,
                code=McpServer.PARSE_ERROR,
                message="Parse error",
            )
        if (
            not isinstance(
                map_request,  # obj
                dict,         # class_or_tuple
            )
            or map_request.get("jsonrpc") != McpServer.JSONRPC_VERSION
            or not isinstance(
                map_request.get("method"),  # obj
                str,                        # class_or_tuple
            )
        ):
            return McpServer._error(
                request_id=None,
                code=McpServer.INVALID_REQUEST,
                message="Invalid Request",
            )
        if "id" not in map_request:
            return None
        request_id = map_request["id"]
        map_params = map_request.get("params") or {}
        method = map_request["method"]
        if method == "initialize":
            requested = map_params.get("protocolVersion")
            return McpServer._result(
                request_id=request_id,
                value={
                    "protocolVersion": (
                        requested
                        if requested in McpServer.LIST_PROTOCOL_VERSION
                        else McpServer.LIST_PROTOCOL_VERSION[-1]
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": McpServer.SERVER_NAME,
                        "version": McpServer.SERVER_VERSION,
                    },
                    "instructions": McpServer.INSTRUCTIONS,
                },
            )
        if method == "ping":
            return McpServer._result(
                request_id=request_id,
                value={},
            )
        if method == "tools/list":
            return McpServer._result(
                request_id=request_id,
                value={
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": tool.map_input_schema,
                        }
                        for tool in self._map_tool.values()
                    ],
                },
            )
        if method == "tools/call":
            return self._call(
                request_id=request_id,
                map_params=map_params,
            )
        return McpServer._error(
            request_id=request_id,
            code=McpServer.METHOD_NOT_FOUND,
            message="Method not found",
        )

    @staticmethod
    def _result(
        request_id: Any,
        value: Any,
    ) -> dict[str, Any]:
        """A JSON-RPC success response."""
        return {
            "jsonrpc": McpServer.JSONRPC_VERSION,
            "id": request_id,
            "result": value,
        }

    @staticmethod
    def _error(
        request_id: Any,
        code: int,
        message: str,
    ) -> dict[str, Any]:
        """A JSON-RPC error response."""
        return {
            "jsonrpc": McpServer.JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }

    def _call(
        self,
        request_id: Any,
        map_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a tool; a refusal of the client or the service is a tool result with isError."""
        tool = self._map_tool.get(map_params.get("name", ""))
        if tool is None:
            return McpServer._error(
                request_id=request_id,
                code=McpServer.INVALID_PARAMS,
                message=f"Unknown tool {map_params.get('name')}",
            )
        map_arguments = map_params.get("arguments") or {}
        try:
            tool.check_arguments(map_arguments)
        except ToolArgumentError as err:
            return McpServer._error(
                request_id=request_id,
                code=McpServer.INVALID_PARAMS,
                message=str(err),
            )
        try:
            output = tool.handler(map_arguments)
        except ServiceError as err:
            return McpServer._result(
                request_id=request_id,
                value=McpServer._content(
                    output={"error": err.to_json()},
                    is_error=True,
                ),
            )
        except (
            MarsTraceError,
            OSError,
        ) as err:
            return McpServer._result(
                request_id=request_id,
                value=McpServer._content(
                    output={"error": {"message": str(err)}},
                    is_error=True,
                ),
            )
        return McpServer._result(
            request_id=request_id,
            value=McpServer._content(
                output=output,
                is_error=False,
            ),
        )

    @staticmethod
    def _content(
        output: Any,
        is_error: bool,
    ) -> dict[str, Any]:
        """A tool result: text for the model, and the structured value when it is an object."""
        if isinstance(
            output,  # obj
            str,     # class_or_tuple
        ):
            text = output
            if len(text) > McpServer.TEXT_CHARACTERS_MAX:
                text = (
                    text[:McpServer.TEXT_CHARACTERS_MAX]
                    + "\n[cut; narrow the view with filters such as --block, --warp or --tid]"
                )
            return {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            }
        return {
            "content": [{"type": "text", "text": json.dumps(output)}],
            "structuredContent": output,
            "isError": is_error,
        }

    @staticmethod
    def _schema(
        map_property: dict[str, dict[str, Any]],
        list_required: list[str],
    ) -> dict[str, Any]:
        """The JSON Schema of a tool's arguments."""
        return {
            "type": "object",
            "properties": map_property,
            "required": list_required,
            "additionalProperties": False,
        }

    def _tools(self) -> list[McpTool]:
        """Every tool the server offers."""
        map_directory = {
            "type": "string",
            "description": "the job directory, holding marstrace.toml",
        }
        map_hardware = {
            "type": "string",
            "description": "a hardware id overriding the manifest's hardware",
        }
        map_cuda_version = {
            "type": "string",
            "description": "a CUDA version, such as 13.0, overriding the manifest's cuda_version; "
            "one the hardware offers in marstrace_hardware",
        }
        map_job_id = {
            "type": "string",
            "description": "the job id",
        }
        return [
            McpTool(
                name="marstrace_guide",
                description="How to use MarsTrace, in full: topic remote is setting up an account and "
                "a key, finding the cubin, gathering the workspace, sizing the caps, estimating, "
                "submitting and fetching; topic read is gating a trace and reading it with "
                "marstrace_show. Needs no key. Read remote before the first job and read before the "
                "first trace.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "topic": {
                            "type": "string",
                            "enum": AgentGuide.topics(),
                            "description": "remote or read",
                        },
                    },
                    list_required=["topic"],
                ),
                handler=lambda map_arguments: AgentGuide.text(map_arguments["topic"]),
            ),
            McpTool(
                name="marstrace_hardware",
                description="List the hardware a job can run on: id, GPU, architecture, availability "
                "and the CUDA versions a job can choose with the default one; the service's agent token "
                "rates, the same on every hardware; and the agent model with its default and maximum limits.",
                map_input_schema=McpServer._schema(
                    map_property={},
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow().hardware(),
            ),
            McpTool(
                name="marstrace_credit",
                description="The workspace's available balance, what running jobs hold, the minimum "
                "balance a job needs to start and whether it is reached (micro-USD; 1000000 is $1), and "
                "the page where credit is added.",
                map_input_schema=McpServer._schema(
                    map_property={},
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow().credit(),
            ),
            McpTool(
                name="marstrace_cubin_find",
                description="Find a kernel's cubin in the Triton, torch.compile and TileLang caches "
                "and given directories, newest first; optionally copy one into a job directory for "
                "[workspace] cubins.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "pattern": {"type": "string", "description": "a regex searched in kernel names"},
                        "dirs": {"type": "array", "items": {"type": "string"}},
                        "copy_to": {"type": "string", "description": "a job directory"},
                        "pick": {"type": "integer", "description": "the match to copy"},
                    },
                    list_required=["pattern"],
                ),
                handler=lambda map_arguments: DslCache.search(
                    pattern=map_arguments["pattern"],
                    list_directory=map_arguments.get("dirs", []),
                    copy_to=map_arguments.get("copy_to"),
                    pick=map_arguments.get("pick", 0),
                ),
            ),
            McpTool(
                name="marstrace_cubin_show",
                description="A cubin's target, digest and kernel symbols.",
                map_input_schema=McpServer._schema(
                    map_property={"binary": {"type": "string", "description": "the cubin"}},
                    list_required=["binary"],
                ),
                handler=self._cubin_show,
            ),
            McpTool(
                name="marstrace_estimate",
                description="Upload a job directory's workspace and price it; nothing is held. Returns "
                "the CUDA version, the token rates, the token bounds of one attempt, the cost at those bounds over every "
                "attempt (cost_at_bounds.total_usd, the most the job can cost), the minimum balance a job "
                "needs, the available balance, and whether the job would start now. Show the rates and "
                "the cost at the bounds to the user before submitting.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "directory": map_directory,
                        "hardware": map_hardware,
                        "cuda_version": map_cuda_version,
                    },
                    list_required=["directory"],
                ),
                handler=lambda map_arguments: self._workflow().quote(
                    directory=map_arguments["directory"],
                    hardware=map_arguments.get("hardware"),
                    cuda_version=map_arguments.get("cuda_version"),
                ).summary(),
            ),
            McpTool(
                name="marstrace_submit",
                description="Price and submit a job directory. Show the user the estimate's rates and "
                "cost_at_bounds.total_usd first, say that the job is charged the tokens it actually used over every "
                "attempt, failed ones included, even past the available balance, and submit only with "
                "their consent. accept_cost_at_bounds_usd is the cost at the bounds they agreed to; a "
                "higher one is refused. Reuse idempotency_key when retrying a submission whose answer "
                "was lost.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "directory": map_directory,
                        "accept_cost_at_bounds_usd": {
                            "type": "number",
                            "description": "the cost at the bounds, in dollars, the user agreed to",
                        },
                        "hardware": map_hardware,
                        "cuda_version": map_cuda_version,
                        "idempotency_key": {
                            "type": "string",
                            "description": "reuse it when retrying a submission whose answer was lost",
                        },
                    },
                    list_required=[
                        "directory",
                        "accept_cost_at_bounds_usd",
                    ],
                ),
                handler=self._submit,
            ),
            McpTool(
                name="marstrace_jobs",
                description="A page of the account's jobs, newest first.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "limit": {"type": "integer"},
                        "cursor": {"type": "string"},
                    },
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow().jobs(
                    limit=map_arguments.get("limit", McpServer.JOBS_LIMIT_DEFAULT),
                    cursor=map_arguments.get("cursor"),
                ),
            ),
            McpTool(
                name="marstrace_job",
                description="One job: its status, attempts with their failures, progress with the "
                "usage so far, the amount held at admission, the charge and its lines once settled, "
                "and its error.",
                map_input_schema=McpServer._schema(
                    map_property={"job_id": map_job_id},
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow().job(map_arguments["job_id"]),
            ),
            McpTool(
                name="marstrace_wait",
                description="Wait for a job to end (succeeded, failed or cancelled), up to "
                "timeout_seconds; returns the job as last seen.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "job_id": map_job_id,
                        "timeout_seconds": {"type": "number"},
                    },
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow().wait(
                    job_id=map_arguments["job_id"],
                    timeout_seconds=min(
                        map_arguments.get("timeout_seconds", McpServer.WAIT_SECONDS_DEFAULT),
                        McpServer.WAIT_SECONDS_MAX,
                    ),
                ),
            ),
            McpTool(
                name="marstrace_cancel",
                description="Ask a job to stop; it is charged what its attempts used up to the stop.",
                map_input_schema=McpServer._schema(
                    map_property={"job_id": map_job_id},
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow().cancel(map_arguments["job_id"]),
            ),
            McpTool(
                name="marstrace_result",
                description="A succeeded job's result manifest (quality, reason codes, device) and a "
                "download URL of its marstrace.json, valid for 10 minutes.",
                map_input_schema=McpServer._schema(
                    map_property={"job_id": map_job_id},
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow().result(map_arguments["job_id"]),
            ),
            McpTool(
                name="marstrace_fetch",
                description="Download a succeeded job's marstrace.json and its result manifest into a "
                "local directory.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "job_id": map_job_id,
                        "output": {"type": "string", "description": "marstrace-results/<job> by default"},
                    },
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: {
                    "files": self._workflow().fetch(
                        job_id=map_arguments["job_id"],
                        output_directory=map_arguments.get("output") or os.path.join(
                            MarsTraceWorkflow.RESULTS_DIRECTORY,
                            map_arguments["job_id"],
                        ),
                    ),
                },
            ),
            McpTool(
                name="marstrace_show",
                description="Read a fetched marstrace.json with the trace viewer; options are the "
                "'marstrace show' options, e.g. ['--stats'], ['--bubbles'], ['--json'], "
                "['--select', '<mtrace/1 token>']. Check quality and reason_codes in the "
                "result-manifest.json beside it first: an unusable or partial trace limits what it "
                "can support. Times are in the report's clock (ns, or cyc for an sm clock), and a "
                "warp-level run's rows are warps. With --json the answer is the JSON alone. "
                "marstrace_guide topic read says how to find the bubbles.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "trace": {"type": "string", "description": "the marstrace.json"},
                        "options": {"type": "array", "items": {"type": "string"}},
                    },
                    list_required=["trace"],
                ),
                handler=self._show,
            ),
        ]

    def _cubin_show(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """A cubin's target, digest and kernels."""
        binary = KernelBinary.load(map_arguments["binary"])
        return {
            "target": binary.target(),
            "sha256": binary.sha256,
            "kernels": binary.list_kernel,
        }

    @staticmethod
    def _check_accepted(
        cost_micro_usd: Any,
        accepted_usd: Any,
    ) -> None:
        """Refuse a cost at the bounds above what the user accepted."""
        cost = decimal.Decimal(str(cost_micro_usd))
        if cost > decimal.Decimal(str(accepted_usd)) * MarsTraceWorkflow.MICRO_PER_UNIT:
            raise MarsTraceError(
                f"the cost at the bounds is {MarsTraceWorkflow.usd(cost)}, above the accepted ${accepted_usd}; "
                "ask the user, or lower [agent] or [limits] max_attempts"
            )

    def _submit(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """Price, check the cost at the bounds against what the user accepted, and submit; a
        key the job directory already submitted under is submitted again as recorded."""
        idempotency_key = map_arguments.get("idempotency_key")
        if idempotency_key is not None:
            map_job = self._workflow().resubmit(
                directory=map_arguments["directory"],
                idempotency_key=idempotency_key,
            )
            if map_job is not None:
                return {
                    "estimate": None,
                    "job": map_job,
                }
        quoted = self._workflow().quote(
            directory=map_arguments["directory"],
            hardware=map_arguments.get("hardware"),
            cuda_version=map_arguments.get("cuda_version"),
        )
        self._workflow().check_admissible(quoted)
        McpServer._check_accepted(
            cost_micro_usd=quoted.map_quote["cost_at_bounds_micro_usd"]["total"],
            accepted_usd=map_arguments["accept_cost_at_bounds_usd"],
        )
        return {
            "estimate": quoted.summary(),
            "job": self._workflow().submit(
                quoted=quoted,
                idempotency_key=idempotency_key,
            ),
        }

    def _show(self, map_arguments: dict[str, Any]) -> str:
        """The viewer's output on a trace."""
        parser = argparse.ArgumentParser(prog="marstrace")
        ShowCommand().register(parser.add_subparsers())
        list_option = map_arguments.get("options", [])
        output = io.StringIO()
        notes = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(notes):
            try:
                args = parser.parse_args(["show", map_arguments["trace"], *list_option])
                args.func(args)
            except SystemExit as err:
                raise MarsTraceError(
                    f"marstrace show refused the options: {(notes.getvalue() + output.getvalue()).strip()}"
                ) from err
            except ValueError as err:
                raise MarsTraceError(f"marstrace show cannot read {map_arguments['trace']}: {err}") from err
        # a JSON answer is the JSON alone; otherwise the viewer's notes lead, as on a terminal
        if "--json" in list_option:
            return output.getvalue()
        return notes.getvalue() + output.getvalue()
