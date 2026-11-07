import argparse
import contextlib
import decimal
import io
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO
from xtrace_harness.binary import DslCache, KernelBinary
from xtrace_harness.error import HarnessError, PlatformError, ToolArgumentError
from xtrace_harness.listing import KernelListing
from xtrace_harness.viewer.cli.show import ShowCommand
from xtrace_harness.workflow import XtraceWorkflow


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
    ``xtrace`` operations on local paths, so an agent can trace a kernel through MCP alone.

    Args:
        workflow: the operations the tools run
    """

    JSONRPC_VERSION = "2.0"
    LIST_PROTOCOL_VERSION = (
        "2025-03-26",
        "2025-06-18",
        "2025-11-25",
    )
    SERVER_NAME = "xtrace-harness"
    SERVER_VERSION = "0.1.0"
    INSTRUCTIONS = (
        "Trace CUDA kernels on MarsCompute GPUs. Inspect the cubin, pick sites from xtrace_sass, "
        "estimate, get the user's yes to the cost cap, submit with accept_max_cost_usd, wait, "
        "fetch, then read the trace with xtrace_show."
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

    def __init__(self, workflow: XtraceWorkflow) -> None:
        self._workflow = workflow
        self._map_tool = {tool.name: tool for tool in self._tools()}

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
        """Run a tool; a refusal of the harness or the platform is a tool result with isError."""
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
        except PlatformError as err:
            return McpServer._result(
                request_id=request_id,
                value=McpServer._content(
                    output={"error": err.to_json()},
                    is_error=True,
                ),
            )
        except (
            HarnessError,
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
                    + "\n[cut; narrow the request with line, grep, first or last]"
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
            "description": "the job directory, holding xtrace.toml",
        }
        map_hardware = {
            "type": "string",
            "description": "an environment id overriding the manifest's hardware",
        }
        map_job_id = {
            "type": "string",
            "description": "the job id",
        }
        return [
            McpTool(
                name="xtrace_hardware",
                description="List the hardware a job can run on: environment id, GPU, architecture, "
                "rate per GPU-hour, stages; and the planner models and default limits.",
                map_input_schema=McpServer._schema(
                    map_property={},
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow.hardware(),
            ),
            McpTool(
                name="xtrace_credit",
                description="The account's available and reserved credit, in micro-USD, and the "
                "page where credit is added.",
                map_input_schema=McpServer._schema(
                    map_property={},
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow.credit(),
            ),
            McpTool(
                name="xtrace_cubin_find",
                description="Find a kernel's cubin in the Triton, torch.compile and TileLang caches "
                "and given directories, newest first; optionally copy one into a job directory.",
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
                name="xtrace_cubin_show",
                description="A cubin's target, digest and kernel symbols.",
                map_input_schema=McpServer._schema(
                    map_property={"binary": {"type": "string", "description": "the cubin"}},
                    list_required=["binary"],
                ),
                handler=self._cubin_show,
            ),
            McpTool(
                name="xtrace_inspect",
                description="Inspect a cubin's kernel on the platform (free) and cache the listing "
                "beside it; returns the kernels, registers and parameter layout.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "binary": {"type": "string", "description": "the cubin"},
                        "kernel": {"type": "string", "description": "the kernel symbol"},
                        "refresh": {"type": "boolean"},
                    },
                    list_required=["binary"],
                ),
                handler=self._inspect,
            ),
            McpTool(
                name="xtrace_sass",
                description="A kernel's SASS listing (cached, or inspected on demand), narrowed by "
                "source line, regex or index range. '+' marks instructions that take a probe.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "listing": {"type": "string", "description": "a cubin or an inspection.json"},
                        "kernel": {"type": "string"},
                        "kernels": {"type": "boolean", "description": "list kernels and parameters"},
                        "line": {"type": "string", "description": "<file>:<line> or <line>"},
                        "grep": {"type": "string"},
                        "first": {"type": "integer"},
                        "last": {"type": "integer"},
                    },
                    list_required=["listing"],
                ),
                handler=self._sass,
            ),
            McpTool(
                name="xtrace_estimate",
                description="Upload a job directory, resolve its sites and quote it: the cost cap "
                "and the GPU rate. Show the cap to the user before submitting.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "directory": map_directory,
                        "hardware": map_hardware,
                    },
                    list_required=["directory"],
                ),
                handler=lambda map_arguments: self._workflow.quote(
                    directory=map_arguments["directory"],
                    hardware=map_arguments.get("hardware"),
                ).summary(),
            ),
            McpTool(
                name="xtrace_submit",
                description="Quote and submit a job directory. Refused when the quote's cost cap "
                "exceeds accept_max_cost_usd, the amount the user agreed to.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "directory": map_directory,
                        "accept_max_cost_usd": {"type": "number"},
                        "hardware": map_hardware,
                        "idempotency_key": {
                            "type": "string",
                            "description": "reuse it when retrying a submission whose answer was lost",
                        },
                    },
                    list_required=[
                        "directory",
                        "accept_max_cost_usd",
                    ],
                ),
                handler=self._submit,
            ),
            McpTool(
                name="xtrace_jobs",
                description="A page of the account's jobs, newest first.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "limit": {"type": "integer"},
                        "cursor": {"type": "string"},
                    },
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow.jobs(
                    limit=map_arguments.get("limit", McpServer.JOBS_LIMIT_DEFAULT),
                    cursor=map_arguments.get("cursor"),
                ),
            ),
            McpTool(
                name="xtrace_job",
                description="One job's state, attempts, accounting and result manifest.",
                map_input_schema=McpServer._schema(
                    map_property={"job_id": map_job_id},
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow.job(map_arguments["job_id"]),
            ),
            McpTool(
                name="xtrace_wait",
                description="Wait for a job to end, up to timeout_seconds; returns its last view.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "job_id": map_job_id,
                        "timeout_seconds": {"type": "number"},
                    },
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow.wait(
                    job_id=map_arguments["job_id"],
                    timeout_seconds=min(
                        map_arguments.get("timeout_seconds", McpServer.WAIT_SECONDS_DEFAULT),
                        McpServer.WAIT_SECONDS_MAX,
                    ),
                ),
            ),
            McpTool(
                name="xtrace_cancel",
                description="Ask a job to stop and release its resources.",
                map_input_schema=McpServer._schema(
                    map_property={"job_id": map_job_id},
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: self._workflow.cancel(map_arguments["job_id"]),
            ),
            McpTool(
                name="xtrace_fetch",
                description="Download a finished job's result files into a local directory.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "job_id": map_job_id,
                        "output": {"type": "string", "description": "xtrace-results/<job> by default"},
                    },
                    list_required=["job_id"],
                ),
                handler=lambda map_arguments: {
                    "files": self._workflow.fetch(
                        job_id=map_arguments["job_id"],
                        output_directory=map_arguments.get("output") or os.path.join(
                            XtraceWorkflow.RESULTS_DIRECTORY,
                            map_arguments["job_id"],
                        ),
                    ),
                },
            ),
            McpTool(
                name="xtrace_show",
                description="Read a fetched xtrace.json with the trace viewer; options are the "
                "'xtrace show' options, e.g. ['--stats'], ['--bubbles'], ['--json'].",
                map_input_schema=McpServer._schema(
                    map_property={
                        "trace": {"type": "string", "description": "the xtrace.json"},
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

    def _inspect(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """Inspect a cubin and summarize the listing."""
        path = self._workflow.inspect(
            binary_path=map_arguments["binary"],
            kernel_symbol=map_arguments.get("kernel"),
            do_refresh=map_arguments.get("refresh", False),
        )
        listing = KernelListing.load(path)
        map_summary: dict[str, Any] = {
            "listing": path,
            "architecture": listing.architecture,
            "kernels": listing.map_kernel_name,
            "kernel": None,
        }
        if listing.kernel is not None:
            map_summary["kernel"] = {
                "symbol": listing.kernel.symbol,
                "registers": listing.kernel.registers,
                "static_shared_bytes": listing.kernel.static_shared_bytes,
                "parameters": [
                    {
                        "index": parameter.index,
                        "offset_bytes": parameter.offset_bytes,
                        "size_bytes": parameter.size_bytes,
                    }
                    for parameter in listing.kernel.list_parameter
                ],
                "instructions": len(listing.kernel.list_instruction),
                "eligible_sites": sum(
                    instruction.is_eligible for instruction in listing.kernel.list_instruction
                ),
            }
        return map_summary

    def _sass(self, map_arguments: dict[str, Any]) -> str:
        """A narrowed SASS listing."""
        path = map_arguments["listing"]
        if not path.endswith(".json"):
            path = self._workflow.inspect(
                binary_path=path,
                kernel_symbol=map_arguments.get("kernel"),
            )
        listing = KernelListing.load(path)
        if map_arguments.get("kernels"):
            return listing.render_kernels()
        return listing.render_listing(
            source=map_arguments.get("line"),
            opcode_pattern=map_arguments.get("grep"),
            first_index=map_arguments.get("first"),
            last_index=map_arguments.get("last"),
        )

    def _submit(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """Quote, check the cap against what the user accepted, and submit."""
        quoted = self._workflow.quote(
            directory=map_arguments["directory"],
            hardware=map_arguments.get("hardware"),
        )
        cap = decimal.Decimal(quoted.map_quote["limits"]["max_total_cost_micro_usd"])
        accepted = decimal.Decimal(str(map_arguments["accept_max_cost_usd"])) * XtraceWorkflow.MICRO_PER_UNIT
        if cap > accepted:
            raise HarnessError(
                f"the quote's cost cap is {XtraceWorkflow.usd(cap)}, above the accepted "
                f"${map_arguments['accept_max_cost_usd']}; ask the user, or lower [limits] max_cost_usd"
            )
        return {
            "quote": quoted.summary(),
            "job": self._workflow.submit(
                quoted=quoted,
                idempotency_key=map_arguments.get("idempotency_key"),
            ),
        }

    def _show(self, map_arguments: dict[str, Any]) -> str:
        """The viewer's output on a trace."""
        parser = argparse.ArgumentParser(prog="xtrace")
        ShowCommand().register(parser.add_subparsers())
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                args = parser.parse_args(["show", map_arguments["trace"], *map_arguments.get("options", [])])
                args.func(args)
            except SystemExit as err:
                raise HarnessError(f"xtrace show refused the options: {output.getvalue().strip()}") from err
            except ValueError as err:
                raise HarnessError(f"xtrace show cannot read {map_arguments['trace']}: {err}") from err
        return output.getvalue()
