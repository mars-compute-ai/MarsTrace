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
from xtrace_harness.listing import InspectionCache, KernelListing
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
        "Trace CUDA kernels on MarsCompute GPUs. Inspect the cubin (a paid inspection: estimate, "
        "get the user's yes to the fee, submit, wait, fetch the result), pick sites from xtrace_sass, "
        "estimate the job, get the user's yes to the cost cap, submit with accept_max_cost_usd, wait, "
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
        map_inspection_id = {
            "type": "string",
            "description": "the inspection id",
        }
        map_inspection_target = {
            "binary": {"type": "string", "description": "the cubin"},
            "kernel": {"type": "string", "description": "the kernel symbol; without it the kernel catalog only"},
            "profile_id": {"type": "string", "description": "an inspection profile; the first one by default"},
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
                name="xtrace_inspection_capabilities",
                description="The inspection profiles open to the account, with their limits and "
                "fixed fee per inspection, or why inspections are unavailable.",
                map_input_schema=McpServer._schema(
                    map_property={},
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow.inspection_capabilities(),
            ),
            McpTool(
                name="xtrace_inspection_estimate",
                description="Upload a cubin and quote its inspection: the profile and the fixed fee. "
                "Show the fee to the user before submitting.",
                map_input_schema=McpServer._schema(
                    map_property=map_inspection_target,
                    list_required=["binary"],
                ),
                handler=lambda map_arguments: self._workflow.quote_inspection(
                    binary_path=map_arguments["binary"],
                    kernel_symbol=map_arguments.get("kernel"),
                    profile_id=map_arguments.get("profile_id"),
                ).summary(),
            ),
            McpTool(
                name="xtrace_inspection_submit",
                description="Quote and accept an inspection of a cubin. Refused when the fee exceeds "
                "accept_max_fee_usd, the amount the user agreed to.",
                map_input_schema=McpServer._schema(
                    map_property={
                        **map_inspection_target,
                        "accept_max_fee_usd": {"type": "number"},
                        "idempotency_key": {
                            "type": "string",
                            "description": "reuse it when retrying an acceptance whose answer was lost",
                        },
                    },
                    list_required=[
                        "binary",
                        "accept_max_fee_usd",
                    ],
                ),
                handler=self._inspection_submit,
            ),
            McpTool(
                name="xtrace_inspection_get",
                description="One inspection's execution, cleanup and accounting.",
                map_input_schema=McpServer._schema(
                    map_property={"inspection_id": map_inspection_id},
                    list_required=["inspection_id"],
                ),
                handler=lambda map_arguments: self._workflow.inspection(map_arguments["inspection_id"]),
            ),
            McpTool(
                name="xtrace_inspection_wait",
                description="Wait for an inspection to end, up to timeout_seconds; returns its last status.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "inspection_id": map_inspection_id,
                        "timeout_seconds": {"type": "number"},
                    },
                    list_required=["inspection_id"],
                ),
                handler=lambda map_arguments: self._workflow.wait_inspection(
                    inspection_id=map_arguments["inspection_id"],
                    timeout_seconds=min(
                        map_arguments.get("timeout_seconds", McpServer.WAIT_SECONDS_DEFAULT),
                        McpServer.WAIT_SECONDS_MAX,
                    ),
                ),
            ),
            McpTool(
                name="xtrace_inspection_query",
                description="A page of the account's inspections, in inspection id order.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "limit": {"type": "integer"},
                        "cursor": {"type": "string"},
                    },
                    list_required=[],
                ),
                handler=lambda map_arguments: self._workflow.inspections(
                    limit=map_arguments.get("limit", McpServer.JOBS_LIMIT_DEFAULT),
                    cursor=map_arguments.get("cursor"),
                ),
            ),
            McpTool(
                name="xtrace_inspection_cancel",
                description="Ask an inspection to stop.",
                map_input_schema=McpServer._schema(
                    map_property={"inspection_id": map_inspection_id},
                    list_required=["inspection_id"],
                ),
                handler=lambda map_arguments: self._workflow.cancel_inspection(map_arguments["inspection_id"]),
            ),
            McpTool(
                name="xtrace_inspection_result",
                description="Fetch a completed inspection of a cubin, download its kernel's listing and "
                "cache both beside the cubin; returns the kernels, parameter layout and listing size.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "inspection_id": map_inspection_id,
                        "binary": {"type": "string", "description": "the cubin it inspected"},
                        "kernel": {"type": "string", "description": "the kernel whose listing to download"},
                    },
                    list_required=[
                        "inspection_id",
                        "binary",
                    ],
                ),
                handler=self._inspection_result,
            ),
            McpTool(
                name="xtrace_inspection_downloads",
                description="Download artifacts an inspection result names into a local directory.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "inspection_id": map_inspection_id,
                        "artifact_ids": {"type": "array", "items": {"type": "string"}},
                        "output": {"type": "string", "description": "xtrace-inspections/<inspection> by default"},
                    },
                    list_required=[
                        "inspection_id",
                        "artifact_ids",
                    ],
                ),
                handler=lambda map_arguments: {
                    "files": self._workflow.download_inspection(
                        inspection_id=map_arguments["inspection_id"],
                        list_artifact_id=map_arguments["artifact_ids"],
                        output_directory=map_arguments.get("output") or os.path.join(
                            XtraceWorkflow.INSPECTIONS_DIRECTORY,
                            map_arguments["inspection_id"],
                        ),
                    ),
                },
            ),
            McpTool(
                name="xtrace_sass",
                description="A kernel's cached SASS listing, narrowed by regex or index range. "
                "Inspect the cubin first with xtrace_inspection_submit and xtrace_inspection_result.",
                map_input_schema=McpServer._schema(
                    map_property={
                        "listing": {"type": "string", "description": "a cubin or a cached .inspection-result.json"},
                        "kernel": {"type": "string"},
                        "kernels": {"type": "boolean", "description": "list kernels and parameters"},
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
                description="Upload a job directory, check its sites and quote it: the cost cap "
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

    @staticmethod
    def _check_accepted(
        cap_micro_usd: Any,
        accepted_usd: Any,
        what: str,
        remedy: str,
    ) -> None:
        """Refuse an amount above what the user accepted."""
        cap = decimal.Decimal(str(cap_micro_usd))
        if cap > decimal.Decimal(str(accepted_usd)) * XtraceWorkflow.MICRO_PER_UNIT:
            raise HarnessError(
                f"the quote's {what} is {XtraceWorkflow.usd(cap)}, above the accepted ${accepted_usd}; {remedy}"
            )

    def _inspection_submit(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """Quote an inspection, check its fee against what the user accepted, and accept it."""
        quoted = self._workflow.quote_inspection(
            binary_path=map_arguments["binary"],
            kernel_symbol=map_arguments.get("kernel"),
            profile_id=map_arguments.get("profile_id"),
        )
        McpServer._check_accepted(
            cap_micro_usd=quoted.map_quote["price"]["fixed_fee_micro_usd"],
            accepted_usd=map_arguments["accept_max_fee_usd"],
            what="fee",
            remedy="ask the user",
        )
        return {
            "quote": quoted.summary(),
            "inspection": self._workflow.submit_inspection(
                quoted=quoted,
                idempotency_key=map_arguments.get("idempotency_key"),
            ),
        }

    def _inspection_result(self, map_arguments: dict[str, Any]) -> dict[str, Any]:
        """Fetch an inspection into the cache and summarize it."""
        path = self._workflow.fetch_inspection(
            inspection_id=map_arguments["inspection_id"],
            binary_path=map_arguments["binary"],
            kernel_symbol=map_arguments.get("kernel"),
        )
        listing = KernelListing.load(path)
        return {
            "listing": path,
            "kernels": {
                symbol: {
                    "architecture": kernel.architecture,
                    "instructions": kernel.instruction_count,
                    "is_listing_complete": kernel.is_listing_complete,
                    "parameters": (
                        None
                        if kernel.list_parameter is None
                        else [
                            {
                                "index": parameter.index,
                                "offset_bytes": parameter.offset_bytes,
                                "size_bytes": parameter.size_bytes,
                            }
                            for parameter in kernel.list_parameter
                        ]
                    ),
                }
                for symbol, kernel in listing.map_kernel.items()
            },
            "listed_kernel": None if listing.kernel is None else listing.kernel.symbol,
        }

    def _sass(self, map_arguments: dict[str, Any]) -> str:
        """A narrowed SASS listing from the cache."""
        path = map_arguments["listing"]
        if not path.endswith(".json"):
            path = InspectionCache.path_of(
                binary_path=path,
                sha256=KernelBinary.load(path).sha256,
                kernel_symbol=map_arguments.get("kernel"),
            )
            if not os.path.isfile(path):
                raise HarnessError(
                    f"no inspection of {map_arguments.get('kernel') or 'the kernel catalog'} is cached "
                    f"beside {map_arguments['listing']}; run xtrace_inspection_submit and "
                    "xtrace_inspection_result first"
                )
        listing = KernelListing.load(path)
        if map_arguments.get("kernels"):
            return listing.render_kernels()
        return listing.render_listing(
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
        McpServer._check_accepted(
            cap_micro_usd=quoted.map_quote["limits"]["max_total_cost_micro_usd"],
            accepted_usd=map_arguments["accept_max_cost_usd"],
            what="cost cap",
            remedy="ask the user, or lower [limits] max_cost_usd",
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
