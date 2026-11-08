import json
import os
import re
from typing import Any
import jsonschema


class SchemaCheck:
    """Validates request bodies against the platform's vendored JSON Schemas, by schema name.

    A schema that is not vendored yet is skipped, so the check grows with the vendored set.
    """

    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "xtrace_harness",
        "schemas",
    )
    SUFFIX = ".schema.json"
    # (method, path pattern) of a request -> the schema its body follows
    LIST_REQUEST_SCHEMA = [
        ("POST", re.compile(r"^/v1/xtrace/artifacts/uploads$"), "upload-begin"),
        ("POST", re.compile(r"^/v1/xtrace/artifacts/[^/]+/complete$"), "upload-complete"),
        ("POST", re.compile(r"^/v1/xtrace/quotes$"), "estimate-input"),
        ("POST", re.compile(r"^/v1/xtrace/jobs$"), "submit-input"),
        ("POST", re.compile(r"^/v1/xtrace/inspections/quotes$"), "inspection-request"),
        ("POST", re.compile(r"^/v1/xtrace/inspections$"), "inspection-accept"),
        ("POST", re.compile(r"^/v1/xtrace/inspections/[^/]+/artifacts/downloads$"), "inspection-downloads-request"),
    ]
    # (method, path pattern) of a response -> the schema its body follows
    LIST_RESPONSE_SCHEMA = [
        ("GET", re.compile(r"^/v1/xtrace/capabilities$"), "capabilities"),
        ("POST", re.compile(r"^/v1/xtrace/quotes$"), "quote"),
        ("GET", re.compile(r"^/v1/xtrace/inspections/capabilities$"), "inspection-capabilities"),
        ("POST", re.compile(r"^/v1/xtrace/inspections/quotes$"), "inspection-quote"),
        ("POST", re.compile(r"^/v1/xtrace/inspections$"), "inspection"),
        ("GET", re.compile(r"^/v1/xtrace/inspections/[0-9a-f-]{36}$"), "inspection"),
        ("POST", re.compile(r"^/v1/xtrace/inspections/[^/]+/cancel$"), "inspection"),
        ("GET", re.compile(r"^/v1/xtrace/inspections/[^/]+/result$"), "inspection-result"),
        ("POST", re.compile(r"^/v1/xtrace/inspections/[^/]+/artifacts/downloads$"), "inspection-downloads"),
    ]
    LAUNCH_MANIFEST = "launch-manifest"

    @classmethod
    def schema_of(cls, name: str) -> dict[str, Any] | None:
        """The vendored schema of a name, or None when it is not vendored."""
        path = os.path.join(
            cls.directory,
            name + SchemaCheck.SUFFIX,
        )
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as schema_file:
            return json.load(schema_file)

    @classmethod
    def validate(
        cls,
        name: str,
        value: Any,
    ) -> None:
        """Validate a value against a named schema when it is vendored; raise when it fails."""
        map_schema = cls.schema_of(name)
        if map_schema is None:
            return
        jsonschema.validators.validator_for(map_schema)(map_schema).validate(value)

    @classmethod
    def validate_response(
        cls,
        method: str,
        path: str,
        value: Any,
    ) -> None:
        """Validate a successful response body against the schema of its route, when it has one."""
        for route_method, pattern, name in SchemaCheck.LIST_RESPONSE_SCHEMA:
            if (
                route_method == method
                and pattern.match(path)
            ):
                cls.validate(
                    name,
                    value,
                )
                return

    @classmethod
    def validate_request(
        cls,
        method: str,
        path: str,
        body: bytes,
    ) -> None:
        """Validate a request body against the schema of its route, when the route has one."""
        for route_method, pattern, name in SchemaCheck.LIST_REQUEST_SCHEMA:
            if (
                route_method == method
                and pattern.match(path)
            ):
                cls.validate(
                    name,
                    json.loads(body),
                )
                return
