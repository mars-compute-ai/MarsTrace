import importlib.util
import json
import os
import tempfile
import unittest
from unittest import mock
import jsonschema
from schema_check import SchemaCheck

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "sync_schemas.py",
)


def load_script() -> object:
    """The sync script as a module."""
    spec = importlib.util.spec_from_file_location("sync_schemas", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: str, value: object) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(value, json_file)


class SchemaSyncTest(unittest.TestCase):
    def test_sync_then_check(self) -> None:
        schema_sync_class = load_script().SchemaSync
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "schemas")
            write_json(os.path.join(source, "quote.schema.json"), {"type": "object"})
            write_json(os.path.join(source, "job.schema.json"), {"type": "object"})
            schema_sync = schema_sync_class(source_directory=source, target_directory=target)
            self.assertEqual(schema_sync.list_stale(), ["job.schema.json", "quote.schema.json"])
            self.assertEqual(schema_sync.sync(), ["job.schema.json", "quote.schema.json"])
            self.assertEqual(schema_sync.list_stale(), [])
            write_json(os.path.join(source, "quote.schema.json"), {"type": "array"})
            os.remove(os.path.join(source, "job.schema.json"))
            self.assertEqual(schema_sync.list_stale(), ["job.schema.json", "quote.schema.json"])
            schema_sync.sync()
            self.assertEqual(sorted(os.listdir(target)), ["quote.schema.json"])


class SchemaCheckTest(unittest.TestCase):
    def test_validates_by_name_when_vendored(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(SchemaCheck, "directory", directory):
            SchemaCheck.validate_request("POST", "/v1/xtrace/quotes", b"{}")
            write_json(
                os.path.join(directory, "estimate-input.schema.json"),
                {"type": "object", "required": ["prompt"]},
            )
            with self.assertRaises(jsonschema.ValidationError):
                SchemaCheck.validate_request("POST", "/v1/xtrace/quotes", b"{}")
            SchemaCheck.validate_request("POST", "/v1/xtrace/quotes", b'{"prompt": "x"}')
            SchemaCheck.validate_request("GET", "/v1/xtrace/quotes", b"")


if __name__ == "__main__":
    unittest.main()
