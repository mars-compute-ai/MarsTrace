import contextlib
import io
import json
import os
import unittest
from marstrace.cli import CommandLine
from marstrace.viewer.common.trace.format.panel import SelectionToken

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "trace_sass_regions.json",
)


def run_show(*list_option: str) -> str:
    """The output of ``marstrace show`` on the fixture with the given options."""
    return run_show_streams(*list_option)[0]


def run_show_streams(*list_option: str) -> tuple[str, str]:
    """What ``marstrace show`` on the fixture writes to stdout and to stderr."""
    output = io.StringIO()
    errors = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
        status = CommandLine.run(["show", FIXTURE, *list_option])
    if status != 0:
        raise AssertionError(f"marstrace show exited with {status}: {errors.getvalue()}")
    return output.getvalue(), errors.getvalue()


class ViewerTest(unittest.TestCase):
    def test_region_stats(self) -> None:
        text = run_show("--stats")
        self.assertIn("mainloop", text)
        self.assertIn("epilogue", text)

    def test_bubbles(self) -> None:
        self.assertIn("bubble ratio", run_show("--bubbles"))

    def test_json_scopes(self) -> None:
        map_view = json.loads(run_show("--json"))
        self.assertEqual(
            sorted(map_stat["scope_label"] for map_stat in map_view["scope_stats"]),
            ["epilogue", "mainloop"],
        )

    def test_selection_with_json_keeps_stdout_to_json(self) -> None:
        kernel = json.loads(run_show("--json"))["kernel_prototype"]
        token = (
            f"mtrace/1 kernel={SelectionToken.kernel_tag(kernel)} "
            f"t=0:1000000000ns lane=block:0 order=thread scopes=mainloop"
        )
        output, errors = run_show_streams("--select", token, "--json")
        self.assertIn(
            "mainloop", [map_stat["scope_label"] for map_stat in json.loads(output)["scope_stats"]]
        )
        self.assertIn("Selection:", errors)

    def test_block_filter(self) -> None:
        self.assertIn("mainloop", run_show("--block", "1", "--stats"))


if __name__ == "__main__":
    unittest.main()
