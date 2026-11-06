import contextlib
import io
import json
import os
import unittest
from xtrace_harness.cli import CommandLine

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "trace_sass_regions.json",
)


def run_show(*list_option: str) -> str:
    """The output of ``xtrace show`` on the fixture with the given options."""
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = CommandLine.run(["show", FIXTURE, *list_option])
    if status != 0:
        raise AssertionError(f"xtrace show exited with {status}")
    return output.getvalue()


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

    def test_block_filter(self) -> None:
        self.assertIn("mainloop", run_show("--block", "1", "--stats"))


if __name__ == "__main__":
    unittest.main()
