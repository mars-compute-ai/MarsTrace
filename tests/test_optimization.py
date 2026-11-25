import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from fake_service import FIXTURE_DIRECTORY
from marstrace.error import ServiceError
from marstrace.optimization import DeliveredOptimization


def load_section() -> dict:
    """The fixture's optimization section."""
    with open(os.path.join(FIXTURE_DIRECTORY, "optimization_section.json")) as section_file:
        return json.load(section_file)


def with_file(section: dict, path: str, text: str) -> dict:
    """The section with one more deliverable whose digest and size match its text."""
    section = copy.deepcopy(section)
    data = text.encode("utf-8")
    section["data"]["list_file"].append(
        {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "media_type": "text/plain", "text": text}
    )
    return section


class DeliveredOptimizationTest(unittest.TestCase):
    """The optimization section of marstrace.json, checked, written and read for a person."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)

    def optimization(self, section: dict) -> DeliveredOptimization:
        optimization = DeliveredOptimization.from_document({"sections": [{"type": "marscompute.other", "data": {}}, section]})
        self.assertIsNotNone(optimization)
        return optimization

    def test_every_deliverable_is_written_under_optimization(self) -> None:
        optimization = self.optimization(load_section())
        list_path = optimization.write_files(self.directory)
        self.assertEqual(len(list_path), 7)
        directory = os.path.join(self.directory, "optimization")
        self.assertEqual(list_path[3], os.path.join(directory, "optimized.patch"))
        with open(os.path.join(directory, "optimized.patch"), encoding="utf-8") as patch_file:
            self.assertTrue(patch_file.read().startswith("diff --git a/matmul.cu b/matmul.cu"))
        self.assertTrue(os.path.isfile(os.path.join(directory, "iterations", "iter_001_shared_tiles", "post_bench.json")))
        text = optimization.text(directory)
        self.assertIn("optimization: improved, correctness passed, best speedup 1.41x", text)
        self.assertRegex(text, r"n2048\s+6400.0\s+4400.0\s+1.46x\s+-\s+yes")
        self.assertIn(f"patch     {os.path.join(directory, 'optimized.patch')}", text)
        self.assertIn(f"reports   {os.path.join(directory, 'performance_report.md')}", text)
        self.assertIn(os.path.join(directory, "optimization_process_report.md"), text)
        self.assertIn("reverted  iter_002_unroll_inner 0.98x", text)
        self.assertIn("iteration 3 file(s) under", text)
        self.assertEqual({key for map_file in optimization.to_json()["list_file"] for key in map_file}, {"path", "sha256", "size_bytes", "media_type"})

    def test_a_deliverable_whose_bytes_differ_from_its_digest_is_refused(self) -> None:
        for key, value in (("text", "# Performance report\n\nchanged\n"), ("size_bytes", 1), ("sha256", "0" * 64)):
            section = load_section()
            section["data"]["list_file"][0][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ServiceError, "performance_report.md with different bytes"):
                self.optimization(section).write_files(self.directory)
        self.assertFalse(os.path.exists(os.path.join(self.directory, "optimization")))

    def test_a_path_out_of_the_directory_is_refused_before_anything_is_written(self) -> None:
        for path in ("../escape.md", "iterations/../../escape.md", "/etc/escape.md", "a..b.md", "reports/", "./x.md", "a//b.md", "a b.md", "x" * 201, ""):
            with self.subTest(path=path), self.assertRaisesRegex(ServiceError, "unsafe path"):
                self.optimization(with_file(load_section(), path, "escape\n")).write_files(self.directory)
        self.assertEqual(os.listdir(self.directory), [])
        with self.assertRaisesRegex(ServiceError, "optimized.patch twice"):
            self.optimization(with_file(load_section(), "optimized.patch", "again\n")).write_files(self.directory)

    def test_a_verified_run_without_gain_says_the_job_is_complete(self) -> None:
        section = load_section()
        section["data"]["summary"].update(status="no_gain", correctness="passed")
        self.assertIn("no faster version found", self.optimization(section).text())
        section["data"]["summary"].update(status="improved")
        self.assertNotIn("no faster version found", self.optimization(section).text())

    def test_a_report_without_the_section_has_no_optimization(self) -> None:
        self.assertIsNone(DeliveredOptimization.from_document({"sections": []}))
        self.assertIsNone(DeliveredOptimization.from_document([]))
        self.assertIsNone(DeliveredOptimization.from_file(os.path.join(self.directory, "absent.json")))
        optimization = self.optimization({"type": "marscompute.optimization", "data": {"status": "incomplete", "correctness": "failed", "best_speedup": None}})
        self.assertEqual(optimization.write_files(self.directory), [])
        self.assertIn("incomplete, correctness failed, best speedup none yet", optimization.text(self.directory))
        self.assertIn("patch     none delivered", optimization.text(self.directory))


if __name__ == "__main__":
    unittest.main()
