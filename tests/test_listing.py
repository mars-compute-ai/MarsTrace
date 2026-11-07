import json
import os
import tempfile
import unittest
from fake_control_plane import FIXTURE_DIRECTORY
from xtrace_harness.error import HarnessError
from xtrace_harness.listing import InspectionCache, KernelListing

FIXTURE = os.path.join(FIXTURE_DIRECTORY, "matmul_inspection.json")


class KernelListingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.listing = KernelListing.load(FIXTURE)

    def test_kernels_show_the_parameter_layout(self) -> None:
        text = self.listing.render_kernels()
        self.assertIn("architecture sm_90a", text)
        self.assertIn("* square_matrix_mul", text)
        self.assertIn("param 3   offset 24    size 4", text)

    def test_listing_marks_eligible_instructions(self) -> None:
        list_line = self.listing.render_listing(source="matmul.cu:8").splitlines()
        self.assertEqual(len(list_line), 3)
        self.assertRegex(list_line[0], r"^\s+5\s{3}B1")
        self.assertRegex(list_line[1], r"^\s+6 \+ B1")
        self.assertIn("LDG.E", self.listing.render_listing(opcode_pattern="LDG"))
        self.assertEqual(len(self.listing.render_listing(first_index=10, last_index=11).splitlines()), 2)

    def test_line_resolves_to_its_first_eligible_instruction(self) -> None:
        self.assertEqual(self.listing.resolve_line("matmul.cu:8"), 6)
        self.assertEqual(self.listing.resolve_line("cu:9"), 7)

    def test_kernel_list_only(self) -> None:
        with open(FIXTURE) as fixture_file:
            map_inspection = json.load(fixture_file)
        map_inspection["kernel"] = None
        listing = KernelListing.from_json(map_inspection=map_inspection, where="test")
        self.assertIn("  square_matrix_mul", listing.render_kernels())
        with self.assertRaisesRegex(HarnessError, "--kernel"):
            listing.render_listing()

    def test_other_schemas_are_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "is not a marscompute.xtrace.inspection.v1 result"):
            KernelListing.from_json(map_inspection={"schema": "xtrace.inspect/1"}, where="old.json")
        with self.assertRaisesRegex(HarnessError, "malformed"):
            KernelListing.from_json(
                map_inspection={"schema_version": KernelListing.SCHEMA, "kernels": []},
                where="partial.json",
            )


class InspectionCacheTest(unittest.TestCase):
    def test_key_is_digest_and_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "k.cubin")
            with open(FIXTURE, "rb") as fixture_file:
                data = fixture_file.read()
            path = InspectionCache.store(binary_path=binary, sha256="a" * 64, kernel_symbol="k", data=data)
            self.assertTrue(path.startswith(os.path.join(directory, ".xtrace-cache", "a" * 64)))
            self.assertIsNotNone(InspectionCache.load(binary_path=binary, sha256="a" * 64, kernel_symbol="k"))
            self.assertIsNone(InspectionCache.load(binary_path=binary, sha256="b" * 64, kernel_symbol="k"))
            self.assertIsNone(InspectionCache.load(binary_path=binary, sha256="a" * 64, kernel_symbol=None))


if __name__ == "__main__":
    unittest.main()
