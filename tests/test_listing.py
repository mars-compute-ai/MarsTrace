import json
import os
import tempfile
import unittest
from xtrace_harness.listing import KernelListing

INSPECT_RESULT = {
    "schema": "xtrace.inspect/1",
    "kernels": [
        {"name": "other_kernel", "matched": False},
        {
            "name": "square_matrix_mul",
            "matched": True,
            "registers": 16,
            "static_shared_bytes": 0,
            "params": [
                {"index": 0, "offset": 0, "size": 8},
                {"index": 1, "offset": 8, "size": 8},
                {"index": 2, "offset": 16, "size": 8},
                {"index": 3, "offset": 24, "size": 4},
            ],
            "instructions": [
                {"index": 0, "pc": 0, "text": "LDC R1, c[0x0][0x28]", "block": 0},
                {"index": 7, "pc": 112, "text": "IMAD R4, R3, R2, RZ", "file": "matmul.cu", "line": 8, "block": 1},
                {"index": 9, "pc": 144, "text": "LDG.E R6, desc[UR4][R4.64]", "file": "matmul.cu", "line": 9, "block": 2},
            ],
        },
    ],
}


class KernelListingTest(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as result_file:
            json.dump(INSPECT_RESULT, result_file)
        self.addCleanup(os.unlink, result_file.name)
        self.listing = KernelListing.load(result_file.name)

    def test_kernels_show_the_parameter_layout(self) -> None:
        text = self.listing.render_kernels()
        self.assertIn("* square_matrix_mul", text)
        self.assertIn("param 3   offset 24    size 4", text)
        self.assertNotIn("registers", text.split("\n")[0])

    def test_listing_narrows_by_line_and_opcode(self) -> None:
        self.assertEqual(len(self.listing.render_listing(source="matmul.cu:8").splitlines()), 1)
        self.assertIn("LDG.E", self.listing.render_listing(opcode_pattern="LDG"))
        self.assertEqual(self.listing.render_listing(first_index=8), self.listing.render_listing(opcode_pattern="LDG"))


if __name__ == "__main__":
    unittest.main()
