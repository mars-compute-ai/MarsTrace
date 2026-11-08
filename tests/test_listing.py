import hashlib
import json
import os
import tempfile
import unittest
from fake_control_plane import matmul_documents
from schema_check import SchemaCheck
from xtrace_harness.client import MarsComputeClient
from xtrace_harness.error import HarnessError
from xtrace_harness.listing import InspectionCache, KernelListing


class KernelListingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result, self.instructions, _ = matmul_documents()
        self.listing = KernelListing.from_documents(self.result, self.instructions, "test")

    def test_the_fixture_follows_the_schema(self) -> None:
        SchemaCheck.validate("inspection-result", self.result)

    def test_kernels_show_the_parameter_layout(self) -> None:
        text = self.listing.render_kernels()
        self.assertIn("inspection 0c9d2a4e-1b3f-4e5a-8c6d-7e8f9a0b1c2d", text)
        self.assertIn("* square_matrix_mul  (sm_90a, 216 instructions)", text)
        self.assertIn("param 3   offset 24    size 4", text)
        self.assertNotIn("registers", text)

    def test_listing_shows_index_address_and_predicate(self) -> None:
        list_line = self.listing.render_listing(first_index=12, last_index=13).splitlines()
        self.assertEqual(list_line, ["    12  0x0000c0  @P0 EXIT", "    13  0x0000d0  ISETP.GE.AND P0, PT, R0, 0x1, PT"])
        self.assertIn("STG.E desc[UR6][R2.64], R17", self.listing.render_listing(opcode_pattern="STG"))
        self.assertEqual(len(self.listing.render_listing(opcode_pattern="^@!P0 BRA").splitlines()), 5)

    def test_index_must_be_listed(self) -> None:
        self.listing.check_index(199)
        with self.assertRaisesRegex(HarnessError, "site index 216 is not an instruction of square_matrix_mul"):
            self.listing.check_index(216)

    def test_unknown_layout_is_kept_unknown(self) -> None:
        self.result["kernel_catalog"]["kernels"][0]["parameter_layout"] = {"state": "unknown", "reason": "parameter_metadata_absent"}
        listing = KernelListing.from_documents(self.result, None, "test")
        self.assertIsNone(listing.map_kernel["square_matrix_mul"].list_parameter)
        self.assertIn("parameter layout unknown: parameter_metadata_absent", listing.render_kernels())

    def test_kernel_catalog_only(self) -> None:
        listing = KernelListing.from_documents(self.result, None, "test")
        self.assertIn("  square_matrix_mul", listing.render_kernels())
        with self.assertRaisesRegex(HarnessError, "--kernel"):
            listing.render_listing()

    def test_other_documents_are_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "is not a marscompute.xtrace.inspection-result.v1 document"):
            KernelListing.from_documents({"schema_version": "marscompute.xtrace.inspection.v1"}, None, "old.json")
        with self.assertRaisesRegex(HarnessError, "malformed"):
            KernelListing.from_documents({"schema_version": KernelListing.RESULT_SCHEMA}, None, "partial.json")
        with self.assertRaisesRegex(HarnessError, "names other_kernel, which the result lacks"):
            KernelListing.from_documents(self.result, dict(self.instructions, symbol="other_kernel"), "test")


class InspectionCacheTest(unittest.TestCase):
    def test_key_is_digest_and_kernel(self) -> None:
        result, _, listing = matmul_documents()
        with tempfile.TemporaryDirectory() as directory:
            binary = os.path.join(directory, "k.cubin")
            path = InspectionCache.store(
                binary_path=binary,
                sha256="a" * 64,
                kernel_symbol="k",
                result_data=json.dumps(result).encode(),
                instructions_data=listing,
            )
            self.assertTrue(path.startswith(os.path.join(directory, ".xtrace-cache", "a" * 64)))
            self.assertTrue(path.endswith(".inspection-result.json"))
            self.assertTrue(os.path.isfile(InspectionCache.instructions_path_of(path)))
            self.assertEqual(InspectionCache.load(binary_path=binary, sha256="a" * 64, kernel_symbol="k").kernel.symbol, "square_matrix_mul")
            self.assertIsNone(InspectionCache.load(binary_path=binary, sha256="b" * 64, kernel_symbol="k"))
            self.assertIsNone(InspectionCache.load(binary_path=binary, sha256="a" * 64, kernel_symbol=None))


class CanonicalDigestTest(unittest.TestCase):
    def test_keys_sort_and_text_stays_unescaped(self) -> None:
        value = {"b": [1, True, None], "a": {"é": "x\"y"}, "n": 2.0}
        self.assertEqual(MarsComputeClient.canonical_json(value), '{"a":{"é":"x\\"y"},"b":[1,true,null],"n":2}')
        self.assertEqual(
            MarsComputeClient.canonical_digest(value),
            hashlib.sha256('{"a":{"é":"x\\"y"},"b":[1,true,null],"n":2}'.encode()).hexdigest(),
        )

    def test_fractions_are_refused(self) -> None:
        with self.assertRaisesRegex(HarnessError, "no canonical form"):
            MarsComputeClient.canonical_json({"atol": 0.5})


if __name__ == "__main__":
    unittest.main()
