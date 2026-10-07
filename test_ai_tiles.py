import unittest
from ai_restore import positions


class TileCoverageTests(unittest.TestCase):
    def test_coverage_and_overlap_at_non_multiple_edges(self):
        for length in [1, 82, 256, 257, 511, 700]:
            starts = positions(length, 256, 64)
            covered = set()
            for start in starts:
                covered.update(range(start, min(start + 256, length)))
            self.assertEqual(covered, set(range(length)))
            self.assertEqual(len(starts), len(set(starts)))
            self.assertEqual(starts[0], 0)
            self.assertEqual(starts[-1], max(0, length - 256))
            for left, right in zip(starts, starts[1:]):
                self.assertLessEqual(right - left, 192)


if __name__ == '__main__':
    unittest.main()
