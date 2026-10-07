import unittest
from ai_restore import positions, orient
import numpy as np


class TileCoverageTests(unittest.TestCase):
    def test_clockwise_rotation_moves_pixel_and_returns_contiguous(self):
        image = np.zeros((2, 3, 3), np.uint8)
        image[0, 0] = [10, 20, 30]
        result = orient(image, 90)
        self.assertEqual(result.shape, (3, 2, 3))
        np.testing.assert_array_equal(result[0, 1], [10, 20, 30])
        np.testing.assert_array_equal(orient(orient(image, 180), 180), image)
        self.assertTrue(result.flags.c_contiguous)

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
