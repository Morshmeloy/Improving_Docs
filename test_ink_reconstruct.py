import unittest
import numpy as np
from ink_reconstruct import lift_ink, reconstruct_patch, bounds


class InkReconstructionTests(unittest.TestCase):
    def test_colored_ink_is_exact_and_unselected_source_strokes_are_not_erased(self):
        source=np.full((80,120,3),255,np.uint8)
        source[25:30,10:40]=[20,40,180]
        source[50:52,60:100]=170
        mask=np.full_like(source,255)
        mask[25:30,10:40]=0
        mask[10:12,50:80]=0
        result,overlay,drawn,stats=reconstruct_patch(source,mask,0)
        np.testing.assert_array_equal(result[25:30,10:40],source[25:30,10:40])
        np.testing.assert_array_equal(result[50:52,60:100],source[50:52,60:100])
        self.assertTrue(np.all(result[10:12,50:80]==0))
        self.assertTrue(stats['colored_pixels_equal'])

    def test_white_page_lifting_remains_white(self):
        source=np.full((100,150,3),255,np.uint8)
        np.testing.assert_array_equal(lift_ink(source,10),source)

    def test_normalized_box_bounds_and_rejection(self):
        self.assertEqual(bounds([.1,.2,.9,.8],101,99),(10,19,91,80))
        with self.assertRaises(ValueError):bounds([.8,.2,.1,.9],101,99)


if __name__=='__main__':unittest.main()
