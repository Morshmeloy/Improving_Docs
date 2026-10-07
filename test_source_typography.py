import unittest
import numpy as np
from source_typography import repair_source_label, observed_ink, ink_bounds
from template_restore import transfer_static

class SourceTypographyTests(unittest.TestCase):
    def test_reference_cannot_create_a_label_without_source_evidence(self):
        original=np.full((40,160,3),255,np.uint8)
        reference=np.zeros_like(original)
        result,stats=repair_source_label(original,reference)
        np.testing.assert_array_equal(result,original)
        self.assertEqual(stats['status'],'no_source_ink')
        self.assertFalse(stats['font_substituted'])

    def test_original_geometry_and_colored_pixels_survive(self):
        original=np.full((60,180,3),250,np.uint8)
        original[15:35,20:23]=180;original[15:18,20:45]=180
        original[32:35,20:45]=180;original[15:35,42:45]=180
        original[40:45,100:145]=[25,40,180]
        reference=np.full((80,250,3),255,np.uint8);reference[10:50,15:190]=0
        before=original.copy();source=observed_ink(original)
        result,stats=repair_source_label(original,reference)
        np.testing.assert_array_equal(original,before)
        np.testing.assert_array_equal(result[40:45,100:145],original[40:45,100:145])
        self.assertTrue(stats['ink_envelope_equal'])
        self.assertEqual(stats['source_ink_bbox'],ink_bounds(source))
        self.assertFalse(stats['source_scaled'])
        self.assertLess(result[20,21,0],original[20,21,0])
        x0,y0,x1,y1=stats['source_ink_bbox'];outside=np.ones(source.shape,bool);outside[y0:y1,x0:x1]=False
        np.testing.assert_array_equal(result[outside],original[outside])

    def test_original_raster_is_required_for_source_guided_mode(self):
        image=np.full((80,180,3),255,np.uint8)
        profile={'schema':2,'protected':[[0,.5,1,1]],'regions':[{'name':'test','kind':'static_label','mode':'source_guided_label','reviewed':True,'verified_text':'ЗАЯВИТЕЛЬ','target':[.1,.1,.8,.3],'reference':[.1,.1,.8,.3]}]}
        with self.assertRaises(ValueError):transfer_static(image,image,profile)

if __name__=='__main__':unittest.main()
