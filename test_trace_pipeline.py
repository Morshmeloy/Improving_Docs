import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image
from enhance_batch import enhance, main
from trace_pipeline import trace_strokes

class TraceTests(unittest.TestCase):
    def test_faint_stroke_darkening_without_completing_gap(self):
        a=np.full((160,240,3),255,np.uint8)
        # A true absent segment, unlike pale evidence, must not be connected.
        a[30:130,40:43]=238;a[30:130,90:93]=242
        a[65:85,40:43]=255
        out,_=trace_strokes(a,.8)
        self.assertLess(float(out[35:60,40:43].mean()),215)
        self.assertTrue(np.all(out[70:80,40:43]==255))
        self.assertTrue(np.all(out[:,150:]==255))

    def test_manual_signature_pixels_are_exact(self):
        a=np.full((120,180,3),250,np.uint8)
        cv2.line(a,(30,70),(130,90),(225,225,225),1)
        a[20:35,40:80]=[200,210,235]
        out,mask,stats=enhance(a,'trace_study',[[.1,.5,.9,.95]],.8)
        self.assertTrue(np.array_equal(a[mask],out[mask]))
        self.assertEqual(stats['protected_max_error'],0)
        self.assertFalse(stats['missing_segments_completed'])
        self.assertFalse(np.array_equal(a[20:35,40:80],out[20:35,40:80]))

    def test_cli_candidate_is_explicit_and_pdf_verified(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'input.png';out=root/'out'
            a=np.full((100,160,3),255,np.uint8);a[50:53,20:140]=240
            Image.fromarray(a).save(src)
            self.assertEqual(main([str(src),'--output',str(out),'--trace-study']),0)
            folder=next(out.glob('input__*'))
            report=json.loads((folder/'report.json').read_text())
            self.assertTrue((folder/'enhanced_trace_study.pdf').is_file())
            self.assertTrue(report['saved_pdf_rasters_verified'])
            self.assertEqual(report['pages'][0]['variants']['trace_study']['missing_segments_completed'],False)

    def test_empty_small_and_invalid_inputs(self):
        for shape in [(1,1),(2,6),(100,100)]:
            out,_=trace_strokes(np.full((*shape,3),240,np.uint8))
            self.assertTrue(np.all(out==255))
        for val in (float('nan'),0,3):
            with self.assertRaises(ValueError):trace_strokes(np.full((10,10,3),255,np.uint8),val)

if __name__=='__main__':unittest.main()
