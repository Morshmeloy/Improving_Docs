import unittest
import numpy as np
from ink_reconstruct import lift_ink, reconstruct_patch, bounds, page_regions, main
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch
from PIL import Image


class InkReconstructionTests(unittest.TestCase):
    def test_whole_document_includes_base_and_keeps_refinements(self):
        ref=[{'box':[.2,.1,.8,.2],'gain':10}]
        planned=page_regions(ref,True,5)
        self.assertEqual(planned[0],{'box':[0,0,1,1],'gain':5})
        self.assertEqual(planned[1:],ref)
        self.assertEqual(page_regions([],True,5)[0]['box'],[0,0,1,1])

    def test_whole_document_processes_each_tiff_page_without_manual_regions(self):
        class FakeSegmentation:
            def restore(self,rgb,tile,overlap):
                gray=rgb.mean(axis=2)
                return np.repeat(np.where(gray<240,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);frames=[]
            for y in [20,60]:
                rgb=np.full((100,150,3),255,np.uint8);rgb[y:y+3,10:130]=235
                frames.append(Image.fromarray(rgb))
            source=root/'scan.tif';frames[0].save(source,save_all=True,append_images=frames[1:])
            fake=types.SimpleNamespace(DocResCPU=lambda threads,task:FakeSegmentation())
            with patch.dict('sys.modules',{'docres_cpu':fake}):
                main([str(source),'--whole-document','--output',str(root/'out'),'--gain','5'])
            report=json.loads((root/'out'/'report.json').read_text())
            self.assertEqual(len(report['pages']),2)
            self.assertTrue(all(x['processing_coverage']==1 for x in report['pages']))
            self.assertTrue(all(x['regions'][0]['drawn_pixels']>0 for x in report['pages']))
            self.assertTrue(report['saved_pdf_rasters_verified'])

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
