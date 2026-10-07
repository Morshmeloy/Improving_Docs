import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from redraw_pipeline import connect_endpoints,redraw
from redraw_batch import main

class DrawingTests(unittest.TestCase):
    def test_pixels_outside_drawing_regions_remain_exact(self):
        a=np.full((100,160,3),250,np.uint8);a[50:53,10:100]=238
        areas=np.zeros((100,160),bool);areas[:,10:40]=True
        result,_,stats=redraw(a,areas)
        np.testing.assert_array_equal(result[~areas],a[~areas])
        self.assertGreater(stats['drawn_pixels'],0)

    def test_missing_regions_is_error_instead_of_successful_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=np.full((30,40,3),255,np.uint8)
            Image.fromarray(a).save(root/'a.png')
            (root/'regions.json').write_text('{}')
            out=root/'out'
            code=main([str(root/'a.png'),'--output',str(out),'--redraw-regions',str(root/'regions.json')])
            self.assertEqual(code,1)
            report=json.loads(next(out.glob('batch_*.json')).read_text())
            self.assertIn('No drawing regions',report[0]['error'])
            self.assertFalse(any(x.is_dir() for x in out.iterdir()))

    def test_facing_endpoints_are_connected(self):
        skeleton=np.zeros((40,60),bool);skeleton[20,5:25]=True;skeleton[20,29:50]=True
        added,connections=connect_endpoints(skeleton,6)
        self.assertTrue(added[20,25:29].all());self.assertEqual(len(connections),1)
        unchanged,_=connect_endpoints(skeleton,3);self.assertFalse(unchanged.any())

    def test_parallel_strokes_not_connected_sideways(self):
        skeleton=np.zeros((40,60),bool);skeleton[20,5:45]=True;skeleton[24,5:45]=True
        added,connections=connect_endpoints(skeleton,6)
        self.assertFalse(added.any());self.assertEqual(connections,[])

    def test_black_pencil_and_locks(self):
        a=np.full((100,160,3),255,np.uint8);a[50:53,10:100]=238
        areas=np.ones((100,160),bool);locks=np.zeros_like(areas);locks[:,55:70]=True
        b,overlay,stats=redraw(a,areas,locked=locks)
        self.assertTrue(np.any(np.all(b==0,axis=2)))
        self.assertTrue(np.array_equal(a[locks],b[locks]))
        self.assertTrue(np.all(b[:,120:]==255))
        self.assertGreater(stats['drawn_pixels'],0)

    def test_batch_corrupt_file_continues(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);inputs=root/'in';inputs.mkdir();out=root/'out'
            a=np.full((100,150,3),255,np.uint8);a[30:33,20:120]=240
            Image.fromarray(a).save(inputs/'a.png');(inputs/'bad.pdf').write_bytes(b'broken')
            code=main([str(inputs),'--output',str(out),'--redraw-all'])
            self.assertEqual(code,1)
            report=json.loads(next(out.glob('batch_*.json')).read_text())
            self.assertEqual(sum(r['status']=='ok' for r in report),1)
            folder=Path(next(r['output'] for r in report if r['status']=='ok'))
            self.assertTrue(json.loads((folder/'report.json').read_text())['saved_pdf_verified'])

if __name__=='__main__':unittest.main()
