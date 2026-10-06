"""Run: python -m unittest -v test_enhancer.py"""
import json
from pathlib import Path
import tempfile
import unittest
import fitz
import numpy as np
from PIL import Image
from enhance_batch import enhance, main

class EnhancementTests(unittest.TestCase):
    def test_exact_color_and_manual_signature_locks(self):
        a = np.full((160, 220, 3), 250, np.uint8)
        a[20:50, 20:90] = [130, 150, 220]  # blue ink
        a[100:110, 100:150] = 210  # gray signature
        a[60:65, 30:190] = 190  # faint text
        for mode in ['safe', 'readable']:
            b, mask, stats = enhance(a, mode, [[.4,.55,.8,.8]])
            self.assertTrue(np.array_equal(a[mask], b[mask]))
            self.assertTrue(np.array_equal(a[100:110,100:150], b[100:110,100:150]))
            self.assertTrue(np.all(b[60:65,30:190] < a[60:65,30:190]))
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(stats['protected_pixels_equal'])

    def test_four_files_and_corrupt_input_continue(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); src = root/'input'; src.mkdir(); out = root/'out'
            a = Image.new('RGB', (128,160), (235,235,235))
            a.save(src/'a.png'); a.save(src/'b.jpg')
            a.save(src/'c.tiff', save_all=True, append_images=[a])
            d=fitz.open(); d.new_page(); d.save(src/'d.pdf'); d.close()
            (src/'broken.pdf').write_bytes(b'corrupt')
            hashes = {p.name:p.read_bytes() for p in src.iterdir()}
            self.assertEqual(main([str(src),'--output',str(out),'--dpi','72']),1)
            reports = json.loads(next(out.glob('batch_*.json')).read_text())
            self.assertEqual(sum(r['status']=='ok' for r in reports),4)
            for r in reports:
                if r['status']=='ok':
                    for mode in ['safe','readable']:
                        with fitz.open(Path(r['output'])/f'enhanced_{mode}.pdf') as pdf:
                            self.assertEqual(len(pdf),r['pages'])
            self.assertEqual(hashes,{p.name:p.read_bytes() for p in src.iterdir()})
            # Repeat cannot overwrite outputs; existing success folders survive.
            self.assertEqual(main([str(src),'--output',str(out),'--dpi','72']),1)
            self.assertEqual(len([p for p in out.iterdir() if p.is_dir()]),4)

if __name__ == '__main__':
    unittest.main()
