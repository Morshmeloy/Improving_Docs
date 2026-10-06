import json
from pathlib import Path
import tempfile
import unittest
import cv2
import fitz
import numpy as np
from PIL import Image
from enhance_batch import enhance, main, verify_pdf_rasters
from cv_pipeline import restore

class CVTests(unittest.TestCase):
    def test_paper_flattening_with_known_faint_strokes(self):
        h,w=360,640
        bg=np.tile(np.linspace(170,250,w,dtype=np.float32),(h,1))
        a=np.repeat(bg[...,None],3,2)
        # Known ground truth strokes, not inferred from result.
        a[100:104,80:550]-=12
        a[200:203,80:550]-=8
        a=a.round().astype(np.uint8)
        source_range=float(np.ptp(a[30,:,0]))
        for mode in ('cv_balanced','cv_detail'):
            out,_,_=enhance(a,mode)
            self.assertLess(float(np.ptp(out[30,40:-40,0])),source_range/3)
            original_contrast=float(a[90,100:500,0].mean()-a[101,100:500,0].mean())
            contrast=float(out[90,100:500,0].mean()-out[101,100:500,0].mean())
            self.assertGreater(contrast,2*original_contrast)

    def test_all_variants_locks_and_ink_study(self):
        a=np.full((200,300,3),245,np.uint8)
        a[20:50,20:90]=[145,170,230]
        a[140:150,120:200]=220
        for mode in ('safe','readable','cv_balanced','cv_detail','ink_study','photo'):
            out,mask,stats=enhance(a,mode,[[.35,.65,.8,.9]])
            self.assertTrue(np.array_equal(a[mask],out[mask]))
            self.assertEqual(stats['protected_max_error'],0)
            self.assertEqual(a.shape,out.shape)
        study,_,_=enhance(a,'ink_study')
        self.assertFalse(np.array_equal(study[20:50,20:90],a[20:50,20:90]))
        cv,_,_=enhance(a,'cv_detail')
        self.assertTrue(np.array_equal(cv[20:50,20:90],a[20:50,20:90]))

    def test_saved_pdf_pixels_and_masks(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'scan.png';out=root/'out'
            a=np.full((160,240,3),238,np.uint8)
            a[20:35,10:80]=[150,170,230];a[110:115,130:180]=210
            Image.fromarray(a).save(src)
            region=root/'regions.json'
            region.write_text(json.dumps({'scan.png':{'1':[[.5,.6,.9,.9]]}}))
            self.assertEqual(main([str(src),'--output',str(out),'--regions',str(region),'--ink-study']),0)
            folder=next(out.glob('scan__*'))
            for mode in ('safe','readable','cv_balanced','cv_detail','ink_study'):
                image=np.array(Image.open(folder/f'page_0001_{mode}.png'))
                mask=np.array(Image.open(folder/f'page_0001_{mode}_protected.png'))>0
                with fitz.open(folder/f'enhanced_{mode}.pdf') as pdf:
                    # Extract actual lossless embedded raster, not resampled preview.
                    xref=pdf[0].get_images()[0][0]
                    pix=fitz.Pixmap(pdf,xref)
                    saved=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3)
                self.assertTrue(np.array_equal(image,saved))
                self.assertTrue(np.array_equal(a[mask],saved[mask]))

    def test_blank_and_tiny_images(self):
        for size in ((1,1),(5,9),(100,100)):
            a=np.full((*size,3),230,np.uint8)
            for mode in ('cv_balanced','cv_detail','ink_study'):
                out,_=restore(a,mode)
                self.assertEqual(out.shape,a.shape)
                self.assertTrue(np.all(out==255))

    def test_pdf_validation_rejects_pixel_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);png=root/'expected.png';pdfpath=root/'out.pdf'
            Image.new('RGB',(20,30),'white').save(png)
            doc=fitz.open();page=doc.new_page(width=20,height=30)
            page.insert_image(page.rect,filename=str(png));doc.save(pdfpath);doc.close()
            self.assertTrue(verify_pdf_rasters(pdfpath,[png]))
            Image.new('RGB',(20,30),'black').save(png)
            with self.assertRaisesRegex(RuntimeError,'pixel verification'):
                verify_pdf_rasters(pdfpath,[png])

    def test_strength_validation(self):
        for val in ('nan','0','3'):
            with self.assertRaises(SystemExit):main(['missing','--output','unused','--strength',val])

if __name__=='__main__':unittest.main()
