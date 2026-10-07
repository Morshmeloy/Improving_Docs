import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import fitz
import numpy as np
from PIL import Image
from auto_form import candidates, identity, restore_labels, shared_score, restore_frame, reviewed_cache
from auto_restore import AutoRestorer, run_auto
from batch_review import select_files
from enhance_batch import sha256, verify_pdf_rasters
from ocr_engine import NeuralOCR


def row(text,x=10,y=10,w=100,h=12):
    return {'box':[[x,y],[x+w,y],[x+w,y+h],[x,y+h]],'text':text,'confidence':.95}


class AutomaticTests(unittest.TestCase):
    def test_only_static_complete_labels_are_candidates(self):
        self.assertEqual(identity('ЗАЯВИТЕЛЬ')[0],'ЗАЯВИТЕЛЬ')
        for text in ['ЗАЯВИТЕЛЬ ООО РОМАШКА','ЗАЯВИТЕЛЬ: Иванов','ЗАЯВИТЕЛЬ И','12.04.2026','Иванов Иван','ПО','Серия 123456']:
            self.assertIsNone(identity(text),text)

    def test_matching_never_copies_common_variable_content(self):
        rows=[row('Иванов'),row('ЗАЯВИТЕЛЬ',y=30),row('ИЗГОТОВИТЕЛЬ',y=50),row('ПРОДУКЦИЯ',y=70)]
        rgb=np.full((100,150,3),255,np.uint8)
        ref={'rgb':rgb,'rows':rows}
        self.assertEqual(shared_score(rows,rgb.shape,ref),3)
        self.assertEqual(len(candidates(rows,rgb.shape)),3)

    def test_labels_preserve_all_pixels_outside_their_mask(self):
        original=np.full((100,150,3),225,np.uint8);base=original.copy()
        rows=[row('ЗАЯВИТЕЛЬ'),row('Иванов',y=50)]
        ref={'rgb':original,'rows':rows}
        with patch('auto_form.matched_label',return_value=(np.full((12,100,3),0,np.uint8),{})):
            final,mask,records=restore_labels(base,original,rows,ref)
        self.assertTrue(mask.any());self.assertTrue(np.array_equal(final[~mask],base[~mask]))
        self.assertEqual([r['label'] for r in records],['ЗАЯВИТЕЛЬ'])
        self.assertTrue(np.array_equal(final[50:62],base[50:62]))

    def test_no_reference_and_overlapping_data_are_not_retyped(self):
        rgb=np.full((100,150,3),225,np.uint8);labels=[row('ЗАЯВИТЕЛЬ'),row('Иванов',x=40)]
        with patch('auto_form.matched_label') as draw:
            final,mask,records=restore_labels(rgb,rgb,labels,{'rgb':rgb,'rows':[labels[0]]})
            self.assertFalse(mask.any());draw.assert_not_called()
            self.assertEqual(records[0]['status'],'skipped_overlap')
            final,mask,records=restore_labels(rgb,rgb,labels,None)
            self.assertFalse(mask.any());self.assertTrue(np.array_equal(final,rgb))

    def test_two_dialogs_no_manual_profile_or_orientation(self):
        with patch('batch_review.filedialog.askopenfilenames',return_value=['a.pdf','b.pdf']) as docs,patch('batch_review.filedialog.askopenfilename',return_value='') as ref:
            files,reference=select_files(object())
        self.assertEqual(len(files),2);self.assertIsNone(reference)
        docs.assert_called_once();ref.assert_called_once()
        with patch('batch_review.filedialog.askopenfilenames',return_value=[]),patch('batch_review.filedialog.askopenfilename') as ref:
            self.assertEqual(select_files(object()),([],None));ref.assert_not_called()

    def test_neural_ocr_detects_full_page_rotation(self):
        image=np.full((80,100,3),255,np.uint8);image[:10,:10]=0
        class Reader:
            def readtext(self,rgb,**kwargs):
                text='ДОКУМЕНТ ПРАВИЛЬНО ПОВЕРНУТ' if rgb[-1,-1].mean()<100 else 'x'
                return [([[10,10],[30,10],[30,20],[10,20]],text,.95)]
        angle,audit=NeuralOCR(Reader()).orientation(image)
        self.assertEqual(angle,180);self.assertTrue(audit['reliable'])

    def test_ornament_transfer_excludes_interior_data_and_dark_strokes(self):
        rgb=np.full((300,200,3),255,np.uint8)
        rgb[:,1:4]=[200,160,50];rgb[100:120,90:110]=[200,160,50]
        source=np.full_like(rgb,255);source[140:150,1:4]=0
        rows=[row('ЗАЯВИТЕЛЬ',20,60,100,12),row('ИЗГОТОВИТЕЛЬ',20,120,100,12),
              row('ПРОДУКЦИЯ',20,180,100,12),row('ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ',20,240,160,12)]
        final,mask,audit=restore_frame(source,source,rows,{'rgb':rgb,'rows':rows})
        self.assertEqual(audit['status'],'restored');self.assertTrue(mask[:,1:4].any())
        self.assertFalse(mask[100:120,90:110].any());self.assertFalse(mask[140:150,1:4].any())
        self.assertTrue(np.array_equal(final[~mask],source[~mask]))

    def test_reviewed_geometry_never_applies_to_another_document(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'source.pdf';b=Path(td)/'ref.png';a.write_bytes(b'unrelated');b.write_bytes(b'unrelated')
            self.assertIsNone(reviewed_cache(a,b,1,180))

    def test_reference_ocr_is_cached_and_unmatched_sample_is_ignored(self):
        class OCR:
            reads=0
            def orientation(self,rgb):return 0,{'chosen_clockwise':0}
            def read(self,rgb,**kwargs):self.reads+=1;return [row('Different private contents')]
        class Model:
            def restore(self,rgb,*args):return np.full_like(rgb,255)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);reference=root/'ref.tif';source=root/'source.tif'
            image=Image.fromarray(np.full((60,150,3),255,np.uint8))
            image.save(reference,save_all=True,append_images=[image]);image.save(source)
            ocr=OCR();restorer=AutoRestorer(ocr,Model())
            output,results=run_auto([source,source],reference,root/'out',restorer)
            self.assertEqual(ocr.reads,4) # Two reference pages once, two source pages.
            self.assertTrue(all(r['status']=='ok' and r['reference_pages']==0 for r in results))

    def test_sequential_mixed_orientation_all_pages_and_errors(self):
        class OCR:
            ref_reads=0
            def orientation(self,rgb):
                angle=90 if rgb.shape[0]>rgb.shape[1] else 0
                return angle,{'chosen_clockwise':angle,'reliable':True}
            def read(self,rgb,**kwargs):
                self.ref_reads+=1;return []
        class Model:
            calls=0
            def restore(self,rgb,*args):
                self.calls+=1
                return np.repeat(np.where(rgb.mean(2)<240,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=np.full((60,90,3),255,np.uint8);a[20:22,10:70]=210
            b=np.full((90,60,3),255,np.uint8);b[20:22,10:50]=210
            source=root/'two.tif';Image.fromarray(a).save(source,save_all=True,append_images=[Image.fromarray(b)])
            bad=root/'bad.pdf';bad.write_bytes(b'bad pdf');digest=sha256(source)
            ocr=OCR();model=Model();restorer=AutoRestorer(ocr,model)
            output,results=run_auto([bad,source,source],output=root/'out',restorer=restorer)
            self.assertEqual([r['status'] for r in results],['error','ok','ok'])
            self.assertEqual(model.calls,4);self.assertEqual(sha256(source),digest)
            for result in results[1:]:
                folder=Path(result['output']);report=json.loads((folder/'report.json').read_text())
                self.assertEqual([p['orientation']['chosen_clockwise'] for p in report['pages']],[0,90])
                self.assertEqual([p['mode'] for p in report['pages']],['neural_only','neural_only'])
                self.assertTrue(all(p['outside_static_regions_equal_to_neural'] for p in report['pages']))
                verify_pdf_rasters(folder/'restored_draft.pdf',[folder/'page_0001_final.png',folder/'page_0002_final.png'])
                with fitz.open(folder/'restored_draft.pdf') as pdf:
                    self.assertEqual(len(pdf),2);self.assertGreater(pdf[1].rect.width,pdf[1].rect.height)
            self.assertEqual(len(json.loads((output/'batch_report.json').read_text())['documents']),3)
            self.assertFalse(list(output.glob('.auto_*')))


if __name__=='__main__':unittest.main()
