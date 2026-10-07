import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import fitz
import numpy as np
from PIL import Image
from enhance_batch import add_page,png_bytes,sha256,verify_pdf_rasters
from restore_batch import normalize_job,run_jobs

class SequentialBatchTests(unittest.TestCase):
    def test_all_pages_reference_pdf_and_failed_file_continuation(self):
        instances=[]
        class FakeModel:
            def restore(self,rgb,tile,overlap):
                gray=rgb.mean(axis=2);return np.repeat(np.where(gray<240,0,255).astype(np.uint8)[:,:,None],3,axis=2)
        def factory(threads,task):instances.append((threads,task));return FakeModel()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);frames=[]
            for y in [30,60]:
                rgb=np.full((100,150,3),255,np.uint8);rgb[y:y+3,20:120]=210;rgb[80:85,110:140]=[20,40,180];frames.append(Image.fromarray(rgb))
            source=root/'two.tif';frames[0].save(source,save_all=True,append_images=frames[1:]);digest=sha256(source)
            ref=np.full((100,150,3),255,np.uint8);ref[5:15,15:105]=60
            pdf=fitz.open();add_page(pdf,(54,36),png_bytes(ref));reference=root/'reference.pdf';pdf.save(reference);pdf.close()
            profile=root/'profile.json';profile.write_text(json.dumps({'schema':3,'protection_policy':'outside_approved_regions','protected':[[0,.3,1,1]],'source_sha256':digest,'reference_sha256':sha256(reference),'expected_dpi':200,'expected_rotation':0,'source_page':1,'reference_page':1,'reference_rotation':0,'regions':[{'name':'static_form','kind':'static_heading','target':[.1,.05,.7,.15],'reference':[.1,.05,.7,.15]}]}))
            bad=root/'bad.pdf';bad.write_bytes(b'bad PDF')
            bad_job=normalize_job({'input':'bad.pdf'},root,{})
            good=normalize_job({'input':'two.tif','reference':'reference.pdf','page_profiles':{'1':{'profile':'profile.json','reference_page':1}}},root,{})
            fake=types.SimpleNamespace(DocResCPU=factory)
            with patch.dict('sys.modules',{'docres_cpu':fake}):results=run_jobs([bad_job,good,good],root/'out')
            self.assertEqual([x['status'] for x in results],['error','ok','ok'])
            self.assertEqual(len(instances),1)
            self.assertEqual(sha256(source),digest)
            for result in results[1:]:
                folder=Path(result['output']);report=json.loads((folder/'report.json').read_text())
                self.assertEqual(result['pages'],2);self.assertEqual(result['reference_pages'],1)
                self.assertEqual([x['mode'] for x in report['pages']],['neural_and_reference','neural_only'])
                self.assertFalse(report['reference_is_neural_model_input'])
                self.assertTrue(report['pages'][0]['template_audit']['outside_transfer_equal_to_base'])
                verify_pdf_rasters(folder/'restored_draft.pdf',[folder/'page_0001_final.png',folder/'page_0002_final.png'])
                self.assertEqual((folder/'page_0002_final.png').read_bytes(),(folder/'neural'/'page_0002_reconstructed.png').read_bytes())
            self.assertFalse(list((root/'out').glob('.doc_*')))
            checkpoint=json.loads((root/'out'/'batch_report.json').read_text());self.assertEqual(len(checkpoint['documents']),3)

    def test_config_paths_are_relative_to_config_not_terminal(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            job=normalize_job({'input':'input/a.pdf','reference':'samples/a.png','page_profiles':{'2':'profiles/a.json'}},root,{'rotate':180})
            self.assertEqual(job['input'],str(root/'input'/'a.pdf'))
            self.assertEqual(job['page_profiles']['2']['reference'],str(root/'samples'/'a.png'))
            self.assertEqual(job['rotate'],180)
            with self.assertRaises(ValueError):normalize_job({'input':'x.pdf','page_profiles':{'0':'p.json'}},root,{})

if __name__=='__main__':unittest.main()
