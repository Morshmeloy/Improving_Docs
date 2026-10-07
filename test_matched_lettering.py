import unittest
import hashlib
import numpy as np
from matched_lettering import fit_glyph
from template_restore import transfer_static
from setup_fonts import verify

class MatchedLetteringTests(unittest.TestCase):
    def test_glyph_uses_source_dimensions_not_cleanup_rectangle(self):
        glyph=np.zeros((20,80),np.uint8)
        result=fit_glyph(glyph,(60,200),[12,17,112,32])
        yy,xx=np.where(result<128)
        self.assertEqual([int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],[12,17,112,32])
        self.assertTrue(np.all(result[:17]==255));self.assertTrue(np.all(result[32:]==255))
        with self.assertRaises(ValueError):fit_glyph(glyph,(60,200),[0,0,201,20])

    def test_matched_typesetting_cannot_run_without_original_metrics(self):
        image=np.full((80,180,3),255,np.uint8)
        profile={'schema':2,'protected':[[0,.5,1,1]],'regions':[{'name':'test','kind':'static_label','mode':'metric_matched_text','reviewed':True,'verified_text':'ЗАЯВИТЕЛЬ','target':[.1,.1,.8,.3],'reference':[.1,.1,.8,.3]}]}
        with self.assertRaises(ValueError):transfer_static(image,image,profile)

    def test_font_version_and_size_are_verified(self):
        data=b'font-fixture';sha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
        self.assertTrue(verify(data,sha,len(data)))
        self.assertFalse(verify(data+b'x',sha,len(data)))
        self.assertFalse(verify(b'other',sha,len(data)))

if __name__=='__main__':unittest.main()
