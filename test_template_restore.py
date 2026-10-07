import unittest
import numpy as np
from template_restore import transfer_static

class TemplateTests(unittest.TestCase):
    def test_reference_text_cannot_leak_outside_allowlist(self):
        base=np.full((100,100,3),230,np.uint8)
        base[50:70,20:80]=[20,40,180] # source seal/signature/content
        reference=np.zeros((100,100,3),np.uint8) # hostile entirely different content
        profile={'schema':1,'protected':[[0,.3,1,1]],'regions':[{'name':'title','kind':'static_heading','target':[.1,.05,.9,.2],'reference':[0,0,1,1]}]}
        result,mask,protected,_=transfer_static(base,reference,profile)
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(result[~mask],base[~mask])
        np.testing.assert_array_equal(result[protected],base[protected])
        self.assertTrue(np.all(result[mask]==0))

    def test_overlap_and_nonstatic_region_rejected(self):
        image=np.full((100,100,3),255,np.uint8)
        for kind in ['static_heading','signature']:
            profile={'schema':1,'protected':[[0,.1,1,1]],'regions':[{'name':'bad','kind':kind,'target':[.2,.05,.8,.3],'reference':[0,0,1,1]}]}
            with self.assertRaises(ValueError):transfer_static(image,image,profile)

    def test_repeated_ornament_records_provenance(self):
        base=np.full((100,100,3),255,np.uint8);ref=base.copy();ref[0:10,0:10]=100
        profile={'schema':1,'protected':[[0,.3,1,1]],'regions':[{'name':'frame','kind':'ornamental_frame','mode':'tile_x','target':[.1,.01,.9,.1],'reference':[0,0,.1,.1]}]}
        result,mask,_,records=transfer_static(base,ref,profile)
        self.assertTrue(np.all(result[mask]==100));self.assertIn('tile_width',records[0]['transform'])

if __name__=='__main__':unittest.main()
