"""Automatic local neural restoration, sequential files and per-page orientation."""
from datetime import datetime
import html
from pathlib import Path
import shutil
import tempfile
import fitz
import numpy as np
from ai_restore import orient
from enhance_batch import pages, png_bytes, add_page, verify_pdf_rasters, sha256
from ink_reconstruct import lift_ink, reconstruct_patch
from auto_form import shared_score, restore_labels, restore_frame, reviewed_cache
from restore_batch import write_json, update_index, document_key


class AutoRestorer:
    def __init__(self, ocr=None, model=None):
        if ocr is None:
            from ocr_engine import NeuralOCR
            ocr = NeuralOCR()
        if model is None:
            from setup_docres import main as setup_docres
            from setup_fonts import main as setup_fonts
            setup_docres(); setup_fonts()
            from docres_cpu import DocResCPU
            model = DocResCPU(4,'binarization')
        self.ocr = ocr; self.model = model; self.references = {}

    def reference_pages(self, path):
        if path is None:
            return []
        key = str(Path(path).resolve())
        if key not in self.references:
            found = []
            for n,rgb,_,_ in pages(Path(path),200,40):
                print(f'  Reference page {n}: automatic orientation and OCR',flush=True)
                angle,audit = self.ocr.orientation(rgb)
                image = orient(rgb,angle)
                found.append({'page':n,'rgb':image,'rows':self.ocr.read(image,enhance=True),
                              'orientation':audit,'angle':angle})
            self.references[key] = found
        return self.references[key]

    def document(self, source, reference_path, output, index):
        source = Path(source); job = {'input':str(source)}
        destination = output/document_key(job,index)
        if destination.exists():
            raise ValueError('Output already exists: '+str(destination))
        source_hash = sha256(source)
        stage = Path(tempfile.mkdtemp(prefix='.auto_',dir=output)); doc = fitz.open()
        expected = []; reports = []; cards = []
        try:
            references = self.reference_pages(reference_path)
            shutil.copy2(source,stage/('source'+source.suffix))
            for n,rgb,size,_ in pages(source,200,40):
                print(f'[{index}] {source.name}, page {n}: orientation',flush=True)
                angle,orientation = self.ocr.orientation(rgb)
                original = orient(rgb,angle)
                print(f'  page {n}: DocRes full-page neural reconstruction ({angle} degrees)',flush=True)
                segmented = self.model.restore(lift_ink(original,5),384,48)
                base,overlay,_,ink = reconstruct_patch(original,segmented,4)
                # Observe source text without transcribing its individual data.
                rows = self.ocr.read(original,enhance=True)
                scores = [(shared_score(rows,original.shape,ref),ref) for ref in references]
                best = max(scores,key=lambda pair:pair[0]) if scores else (0,None)
                ref = best[1] if best[0]>=3 else None
                profile = reviewed_cache(source,reference_path,n,angle)
                cache_ref = next((r for r in references if r['page']==1 and r['angle']==0),None)
                if profile and cache_ref is not None:
                    from template_restore import transfer_static
                    final,mask,_,label_audit = transfer_static(base,cache_ref['rgb'],profile,original)
                    ref = cache_ref; frame_audit = {'status':'exact_file_pair_reviewed_cache'}
                    mode = 'reviewed_exact_pair'
                else:
                    final,frame_mask,frame_audit = restore_frame(base,original,rows,ref)
                    final,label_mask,label_audit = restore_labels(final,original,rows,ref)
                    mask = frame_mask|label_mask; mode = 'automatic_ocr' if ref else 'neural_only'
                outside_equal = bool(np.array_equal(final[~mask],base[~mask]))
                if not outside_equal:
                    raise AssertionError('Reference modified pixels outside its static mask')
                overlay[mask] = [0,80,255]
                prefix = f'page_{n:04d}'
                for suffix,image in [('original',original),('neural',base),('final',final),('overlay',overlay),
                                     ('static_mask',np.repeat((mask.astype(np.uint8)*255)[:,:,None],3,axis=2))]:
                    (stage/f'{prefix}_{suffix}.png').write_bytes(png_bytes(image))
                expected.append(stage/f'{prefix}_final.png')
                add_page(doc,size[::-1] if angle in (90,270) else size,png_bytes(final))
                reports.append({'page':n,'mode':mode,'orientation':orientation,'ink':ink,
                                'reference_page':ref['page'] if ref else None,
                                'shared_static_labels':best[0],'labels':label_audit,'frame':frame_audit,
                                'outside_static_regions_equal_to_neural':outside_equal,
                                'static_pixels':int(mask.sum())})
                cards.append(f'<h2>Страница {n}</h2><div class="row">'+''.join(
                    f'<figure><figcaption>{label}</figcaption><a href="{prefix}_{suffix}.png"><img src="{prefix}_{suffix}.png"></a></figure>'
                    for suffix,label in [('original','Оригинал'),('final','Улучшенный документ')])+'</div>')
            if not reports:
                raise ValueError('Document contains no pages')
            doc.set_metadata({'title':'Edited document reconstruction draft','producer':'Improving Docs automatic local AI'})
            doc.save(stage/'restored_draft.pdf',deflate=True); doc.close()
            verify_pdf_rasters(stage/'restored_draft.pdf',expected)
            write_json(stage/'report.json',{'source_sha256':source_hash,'pages':reports,
                       'engines':['DocRes binarization','EasyOCR CRAFT + Cyrillic recognition'],
                       'reference_sha256':sha256(reference_path) if reference_path else None,
                       'reference_is_neural_model_input':False,'pdf_rasters_verified':True,
                       'semantic_accuracy_verified':False})
            (stage/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>.row{display:flex}figure{width:48%;margin:1%}img{width:100%}</style><h1>'+html.escape(source.name)+'</h1>'+''.join(cards),encoding='utf-8')
            if sha256(source)!=source_hash:
                raise AssertionError('Input file changed')
            stage.rename(destination)
            return {'status':'ok','input':str(source),'output':str(destination),'pages':len(reports),
                    'reference_pages':sum(p['reference_page'] is not None for p in reports)}
        except BaseException:
            if not doc.is_closed:doc.close()
            shutil.rmtree(stage,ignore_errors=True);raise


def run_auto(files,reference=None,output=None,restorer=None):
    reference = Path(reference).resolve() if reference else None
    output = Path(output or Path('output')/('auto_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    output.mkdir(parents=True,exist_ok=False); results = []
    for index,path in enumerate(files,1):
        try:
            if restorer is None:
                print('Loading local neural models (CPU)...',flush=True)
                restorer = AutoRestorer()
            result = restorer.document(Path(path).resolve(),reference,output,index)
        except (Exception,SystemExit) as error:
            result = {'status':'error','input':str(path),'error':str(error)}
            print('ERROR:',path,error,flush=True)
        results.append(result); update_index(output,results)
    return output,results
