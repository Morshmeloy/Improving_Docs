"""Sequential DocRes + reference form reconstruction, all documents and pages."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import tempfile
import fitz
from enhance_batch import add_page, png_bytes, verify_pdf_rasters, sha256
from template_restore import main as restore_template, load_rgb
from ink_reconstruct import main as restore_ink

DEFAULTS={'dpi':200,'gain':5,'threads':4,'tile':384,'overlap':48,'rotate':0}


def write_json(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def resolve_path(value,root):
    p=Path(value).expanduser();return (root/p).resolve() if not p.is_absolute() else p.resolve()


def normalize_job(job,root,defaults):
    out={**DEFAULTS,**defaults,**job}
    out['input']=str(resolve_path(job['input'],root))
    if out.get('reference'):out['reference']=str(resolve_path(out['reference'],root))
    out['page_profiles']={}
    for page,entry in job.get('page_profiles',{}).items():
        if int(page)<1:raise ValueError('Page numbers start at 1')
        entry={'profile':entry} if isinstance(entry,str) else dict(entry)
        entry['profile']=str(resolve_path(entry['profile'],root))
        reference=entry.get('reference',out.get('reference'))
        if not reference:raise ValueError('Template profile needs a reference file')
        entry['reference']=str(resolve_path(reference,root))
        out['page_profiles'][str(int(page))]=entry
    return out


def check_job(job):
    source=Path(job['input'])
    if not source.is_file():raise ValueError('Input missing: '+str(source))
    source_hash=sha256(source)
    if job['rotate'] not in [0,90,180,270] or not 72<=job['dpi']<=600:raise ValueError('Invalid rotation or dpi')
    for page,entry in job['page_profiles'].items():
        p=Path(entry['profile']);ref=Path(entry['reference'])
        if not p.is_file() or not ref.is_file():raise ValueError('Profile or reference missing on page '+page)
        profile=json.loads(p.read_text(encoding='utf-8-sig'))
        if profile.get('source_sha256') and profile['source_sha256']!=source_hash:raise ValueError('Profile belongs to another source. Review it in the editor: '+str(p))
        if profile.get('reference_sha256') and profile['reference_sha256']!=sha256(ref):raise ValueError('Profile belongs to another reference: '+str(p))
        actual={'expected_dpi':job['dpi'],'expected_rotation':job['rotate'],'source_page':int(page),'reference_page':entry.get('reference_page',1),'reference_rotation':entry.get('reference_rotate',0)}
        for key,value in actual.items():
            if key in profile and profile[key]!=value:raise ValueError('Profile mismatch: '+key)
    return source_hash


def document_key(job,index):
    name=re.sub(r'[^\w.-]+','_',Path(job['input']).stem,flags=re.UNICODE)[:65]
    return f'{index:03d}_{name}'


def run_document(job,output,index,model_holder):
    """One file, all pages; publish complete results atomically."""
    source_hash=check_job(job)
    destination=output/document_key(job,index)
    if destination.exists():raise ValueError('Document output already exists')
    if not model_holder:
        from docres_cpu import DocResCPU
        model_holder.append(DocResCPU(job['threads'],'binarization'))
    stage=Path(tempfile.mkdtemp(prefix='.doc_',dir=output))
    try:
        print(f'[{index}] {Path(job["input"]).name}: full-page neural reconstruction',flush=True)
        with (stage/'neural.log').open('w',encoding='utf-8') as log, contextlib.redirect_stdout(log):
            restore_ink([job['input'],'--whole-document','--output',str(stage/'neural'),'--dpi',str(job['dpi']),'--rotate',str(job['rotate']),'--gain',str(job['gain']),'--threads',str(job['threads']),'--tile',str(job['tile']),'--overlap',str(job['overlap'])],model=model_holder[0])
        neural=stage/'neural';expected=[];document=fitz.open();cards=[];page_reports=[]
        try:
            with fitz.open(neural/'reconstructed_draft.pdf') as base_pdf:
                if any(int(p)>len(base_pdf) for p in job['page_profiles']):raise ValueError('Profile references a nonexistent source page')
                for n,page in enumerate(base_pdf,1):
                    before=neural/f'page_{n:04d}_reconstructed.png';final=before
                    entry=job['page_profiles'].get(str(n))
                    if entry:
                        print(f'  page {n}: approved reference form',flush=True)
                        target=stage/f'template_page_{n:04d}'
                        with (stage/f'template_page_{n:04d}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
                            restore_template(['--source',job['input'],'--base',str(before),'--reference',entry['reference'],'--profile',entry['profile'],'--output',str(target),'--page',str(n),'--dpi',str(job['dpi']),'--rotate',str(job['rotate']),'--reference-page',str(entry.get('reference_page',1)),'--reference-rotate',str(entry.get('reference_rotate',0))])
                        final=target/'reconstructed.png'
                        page_reports.append({'page':n,'mode':'neural_and_reference','template_audit':json.loads((target/'report.json').read_text())})
                    else:
                        print(f'  page {n}: neural image only; no template profile',flush=True)
                        page_reports.append({'page':n,'mode':'neural_only'})
                    name=f'page_{n:04d}_final.png';shutil.copy2(final,stage/name);expected.append(stage/name)
                    add_page(document,(page.rect.width,page.rect.height),png_bytes(load_rgb(final)))
                    cards.append(f'<h2>Page {n}: {page_reports[-1]["mode"]}</h2><div class="row"><figure><figcaption>Original</figcaption><img src="neural/page_{n:04d}_original.png"></figure><figure><figcaption>Final draft</figcaption><img src="{name}"></figure></div>')
            document.set_metadata({'title':'EDITED reconstruction draft: sequential neural and reviewed reference form','producer':'Improving Docs batch'})
            document.save(stage/'restored_draft.pdf',deflate=True)
        finally:document.close()
        verify_pdf_rasters(stage/'restored_draft.pdf',expected)
        report={'source_sha256':source_hash,'input':job['input'],'pages':page_reports,'reference_is_neural_model_input':False,'pdf_rasters_verified':True,'semantic_accuracy_verified':False}
        write_json(stage/'report.json',report)
        (stage/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>.row{display:flex}figure{width:48%}img{width:100%}</style><h1>'+html.escape(Path(job['input']).name)+'</h1><p>Edited draft; reference applies only to reviewed static regions.</p>'+''.join(cards),encoding='utf-8')
        stage.rename(destination)
        return {'status':'ok','input':job['input'],'source_sha256':source_hash,'output':str(destination),'pages':len(page_reports),'reference_pages':sum(p['mode']=='neural_and_reference' for p in page_reports)}
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise


def update_index(output,results):
    write_json(output/'batch_report.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'documents':results})
    items=[]
    for item in results:
        name=html.escape(Path(item['input']).name)
        if item['status']=='ok':
            directory=Path(item['output']).name
            items.append(f'<li>{name}: <a href="{directory}/comparison.html">compare</a> | <a href="{directory}/restored_draft.pdf">PDF</a> ({item["reference_pages"]}/{item["pages"]} pages with reference)</li>')
        else:items.append('<li>'+name+': '+html.escape(item.get('error',item['status']))+'</li>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Sequential document restoration</h1><ul>'+''.join(items)+'</ul>',encoding='utf-8')


def run_jobs(jobs,output,step=False):
    output.mkdir(parents=True,exist_ok=False);results=[];model_holder=[]
    write_json(output/'batch_config.json',{'schema':1,'documents':jobs})
    for index,job in enumerate(jobs,1):
        if step:input(f'Press Enter to process document {index}/{len(jobs)}: {Path(job["input"]).name} ')
        try:result=run_document(job,output,index,model_holder)
        except (Exception,SystemExit) as error:
            result={'status':'error','input':job['input'],'error':str(error)};print('ERROR:',error,flush=True)
        results.append(result);update_index(output,results)
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--step',action='store_true');a=p.parse_args(argv)
    config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    if config.get('schema')!=1 or not config.get('documents'):p.error('Config requires schema=1 and documents')
    jobs=[normalize_job(j,a.config.resolve().parent,config.get('defaults',{})) for j in config['documents']]
    results=run_jobs(jobs,a.output.resolve(),a.step)
    print('Open:',(a.output/'index.html').resolve())
    return int(any(x['status']!='ok' for x in results))


if __name__=='__main__':raise SystemExit(main())
