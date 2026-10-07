"""Sequential explicit black tracing in selected regions, with source and change overlay."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path
import shutil
import tempfile
import fitz
import numpy as np
from enhance_batch import SUPPORTED,pages,sha256,png_bytes,add_page,verify_pdf_rasters,protection_mask
from redraw_pipeline import redraw


def process(path,output,args,regions,locks):
    digest=sha256(path);target=output/(path.stem[:80]+'__'+digest[:12])
    if target.exists():raise FileExistsError('Output exists: '+str(target))
    stage=Path(tempfile.mkdtemp(prefix='.drawing_',dir=output));doc=fitz.open()
    try:
        shutil.copy2(path,stage/('source'+path.suffix.lower()))
        report={'source_sha256':digest,'kind':'black tracing with inferred gap connections','pages':[]};expected=[];cards=[]
        for n,rgb,size,info in pages(path,args.dpi,40):
            boxes=regions.get(path.name,{}).get(str(n),[])
            areas=np.ones(rgb.shape[:2],bool) if args.redraw_all else protection_mask(rgb,boxes,color=False)
            locked=protection_mask(rgb,locks.get(path.name,{}).get(str(n),[]),color=False)
            out,overlay,stats=redraw(rgb,areas,args.max_gap,args.line_width,args.threshold,locked)
            prefix=f'page_{n:04d}'
            for suffix,array in [('original',rgb),('redrawn',out),('drawing_overlay',overlay)]:
                (stage/f'{prefix}_{suffix}.png').write_bytes(png_bytes(array))
            expected.append(stage/f'{prefix}_redrawn.png');add_page(doc,size,png_bytes(out))
            stats['page']=n;stats['protected_equal']=bool(np.array_equal(out[locked],rgb[locked]));report['pages'].append(stats)
            cards.append(f'<h2>Page {n}</h2><div>'+''.join(f'<figure><figcaption>{s}</figcaption><img src="{prefix}_{s}.png"></figure>' for s in ['original','redrawn','drawing_overlay'])+'</div>')
        if not expected:raise ValueError('No pages')
        doc.set_metadata({'title':'EDITED: black tracing, inferred short connections','producer':'Improving Docs redraw'});doc.save(stage/'redrawn.pdf',deflate=True);doc.close()
        verify_pdf_rasters(stage/'redrawn.pdf',expected);report['saved_pdf_verified']=True
        (stage/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>div{display:flex}figure{width:32%;margin:5px}img{width:100%}</style><h1>'+html.escape(path.name)+'</h1><p>Edited drawing. Overlay: green = traced strokes, red = proposed connections. Original preserved.</p>'+''.join(cards),encoding='utf-8')
        stage.rename(target);return {'source':path.name,'status':'ok','output':str(target)}
    except Exception:
        if not doc.is_closed:doc.close()
        shutil.rmtree(stage,ignore_errors=True);raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    scope=p.add_mutually_exclusive_group(required=True);scope.add_argument('--redraw-regions',type=Path);scope.add_argument('--redraw-all',action='store_true')
    p.add_argument('--regions',type=Path,help='Exact locks override drawing')
    p.add_argument('--max-gap',type=int,default=6);p.add_argument('--line-width',type=int,default=1)
    p.add_argument('--threshold',type=float,default=.025);p.add_argument('--dpi',type=int,default=300)
    a=p.parse_args(argv)
    if not 0<=a.max_gap<=20 or not 1<=a.line_width<=3 or not .005<=a.threshold<=.2 or not 72<=a.dpi<=600:p.error('Invalid drawing parameters')
    regions=json.loads(a.redraw_regions.read_text(encoding='utf-8-sig')) if a.redraw_regions else {}
    locks=json.loads(a.regions.read_text(encoding='utf-8-sig')) if a.regions else {}
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True);files=[];results=[]
    for item in a.inputs:
        if item.is_dir():files.extend(sorted(x for x in item.iterdir() if x.is_file() and x.suffix.lower() in SUPPORTED))
        else:files.append(item)
    for path in dict.fromkeys(files):
        try:
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:raise ValueError('Missing or unsupported input')
            if path.resolve().is_relative_to(output):raise ValueError('Input must be outside output')
            results.append(process(path,output,a,regions,locks))
        except Exception as exc:results.append({'source':str(path),'status':'error','error':str(exc)})
    report=output/('batch_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json')
    report.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(results,ensure_ascii=False))
    return 0 if results and all(x['status']=='ok' for x in results) else 1

if __name__=='__main__':raise SystemExit(main())
