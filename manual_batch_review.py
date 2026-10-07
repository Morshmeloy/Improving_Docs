"""Walk documents sequentially: choose reference, review form, restore, inspect, next."""
import argparse
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog,messagebox,simpledialog
import webbrowser
from enhance_batch import pages,SUPPORTED
from ai_restore import orient
from form_editor import FormEditor
from restore_batch import DEFAULTS,run_document,update_index,write_json,normalize_job,document_key


def choose_rotation(root,title):
    value=simpledialog.askinteger(title,'Поворот по часовой стрелке: 0, 90, 180 или 270.\nДля текущего исходника ВА-103 — 180.',parent=root,initialvalue=0)
    if value is None:return None
    if value not in [0,90,180,270]:messagebox.showerror('Поворот','Нужен 0, 90, 180 или 270.',parent=root);return choose_rotation(root,title)
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path,nargs='?',default=Path('input'));p.add_argument('--output',type=Path);a=p.parse_args(argv)
    if not a.input.exists():p.error('Input file or folder does not exist')
    files=sorted((x.resolve() for x in a.input.iterdir() if x.is_file() and x.suffix.lower() in SUPPORTED),key=lambda x:x.name.casefold()) if a.input.is_dir() else [a.input.resolve()]
    if not files:p.error('No supported input documents')
    output=(a.output or Path('output')/('review_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists():p.error('Output folder must be new')
    output.mkdir(parents=True);root=tk.Tk();root.withdraw();jobs=[];results=[];model_holder=[]
    try:
        for index,source in enumerate(files,1):
            rotation=choose_rotation(root,f'{index}/{len(files)}: {source.name}')
            if rotation is None:break
            job={**DEFAULTS,'input':str(source),'rotate':rotation,'page_profiles':{}}
            use_reference=messagebox.askyesno('Образец для документа',source.name+'\nПредоставить читаемый образец бланка / оригинал формы?',parent=root)
            try:
                if use_reference:
                    ref_name=filedialog.askopenfilename(parent=root,title='Образец бланка: PDF, PNG, JPEG или TIFF',filetypes=[('Документы','*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp')])
                    if not ref_name:break
                    reference=Path(ref_name).resolve();ref_rotate=choose_rotation(root,'Ориентация образца')
                    if ref_rotate is None:break
                    job['reference']=str(reference)
                    for n,rgb,_,_ in pages(source,job['dpi'],40):
                        ref_page=simpledialog.askinteger('Страница образца',f'{source.name}, страница {n}\nНомер соответствующей страницы образца (для картинки — 1).',initialvalue=1,minvalue=1,parent=root)
                        if ref_page is None:break
                        ref_entry=next((x for x in pages(reference,job['dpi'],40) if x[0]==ref_page),None)
                        if ref_entry is None:raise ValueError('Нет такой страницы в образце')
                        profile_path=output/'profiles'/f'{index:03d}_page_{n:04d}.json'
                        editor=FormEditor(root,orient(rgb,rotation),orient(ref_entry[1],ref_rotate),source,reference,n,ref_page,job['dpi'],rotation,ref_rotate,profile_path)
                        if editor.result:job['page_profiles'][str(n)]={'profile':editor.result,'reference':str(reference),'reference_page':ref_page,'reference_rotate':ref_rotate}
                job=normalize_job(job,Path.cwd(),{})
                jobs.append(job);write_json(output/'batch_config.json',{'schema':1,'documents':jobs})
                result=run_document(job,output,index,model_holder)
            except (Exception,SystemExit) as error:
                result={'status':'error','input':str(source),'error':str(error)};messagebox.showerror('Ошибка документа',str(error),parent=root)
            results.append(result);update_index(output,results)
            if result['status']=='ok':webbrowser.open((Path(result['output'])/'comparison.html').as_uri())
            if index<len(files) and not messagebox.askyesno('Следующий документ',f'Готово: {source.name}\nПерейти к следующему документу?',parent=root):break
        update_index(output,results);print('Results:',output/'index.html');webbrowser.open((output/'index.html').as_uri())
    finally:root.destroy()
    return int(any(x['status']=='error' for x in results))


if __name__=='__main__':raise SystemExit(main())
