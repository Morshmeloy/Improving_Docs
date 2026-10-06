"""Optional manual pixel protection selector (Tkinter, Windows/Linux desktop)."""
import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from enhance_batch import pages, SUPPORTED


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,default=Path('protected_regions.json'))
    args=p.parse_args()
    paths=sorted(x for x in args.input.iterdir() if x.is_file() and x.suffix.lower() in SUPPORTED) if args.input.is_dir() else [args.input]
    if not paths: p.error('No supported files')
    result={}
    if args.output.exists():
        result=json.loads(args.output.read_text(encoding='utf-8-sig'))
    root=tk.Tk();root.title('Protect signatures and stamps — drag rectangles; Next to confirm')
    state={'image':None,'start':None,'rect':None,'boxes':[],'key':None}
    label=tk.Label(root);label.pack()
    canvas=tk.Canvas(root,bg='#555',width=780,height=780);canvas.pack()
    buttons=tk.Frame(root);buttons.pack()
    def save():
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def next_page():
        if state['key']:
            name,n=state['key']
            result.setdefault(name,{})[str(n)]=state['boxes']
            save()
        try:
            name,n,rgb=next(iterator)
        except StopIteration:
            messagebox.showinfo('Saved',str(args.output.resolve()));root.destroy();return
        im=Image.fromarray(rgb);im.thumbnail((780,780))
        state.update(image=ImageTk.PhotoImage(im),key=(name,n),boxes=[],start=None,rect=None,w=im.width,h=im.height)
        canvas.config(width=im.width,height=im.height);canvas.delete('all')
        canvas.create_image(0,0,anchor='nw',image=state['image'])
        state['boxes'] = result.get(name, {}).get(str(n), []).copy()
        for x0,y0,x1,y1 in state['boxes']:
            canvas.create_rectangle(x0*im.width,y0*im.height,x1*im.width,y1*im.height,outline='red',width=2)
        label.config(text=f'{name} | Page {n}. Drag rectangles over ALL signatures/stamps. Next confirms.')
    def start(e):
        state['start']=(max(0,min(e.x,state['w'])),max(0,min(e.y,state['h'])))
        state['rect']=canvas.create_rectangle(*state['start'],*state['start'],outline='red',width=2)
    def move(e):
        if state['start']:
            canvas.coords(state['rect'],*state['start'],max(0,min(e.x,state['w'])),max(0,min(e.y,state['h'])))
    def end(e):
        if state['start']:
            move(e);x0,y0,x1,y1=canvas.coords(state['rect'])
            x0,x1=sorted((x0,x1));y0,y1=sorted((y0,y1))
            if x1-x0>2 and y1-y0>2:
                state['boxes'].append([x0/state['w'],y0/state['h'],x1/state['w'],y1/state['h']])
            else: canvas.delete(state['rect'])
            state['start']=None
    def reset():
        state['boxes']=[]
        for x in canvas.find_all():
            if canvas.type(x)=='rectangle':canvas.delete(x)
    def frames():
        for path in paths:
            for n,rgb,_,_ in pages(path,100,40):yield path.name,n,rgb
    iterator=iter(frames())
    canvas.bind('<ButtonPress-1>',start);canvas.bind('<B1-Motion>',move);canvas.bind('<ButtonRelease-1>',end)
    tk.Button(buttons,text='Reset current page',command=reset).pack(side='left')
    tk.Button(buttons,text='Next / Save',command=next_page).pack(side='left')
    def close():
        if state['key']:
            name,n=state['key']
            result.setdefault(name,{})[str(n)]=state['boxes']
            save()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW',close)
    next_page();root.mainloop()

if __name__=='__main__':main()
