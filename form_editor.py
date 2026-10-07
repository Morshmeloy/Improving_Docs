"""Visual, reviewed mapping between source static fields and a reference form."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image,ImageTk
from enhance_batch import sha256
from ink_reconstruct import bounds


class FormEditor:
    def __init__(self,parent,source_rgb,reference_rgb,source,reference,page,reference_page,dpi,rotation,reference_rotation,save_path):
        self.result=None;self.rows=[];self.boxes=[None,None];self.images=[Image.fromarray(source_rgb),Image.fromarray(reference_rgb)];self.panes=[];self.start=[None,None];self.factor=1.
        self.metadata={'schema':3,'protection_policy':'outside_approved_regions','source_sha256':sha256(source),'reference_sha256':sha256(reference),'source_page':page,'reference_page':reference_page,'expected_dpi':dpi,'expected_rotation':rotation,'reference_rotation':reference_rotation}
        self.save_path=Path(save_path);self.win=tk.Toplevel(parent);self.win.title(f'{Path(source).name} — страница {page}: каркас по образцу')
        tk.Label(self.win,text='Слева исходник, справа образец. Выделите одну постоянную надпись на обоих изображениях.\nВпишите проверенный текст и нажмите «Добавить». Реквизиты и росчерки не включайте в каркас.').pack()
        controls=tk.Frame(self.win);controls.pack(fill='x')
        tk.Button(controls,text='Загрузить / проверить готовый профиль',command=self.load).pack(side='left')
        tk.Button(controls,text='Увеличить',command=lambda:self.zoom(1.4)).pack(side='left');tk.Button(controls,text='Уменьшить',command=lambda:self.zoom(1/1.4)).pack(side='left')
        body=tk.Frame(self.win);body.pack()
        for i in range(2):
            frame=tk.Frame(body);frame.pack(side='left');tk.Label(frame,text='Исходный документ' if i==0 else 'Образец').grid(row=0,column=0)
            canvas=tk.Canvas(frame,width=510,height=570,bg='#555');canvas.grid(row=1,column=0)
            sy=tk.Scrollbar(frame,command=canvas.yview);sy.grid(row=1,column=1,sticky='ns');sx=tk.Scrollbar(frame,orient='horizontal',command=canvas.xview);sx.grid(row=2,column=0,sticky='ew');canvas.config(xscrollcommand=sx.set,yscrollcommand=sy.set)
            self.panes.append({'canvas':canvas,'base_scale':min(510/self.images[i].width,570/self.images[i].height)})
            canvas.bind('<ButtonPress-1>',lambda e,j=i:self.begin(e,j));canvas.bind('<B1-Motion>',lambda e,j=i:self.drag(e,j));canvas.bind('<ButtonRelease-1>',lambda e,j=i:self.end(e,j))
        bar=tk.Frame(self.win);bar.pack(fill='x');self.kind=tk.StringVar(value='Надпись');tk.OptionMenu(bar,self.kind,'Надпись','Рамка','Заголовок из образца','Знак из образца','Защитить область').pack(side='left')
        self.text=tk.Entry(bar,width=49);self.text.pack(side='left');self.weight=tk.StringVar(value='bold');tk.OptionMenu(bar,self.weight,'bold','regular').pack(side='left');tk.Button(bar,text='Добавить',command=self.add).pack(side='left')
        self.list=tk.Listbox(self.win,width=125,height=7);self.list.pack();bottom=tk.Frame(self.win);bottom.pack(fill='x')
        tk.Button(bottom,text='Удалить выбранное',command=self.remove).pack(side='left');tk.Button(bottom,text='Сохранить проверенный каркас',command=self.save).pack(side='right');tk.Button(bottom,text='Эта страница без каркаса',command=self.win.destroy).pack(side='right')
        self.redraw();parent.wait_window(self.win)

    def redraw(self):
        for i,pane in enumerate(self.panes):
            scale=pane['base_scale']*self.factor;im=self.images[i];w,h=max(1,round(im.width*scale)),max(1,round(im.height*scale));pane['size']=(w,h);pane['photo']=ImageTk.PhotoImage(im.resize((w,h)));c=pane['canvas'];c.delete('all');c.config(scrollregion=(0,0,w,h));c.create_image(0,0,image=pane['photo'],anchor='nw')
            for row in self.rows:
                box=row.get('target') if i==0 else row.get('reference')
                if box:c.create_rectangle(box[0]*w,box[1]*h,box[2]*w,box[3]*h,outline='cyan' if row['kind']=='protect' else '#ff6666',width=2)
            if self.boxes[i]:
                b=self.boxes[i];c.create_rectangle(b[0]*w,b[1]*h,b[2]*w,b[3]*h,outline='lime',width=2)
        self.list.delete(0,'end')
        for row in self.rows:self.list.insert('end',row.get('verified_text',row['name'])+' | '+row['kind'])

    def zoom(self,value):self.factor=max(.5,min(8,self.factor*value));self.redraw()
    def point(self,event,i):
        p=self.panes[i];w,h=p['size'];return max(0,min(1,p['canvas'].canvasx(event.x)/w)),max(0,min(1,p['canvas'].canvasy(event.y)/h))
    def begin(self,event,i):self.start[i]=self.point(event,i);self.boxes[i]=None
    def drag(self,event,i):
        if self.start[i]:
            x,y=self.point(event,i);a,b=self.start[i];self.boxes[i]=[min(a,x),min(b,y),max(a,x),max(b,y)];self.redraw()
    def end(self,event,i):self.drag(event,i);self.start[i]=None

    def add(self):
        kind=self.kind.get();target=self.boxes[0]
        if not target or target[2]<=target[0] or target[3]<=target[1]:messagebox.showerror('Область','Выделите область на исходнике.',parent=self.win);return
        row={'name':f'region_{len(self.rows)+1}','target':target}
        if kind=='Защитить область':row['kind']='protect'
        else:
            ref=self.boxes[1]
            if not ref or ref[2]<=ref[0] or ref[3]<=ref[1]:messagebox.showerror('Образец','Выделите соответствующий фрагмент образца.',parent=self.win);return
            row['reference']=ref
            row['kind']={'Надпись':'static_label','Рамка':'ornamental_frame','Заголовок из образца':'static_heading','Знак из образца':'static_symbol'}[kind]
            row['mode']='metric_matched_text' if kind=='Надпись' else 'projective'
            if kind=='Надпись':
                if not self.text.get().strip():messagebox.showerror('Текст','Впишите точную постоянную надпись, проверенную по исходнику и образцу.',parent=self.win);return
                row.update(verified_text=self.text.get().strip(),reviewed=True,weight=self.weight.get())
        self.rows.append(row);self.boxes=[None,None];self.text.delete(0,'end');self.redraw()

    def remove(self):
        selected=self.list.curselection()
        if selected:del self.rows[selected[0]];self.redraw()

    def load(self):
        path=filedialog.askopenfilename(parent=self.win,title='Профиль для визуальной проверки',filetypes=[('JSON','*.json')])
        if not path:return
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8-sig'));self.rows=list(data['regions'])+[{'name':'protected','kind':'protect','target':b} for b in data.get('protected',[])]
            # Coordinates may be reused ONLY after the user reviews both images
            # and saves a profile bound to this exact source/reference/page.
            self.redraw()
        except Exception as error:messagebox.showerror('Профиль',str(error),parent=self.win)

    def save(self):
        regions=[r for r in self.rows if r['kind']!='protect'];protected=[r['target'] for r in self.rows if r['kind']=='protect']
        if not regions:messagebox.showerror('Каркас','Добавьте хотя бы одну область или выберите обработку без каркаса.',parent=self.win);return
        w,h=self.images[0].size
        try:
            import numpy as np
            locked=np.zeros((h,w),bool)
            for b in protected:
                x0,y0,x1,y1=bounds(b,w,h);locked[y0:y1,x0:x1]=True
            for r in regions:
                x0,y0,x1,y1=bounds(r['target'],w,h)
                if locked[y0:y1,x0:x1].any():raise ValueError('Каркас пересекает защищённую область: '+r['name'])
                bounds(r['reference'],self.images[1].width,self.images[1].height)
            data={**self.metadata,'regions':regions,'protected':protected};self.save_path.parent.mkdir(parents=True,exist_ok=True);self.save_path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');self.result=str(self.save_path.resolve());self.win.destroy()
        except Exception as error:messagebox.showerror('Проверка',str(error),parent=self.win)
