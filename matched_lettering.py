"""Crisp verified labels, using source ink dimensions rather than region dimensions."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from source_typography import observed_ink, ink_bounds
from setup_fonts import ROOT, FILES, verify


def fit_glyph(glyph, shape, box):
    h,w=shape; x0,y0,x1,y1=box
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Glyph box outside label region')
    result=np.full((h,w),255,np.uint8)
    result[y0:y1,x0:x1]=cv2.resize(glyph,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
    return result


def measure_source_ink(original):
    for threshold in [210,225,235]:
        box=ink_bounds(observed_ink(original,threshold))
        if box is not None:return box,threshold
    return None,None


def matched_label(original, text, weight='bold'):
    box,threshold=measure_source_ink(original)
    if box is None:raise ValueError('Cannot fit type size: no source strokes in this label')
    name='Tinos-Bold.ttf' if weight=='bold' else 'Tinos-Regular.ttf'
    font_path=ROOT/name;sha,size=FILES[name]
    if not font_path.is_file() or not verify(font_path.read_bytes(),sha,size):
        raise ValueError('Install verified fonts first: python setup_fonts.py')
    font=ImageFont.truetype(str(font_path),160)
    left,top,right,bottom=font.getbbox(text)
    canvas=Image.new('L',(right-left,bottom-top),255)
    ImageDraw.Draw(canvas).text((-left,-top),text,font=font,fill=0)
    glyph=np.array(canvas)
    mapped=fit_glyph(glyph,original.shape[:2],box)
    x0,y0,x1,y1=box
    transform={'method':'crisp verified text fitted to original observed ink dimensions','font':name,'source_ink_bbox':box,'source_ink_threshold':threshold,'rendered_ink_box':box,'target_ink_width':x1-x0,'target_ink_height':y1-y0,'region_width':original.shape[1],'region_height':original.shape[0],'scale_x':(x1-x0)/glyph.shape[1],'scale_y':(y1-y0)/glyph.shape[0],'font_exactly_identified':False,'content_from_reference':False}
    return np.repeat(mapped[:,:,None],3,axis=2),transform
