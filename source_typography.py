"""Preserve observed glyph geometry; use a reference only to confirm small gaps."""
import cv2
import numpy as np


def observed_ink(rgb, threshold=210):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    channels = rgb.astype(np.int16)
    ink = (gray < threshold) & (channels.max(axis=2)-channels.min(axis=2)<35)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    keep = np.zeros(count,bool)
    if count>1: keep[1:] = stats[1:,cv2.CC_STAT_AREA]>=3
    return keep[labels]


def ink_bounds(mask):
    yy,xx = np.where(mask)
    if not len(xx): return None
    return [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]


def repair_source_label(original, reference):
    """No fonts, no typesetting, no scaling of the source letters.

    Ref strokes may fill only holes that a 3x3 closing also proposes, within the
    source ink envelope. This keeps additions local and prevents wholesale
    reference substitution when two forms use different typefaces.
    """
    source = observed_ink(original)
    src_box = ink_bounds(source)
    if src_box is None:
        return original.copy(), {'status':'no_source_ink', 'inferred_pixels':0, 'font_substituted':False}
    ref = observed_ink(reference,170)
    ref_box = ink_bounds(ref)
    candidates = np.zeros_like(source)
    score = 0.
    if ref_box is not None:
        x0,y0,x1,y1 = src_box
        rx0,ry0,rx1,ry1 = ref_box
        scaled = cv2.resize(ref[ry0:ry1,rx0:rx1].astype(np.uint8),(x1-x0,y1-y0),interpolation=cv2.INTER_NEAREST)>0
        proposal = np.zeros_like(source); proposal[y0:y1,x0:x1] = scaled
        # Translation-only fine alignment: source image is never warped.
        near = cv2.dilate(source.astype(np.uint8),np.ones((3,3),np.uint8))>0
        best = None
        for dy in range(-2,3):
            for dx in range(-2,3):
                moved = cv2.warpAffine(proposal.astype(np.uint8),np.float32([[1,0,dx],[0,1,dy]]),(source.shape[1],source.shape[0]))>0
                quality = float((moved & near).sum()/max(1,moved.sum()))
                if best is None or quality>score:
                    best,score = moved,quality
        closed = cv2.morphologyEx(source.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
        envelope = np.zeros_like(source);envelope[y0:y1,x0:x1]=True
        if score>=.55:
            candidates = best & closed & ~source & envelope
        # Too many proposed repairs means the registration is not trustworthy.
        limit = max(3, int(.25*source.sum()))
        if candidates.sum()>limit:
            # Keep the proposals nearest to source ink rather than dropping
            # every repair when a fragmented word exceeds the budget.
            distance = cv2.distanceTransform((~source).astype(np.uint8),cv2.DIST_L2,3)
            yy,xx = np.where(candidates)
            chosen = np.argsort(distance[yy,xx],kind='stable')[:limit]
            candidates[:]=False; candidates[yy[chosen],xx[chosen]]=True
    # Darken the observed strokes continuously, retaining their original raster
    # edge shape. No white rectangle and no replacement glyph outlines.
    result = original.copy()
    darkness = 255-original.astype(np.float32)
    darkened = np.clip(255-3.4*darkness,0,255).round().astype(np.uint8)
    result[source] = darkened[source]
    result[candidates] = 75
    return result, {'status':'source_strokes_reinforced', 'font_substituted':False, 'source_scaled':False, 'source_ink_bbox':src_box, 'reference_ink_bbox':ref_box, 'reference_support_score':score, 'observed_ink_pixels':int(source.sum()), 'inferred_pixels':int(candidates.sum()), 'output_ink_bbox':ink_bounds(source|candidates), 'ink_envelope_equal':ink_bounds(source|candidates)==src_box}
