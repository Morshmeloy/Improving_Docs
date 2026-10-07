"""Reference-guided blank-form reconstruction, with an explicit static allowlist.

The learned ink reconstruction is the base. Reference content is never globally
blended: only approved static regions are mapped onto that base with OpenCV.
"""
import argparse
import html
import json
from pathlib import Path
import shutil
import cv2
import fitz
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from enhance_batch import pages, png_bytes, add_page, verify_pdf_rasters, sha256
from ai_restore import orient
from ink_reconstruct import bounds


def load_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def transfer_static(base, reference, profile, original=None):
    """Return reconstructed raster and exact provenance mask; reject protected overlap."""
    h, w = base.shape[:2]
    rh, rw = reference.shape[:2]
    if profile.get('schema') not in {1, 2} or not profile.get('regions') or not profile.get('protected'):
        raise ValueError('Profile requires schema=1, static regions and protected content boxes')
    protected = np.zeros((h, w), bool)
    for box in profile['protected']:
        x0, y0, x1, y1 = bounds(box, w, h)
        protected[y0:y1, x0:x1] = True
    result = base.copy()
    mask = np.zeros((h, w), bool)
    records = []
    for region in profile['regions']:
        if region.get('kind') not in {'static_heading', 'ornamental_frame', 'static_label', 'static_symbol'}:
            raise ValueError('Only static form elements may be transferred')
        tx0, ty0, tx1, ty1 = bounds(region['target'], w, h)
        sx0, sy0, sx1, sy1 = bounds(region['reference'], rw, rh)
        if protected[ty0:ty1, tx0:tx1].any():
            raise ValueError('Static transfer overlaps protected content: ' + region['name'])
        patch = reference[sy0:sy1, sx0:sx1]
        height, width = ty1-ty0, tx1-tx0
        mode = region.get('mode', 'projective')
        if mode == 'source_guided_label':
            if original is None or original.shape != base.shape:
                raise ValueError('Source-guided typography requires the original raster')
            if not region.get('verified_text') or not region.get('reviewed'):
                raise ValueError('Source-guided labels require a reviewed reference identity')
            from source_typography import repair_source_label
            mapped, transform = repair_source_label(original[ty0:ty1,tx0:tx1], patch)
        elif mode == 'verified_label_text':
            if not region.get('verified_text') or not region.get('reviewed'):
                raise ValueError('Typesetting requires manually verified static reference text')
            font_paths = [Path('C:/Windows/Fonts/timesbd.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf')]
            font_path = next((x for x in font_paths if x.is_file()), None)
            if font_path is None: raise ValueError('Install a Cyrillic serif bold font: timesbd.ttf or DejaVuSerif-Bold.ttf')
            font = ImageFont.truetype(str(font_path),120)
            text = region['verified_text']
            left,top,right,bottom = font.getbbox(text)
            canvas = Image.new('L',(right-left,bottom-top),255)
            ImageDraw.Draw(canvas).text((-left,-top),text,font=font,fill=0)
            glyph = np.array(canvas)
            mapped_gray = cv2.resize(glyph,(width,height),interpolation=cv2.INTER_AREA)
            mapped = np.repeat(mapped_gray[:,:,None],3,axis=2)
            transform = {'method':'typesetting of manually verified static text from reference', 'font':font_path.name, 'original_typography_exact':False}
        elif mode == 'ink_label':
            if not region.get('verified_text') or not region.get('reviewed'):
                raise ValueError('Ink labels require manually reviewed reference transcription')
            # Isolate achromatic printed ink; reject colored signatures/seals.
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            chroma = patch.max(axis=2).astype(np.int16)-patch.min(axis=2).astype(np.int16)
            ink = (gray < 160) & (chroma < 65)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
            keep = np.zeros(count,bool)
            if count>1: keep[1:] = stats[1:,cv2.CC_STAT_AREA]>=2
            ink = keep[labels]
            ys,xs = np.where(ink)
            if not len(xs): raise ValueError('Reference label has no usable black ink')
            glyph = (255*(~ink[ys.min():ys.max()+1,xs.min():xs.max()+1])).astype(np.uint8)
            mapped_gray = cv2.resize(glyph,(width,height),interpolation=cv2.INTER_CUBIC)
            mapped = np.repeat(mapped_gray[:,:,None],3,axis=2)
            transform = {'method':'achromatic reference glyph segmentation + bounding-box registration', 'reference_ink_bbox':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]}
        elif mode == 'projective':
            ph, pw = patch.shape[:2]
            # Explicit four-corner registration: unrelated body text must not
            # participate in feature matching or bias the registration.
            src = np.float32([[0,0],[pw-1,0],[pw-1,ph-1],[0,ph-1]])
            dst = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
            matrix = cv2.getPerspectiveTransform(src, dst)
            mapped = cv2.warpPerspective(patch, matrix, (width,height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            transform = matrix.tolist()
        elif mode == 'tile_x':
            tile_width = max(1, round(patch.shape[1]*height/patch.shape[0]))
            tile = cv2.resize(patch, (tile_width,height), interpolation=cv2.INTER_CUBIC)
            mapped = np.tile(tile, (1, (width+tile_width-1)//tile_width, 1))[:, :width]
            transform = {'tile_width': tile_width, 'note': 'Repeated ornament; not recovered original pixels'}
        else:
            raise ValueError('Unknown transfer mode')
        result[ty0:ty1,tx0:tx1] = mapped
        mask[ty0:ty1,tx0:tx1] = True
        records.append({**region, 'target_pixels': [tx0,ty0,tx1,ty1], 'reference_pixels': [sx0,sy0,sx1,sy1], 'transform': transform})
    if not np.array_equal(result[~mask], base[~mask]) or not np.array_equal(result[protected], base[protected]):
        raise AssertionError('Content preservation invariant failed')
    return result, mask, protected, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Original document; unchanged and hashed')
    parser.add_argument('--base', required=True, type=Path, help='Upright learned-ink reconstructed page PNG')
    parser.add_argument('--reference', required=True, type=Path, help='User-selected blank-form example PNG/JPEG')
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--rotate', type=int, choices=[0,90,180,270], default=0)
    args = parser.parse_args(argv)
    if args.output.exists() or args.page < 1 or not 72 <= args.dpi <= 600:
        parser.error('Output must be new, page positive, dpi 72..600')
    base, reference = load_rgb(args.base), load_rgb(args.reference)
    profile = json.loads(args.profile.read_text(encoding='utf-8-sig'))
    for key, path in [('source_sha256', args.source), ('reference_sha256', args.reference)]:
        if profile.get(key) and sha256(path) != profile[key]:
            parser.error('Profile is bound to a different ' + key + '; create a matching profile')
    selected = next((entry for entry in pages(args.source, args.dpi, 40) if entry[0] == args.page), None)
    if selected is None:
        parser.error('Page not found')
    _, original, size, _ = selected
    original = orient(original, args.rotate)
    if original.shape != base.shape:
        parser.error('Base shape must match source page at --dpi and --rotate')
    result, mask, protected, records = transfer_static(base, reference, profile, original)
    args.output.mkdir(parents=True)
    for filename, image in [('original.png',original), ('before.png',base), ('reference.png',reference), ('reconstructed.png',result), ('transferred_mask.png',np.repeat((mask*255).astype(np.uint8)[:,:,None],3,axis=2))]:
        (args.output/filename).write_bytes(png_bytes(image))
    overlay = base.copy(); overlay[mask] = [220,50,150]
    (args.output/'transfer_overlay.png').write_bytes(png_bytes(overlay))
    shutil.copy2(args.profile, args.output/'profile.json')
    document = fitz.open()
    if args.rotate in {90,270}: size = (size[1],size[0])
    add_page(document, size, png_bytes(result))
    document.set_metadata({'title':'EDITED DRAFT: static form from reference; content from source-derived base', 'producer':'Improving Docs template reconstruction'})
    document.save(args.output/'template_reconstruction_draft.pdf', deflate=True)
    document.close()
    verify_pdf_rasters(args.output/'template_reconstruction_draft.pdf',[args.output/'reconstructed.png'])
    report = {'method':'user-provided reference + OpenCV region registration on learned-ink base', 'source_sha256':sha256(args.source), 'base_sha256':sha256(args.base), 'reference_sha256':sha256(args.reference), 'profile_sha256':sha256(args.profile), 'page':args.page, 'dpi':args.dpi, 'rotate_clockwise':args.rotate, 'regions':records, 'transferred_pixels':int(mask.sum()), 'outside_transfer_equal_to_base':bool(np.array_equal(result[~mask],base[~mask])), 'protected_content_equal_to_base':bool(np.array_equal(result[protected],base[protected])), 'signature_from_reference':False, 'semantic_accuracy_verified':False, 'pdf_rasters_verified':True}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>.row{display:flex}figure{width:32%;margin:6px}img{width:100%}</style><h1>'+html.escape(args.source.name)+'</h1><p>Edited draft. Only approved static form regions were transferred. Body, numbers, seals and signatures remain exactly equal to the learned-ink base. Pink = transferred reference pixels.</p><div class="row">'+''.join('<figure><figcaption>'+label+'</figcaption><a href="'+name+'"><img src="'+name+'"></a></figure>' for name,label in [('before.png','Learned ink base'),('reconstructed.png','Reference-guided static form'),('transfer_overlay.png','Transfer provenance')])+'</div>',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
