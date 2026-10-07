#!/usr/bin/env python3
"""Sequential, non-generative document enhancement. Python 3.12+."""
from __future__ import annotations
import argparse
import hashlib
import html
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import cv2
import fitz
import numpy as np
from PIL import Image, ImageOps, ImageSequence
from cv_pipeline import restore, diagnostics
from trace_pipeline import trace_strokes

SUPPORTED = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}
LOG = logging.getLogger('enhancer')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def protection_mask(rgb, boxes=(), color=True):
    """Exact pixel locks; color is a heuristic, not a stamp detector."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if color:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Also catch relatively pale colored ink. Dilation keeps antialiased edges.
        mask[(hsv[..., 1] >= 18) & (hsv[..., 2] >= 35)] = 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    for b in boxes:
        if len(b) != 4 or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in b):
            raise ValueError('Each protected box must have four finite numbers')
        x0, y0, x1, y1 = b
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError('Protected boxes require 0 <= x0 < x1 <= 1, same for y')
        mask[math.floor(y0*h):math.ceil(y1*h), math.floor(x0*w):math.ceil(x1*w)] = 255
    return mask.astype(bool)


def enhance(rgb, mode, boxes=(), strength=1.):
    """Generate a contrast candidate and restore exact locked source pixels."""
    extra = {}
    if mode == 'trace_study':
        out, extra = trace_strokes(rgb, strength)
        mask = protection_mask(rgb, boxes, color=False)
    elif mode in {'cv_balanced', 'cv_detail', 'ink_study'}:
        out, extra = restore(rgb, mode, strength)
        mask = protection_mask(rgb, boxes, color=mode != 'ink_study')
    elif mode == 'photo':
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        local = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(lab[..., 0])
        lab[..., 0] = np.round(.65*lab[..., 0] + .35*local).astype(np.uint8)
        base = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
        blurred = cv2.GaussianBlur(base, (0, 0), .8)
        out = np.clip(base + .18*(base-blurred), 0, 255).round().astype(np.uint8)
        mask = protection_mask(rgb, boxes, color=False)
    else:
        gamma, strength = (1.8, .15) if mode == 'safe' else (3.0, .35)
        base = rgb.astype(np.float32)
        y = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        target = 255 * np.power(y/255, gamma)
        # Multiplication applies the same gain to R/G/B, rather than grayscale conversion.
        gain = np.divide(target, y, out=np.ones_like(y), where=y > 0)
        base *= gain[..., None]
        blurred = cv2.GaussianBlur(base, (0, 0), .65)
        # Darken edge cores only: no light halos that would erase faint strokes.
        out = np.clip(base + strength*np.minimum(base-blurred, 0), 0, 255).round().astype(np.uint8)
        mask = protection_mask(rgb, boxes)
    out[mask] = rgb[mask]
    if np.any(out[mask] != rgb[mask]):
        raise RuntimeError('Protected pixels changed')
    stats = {'protected_pixels': int(mask.sum()), 'protected_fraction': float(mask.mean()),
             'protected_pixels_equal': True, 'width': rgb.shape[1], 'height': rgb.shape[0]}
    stats.update(extra)
    stats.update(diagnostics(rgb, out, mask))
    return out, mask, stats


def pages(path, dpi, max_mp):
    if path.suffix.lower() == '.pdf':
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError('Password-protected PDF is not supported')
            signed = doc.get_sigflags() > 0
            for n, page in enumerate(doc, 1):
                area = page.rect.width * page.rect.height
                scale = min(dpi / 72, math.sqrt(max_mp*1e6 / max(area, 1)))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
                rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
                native = [{'width': x[2], 'height': x[3]} for x in page.get_images()]
                yield n, rgb, (page.rect.width, page.rect.height), {'render_dpi': scale*72,
                    'embedded_images': native, 'digital_signature_present': signed}
    else:
        with Image.open(path) as src:
            for n, frame in enumerate(ImageSequence.Iterator(src), 1):
                frame = ImageOps.exif_transpose(frame.copy())
                if frame.width * frame.height > max_mp*1e6:
                    raise ValueError('Image exceeds --max-megapixels; increase limit explicitly')
                rgba = frame.convert('RGBA')
                white = Image.new('RGBA', rgba.size, 'white')
                white.alpha_composite(rgba)
                rgb = np.array(white.convert('RGB'))
                yield n, rgb, (frame.width*72/dpi, frame.height*72/dpi), {'render_dpi': dpi,
                    'note': 'Image pixels unchanged in size; PDF physical size uses --dpi.'}


def png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='PNG', compress_level=6)
    return buf.getvalue()


def add_page(doc, size, data):
    p = doc.new_page(width=size[0], height=size[1])
    p.insert_image(p.rect, stream=data)



def verify_pdf_rasters(pdf_path, expected_paths):
    """Reject damaged PDF structures and altered embedded samples before publishing."""
    with fitz.open(pdf_path) as pdf:
        if len(pdf) != len(expected_paths):
            raise RuntimeError('Saved PDF page count mismatch')
        for page, expected_path in zip(pdf, expected_paths):
            images = page.get_images()
            if len(images) != 1:
                raise RuntimeError('Expected one raster per output page')
            pix = fitz.Pixmap(pdf, images[0][0])
            expected = np.array(Image.open(expected_path).convert('RGB'))
            if pix.n != 3 or pix.alpha:
                raise RuntimeError('Saved PDF is not RGB')
            saved = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            if not np.array_equal(saved, expected):
                raise RuntimeError('Saved PDF pixel verification failed')
    return True


def process(path, output, args, region_map):
    digest = sha256(path)
    target = output / (path.stem[:80] + '__' + digest[:12])
    if target.exists():
        raise FileExistsError(f'Output already exists: {target}; use a new output directory')
    stage = Path(tempfile.mkdtemp(prefix='.working_', dir=output))
    docs = {}
    try:
        shutil.copy2(path, stage / ('source' + path.suffix.lower()))
        modes = ['photo'] if args.mode == 'photo' else ['safe', 'readable', 'cv_balanced', 'cv_detail']
        if args.ink_study and args.mode == 'document':
            modes.append('ink_study')
        if args.trace_study and args.mode == 'document':
            modes.append('trace_study')
        docs = {m: fitz.open() for m in modes}
        record = {'source_name': path.name, 'source_sha256': digest, 'mode': args.mode,
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'pages': [],
                  'notice': 'Derivative images only; source is retained. No proof of authenticity or semantic equivalence.'}
        cards = []
        for n, rgb, size, info in pages(path, args.dpi, args.max_megapixels):
            boxes = region_map.get(path.name, {}).get(str(n), [])
            prefix = f'page_{n:04d}'
            Image.fromarray(rgb).save(stage / f'{prefix}_original.png')
            row = {'page': n, 'source': info, 'variants': {},
                   'source_raster_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
            imgs = [(f'{prefix}_original.png', 'Исходник')]
            for mode in modes:
                out, mask, stats = enhance(rgb, mode, boxes, args.trace_strength if mode == 'trace_study' else args.strength)
                data = png_bytes(out)
                (stage / f'{prefix}_{mode}.png').write_bytes(data)
                Image.fromarray((mask*255).astype(np.uint8)).save(stage / f'{prefix}_{mode}_protected.png')
                delta = np.abs(out.astype(np.int16)-rgb.astype(np.int16)).max(axis=2)
                heat = np.stack((delta, np.zeros_like(delta), np.zeros_like(delta)), axis=2).astype(np.uint8)
                Image.fromarray(heat).save(stage / f'{prefix}_{mode}_changes.png')
                add_page(docs[mode], size, data)
                row['variants'][mode] = stats
                imgs.append((f'{prefix}_{mode}.png', mode))
            record['pages'].append(row)
            cards.append('<h2>Страница '+str(n)+'</h2><div class="row">'+''.join(
                '<figure><figcaption>'+html.escape(label)+'</figcaption><a href="'+name+'"><img src="'+name+'"></a></figure>'
                for name, label in imgs)+'</div>')
            LOG.info('%s: page %s finished', path.name, n)
        if not record['pages']:
            raise ValueError('No pages found')
        for mode, doc in docs.items():
            doc.set_metadata({'title': f'Enhanced derivative ({mode})', 'producer': 'Improving Docs 2.1 experimental'})
            doc.save(stage / f'enhanced_{mode}.pdf', deflate=True, garbage=0)
            doc.close()
            expected_paths = [stage / f'page_{r["page"]:04d}_{mode}.png' for r in record['pages']]
            verify_pdf_rasters(stage / f'enhanced_{mode}.pdf', expected_paths)
        for row in record['pages']:
            with Image.open(stage / f'page_{row["page"]:04d}_original.png') as saved_source:
                samples = np.array(saved_source.convert('RGB'))
            if hashlib.sha256(samples.tobytes()).hexdigest() != row['source_raster_sha256']:
                raise RuntimeError('Saved source PNG pixel verification failed')
        record['saved_source_rasters_verified'] = True
        record['saved_pdf_rasters_verified'] = True
        record['outputs'] = {p.name: sha256(p) for p in stage.iterdir() if p.is_file()}
        (stage / 'report.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        (stage / 'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Сравнение</title>'
            '<style>body{font:16px sans-serif;background:#ddd;margin:20px}.row{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}figcaption{padding:8px;background:white}a{display:block}</style>'
            '<h1>'+html.escape(path.name)+'</h1><p>Нажмите изображение для полного размера. Белые зоны *_protected.png сохранены точно. Серые подписи вне этих зон обработаны вместе с текстом. ink_study и trace_study меняют чернила и требует ручной проверки; ручные прямоугольники защищены во всех вариантах. *_changes.png показывает величину изменения красным. Нет автоматической оценки достоверности букв.</p>'+''.join(cards), encoding='utf-8')
        stage.rename(target)
        return {'source': str(path), 'status': 'ok', 'output': str(target), 'pages': len(record['pages'])}
    except Exception:
        for doc in docs.values():
            if not doc.is_closed:
                doc.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path, help='Files and/or folders')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['document', 'photo'], default='document')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--max-megapixels', type=float, default=40)
    parser.add_argument('--regions', type=Path, help='JSON: filename -> page number -> normalized boxes')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--strength', type=float, default=1., help='CV gain multiplier, 0.25..2')
    parser.add_argument('--ink-study', action='store_true', help='Extra experimental candidate; modifies colored ink outside manual locks')
    parser.add_argument('--trace-study', action='store_true', help='Extra experimental directional stroke enhancement; manual locks only')
    parser.add_argument('--trace-strength', type=float, default=.8, help='Trace gain, 0.25..2; default 0.8')
    args = parser.parse_args(argv)
    if not 72 <= args.dpi <= 600 or not math.isfinite(args.max_megapixels) or args.max_megapixels <= 0:
        parser.error('DPI must be 72..600; max-megapixels must be finite and positive')
    if not math.isfinite(args.strength) or not .25 <= args.strength <= 2:
        parser.error('--strength must be 0.25..2')
    if not math.isfinite(args.trace_strength) or not .25 <= args.trace_strength <= 2:
        parser.error('--trace-strength must be 0.25..2')
    if args.trace_study and args.mode == 'photo':
        parser.error('--trace-study is for document mode')
    cv2.setNumThreads(max(1, min(4, os.cpu_count() or 1)))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    region_map = json.loads(args.regions.read_text(encoding='utf-8-sig')) if args.regions else {}
    files, failures = [], []
    for item in args.inputs:
        item = item.resolve()
        if item.is_dir():
            candidates = sorted(item.rglob('*') if args.recursive else item.iterdir())
            files.extend(p for p in candidates if p.is_file() and p.suffix.lower() in SUPPORTED and not p.is_relative_to(output))
        elif item.is_file() and item.suffix.lower() in SUPPORTED:
            if item.is_relative_to(output):
                parser.error('Input must be outside output folder')
            files.append(item)
        else:
            failures.append({'source': str(item), 'status': 'error', 'error': 'Missing or unsupported input'})
    files = list(dict.fromkeys(files))
    results = list(failures)
    for path in files:
        try:
            results.append(process(path, output, args, region_map))
        except Exception as exc:
            LOG.error('%s: %s', path.name, exc)
            results.append({'source': str(path), 'status': 'error', 'error': str(exc)})
    if not results:
        results = [{'status': 'error', 'error': 'No supported files found'}]
    # Unique run report: never overwrite an earlier run.
    report = output / ('batch_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    good = sum(x['status'] == 'ok' for x in results)
    LOG.info('Finished: %s successful, %s errors. Report: %s', good, len(results)-good, report)
    return 0 if good == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())
