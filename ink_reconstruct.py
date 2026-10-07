"""Selected-region ink lifting -> DocRes segmentation -> black ink reconstruction."""
import argparse
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import shutil
import tempfile
import cv2
import fitz
import numpy as np
from ai_restore import orient
from cv_pipeline import paper_background
from enhance_batch import pages, png_bytes, add_page, verify_pdf_rasters, sha256, protection_mask
from redraw_pipeline import skeletonize, connect_endpoints


def lift_ink(rgb, gain):
    background = np.maximum(paper_background(rgb), 32)
    density = 1 - np.clip(rgb.astype(np.float32) / background, 0, 1)
    return np.rint(255 * (1 - np.clip(density * gain, 0, 1))).astype(np.uint8)


def reconstruct_patch(rgb, neural_mask, max_gap=4):
    """Keep observed pixels; add mask pixels and explicit small inferred bridges."""
    ink = neural_mask.mean(axis=2) < 128
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    keep = np.zeros(count, bool)
    if count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
    ink = keep[labels]
    skeleton = skeletonize(ink)
    bridges, connections = connect_endpoints(skeleton, max_gap) if max_gap else (np.zeros_like(ink), [])
    mask = ink | bridges
    # Preserve colored ink exactly. This is a pixel-color heuristic, not stamp recognition.
    channels = rgb.astype(np.int16)
    color = (channels.max(axis=2) - channels.min(axis=2) > 30) & (channels.min(axis=2) < 235)
    color = cv2.dilate(color.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mask &= ~color
    bridges &= ~color
    out = rgb.copy()
    out[mask] = 0
    overlay = rgb.copy()
    overlay[mask] = [0, 170, 0]
    overlay[bridges] = [255, 0, 0]
    return out, overlay, mask, {'drawn_pixels': int(mask.sum()), 'inferred_bridge_pixels': int(bridges.sum()), 'connections': connections, 'colored_pixels_preserved': int(color.sum()), 'colored_pixels_equal': bool(np.array_equal(out[color], rgb[color]))}


def bounds(box, width, height):
    if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('Invalid normalized box')
    return max(0, math.floor(box[0] * width)), max(0, math.floor(box[1] * height)), min(width, math.ceil(box[2] * width)), min(height, math.ceil(box[3] * height))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--regions', required=True, type=Path, help='filename -> page -> boxes in orientation AFTER rotation; box or {box, gain}')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--gain', type=float, default=6)
    p.add_argument('--max-gap', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--locks', type=Path, help='Exact manual protection, normalized AFTER rotation')
    a = p.parse_args(argv)
    if not a.input.is_file() or a.output.exists() or not 72 <= a.dpi <= 600 or not 1 <= a.gain <= 20 or not 0 <= a.max_gap <= 10 or not 1 <= a.threads <= 32:
        p.error('Missing input, existing output or invalid parameters')
    config = json.loads(a.regions.read_text(encoding='utf-8-sig'))
    chosen = config.get(a.input.name, {})
    if not any(chosen.values()):
        p.error('No drawing regions for this exact input filename')
    locks = json.loads(a.locks.read_text(encoding='utf-8-sig')).get(a.input.name, {}) if a.locks else {}
    from docres_cpu import DocResCPU
    model = DocResCPU(a.threads, 'binarization')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.ink_', dir=a.output.parent))
    document = fitz.open()
    expected, cards = [], []
    report = {'method': 'local ink lifting + pretrained DocRes binarization + explicit endpoint reconstruction', 'source_sha256': sha256(a.input), 'rotate_clockwise': a.rotate, 'time_utc': datetime.now(timezone.utc).isoformat(), 'semantic_accuracy_verified': False, 'pages': []}
    total_drawn = 0
    try:
        shutil.copy2(a.input, stage / ('source' + a.input.suffix))
        for n, rgb, size, info in pages(a.input, a.dpi, 40):
            original = orient(rgb, a.rotate)
            out, overlay = original.copy(), original.copy()
            h, w = original.shape[:2]
            area = np.zeros((h, w), bool)
            lock = protection_mask(original, locks.get(str(n), []), color=False)
            page_stats = {'page': n, 'regions': []}
            for index, entry in enumerate(chosen.get(str(n), []), 1):
                box = entry['box'] if isinstance(entry, dict) else entry
                gain = float(entry.get('gain', a.gain)) if isinstance(entry, dict) else a.gain
                if not 1 <= gain <= 20:
                    raise ValueError('Region gain out of range')
                x0, y0, x1, y1 = bounds(box, w, h)
                patch = original[y0:y1, x0:x1]
                lifted = lift_ink(patch, gain)
                print(f'Page {n}, region {index}: {x1-x0}x{y1-y0}, gain={gain}', flush=True)
                segmented = model.restore(lifted, 256, 64)
                restored, marked, mask, stats = reconstruct_patch(patch, segmented, a.max_gap)
                editable = ~lock[y0:y1, x0:x1]
                out[y0:y1, x0:x1][editable] = restored[editable]
                overlay[y0:y1, x0:x1][editable] = marked[editable]
                area[y0:y1, x0:x1] = True
                total_drawn += int((mask & editable).sum())
                stats.update({'box': box, 'gain': gain, 'drawn_pixels_after_locks': int((mask & editable).sum())})
                page_stats['regions'].append(stats)
                prefix = f'page_{n:04d}_region_{index:02d}'
                for suffix, image in [('original', patch), ('lifted_input', lifted), ('neural_mask', segmented), ('reconstructed', out[y0:y1, x0:x1]), ('overlay', overlay[y0:y1, x0:x1])]:
                    (stage / f'{prefix}_{suffix}.png').write_bytes(png_bytes(image))
            page_stats['outside_regions_equal'] = bool(np.array_equal(out[~area], original[~area]))
            page_stats['manual_locks_equal'] = bool(np.array_equal(out[lock], original[lock]))
            report['pages'].append(page_stats)
            prefix = f'page_{n:04d}'
            for suffix, image in [('original', original), ('reconstructed', out), ('drawing_overlay', overlay)]:
                (stage / f'{prefix}_{suffix}.png').write_bytes(png_bytes(image))
            expected.append(stage / f'{prefix}_reconstructed.png')
            if a.rotate in {90, 270}:
                size = (size[1], size[0])
            add_page(document, size, png_bytes(out))
            cards.append(f'<h2>Page {n}</h2><div>' + ''.join(f'<figure><figcaption>{suffix}</figcaption><a href="{prefix}_{suffix}.png"><img src="{prefix}_{suffix}.png"></a></figure>' for suffix in ['original', 'reconstructed', 'drawing_overlay']) + '</div>')
        if not total_drawn:
            raise ValueError('Neural reconstruction added no black strokes. No result published.')
        document.set_metadata({'title': 'EDITED reconstruction draft: neural ink segmentation and inferred bridges', 'producer': 'Improving Docs ink reconstruction'})
        document.save(stage / 'reconstructed_draft.pdf', deflate=True)
        document.close()
        verify_pdf_rasters(stage / 'reconstructed_draft.pdf', expected)
        report.update({'drawn_pixels': total_drawn, 'saved_pdf_rasters_verified': True})
        (stage / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (stage / 'comparison.html').write_text('<!doctype html><meta charset="utf-8"><style>div{display:flex}figure{width:32%;margin:5px}img{width:100%}</style><h1>' + html.escape(a.input.name) + '</h1><p>Edited reconstruction draft. Green: neural ink mask; red: inferred bridges. Color pixels and areas outside selected regions preserved.</p>' + ''.join(cards), encoding='utf-8')
        stage.rename(a.output)
        print('Saved:', a.output.resolve())
    except Exception:
        if not document.is_closed:
            document.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == '__main__':
    main()
