"""Find static form labels with neural OCR; never copy reference body text."""
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import cv2
import numpy as np
from matched_lettering import matched_label
from enhance_batch import sha256

# A semantic allowlist, not coordinates. New layouts need no manual rectangles.
LABELS = (
    'ЕВРАЗИЙСКИЙ ЭКОНОМИЧЕСКИЙ СОЮЗ', 'СЕРТИФИКАТ СООТВЕТСТВИЯ',
    'ОРГАН ПО СЕРТИФИКАЦИИ', 'ЗАЯВИТЕЛЬ', 'ИЗГОТОВИТЕЛЬ', 'ПРОДУКЦИЯ',
    'КОД ТН ВЭД ЕАЭС', 'СООТВЕТСТВУЕТ ТРЕБОВАНИЯМ',
    'СЕРТИФИКАТ СООТВЕТСТВИЯ ВЫДАН НА ОСНОВАНИИ',
    'ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ', 'СРОК ДЕЙСТВИЯ С', 'ВКЛЮЧИТЕЛЬНО',
    'Руководитель (уполномоченное лицо) органа по сертификации',
    'Руководитель (уполномоченное', 'лицо) органа по сертификации',
    'Эксперт (эксперт-аудитор)', '(эксперты-аудиторы)',
    '(эксперты (эксперты-аудиторы))', '(подпись)', '(ф.и.о.)', 'Серия', '№ ЕАЭС',
)


def normalize(text):
    text = text.upper().translate(str.maketrans('ABCEHKMOPTXY', 'АВСЕНКМОРТХУ'))
    return re.sub(r'[^А-ЯЁ0-9]', '', text)


def identity(text):
    normalized = normalize(text)
    matches = [(SequenceMatcher(None, normalized, normalize(label)).ratio(), label)
               for label in LABELS
               if not (normalized.startswith(normalize(label)) and len(normalized)>len(normalize(label)))
               and not (any(c.isdigit() for c in normalized) and not any(c.isdigit() for c in normalize(label)))
               if abs(len(normalized) - len(normalize(label))) <= max(1, len(normalize(label)) * .12)]
    if not matches:
        return None
    score, label = max(matches)
    threshold = 1 if len(normalized) < 8 else .88
    return (label, score) if score >= threshold else None


def rectangle(row, shape):
    points = np.array(row['box']); h, w = shape[:2]
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def candidates(rows, shape):
    found = []
    for row in rows:
        match = identity(row['text'])
        if match and row['confidence'] >= .35:
            box = rectangle(row, shape)
            if box[2] > box[0] and box[3] > box[1]:
                found.append({**row, 'label': match[0], 'match': match[1], 'rect': box})
    return found


def shared_score(source_rows, source_shape, reference):
    a = {r['label'] for r in candidates(source_rows, source_shape)}
    b = {r['label'] for r in candidates(reference['rows'], reference['rgb'].shape)}
    return len(a & b)


def restore_labels(base, original, source_rows, reference):
    """Separate exact labels from glued body lines; skip uncertain overlapping OCR."""
    out = base.copy(); area = np.zeros(base.shape[:2], bool); records = []
    if reference is None:
        return out, area, records
    ref_candidates = candidates(reference['rows'], reference['rgb'].shape)
    ref_labels = {r['label'] for r in ref_candidates}
    source_candidates = candidates(source_rows, base.shape)
    for row in source_candidates:
        if row['label'] not in ref_labels:
            continue
        # A word inside body text is not another occurrence of a form heading.
        # Repeated roles are matched by their relative position, independently.
        def distance(target, sample):
            ah,aw = base.shape[:2]; bh,bw = reference['rgb'].shape[:2]
            ax,ay = np.mean(np.array(target['box']),axis=0)/[aw,ah]
            bx,by = np.mean(np.array(sample['box']),axis=0)/[bw,bh]
            return abs(ax-bx)+abs(ay-by)*2
        nearest = min((r for r in ref_candidates if r['label']==row['label']),key=lambda r:distance(row,r))
        if distance(row,nearest)>.35 or any(
            distance(other,nearest)<distance(row,nearest)
            for other in source_candidates if other['label']==row['label'] and other['box']!=row['box']):
            records.append({'label':row['label'],'status':'skipped_layout_mismatch'});continue
        x0, y0, x1, y1 = row['rect']
        overlapping = False
        for other in source_rows:
            if other is row or other['box'] == row['box']:
                continue
            bx0, by0, bx1, by1 = rectangle(other, base.shape)
            intersection = max(0, min(x1,bx1)-max(x0,bx0)) * max(0, min(y1,by1)-max(y0,by0))
            if intersection > (x1-x0)*(y1-y0)*.1:
                overlapping = True; break
        if overlapping or area[y0:y1,x0:x1].any():
            records.append({'label': row['label'], 'status': 'skipped_overlap'}); continue
        patch = original[y0:y1,x0:x1]
        # Do not white out a colored stamp/signature crossing a field title.
        colors = patch.astype(np.int16)
        if ((colors.max(2)-colors.min(2) > 30) & (colors.min(2) < 235)).any():
            records.append({'label': row['label'], 'status': 'skipped_colored_overlap'}); continue
        try:
            weight = 'bold' if row['label'].isupper() or row['label'].startswith(('Руководитель','лицо)','Эксперт')) else 'regular'
            mapped, metrics = matched_label(patch, row['label'], weight)
        except ValueError as error:
            records.append({'label': row['label'], 'status': 'skipped_no_metrics', 'reason': str(error)}); continue
        out[y0:y1,x0:x1] = mapped; area[y0:y1,x0:x1] = True
        records.append({'label': row['label'], 'status': 'restored', 'target': [x0,y0,x1,y1],
                        'ocr_confidence': row['confidence'], 'metrics': metrics})
    return out, area, records


def reviewed_cache(source, reference, page, angle):
    """Reuse previously reviewed geometry ONLY for an identical pair of files."""
    if reference is None or page != 1 or angle != 180:
        return None
    path = Path(__file__).resolve().parent / 'profiles' / 'va103_matched_lettering.json'
    if not path.is_file():
        return None
    profile = json.loads(path.read_text(encoding='utf-8'))
    if profile.get('source_sha256') == sha256(source) and profile.get('reference_sha256') == sha256(reference):
        return profile
    return None


def restore_frame(base, original, rows, reference):
    """Align golden decoration from independently matched static OCR anchors."""
    mask = np.zeros(base.shape[:2], bool)
    if reference is None:
        return base, mask, {'status':'no_reference'}
    a = candidates(rows, base.shape)
    b = candidates(reference['rows'], reference['rgb'].shape)
    pairs = []
    for label in {r['label'] for r in a} & {r['label'] for r in b}:
        aa = [r for r in a if r['label']==label]; bb = [r for r in b if r['label']==label]
        if len(aa)==len(bb)==1:
            pairs.append((np.mean(np.array(bb[0]['box']),axis=0),np.mean(np.array(aa[0]['box']),axis=0)))
    if len(pairs)<4:
        return base, mask, {'status':'insufficient_static_anchors'}
    src = np.float32([p[0] for p in pairs]); dst = np.float32([p[1] for p in pairs])
    matrix, inliers = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=8)
    if matrix is None or int(inliers.sum()) < 4 or np.ptp(src[inliers.ravel()>0,1]) < reference['rgb'].shape[0]*.35:
        return base, mask, {'status':'unreliable_alignment'}
    rgb = reference['rgb']; hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    gold = (hsv[:,:,0]>=8)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=100)
    h,w = rgb.shape[:2]
    # Only outer ornament; interior colored data are excluded even if golden.
    outside = np.zeros((h,w),bool)
    outside[:max(1,round(h*.025))] = True; outside[-max(1,round(h*.025)):] = True
    outside[:,:max(1,round(w*.025))] = True; outside[:,-max(1,round(w*.025)):] = True
    gold &= outside
    target_h,target_w = base.shape[:2]
    warped = cv2.warpAffine(rgb,matrix,(target_w,target_h),borderValue=(255,255,255))
    mask = cv2.warpAffine(gold.astype(np.uint8),matrix,(target_w,target_h),flags=cv2.INTER_NEAREST)>0
    # Further restrict to margins on the source; protect all original dark ink.
    margin = np.zeros_like(mask)
    margin[:round(target_h*.05)] = True; margin[-round(target_h*.05):] = True
    margin[:,:round(target_w*.06)] = True; margin[:,-round(target_w*.06):] = True
    mask &= margin & (original.min(2)>160)
    out = base.copy(); out[mask] = warped[mask]
    return out,mask,{'status':'restored' if mask.any() else 'no_ornament_in_margins',
                     'static_anchor_inliers':int(inliers.sum()),'pixels':int(mask.sum())}
