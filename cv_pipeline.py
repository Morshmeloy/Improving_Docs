"""Local paper estimation and multiscale ink enhancement, without geometry changes.
All transforms operate on observed samples. They may still amplify noise.
"""
from __future__ import annotations
import cv2
import numpy as np


def paper_background(rgb):
    """Estimate low-frequency paper via closing on a bounded working image.

    Closing is used ONLY to estimate illumination, never on final strokes.
    A 31 px kernel at 900 px long-side can confuse large dark areas with paper;
    this is an explicit limitation, hence several candidates instead of auto selection.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, 900 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    k = min(31, max(3, (min(size) // 12) | 1))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(closed.astype(np.float32), (0, 0), max(1., k/6))
    return cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)


def restore(rgb, mode, strength=1.):
    if mode not in {'cv_balanced', 'cv_detail', 'ink_study'}:
        raise ValueError(f'Unknown CV mode: {mode}')
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image')
    src = rgb.astype(np.float32)
    bg = np.maximum(paper_background(rgb), 32.)
    # Ratio corrects shaded paper per channel, retaining RGB residuals of ink.
    normalized = np.clip(src / bg, 0, 1)
    density = 1 - normalized
    y = cv2.cvtColor((normalized * 255).astype(np.float32), cv2.COLOR_RGB2GRAY)
    # Positive-only detail: do not whiten thin dark stroke cores.
    fine = np.maximum(cv2.GaussianBlur(y, (0, 0), .8) - y, 0) / 255
    medium = np.maximum(cv2.GaussianBlur(y, (0, 0), 2.4) - y, 0) / 255
    exponent, detail_weight = (2.6, .35) if mode == 'cv_balanced' else (4.5, .65)
    exponent = 1 + (exponent - 1) * strength
    # Monotone nonlinear density gain; no threshold removes faint lines.
    enhanced = 1 - np.power(1 - density, exponent)
    detail = (.65 * fine + .35 * medium) * detail_weight * strength
    result = 255 * (1 - np.clip(enhanced + detail[..., None], 0, 1))
    # Diagnostic contrast of colored ink; separate output, never exact-color claim.
    if mode == 'ink_study':
        channel_density = np.maximum(bg - src, 0)
        maximum = channel_density.max(axis=2)
        minimum = channel_density.min(axis=2)
        chroma = np.clip((maximum - minimum) / 24, 0, 1)
        ink = 255 * np.power(normalized, 1 + 3 * strength)
        result = result * (1 - .6 * chroma[..., None]) + ink * (.6 * chroma[..., None])
    out = np.clip(result, 0, 255).round().astype(np.uint8)
    return out, {'paper_estimator': 'RGB closing at <=900px long side + Gaussian smoothing',
                 'density_exponent': exponent, 'detail_weight': detail_weight,
                 'background_p05': float(np.percentile(bg, 5)),
                 'background_p95': float(np.percentile(bg, 95)),
                 'color_ink_modified': mode == 'ink_study',
                 'warning': 'Noise and copying artifacts can be amplified; compare to source.'}


def diagnostics(source, result, locked):
    gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    diff = np.abs(result.astype(np.int16) - source.astype(np.int16)).max(axis=2)
    return {'mean_absolute_channel_change': float(np.abs(result.astype(np.float32)-source).mean()),
            'changed_fraction': float((diff > 0).mean()),
            'black_clipped_fraction': float((gray == 0).mean()),
            'white_clipped_fraction': float((gray == 255).mean()),
            'protected_max_error': int(diff[locked].max()) if np.any(locked) else 0,
            'semantic_equivalence_verified': False}
