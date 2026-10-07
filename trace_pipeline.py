"""Experimental evidence-guided stroke contrast. Never complete a guessed frame."""
import math
import cv2
import numpy as np
from cv_pipeline import paper_background


def trace_strokes(rgb, strength=1.):
    """Boost observed density with directional support from a structure tensor.

    No path tracing, inpainting, dilation of output strokes or symmetric copying.
    Flat paper and clipped-white absent strokes provide no evidence to boost.
    Texture can also be directional: support is not a signature detector.
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB image')
    if not math.isfinite(strength) or not .25 <= strength <= 2:
        raise ValueError('Trace strength must be finite and 0.25..2')
    src = rgb.astype(np.float32)
    bg = np.maximum(paper_background(rgb), 32.)
    normalized = np.clip(src / bg, 0., 1.)
    density = 1. - normalized
    y = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.GaussianBlur(y, (0, 0), .65)
    dx = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0) / 32.
    dy = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1) / 32.
    xx = cv2.GaussianBlur(dx * dx, (0, 0), 1.4)
    yy = cv2.GaussianBlur(dy * dy, (0, 0), 1.4)
    xy = cv2.GaussianBlur(dx * dy, (0, 0), 1.4)
    coherence = np.sqrt(np.maximum((xx-yy)**2 + 4*xy**2, 0)) / (xx+yy+1e-7)
    # Smooth gates, not deletion thresholds. Only the EXTRA gain is gated.
    energy = np.sqrt(xx+yy)
    reliability = energy / (energy + .007)
    contrast = density.max(axis=2)
    evidence = contrast / (contrast + .018)
    support = evidence * reliability * (.25 + .75*coherence)
    # Strong strokes get less extra gain, reducing merges and saturation.
    gain = 1 + strength * (2 + 8*support) * (1-contrast)**2
    enhanced = np.clip(density * gain[..., None], 0., 1.)
    # Weak baseline tonal gain keeps dots/crossings with low coherence visible.
    baseline = 1 - normalized ** (1 + 1.8*strength)
    enhanced = np.maximum(enhanced, baseline)
    result = (255 * (1-enhanced)).round().astype(np.uint8)
    return result, {'method':'observed RGB density + multiscale structure tensor support',
                    'mean_directional_support':float(support.mean()),
                    'color_ink_modified':True,
                    'geometry_changed':False,
                    'missing_segments_completed':False,
                    'warning':'Experimental: texture and printing artifacts can be strengthened; manual review required.'}
