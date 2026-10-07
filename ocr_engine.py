"""Lazy, local neural OCR; no cloud account and no CUDA required."""
from pathlib import Path
import cv2
import numpy as np
from ai_restore import orient
from ink_reconstruct import lift_ink

MODEL_DIR = Path(__file__).resolve().parent / 'models' / 'ocr'


class NeuralOCR:
    def __init__(self, reader=None):
        if reader is None:
            import easyocr
            reader = easyocr.Reader(['ru', 'en'], gpu=False,
                                    model_storage_directory=str(MODEL_DIR),
                                    download_enabled=True, verbose=False)
        self.reader = reader

    def read(self, rgb, max_side=1800, enhance=False):
        height, width = rgb.shape[:2]
        scale = min(1., max_side / max(height, width))
        image = cv2.resize(rgb, (round(width * scale), round(height * scale))) if scale < 1 else rgb
        if enhance:
            image = lift_ink(image, 4)
        found = self.reader.readtext(image, detail=1, paragraph=False, batch_size=1,
                                     workers=0, width_ths=.1, canvas_size=max_side,
                                     min_size=5, text_threshold=.6, low_text=.3,
                                     contrast_ths=.2, adjust_contrast=.7)
        return [{'box': (np.array(box, dtype=float) / scale).tolist(),
                 'text': str(text), 'confidence': float(confidence)}
                for box, text, confidence in found]

    def orientation(self, rgb):
        """Compare full-page rotations, not just rotation of individual OCR boxes."""
        scores = []
        for angle in (0, 90, 180, 270):
            rows = self.read(orient(rgb, angle), max_side=1100, enhance=True)
            score = sum(len(row['text'].replace(' ', '')) * row['confidence'] ** 2
                        for row in rows if row['confidence'] >= .35)
            scores.append((score, angle))
        scores.sort(reverse=True)
        best, angle = scores[0]
        # With no legible text, preserve input orientation instead of a guess.
        reliable = best >= 5 and best > scores[1][0] * 1.15
        if not reliable:
            angle = 0
        return angle, {'chosen_clockwise': angle,
                       'scores': {str(a): round(s, 3) for s, a in scores},
                       'reliable': reliable}


if __name__ == '__main__':
    NeuralOCR()
    print('Russian/English neural OCR weights ready (CPU).')
