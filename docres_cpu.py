"""DocRes appearance task, BGR + global appearance prompt, CPU FP32."""
import hashlib
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from ai_restore import DocDiffCPU, ROOT
from setup_docdiff import blob_sha


def appearance_prompt(bgr):
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (1024, 1024))
    planes = []
    for channel in cv2.split(small):
        bg = cv2.medianBlur(cv2.dilate(channel, np.ones((7, 7), np.uint8)), 21)
        difference = 255 - cv2.absdiff(channel, bg)
        planes.append(cv2.normalize(difference, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1))
    return cv2.resize(cv2.merge(planes), (w, h))


def binarization_prompt(bgr):
    from skimage.filters import threshold_sauvola
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    side = min(gray.shape)
    n1, n2 = max(3, int(.05 * side) | 1), max(3, int(.1 * side) | 1)
    first = threshold_sauvola(gray, window_size=n1, k=.5)
    normalized = np.zeros(gray.shape, np.float32)
    selected = gray > first
    normalized[selected] = (gray[selected] - first[selected]) / np.maximum(float(gray.max()) - first[selected], 1e-6)
    normalized = (normalized * 255).clip(0, 255).astype(np.uint8)
    second = threshold_sauvola(normalized, window_size=n2, k=.5)
    binary = np.where(normalized <= second, 0, 255).astype(np.uint8)
    dx = cv2.convertScaleAbs(cv2.Sobel(bgr, cv2.CV_16S, 1, 0))
    dy = cv2.convertScaleAbs(cv2.Sobel(bgr, cv2.CV_16S, 0, 1))
    gradient = cv2.cvtColor(cv2.addWeighted(dx, .5, dy, .5, 0), cv2.COLOR_BGR2GRAY)
    return np.stack((second.astype(np.uint8), gradient, binary), axis=2)


class DocResCPU(DocDiffCPU):
    name = 'DocRes pretrained appearance, CPU float32'
    steps = 0
    manifest_name = 'docres_manifest.json'

    def __init__(self, threads=4, task='appearance'):
        if task not in {'appearance', 'binarization'}:
            raise ValueError('Unknown DocRes task')
        self.task = task
        self.name = 'DocRes pretrained ' + task + ', CPU float32'
        import torch
        from setup_docres import DEST
        manifest = json.loads((ROOT / self.manifest_name).read_text())
        for entry in manifest['files']:
            path = DEST / entry['path']
            if not path.exists() or blob_sha(path.read_bytes()) != entry['sha']:
                raise ValueError('Run setup_docres.py: missing or invalid ' + entry['path'])
        weight = DEST / 'docres.pkl'
        if not weight.exists():
            raise ValueError('Run setup_docres.py: weights missing')
        with weight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != manifest['weights']['sha256']:
            raise ValueError('DocRes weights SHA-256 mismatch')
        torch.set_num_threads(threads)
        spec = importlib.util.spec_from_file_location('docres_restormer', DEST / 'models' / 'restormer_arch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.Restormer(inp_channels=6, out_channels=3, dim=48, num_blocks=[2, 3, 3, 4], num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=True)
        state = torch.load(weight, map_location='cpu', weights_only=True)['model_state']
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.torch = torch

    def prepare(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # Compute one prompt over the selected region, not differently for every tile.
        prompt = appearance_prompt(bgr) if self.task == 'appearance' else binarization_prompt(bgr)
        return np.concatenate((bgr, prompt), axis=2)

    def predict(self, six_channels, seed=0):
        torch = self.torch
        h, w = six_channels.shape[:2]
        padded = np.pad(six_channels, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)), mode='edge')
        tensor = torch.from_numpy(padded.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.inference_mode():
            result = self.model(tensor)
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite DocRes output')
            if self.task == 'binarization':
                gray = result[:, :2].argmax(dim=1)[0].numpy().astype(np.uint8) * 255
                return np.repeat(gray[:h, :w, None], 3, axis=2)
            bgr = result[0].permute(1, 2, 0).clamp(0, 1).numpy()
        bgr = np.rint(bgr[:h, :w] * 255).astype(np.uint8)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
