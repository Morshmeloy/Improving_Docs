"""DocDiff deblurring / DocRes appearance, CPU float32, overlapping native-scale tiles."""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from enhance_batch import pages, png_bytes, SUPPORTED

ROOT = Path(__file__).resolve().parent


def positions(length, tile, overlap):
    if length <= tile:
        return [0]
    return sorted(set(list(range(0, length - tile + 1, tile - overlap)) + [length - tile]))


class DocDiffCPU:
    name = 'DocDiff pretrained deblurring, CPU float32'
    steps = 100
    manifest_name = 'docdiff_manifest.json'

    def __init__(self, threads=4):
        import torch
        from setup_docdiff import DEST, blob_sha
        manifest = json.loads((ROOT / 'docdiff_manifest.json').read_text())
        for entry in manifest['files']:
            path = DEST / entry['path']
            if not path.exists() or blob_sha(path.read_bytes()) != entry['sha']:
                raise ValueError('Run setup_docdiff.py: missing or invalid ' + entry['path'])
        source = (DEST / 'model' / 'DocDiff.py').read_text(encoding='utf-8')
        expected = source.replace('t: torch.Tensor=torch.tensor([0]).cuda()', 't: torch.Tensor=None')
        if not (DEST / 'model' / 'DocDiff_cpu.py').exists() or (DEST / 'model' / 'DocDiff_cpu.py').read_text(encoding='utf-8') != expected:
            raise ValueError('Run setup_docdiff.py: invalid CPU adapter')
        torch.set_num_threads(threads)
        sys.path.insert(0, str(DEST))
        spec = importlib.util.spec_from_file_location('docdiff_cpu', DEST / 'model' / 'DocDiff_cpu.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.DocDiff(input_channels=6, output_channels=3, n_channels=32, ch_mults=[1, 2, 3, 4], n_blocks=1)
        for name, filename in [('init_predictor', 'init.pth'), ('denoiser', 'denoiser.pth')]:
            state = torch.load(DEST / 'checksave' / filename, map_location='cpu', weights_only=True)
            getattr(self.model, name).load_state_dict(state, strict=True)
        self.model.eval()
        # Exactly the author's linear schedule for 100 diffusion steps.
        beta = torch.linspace(1e-5, .2, 100)
        gamma = torch.cumprod(1 - beta, dim=0)
        self.sqrt_gamma = gamma.sqrt()
        self.sqrt_noise = (1 - gamma).sqrt()
        self.torch = torch

    def predict(self, rgb, seed=0):
        torch = self.torch
        h, w = rgb.shape[:2]
        padded = np.pad(rgb, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)), mode='edge')
        x = torch.from_numpy(padded.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
        with torch.inference_mode():
            condition = self.model.init_predictor(x, torch.zeros(1, dtype=torch.long))
            generator = torch.Generator(device='cpu').manual_seed(seed)
            residual = torch.randn(x.shape, generator=generator)
            for step in reversed(range(100)):
                t = torch.full((1,), step, dtype=torch.long)
                original = self.model.denoiser(torch.cat((residual, condition), dim=1), t)
                if step:
                    noise = (residual - self.sqrt_gamma[step] * original) / self.sqrt_noise[step]
                    residual = self.sqrt_gamma[step - 1] * original + self.sqrt_noise[step - 1] * noise
                else:
                    residual = original
            result = condition + residual
            if not torch.isfinite(result).all():
                raise ValueError('Nonfinite neural output')
            array = result[0].permute(1, 2, 0).clamp(0, 1).numpy()
        return np.rint(array[:h, :w] * 255).astype(np.uint8)

    def prepare(self, rgb):
        return rgb

    def restore(self, rgb, tile=256, overlap=64, seed=0):
        h, w = rgb.shape[:2]
        work = self.prepare(rgb)
        ys, xs = positions(h, tile, overlap), positions(w, tile, overlap)
        sums = np.zeros(rgb.shape, dtype=np.float32)
        weights = np.zeros((h, w), dtype=np.float32)
        total = len(ys) * len(xs)
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs):
            patch = work[y:y + tile, x:x + tile]
            print(f'Tile {index + 1}/{total}: {patch.shape[1]}x{patch.shape[0]}, {self.name}', flush=True)
            result = self.predict(patch, seed + index)
            ph, pw = patch.shape[:2]
            wy = np.maximum(np.hanning(ph), .05)
            wx = np.maximum(np.hanning(pw), .05)
            weight = (wy[:, None] * wx[None, :]).astype(np.float32)
            sums[y:y + ph, x:x + pw] += result.astype(np.float32) * weight[:, :, None]
            weights[y:y + ph, x:x + pw] += weight
        return np.rint(sums / weights[:, :, None]).clip(0, 255).astype(np.uint8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--page', type=int, default=1)
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--crop', nargs=4, type=float, metavar=('X0', 'Y0', 'X1', 'Y1'), help='Normalized page box, 0..1; omitted processes the whole selected page')
    parser.add_argument('--tile', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=64)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--engine', choices=['docdiff', 'docres'], default='docdiff')
    parser.add_argument('--docres-task', choices=['appearance', 'binarization'], default='appearance')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0, help='Clockwise rotation BEFORE normalized crop')
    a = parser.parse_args(argv)
    if not a.input.is_file() or a.input.suffix.lower() not in SUPPORTED:
        parser.error('Input file missing or unsupported')
    if not 72 <= a.dpi <= 600 or a.page < 1 or not 1 <= a.threads <= 32:
        parser.error('Invalid page, dpi or threads')
    if a.tile < 128 or a.tile > 512 or a.tile % 8 or not 16 <= a.overlap < a.tile:
        parser.error('Tile must be 128..512, multiple of 8; overlap 16..tile-1')
    if a.crop and not (0 <= a.crop[0] < a.crop[2] <= 1 and 0 <= a.crop[1] < a.crop[3] <= 1):
        parser.error('Crop must be normalized X0 Y0 X1 Y1')
    if a.output.exists():
        parser.error('Output exists; choose a new folder')
    started = time.monotonic()
    if a.engine == 'docres':
        from docres_cpu import DocResCPU
        model = DocResCPU(a.threads, a.docres_task)
    else:
        model = DocDiffCPU(a.threads)
    selected = None
    for n, rgb, size, info in pages(a.input, a.dpi, 40):
        if n == a.page:
            selected = rgb
            break
    if selected is None:
        parser.error('Page not found')
    selected = orient(selected, a.rotate)
    if a.crop:
        h, w = selected.shape[:2]
        x0, y0, x1, y1 = a.crop
        selected = selected[math.floor(y0 * h):math.ceil(y1 * h), math.floor(x0 * w):math.ceil(x1 * w)].copy()
    restored = model.restore(selected, a.tile, a.overlap, a.seed)
    a.output.mkdir(parents=True)
    for name, image in [('original', selected), (a.engine, restored)]:
        (a.output / (name + '.png')).write_bytes(png_bytes(image))
    changes = np.abs(restored.astype(np.int16) - selected.astype(np.int16))
    report = {'method': model.name, 'upstream_commit': json.loads((ROOT / model.manifest_name).read_text())['commit'], 'source': a.input.name, 'source_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(), 'page': a.page, 'rotate_clockwise': a.rotate, 'crop_after_rotation': a.crop, 'dpi': a.dpi, 'tile': a.tile, 'overlap': a.overlap, 'seed': a.seed, 'diffusion_steps': model.steps, 'seconds': round(time.monotonic() - started, 2), 'mean_absolute_channel_change': float(changes.mean()), 'pixels_changed_more_than_10': float((changes.max(axis=2) > 10).mean()), 'quality_verified': False}
    (a.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    import html
    label = html.escape(model.name)
    filename = html.escape(a.input.name)
    (a.output / 'comparison.html').write_text(f'<!doctype html><meta charset="utf-8"><title>AI comparison</title><style>div{{display:flex;gap:16px}}figure{{margin:0;width:48%}}img{{max-width:100%}}</style><h1>Original / {label}</h1><p>File: {filename}; page {a.page}; rotation {a.rotate} clockwise. Click an image to inspect full resolution.</p><p>Mean channel change: {changes.mean():.2f}/255. Experimental reconstruction; quality requires inspection.</p><div><figure><figcaption>Original</figcaption><a href="original.png"><img src="original.png"></a></figure><figure><figcaption>{label}</figcaption><a href="{a.engine}.png"><img src="{a.engine}.png"></a></figure></div>', encoding='utf-8')
    print('Saved:', a.output.resolve(), flush=True)


def orient(rgb, clockwise):
    return np.rot90(rgb, k=-(clockwise // 90)).copy()


if __name__ == '__main__':
    main()
