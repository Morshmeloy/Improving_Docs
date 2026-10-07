"""Download pinned official DocDiff deblurring assets; verify Git blob hashes."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models' / 'docdiff'


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def main():
    manifest = json.loads((ROOT / 'docdiff_manifest.json').read_text())
    for entry in manifest['files']:
        path = DEST / entry['path']
        if path.exists() and blob_sha(path.read_bytes()) == entry['sha']:
            print('Verified:', entry['path'], flush=True)
            continue
        url = 'https://raw.githubusercontent.com/Royalvice/DocDiff/' + manifest['commit'] + '/' + entry['path']
        print('Downloading:', entry['path'], flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(entry['size'] + 1)
        if len(data) != entry['size'] or blob_sha(data) != entry['sha']:
            raise ValueError('Asset verification failed: ' + entry['path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_bytes(data)
        temporary.replace(path)
    source = (DEST / 'model' / 'DocDiff.py').read_text(encoding='utf-8')
    old = 't: torch.Tensor=torch.tensor([0]).cuda()'
    if source.count(old) != 1:
        raise ValueError('Unexpected upstream CUDA default')
    # Change only the default argument; our runner always supplies the timestep.
    (DEST / 'model' / 'DocDiff_cpu.py').write_text(source.replace(old, 't: torch.Tensor=None'), encoding='utf-8')
    print('DocDiff CPU assets ready. Only deblurring weights were downloaded.')


if __name__ == '__main__':
    main()
