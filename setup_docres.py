"""Pinned DocRes architecture and weights from the demo linked by its authors."""
import hashlib
import json
from pathlib import Path
import urllib.request
from setup_docdiff import blob_sha

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'models' / 'docres'


def main():
    manifest = json.loads((ROOT / 'docres_manifest.json').read_text())
    for entry in manifest['files']:
        path = DEST / entry['path']
        if path.exists() and blob_sha(path.read_bytes()) == entry['sha']:
            continue
        url = 'https://raw.githubusercontent.com/ZZZHANG-jx/DocRes/' + manifest['commit'] + '/' + entry['path']
        print('Downloading:', entry['path'], flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read(entry['size'] + 1)
        if len(data) != entry['size'] or blob_sha(data) != entry['sha']:
            raise ValueError('DocRes source hash mismatch')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    weight = manifest['weights']
    path = DEST / 'docres.pkl'
    if path.exists() and path.stat().st_size == weight['size']:
        with path.open('rb') as stream:
            existing_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        if existing_hash == weight['sha256']:
            print('DocRes weights verified')
            return
    tmp = path.with_suffix('.partial')
    digest = hashlib.sha256()
    count = 0
    try:
        with urllib.request.urlopen(weight['url'], timeout=120) as response, tmp.open('wb') as out:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > weight['size']:
                    raise ValueError('Unexpected weight size')
                digest.update(block)
                out.write(block)
                print(f'DocRes weights: {count / 1048576:.1f}/{weight["size"] / 1048576:.1f} MiB', flush=True)
        if count != weight['size'] or digest.hexdigest() != weight['sha256']:
            raise ValueError('DocRes weights SHA-256 mismatch')
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print('DocRes appearance CPU assets ready')


if __name__ == '__main__':
    main()
