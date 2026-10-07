"""Install pinned Tinos fonts from Google Fonts for reproducible form lettering."""
import hashlib
from pathlib import Path
import urllib.request

REVISION='ba95515f1333efe9342c2ad988b9c2f6bef6dbad'
FILES={'Tinos-Bold.ttf':('887fc97a95d30a08190217b6075a5e0a0f92bec7',597880),'Tinos-Regular.ttf':('893a1b525962984481aee5ce275516d384eb4865',521588)}
ROOT=Path(__file__).resolve().parent/'models'/'fonts'


def verify(data, sha, size):
    return len(data)==size and hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()==sha


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,(sha,size) in FILES.items():
        path=ROOT/name
        if path.is_file() and verify(path.read_bytes(),sha,size):
            print('Verified:',name);continue
        url=f'https://raw.githubusercontent.com/google/fonts/{REVISION}/ofl/tinos/{name}'
        with urllib.request.urlopen(url,timeout=60) as response: data=response.read(size+1)
        if not verify(data,sha,size):raise ValueError('Font content verification failed: '+name)
        partial=path.with_suffix('.partial');partial.write_bytes(data);partial.replace(path)
        print('Installed:',name)


if __name__=='__main__':main()
