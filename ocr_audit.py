"""Optional local OCR audit. OCR is evidence for review, never replacement text."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess


def audit(path, lang, executable='tesseract', psm=3):
    run = subprocess.run([executable, str(path), 'stdout', '-l', lang,
                          '--psm', str(psm), 'tsv'], capture_output=True, text=True,
                         encoding='utf-8', timeout=300, check=True)
    words = []
    for row in csv.DictReader(io.StringIO(run.stdout), delimiter='\t'):
        text = row.get('text', '').strip()
        confidence = float(row.get('conf', '-1'))
        if text and confidence >= 0:
            words.append({'text': text, 'confidence': confidence,
                          'box': [int(row[k]) for k in ('left','top','width','height')]})
    return {'image': path.name, 'word_count': len(words),
            'mean_confidence': sum(x['confidence'] for x in words)/len(words) if words else None,
            'words': words, 'warning': 'Confidence is not accuracy; compare same crop and settings. No pixel edits.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--lang',default='rus+eng')
    p.add_argument('--psm',type=int,choices=range(3,14),default=3)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    exe=shutil.which('tesseract')
    if not exe:p.error('Install Tesseract and language packs first')
    if a.output.exists():p.error('Output exists; choose a new report path')
    files=sorted(x for x in a.directory.glob('page_*.png')
                 if not x.stem.endswith(('_protected','_changes')))
    if not files:p.error('No page images found')
    results=[]
    for file in files:
        try:results.append(audit(file,a.lang,exe,a.psm))
        except (subprocess.SubprocessError, ValueError) as exc:
            results.append({'image':file.name,'error':str(exc)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return int(any('error' in r for r in results))

if __name__=='__main__':raise SystemExit(main())
