"""Choose documents and an optional reference. Everything else is automatic."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import webbrowser
from enhance_batch import SUPPORTED
from auto_restore import run_auto

FILE_TYPES = [('Документы','*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp')]


def select_files(root):
    files = filedialog.askopenfilenames(parent=root,title='Выберите документы для улучшения',filetypes=FILE_TYPES)
    if not files:
        return [],None
    reference = filedialog.askopenfilename(parent=root,
        title='Выберите образец (необязательно). Отмена — обработать без образца',filetypes=FILE_TYPES)
    return [Path(p).resolve() for p in files],Path(reference).resolve() if reference else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='*',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args(argv)
    if args.input:
        files = []
        for path in args.input:
            files.extend(sorted((p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED),key=lambda p:p.name.casefold()) if path.is_dir() else [path.resolve()])
        reference = args.reference.resolve() if args.reference else None
    else:
        root = tk.Tk();root.withdraw()
        try:files,reference = select_files(root)
        finally:root.destroy()
    if not files:
        print('No documents selected.'); return 0
    if reference and not reference.is_file():
        parser.error('Reference file does not exist')
    output,results = run_auto(files,reference,args.output)
    print('Results:',output/'index.html',flush=True)
    webbrowser.open((output/'index.html').as_uri())
    return int(any(r['status']=='error' for r in results))


if __name__=='__main__':raise SystemExit(main())
