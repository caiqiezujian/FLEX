"""Build an independent, file-only FLEX-S2TT folder and ZIP from tested sources.

No Web files, checkpoints, recordings or outputs are included. The only code
rewrite changes the CLI's shared-model import to the bundled flex_model module.
"""
import argparse
from pathlib import Path
import zipfile


def bundle_files(root):
    entry = (root / 'infer_audio.py').read_text(encoding='utf-8')
    original = 'from flex_web.model import '
    if entry.count(original) != 1:
        raise RuntimeError('Expected exactly one shared model import; review the packager')
    files = {
        'infer_audio.py': entry.replace(original, 'from flex_model import ', 1),
        'flex_model.py': (root / 'flex_web/model.py').read_text(encoding='utf-8'),
        'run_infer.sh': (root / 'run_infer.sh').read_text(encoding='utf-8'),
        'README.md': (root / 'docs/S2TT-QUICKSTART.md').read_text(encoding='utf-8'),
        'requirements-infer.txt': (root / 'requirements-infer.txt').read_text(encoding='utf-8'),
    }
    for name, content in files.items():
        if name.endswith('.py'):
            compile(content, name, 'exec')
    return {name: content.replace('\r\n', '\n').encode('utf-8')
            for name, content in files.items()}


def build(root, directory, archive_path):
    root, directory, archive_path = map(Path, (root, directory, archive_path))
    if directory.exists() or archive_path.exists():
        raise FileExistsError('Choose a NEW output directory and ZIP; nothing is overwritten')
    files = bundle_files(root)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (directory / name).open('xb') as handle:
            handle.write(content)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            entry = zipfile.ZipInfo('FLEX-S2TT/' + name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            archive.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity verification failed')
    return len(files)


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=root.parent / 'FLEX-S2TT')
    parser.add_argument('--output', type=Path, default=root.parent / 'FLEX-S2TT.zip')
    args = parser.parse_args()
    count = build(root, args.directory, args.output)
    print(f'{args.directory.resolve()} ({count} files)')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
