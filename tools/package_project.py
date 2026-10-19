"""Build a portable source ZIP, excluding models, logs, recordings and caches."""
import argparse
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root.parent / 'FLEX-Web.zip')
    args = parser.parse_args()
    top_files = {'run.py', 'infer_audio.py', 'start_server.sh', 'run_infer.sh', 'README.md', 'requirements.txt',
                 'requirements-dev.txt', 'requirements-infer.txt', '.gitignore', '.gitattributes',
                 'CONTRIBUTING.md', 'SECURITY.md', '.github/workflows/tests.yml'}
    folders = {'flex_web', 'static', 'tests', 'docs', 'tools'}
    extensions = {'.py', '.js', '.mjs', '.html', '.css', '.md', '.jpg'}
    files = [path for path in root.rglob('*') if path.is_file() and (
        path.relative_to(root).as_posix() in top_files or (
            path.relative_to(root).parts[0] in folders and path.suffix in extensions
            and not any(part.startswith('.') or part == '__pycache__'
                        for part in path.relative_to(root).parts)))]
    # Exclusive creation: never overwrite a user's previous archive.
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, 'FLEX-Web/' + path.relative_to(root).as_posix())
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP integrity verification failed')
    print(f'{args.output.resolve()} ({len(files)} files)')


if __name__ == '__main__':
    main()
