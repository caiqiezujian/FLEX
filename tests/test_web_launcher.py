"""The simple Bash launcher forwards HTTPS without selecting another GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from test_infer_launcher import bash


ROOT = Path(__file__).resolve().parents[1]


def test_https_options_survive_bash_launcher(bash, tmp_path):
    project = tmp_path / 'web with spaces'
    project.mkdir()
    shutil.copy2(ROOT / 'start_server.sh', project / 'start_server.sh')
    (project / 'run.py').write_text(
        'import json, os, sys\n'
        'print(json.dumps({"args": sys.argv[1:], "cuda": os.getenv("CUDA_VISIBLE_DEVICES"),\n'
        ' "worker": os.getenv("VLLM_WORKER_MULTIPROC_METHOD")}))\n', encoding='utf-8')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='6', PYTHONIOENCODING='utf-8',
                       PATH=str(Path(sys.executable).parent) + os.pathsep + environment.get('PATH', ''))
    extra = ['--host', '0.0.0.0', '--port', '18889', '--https-host', '10.0.0.8',
             '--cert-dir', 'persistent certs']
    model = str(project / 'model with spaces')
    result = subprocess.run([bash, (project / 'start_server.sh').as_posix(), model, *extra],
                            env=environment, cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output['args'][-len(extra):] == extra
    assert output['args'][:4] == ['--backend', 'vllm', '--model', model]
    assert output['cuda'] == '6' and output['worker'] == 'spawn'
