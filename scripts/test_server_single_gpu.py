"""Verify multi-GPU rejection and single-GPU selection on a two-CUDA-GPU host.

Usage: python test_server_single_gpu.py --server PATH --model SMALL_GGUF --output DIR
No downloads; starts and stops only its own loopback services.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.pop('CUDA_VISIBLE_DEVICES', None)
checks = []
def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)

try:
    r = subprocess.run([a.server, '--list-devices'], env=env, capture_output=True, timeout=30)
    listing = r.stdout.decode(errors='replace')
    names = re.findall(r'^\s*(CUDA\d+):', listing, re.M)
    check('two CUDA devices available', r.returncode == 0 and len(names) >= 2)
    for flags in [[], ['--no-kvmem'], ['--device', ','.join(names[:2])],
                  ['--no-kvmem', '--device', ','.join(names[:2]), '--split-mode', 'none'],
                  ['--tensor-split', '1,1'], ['--split-mode', 'row'], ['--split-mode', 'tensor']]:
        r = subprocess.run([a.server, '-m', '__missing__.gguf', *flags], env=env,
                           capture_output=True, timeout=30)
        check('reject multi-GPU ' + str(flags), r.returncode != 0 and
              b'multi-GPU is not supported yet' in r.stderr and b'failed to load model' not in r.stderr)
    # Both explicit device selection and main-gpu indexing must target CUDA1.
    for name, flags in [('device', ['--device', names[1]]),
                        ('main-gpu', ['--split-mode', 'none', '--main-gpu', '1'])]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
                '-c', '2048', '--no-kvmem', '--kv-dtype', 'f16', '--gpu-layers', 'all',
                '--no-webui', '--predict', '32', '-s', '123', '--threads', '2',
                '--threads-batch', '2', '--reasoning-effort', 'none', *flags]
        logpath = out / (name + '.log')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f'http://127.0.0.1:{port}'
        with logpath.open('wb') as log:
            proc = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                deadline = time.monotonic() + 180
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{name} server exited; see {logpath}')
                    try:
                        with opener.open(base + '/health', timeout=2) as res:
                            if res.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(name)
                    time.sleep(0.5)
                payload = {'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with the number only.'}],
                           'temperature': 0, 'max_tokens': 32, 'reasoning_effort': 'none'}
                req = urllib.request.Request(base + '/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
                with opener.open(req, timeout=60) as res:
                    body = json.load(res)
                check(name + ' inference', body['choices'][0]['message']['content'].strip() == '5')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        text = logpath.read_text(encoding='utf-8', errors='replace')
        check(name + ' selects requested GPU', 'using device ' + names[1] in text and
              'using device ' + names[0] not in text)
finally:
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
