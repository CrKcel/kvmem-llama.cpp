#!/usr/bin/env python3
"""Run the issue #1 prompt against isolated, sequential server configurations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

PROMPT = 'What is 2+3? Answer with the number only.'
ROOT = Path(__file__).resolve().parents[2]


def swap_kib():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(info['SwapTotal'].split()[0]) - int(info['SwapFree'].split()[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--gpu', required=True, help='CUDA GPU UUID or index')
    ap.add_argument('--port', type=int, default=18250)
    ap.add_argument('--output', type=Path, required=True, help='new diagnostic directory')
    ap.add_argument('--cases', nargs='+', choices=['plain', 'retrieval', 'mtp-snapshots', 'mtp-replay'],
                    default=['plain', 'retrieval', 'mtp-snapshots', 'mtp-replay'])
    ap.add_argument('--hash-model', action='store_true', help='also compute full GGUF SHA-256')
    args = ap.parse_args()
    model = args.model.resolve()
    if not model.is_file() or not 1 <= args.port <= 65536 - len(args.cases):
        ap.error('model must exist and the case ports must fit in 1..65535')
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['LD_LIBRARY_PATH'] = str(ROOT / 'lib')
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = {'messages': [{'role': 'user', 'content': PROMPT}], 'temperature': 0, 'max_tokens': 32}
    (args.output / 'request.json').write_text(json.dumps(payload, indent=2) + '\n')
    info = {'gpu': args.gpu, 'model': str(model), 'model_bytes': model.stat().st_size,
            'os_release': Path('/etc/os-release').read_text(), 'cases': args.cases}
    if args.hash_model:
        with model.open('rb') as f:
            info['model_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hash_file(f)
    (args.output / 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
    results = []
    for index, case in enumerate(args.cases):
        case_port = args.port + index
        # Never kill a listener or use a potentially foreign service's /health.
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('127.0.0.1', case_port))
        spec = 'draft-mtp' if case.startswith('mtp-') else 'none'
        cmd = [str(ROOT / 'bin/llama-kvmem-server'), '-m', str(model), '--host', '127.0.0.1',
               '--port', str(case_port), '-c', '32768', '-n', '32', '-b', '512', '-ngl', '99',
               '--enable-thinking', '--reasoning-budget', '4096', '--spec-type', spec,
               '--spec-draft-n-max', '2', '--spec-kv-dtype', 'f16', '--kvmem-block-tokens', '128',
               '--kvmem-mtp-state', 'replay' if case == 'mtp-replay' else 'snapshots']
        if case == 'plain':
            cmd += ['--no-kvmem', '--kv-dtype', 'f16']
        else:
            cmd += ['--kvmem', '--kv-dtype', 'q8_0', '--kvmem-budget', '8192', '--kvmem-gen-reserve', '512']
        row = {'case': case, 'argv': cmd}
        (args.output / f'{case}.command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        baseline = swap_kib()
        stopped = threading.Event()
        done = threading.Event()
        with (args.output / f'{case}.stderr.log').open('wb') as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            def monitor():
                while not done.wait(0.5):
                    try:
                        lines = Path(f'/proc/{proc.pid}/status').read_text().splitlines()
                        proc_swap = next(int(l.split()[1]) for l in lines if l.startswith('VmSwap:'))
                        if proc_swap >= 512 * 1024 or swap_kib() - baseline >= 1024 * 1024:
                            row['swap_stop'] = True
                            stopped.set()
                            proc.terminate()
                            return
                    except (FileNotFoundError, StopIteration, ProcessLookupError):
                        return
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f'server exited {proc.returncode}; see log')
                    try:
                        op.open(f'http://127.0.0.1:{case_port}/health', timeout=1).close()
                        break
                    except OSError:
                        time.sleep(0.5)
                else:
                    raise RuntimeError('startup timeout')
                req = urllib.request.Request(f'http://127.0.0.1:{case_port}/v1/chat/completions',
                    json.dumps(payload).encode(), {'Content-Type': 'application/json'})
                try:
                    with op.open(req, timeout=120) as response:
                        row['http_status'] = response.status
                        raw = response.read().decode()
                except urllib.error.HTTPError as error:
                    row['http_status'] = error.code
                    raw = error.read().decode()
                (args.output / f'{case}.response.json').write_text(raw)
                response = json.loads(raw)
                choice = response.get('choices', [{}])[0]
                message = choice.get('message', {})
                content = message.get('content') or ''
                row.update(finish_reason=choice.get('finish_reason'), content=content, usage=response.get('usage'))
                row['classification'] = ('correct' if content.strip() == '5' else
                    'thinking_only_at_limit' if choice.get('finish_reason') == 'length' and not content.strip()
                    and message.get('reasoning_content') else 'needs_review')
            except Exception as error:
                row['error'] = str(error)
            finally:
                done.set()
                thread.join(timeout=2)
                if proc.poll() is None:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        results.append(row)
        (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(case, row.get('classification', row.get('error')), flush=True)
        if stopped.is_set():
            raise SystemExit('Stopped on swap threshold; remaining cases not run.')


def hash_file(f):
    digest = hashlib.sha256()
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


if __name__ == '__main__':
    main()
