#!/usr/bin/env python3
"""Exercise sampling defaults, overrides and HTTP validation on 0.8B / RTX 5050.

Starts isolated servers for ordinary and MTP generation; leaves IQ4 untouched.
"""
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request

from mtp_canary import find_mtp_model

ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base, body):
    req = urllib.request.Request(base + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with OPENER.open(req, timeout=120) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def main():
    binary = ROOT / 'build/bin/llama-kvmem-server'
    model = find_mtp_model()
    folder = ROOT / 'logs' / ('chat_sampling_' + time.strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    print('ARTIFACTS', folder, flush=True)
    env = os.environ.copy()
    env['KVMEM_TRACE'] = '1'  # Sampling records are opt-in diagnostics.
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = 'GPU-14f08a8c-8d62-4338-8ae4-c669889cdb29'
    env['LD_LIBRARY_PATH'] = str(ROOT / 'build/bin') + ':/home/leye/kvmem_qw3/.cu13-env/lib'
    results = []
    for mode in ('none', 'draft-mtp'):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        log = folder / (mode + '.stderr.log')
        cmd = [str(binary), '-m', str(model), '--port', str(port), '-c', '2048',
               '-b', '128', '-ngl', '99', '--kvmem', '--kvmem-budget', '1024',
               '--kvmem-gen-reserve', '256', '--kvmem-block-tokens', '32',
               '--kv-dtype', 'q5_0', '--spec-kv-dtype', 'f16',
               '--spec-type', mode, '--spec-draft-n-max', '2',
               '--enable-thinking', '--reasoning-budget', '0', '--seed', '42']
        # Verify CLI default inheritance as well as per-request override.
        if mode == 'draft-mtp':
            cmd += ['--top-p', '0.91']
        with log.open('w') as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=fh)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(log.read_text()[-2000:])
                try:
                    with OPENER.open(base + '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError('health timeout')

            def check(label, fields, expected, stream=False):
                offset = log.stat().st_size
                body = {'messages': [{'role': 'user', 'content': 'List the first ten positive integers.'}],
                        'max_tokens': 16, 'stream': stream, **fields}
                status, raw = post(base, body)
                assert status == 200, (label, status, raw)
                if stream:
                    assert 'data: [DONE]' in raw, raw
                    chunks = [json.loads(s[6:]) for s in raw.splitlines()
                              if s.startswith('data: ') and s != 'data: [DONE]']
                    assert all('error' not in c for c in chunks), chunks
                    content = ''.join(c.get('delta', {}).get('content') or ''
                                      for chunk in chunks for c in chunk.get('choices', []))
                else:
                    response = json.loads(raw)
                    assert 'error' not in response and response['usage']['completion_tokens'] > 0, response
                    content = response['choices'][0]['message'].get('content') or ''
                time.sleep(0.05)
                tail = log.read_text()[offset:]
                line = next(s for s in tail.splitlines() if 'KVMEM_TRACE sampling ' in s)
                actual = dict(re.findall(r'(\w+)=([^ ]+)', line))
                for key, value in expected.items():
                    assert abs(float(actual[key]) - value) < 1e-6, (label, key, actual, expected)
                if mode == 'draft-mtp':
                    assert 'spec_verify ' in tail, (label, tail[-1000:])
                results.append({'mode': mode, 'case': label, 'sampling': actual})
                print('PASS', mode, label, flush=True)
                return content

            tp = 0.91 if mode == 'draft-mtp' else 0.95
            check('thinking-default', {}, {'thinking': 1, 'temperature': 1, 'top_p': tp,
                  'top_k': 20, 'min_p': 0, 'presence_penalty': 0, 'frequency_penalty': 0,
                  'repetition_penalty': 1, 'seed': 42})
            check('nonthinking-default', {'chat_template_kwargs': {'enable_thinking': False}},
                  {'thinking': 0, 'temperature': .7, 'top_p': .91 if mode == 'draft-mtp' else .8,
                   'presence_penalty': 1.5}, stream=True)
            check('override-all', {'temperature': .6, 'top_p': .77, 'top_k': 7, 'min_p': .02,
                  'presence_penalty': .3, 'frequency_penalty': .2, 'repeat_penalty': 1.1, 'seed': 123},
                  {'temperature': .6, 'top_p': .77, 'top_k': 7, 'min_p': .02,
                   'presence_penalty': .3, 'frequency_penalty': .2, 'repetition_penalty': 1.1, 'seed': 123}, stream=True)
            greedy = check('greedy', {'temperature': 0, 'top_k': 7, 'min_p': .5,
                           'enable_thinking': False, 'presence_penalty': 0},
                           {'temperature': 0, 'top_k': 0, 'top_p': 1, 'min_p': 0})
            single = check('top-k-one', {'temperature': 1.7, 'top_k': 1,
                           'enable_thinking': False, 'presence_penalty': 0}, {'temperature': 1.7, 'top_k': 1})
            assert single == greedy and single, (mode, 'top_k=1 must select the greedy token', single, greedy)
            check('null-inherits', {'temperature': None, 'top_p': None, 'seed': None},
                  {'temperature': 1, 'top_p': tp, 'seed': 42})
            invalid = [('temperature', '1'), ('temperature', -1), ('temperature', 2.1),
                       ('top_p', 1.1), ('top_p', False), ('top_k', 2.5), ('top_k', -1),
                       ('top_k', 2**40), ('min_p', -.1), ('presence_penalty', 3),
                       ('frequency_penalty', -3), ('repetition_penalty', 0),
                       ('seed', -1), ('seed', 2**32), ('seed', 1.5)]
            for key, value in invalid:
                status, raw = post(base, {'messages': [{'role': 'user', 'content': 'Hi'}], key: value})
                assert status == 400 and key in json.loads(raw)['error'], (key, value, status, raw)
            status, raw = post(base, {'messages': [{'role': 'user', 'content': 'Hi'}],
                                     'repeat_penalty': 1, 'repetition_penalty': 1.1})
            assert status == 400, raw
            print('PASS', mode, '16 invalid requests return HTTP 400', flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print('PASS all sampling checks', flush=True)


if __name__ == '__main__':
    main()
