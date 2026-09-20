"""Real-model logging regression. Uses a supplied model and private loopback servers."""
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
env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'KVMEM_TRACE', 'KVMEM_PERF'))}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
checks = []

def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)

try:
    for value in ('-1', '6', '1.5', '2x', '999999999999'):
        r = subprocess.run([a.server, '--verbosity', value], env=env, capture_output=True, timeout=30)
        check('reject verbosity ' + value, r.returncode != 0 and b'requires an integer' in r.stderr)
    baseline = None
    cases = [('default', {}, [], False, True),
             ('env-zero', {'KVMEM_TRACE': '0'}, [], False, True),
             ('env-trace', {'KVMEM_TRACE': '1'}, [], True, True),
             ('cli-off', {'KVMEM_TRACE': '1'}, ['--no-kvmem-trace'], False, True),
             ('cli-on', {'KVMEM_TRACE': '0'}, ['--kvmem-trace'], True, True),
             ('perf-only', {'KVMEM_PERF': '1'}, ['--no-kvmem-trace'], False, True),
             ('errors-only', {}, ['-lv', '1'], False, False)]
    for name, extra_env, flags, trace, info in cases:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        args = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(port),
                '-c', '4096', '-n', '16', '--kvmem-budget', '2048', '--kvmem-gen-reserve', '512',
                '--kv-dtype', 'q8_0', '--threads', '2', '--threads-batch', '2', '--no-webui',
                '--reasoning-effort', 'none', '--temp', '0', *flags]
        logpath = out / (name + '.log')
        # A value beginning with '-' must never be mistaken for an option in config logs.
        secret = '-logging-test-secret'
        run_env = env | extra_env | {'LLAMA_API_KEY': secret}
        responses = []
        with logpath.open('wb') as log:
            proc = subprocess.Popen(args, env=run_env, stdout=log, stderr=log,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                deadline = time.monotonic() + 180
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'{name} exited; see {logpath}')
                    try:
                        with opener.open(base + '/health', timeout=2) as res:
                            if res.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(name)
                    time.sleep(0.2)
                for _ in range(2):
                    body = {'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with the number only.'}],
                            'max_tokens': 16, 'temperature': 0, 'seed': 123, 'reasoning_effort': 'none'}
                    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                          {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + secret})
                    with opener.open(req, timeout=120) as res:
                        responses.append(json.load(res))
                time.sleep(0.2)  # Allow the asynchronous logger to drain before process termination.
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(timeout=30)
        text = logpath.read_text(encoding='utf-8', errors='replace')
        (out / (name + '.json')).write_text(json.dumps(responses, indent=2), encoding='utf-8')
        check(name + ' raw diagnostics gate', bool(re.search(r'^KVMEM_PREFILL_DECISION ', text, re.M)) == trace)
        perf = extra_env.get('KVMEM_PERF') == '1'
        check(name + ' all raw records gated', bool(re.search(r'^KVMEM_', text, re.M)) == (trace or perf))
        if perf:
            check(name + ' performance counters independent', 'KVMEM_HARVEST' in text and
                  'KVMEM_PREFILL_DIAGNOSTIC' in text and not re.search(r'^KVMEM_TRACE ', text, re.M))
        check(name + ' adapter diagnostics gate', bool(re.search(r'^KVMEM_KV_BYTES ', text, re.M)) == trace)
        check(name + ' info level', ('prompt eval time =' in text) == info)
        check(name + ' startup summary', ('listening on http' in text) == info)
        check(name + ' key redaction', secret not in text)
        counts = [r['timings'] for r in responses]
        check(name + ' cache hit counted', counts[1]['cache_n'] > 0 and counts[1]['prompt_n'] < counts[0]['prompt_n'])
        if info:
            prompt_counts = [int(v) for v in re.findall(r'prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens', text)]
            check(name + ' log/API token counts agree', prompt_counts == [t['prompt_n'] for t in counts])
        content = responses[0]['choices'][0]['message']['content']
        if baseline is None:
            baseline = content
        check(name + ' generation unchanged', content == baseline and bool(content))
    for flags in ([], ['--no-kvmem-trace'], ['-lv', '1']):
        r = subprocess.run([a.server, '-m', '__missing_logging_model__.gguf', *flags], env=env,
                           capture_output=True, timeout=30)
        check('startup error visible ' + str(flags), r.returncode != 0 and b'failed to load model' in r.stderr)
finally:
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
