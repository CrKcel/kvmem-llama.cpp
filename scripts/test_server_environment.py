"""Environment precedence, fail-closed authentication and startup diagnostic tests.
Pass --model SMALL_GGUF for live HTTP tests; no model downloads are performed.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', required=True)
p.add_argument('--model')
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output).resolve()
out.mkdir(parents=True, exist_ok=True)
base_env = {k: v for k, v in os.environ.items() if not k.upper().startswith('LLAMA_')}
checks = []
def check(name, ok):
    checks.append({'name': name, 'pass': bool(ok)})
    print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    if not ok:
        raise AssertionError(name)

def run(values, args=()):
    env = base_env | {'LLAMA_ARG_MODEL': '__missing__.gguf', 'LLAMA_ARG_N_GPU_LAYERS': '0'} | values
    r = subprocess.run([a.server, *args], env=env, capture_output=True, timeout=30)
    return r.returncode, (r.stdout + r.stderr).decode(errors='replace')

try:
    for name, value in [('LLAMA_ARG_PORT', 'abc'), ('LLAMA_ARG_CTX_SIZE', '-1'),
                        ('LLAMA_ARG_UI', 'maybe'), ('LLAMA_ARG_JINJA', 'false'),
                        ('LLAMA_ARG_N_PARALLEL', '2'), ('LLAMA_ARG_DEVICE', 'CUDA0,CUDA1'),
                        ('LLAMA_API_KEY', ''), ('LLAMA_ARG_API_KEY_FILE', ''),
                        ('LLAMA_API_KEY', ','), ('LLAMA_API_KEY', 'private\ninvalid'),
                        ('LLAMA_ARG_API_KEY_FILE', '__missing-key-file__'),
                        ('LLAMA_ARG_API_KEY', 'misplaced-private-key'),
                        ('LLAMA_ARG_SSL_CERT_FILE', '__certificate__')]:
        code, text = run({name: value})
        check('reject ' + name, code != 0 and 'failed to load model' not in text and
              ('invalid' in text or 'unsupported' in text))
        if 'KEY' in name and value:
            check('no key value disclosure ' + name, value not in text)
    code, text = run({'LLAMA_ARG_FIT': 'on', 'LLAMA_ARG_UNKNOWN_TEST': 'hidden-value'})
    check('unsupported environment diagnostics', code != 0 and 'failed to load model' in text and
          'unsupported environment variable LLAMA_ARG_FIT' in text and
          'LLAMA_ARG_UNKNOWN_TEST' in text and 'hidden-value' not in text)
    code, text = run({'LLAMA_ARG_PORT': '1234', 'LLAMA_ARG_CTX_SIZE': '2048'}, ['--port', '2345', '-c', '4096'])
    check('scalar env plus CLI accepted', code != 0 and 'failed to load model' in text)
    code, text = run({'LLAMA_ARG_CHAT_TEMPLATE': 'environment-template'}, ['--chat-template', 'cli-template'])
    check('CLI template replaces environment template', code != 0 and 'failed to load model' in text)
    code, text = run({'LLAMA_ARG_MMPROJ_OFFLOAD': 'false', 'LLAMA_ARG_MMAP': '0'})
    check('boolean environment flags', code != 0 and 'input=--no-mmproj-offload' in text and 'input=--no-mmap' in text)
    if a.model:
        for values, expected in [({'LLAMA_ARG_MMPROJ': '__missing-projector__'}, '--mmproj file'),
                                 ({'LLAMA_ARG_STATIC_PATH': '__missing-ui__'}, '--ui-dir/--path'),
                                 ({'LLAMA_ARG_IMAGE_MIN_TOKENS': '512', 'LLAMA_ARG_IMAGE_MAX_TOKENS': '128'},
                                  '--image-min-tokens exceeds')]:
            code, text = run({'LLAMA_ARG_MODEL': a.model} | values)
            check('preflight ' + expected, code != 0 and expected in text and 'llama_model_loader:' not in text)
        with tempfile.TemporaryDirectory(prefix='kvmem-env-') as tmp:
            tmp = Path(tmp)
            env_file = tmp / 'environment.keys'
            cli_file = tmp / 'command.keys'
            env_file.write_text('file-env-secret\n', encoding='ascii')
            cli_file.write_text('file-cli-secret\n', encoding='ascii')
            empty_file = tmp / 'empty.keys'
            empty_file.write_text('# no keys\n', encoding='ascii')
            code, text = run({'LLAMA_ARG_API_KEY_FILE': str(empty_file)})
            check('empty environment key file fails closed', code != 0 and 'contains no usable keys' in text)
            ui = tmp / 'ui with spaces'
            ui.mkdir()
            (ui / 'index.html').write_text('environment-ui', encoding='ascii')
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = base_env | {
                'LLAMA_ARG_MODEL': a.model, 'LLAMA_ARG_HOST': '127.0.0.1', 'LLAMA_ARG_PORT': '1',
                'LLAMA_ARG_CTX_SIZE': '2048', 'LLAMA_ARG_N_PREDICT': '-1',
                'LLAMA_ARG_N_GPU_LAYERS': 'all', 'LLAMA_ARG_DEVICE': 'CUDA0',
                'LLAMA_ARG_ALIAS': 'env-model', 'LLAMA_ARG_THREADS': '2',
                'LLAMA_ARG_BATCH': '256', 'LLAMA_ARG_UBATCH': '64', 'LLAMA_ARG_FLASH_ATTN': 'on',
                'LLAMA_ARG_CACHE_TYPE_K': 'f16', 'LLAMA_ARG_CACHE_TYPE_V': 'f16',
                'LLAMA_ARG_UI': 'false', 'LLAMA_ARG_STATIC_PATH': str(ui),
                'LLAMA_API_KEY': 'env-secret', 'LLAMA_ARG_API_KEY_FILE': str(env_file),
                'LLAMA_ARG_TIMEOUT': '30', 'LLAMA_ARG_THREADS_HTTP': '2',
                'LLAMA_ARG_REASONING_EFFORT': 'none',
            }
            args = [a.server, '--port', str(port), '-c', '4096', '-t', '3', '--ui', '--no-kvmem',
                    '--api-key', 'cli-secret', '--api-key-file', str(cli_file)]
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            base = f'http://127.0.0.1:{port}'
            def request(path, key=None, payload=None):
                headers = {'Content-Type': 'application/json'}
                if key is not None:
                    headers['Authorization'] = 'Bearer ' + key
                req = urllib.request.Request(base + path, headers=headers,
                    data=None if payload is None else json.dumps(payload).encode())
                try:
                    with opener.open(req, timeout=60) as res:
                        return res.status, res.read()
                except urllib.error.HTTPError as e:
                    return e.code, e.read()
            logpath = out / 'server.log'
            with logpath.open('wb') as log:
                proc = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    deadline = time.monotonic() + 180
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError('server exited; see server.log')
                        try:
                            if request('/health')[0] == 200:
                                break
                        except OSError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError('startup')
                        time.sleep(0.5)
                    check('CLI overrides environment UI disable', request('/')[1] == b'environment-ui')
                    check('environment auth enforced', request('/v1/models')[0] == 401)
                    for key in ['env-secret', 'cli-secret', 'file-env-secret', 'file-cli-secret']:
                        status, body = request('/v1/models', key)
                        check('appended key ' + str(len(checks)), status == 200 and json.loads(body)['data'][0]['id'] == 'env-model')
                    payload = {'messages': [{'role': 'user', 'content': 'What is 2+3? Answer with the number only.'}],
                               'temperature': 0, 'max_tokens': 32, 'reasoning_effort': 'none'}
                    status, body = request('/v1/chat/completions', 'env-secret', payload)
                    check('environment-configured inference', status == 200 and
                          json.loads(body)['choices'][0]['message']['content'].strip() == '5')
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            text = logpath.read_text(encoding='utf-8', errors='replace')
            ready = json.loads(next(line.split('ready=', 1)[1] for line in text.splitlines()
                                    if line.startswith('KVMEM_STARTUP ready=')))
            check('effective startup values', ready['context_actual'] == 4096 and ready['ubatch_actual'] == 64 and
                  ready['threads_actual'] == 3 and ready['threads_batch_actual'] == 3 and ready['http']['port'] == port)
            check('effective sources', ready['sources']['--ctx-size'] == 'cli' and
                  ready['sources']['--model'] == 'env:LLAMA_ARG_MODEL' and ready['sources']['--ui'] == 'cli')
            check('auth sources and count', ready['auth'] == {'enabled': True, 'key_count': 4} and
                  ready['sources']['--api-key'] == ['env:LLAMA_API_KEY', 'cli'])
            check('output limit diagnostics', ready['n_predict'] == -1 and ready['default_max_tokens'] == 4096)
            check('startup logs redact keys', all(k not in text for k in ['env-secret', 'cli-secret', 'file-env-secret', 'file-cli-secret']))
            with socket.socket() as occupied:
                if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                occupied.bind(('127.0.0.1', 0))
                occupied.listen()
                conflict_port = occupied.getsockname()[1]
                failed = subprocess.run([*args, '--port', str(conflict_port)], env=env,
                                        capture_output=True, timeout=90)
                error = failed.stderr.decode(errors='replace')
                check('bind failure diagnostics', failed.returncode != 0 and
                      f'cannot bind 127.0.0.1:{conflict_port}' in error and
                      'KVMEM_STARTUP ready=' not in error and 'listening on http' not in error)
finally:
    (out / 'results.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
