"""Run sequential Bonsai MTP integration checks against local CUDA servers."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--kvmem', action='store_true')
p.add_argument('--drafts', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--port', type=int, default=18345)
p.add_argument('--context', type=int, default=8192)
p.add_argument('--budget', type=int, default=2048)
p.add_argument('--reserve', type=int, default=1024)
p.add_argument('--long', action='store_true')
p.add_argument('--repeats', type=int, default=1, help='Prose runs; when >1 the first is warmup')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', KVMEM_TRACE='1')
env['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin' + os.pathsep + env['PATH']
creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = f'http://127.0.0.1:{a.port}'


def request(path, data=None, timeout=600):
    body = None if data is None else json.dumps(data).encode()
    with opener.open(urllib.request.Request(url + path, data=body, headers={'Content-Type': 'application/json'}), timeout=timeout) as r:
        return json.load(r)


def chat(messages, limit=64, **options):
    payload = dict(model='bonsai', messages=messages, temperature=0, seed=42, max_tokens=limit,
                   chat_template_kwargs={'enable_thinking': False})
    payload.update(options)
    return request('/v1/chat/completions', payload)


def ask(text, limit=64):
    return chat([{'role': 'user', 'content': text}], limit)


def content(result):
    return result['choices'][0]['message'].get('content', '') or ''


results = {}
for draft in a.drafts:
    # Refuse to reuse another service's port.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', a.port))
    command = [str(a.binary.resolve()), '-m', str(a.model.resolve()), '-ngl', '99',
        '--host', '127.0.0.1', '--port', str(a.port), '-c', str(a.context), '-b', '128', '-ub', '128', '-fa', 'on',
        '--spec-type', 'draft-mtp' if draft else 'none']
    if draft:
        command += ['--spec-draft-n-max', str(draft), '--spec-draft-p-min', '0']
    if a.kvmem:
        command += ['--kv-dtype', 'q8_0', '--spec-kv-dtype', 'q8_0', '--kvmem', '--no-ui',
                    '--kvmem-budget', str(a.budget), '--kvmem-gen-reserve', str(a.reserve),
                    '--kvmem-block-tokens', '128', '--kvmem-mtp-state', 'snapshots']
    else:
        command += ['-ctk', 'q8_0', '-ctv', 'q8_0', '-ctkd', 'q8_0', '-ctvd', 'q8_0', '-np', '1']
    entry = results[str(draft)] = {'command': command, 'checks': {}}
    stop = threading.Event()
    peak = [0]
    telemetry = []

    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '-i', a.gpu,
                    '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem',
                    '--format=csv,noheader,nounits'], text=True, creationflags=creation, timeout=10)
                fields = [v.strip() for v in value.split(',')]
                peak[0] = max(peak[0], int(fields[0]))
                telemetry.append({'time': time.time(), 'values': fields})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stop.wait(0.5)

    log_path = a.out / f'draft-{draft}.log'
    with log_path.open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=log, creationflags=creation)
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(f'Server exited {proc.returncode}; see {log_path}')
                try:
                    request('/health', timeout=2)
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Server startup')
            for label, text, expected in [('math', 'Reply with just the number: 17 + 25 = ?', '42'),
                    ('translation', 'Translate bamboo into Chinese. Reply only with the translation.', '竹')]:
                response = ask(text)
                entry[label] = response
                entry['checks'][label] = expected in content(response)
            history = [{'role': 'user', 'content': 'Remember the identifier BAMBOO-7429. Reply OK.'}]
            first = chat(history)
            history += [first['choices'][0]['message'], {'role': 'user', 'content': 'What identifier did I ask you to remember? Reply only with it.'}]
            second = chat(history)
            entry['multi_turn'] = [first, second]
            entry['checks']['multi_turn'] = 'BAMBOO-7429' in content(second)
            entry['prose_runs'] = [ask('Write a detailed 1000-word explanation of how astronomical observatories work. Use complete paragraphs.', 128)
                                   for _ in range(max(1, a.repeats))]
            entry['prose'] = entry['prose_runs'][-1]
            entry['checks']['repeat_output'] = all(content(r) == content(entry['prose']) for r in entry['prose_runs'])
            entry['checks']['output_limit'] = entry['prose']['usage']['completion_tokens'] <= 128
            if a.long:
                records = [f'Archive entry {i:04d}: the warehouse stores ordinary green bamboo crates.' for i in range(240)]
                records[7] = 'Archive entry 0007: the lunar observatory access code is ORCHID-5831.'
                history = [{'role': 'user', 'content': 'Remember the following archive. Reply OK.\n' + '\n'.join(records)}]
                first = chat(history, 16)
                history += [first['choices'][0]['message'], {'role': 'user', 'content': 'What is the lunar observatory access code? Reply only with the code.'}]
                second = chat(history)
                entry['long'] = [first, second]
                entry['checks']['long_recall'] = 'ORCHID-5831' in content(second)
                entry['checks']['spilled'] = first['usage']['prompt_tokens'] > a.budget + a.reserve
            if a.kvmem:
                thinking = chat([{'role': 'user', 'content': 'What is 23 multiplied by 19?'}], 96,
                                chat_template_kwargs={'enable_thinking': True}, reasoning_budget_tokens=16)
                entry['thinking'] = thinking
                entry['checks']['thinking_budget'] = '437' in content(thinking)
                # Disconnect a streaming client, then require a clean subsequent request.
                data = dict(model='bonsai', messages=[{'role': 'user', 'content': 'Write a very long essay about bamboo.'}],
                            stream=True, max_tokens=512, temperature=0, chat_template_kwargs={'enable_thinking': False})
                with opener.open(urllib.request.Request(url + '/v1/chat/completions',
                        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'}), timeout=120) as r:
                    chunks = []
                    for _ in range(200):
                        line = r.readline()
                        if not line:
                            break
                        if not line.startswith(b'data: {'):
                            continue
                        event = json.loads(line[6:])
                        for choice in event.get('choices', []):
                            text = choice.get('delta', {}).get('content')
                            if text:
                                chunks.append(text)
                        if len(chunks) >= 3:
                            break
                entry['cancel_received_chunks'] = chunks
                entry['checks']['cancel_mid_decode'] = len(chunks) >= 3
                response = ask('Reply with just the number: 8 + 9 = ?')
                entry['after_cancel'] = response
                entry['checks']['after_cancel'] = '17' in content(response)
            print(json.dumps({'draft': draft, 'checks': entry['checks'], 'prose_timings': entry['prose'].get('timings')}, ensure_ascii=False), flush=True)
        except Exception as exc:
            entry['error'] = repr(exc)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=30)
            stop.set()
            watcher.join(timeout=15)
            entry['peak_device_mib'] = peak[0]
            entry['gpu_telemetry_columns'] = ['memory_mib', 'utilization_percent', 'temperature_c', 'power_w', 'clock_sm_mhz', 'clock_mem_mhz']
            entry['gpu_telemetry'] = telemetry
            entry['spec_stats'] = re.findall(r'KVMEM_TRACE spec_stats [^\r\n]+', log_path.read_text(encoding='utf-8', errors='replace'))
            (a.out / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(entry['checks'].values()):
        raise AssertionError(entry['checks'])
print('PASS', flush=True)
