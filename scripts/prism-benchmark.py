"""Compare pristine Prism and KVMem with the same model, requests and CUDA kernels."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--model', type=Path, required=True)
ap.add_argument('--gpu', required=True)
ap.add_argument('--profile', choices=['original-32k','comparison'], default='original-32k')
ap.add_argument('--out', type=Path, default=Path('logs/prism-comparison'))
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=args.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID')
env['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin' + os.pathsep + env['PATH']
env.pop('KVMEM_TRACE', None)
creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = 'http://127.0.0.1:18333'
results = {}

def request(path, data=None, timeout=1800):
    req = urllib.request.Request(base + path, data=None if data is None else json.dumps(data).encode(),
                                 headers={'Content-Type':'application/json'})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)

def chat(messages, count):
    started = time.monotonic()
    response = request('/v1/chat/completions', {'model':'bonsai', 'messages':messages,
        'max_tokens':count, 'temperature':0, 'seed':42, 'cache_prompt':False,
        'chat_template_kwargs':{'enable_thinking':False}, 'enable_thinking':False})
    return {'wall_seconds':time.monotonic()-started, 'response':response}

def save():
    (args.out/'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

def run(name, context, kvmem=False, capacity=False):
    exe = Path('build-win-bonsai/bin/llama-kvmem-server.exe' if kvmem else
               'build-win-prism-reference/build/bin/llama-server.exe').resolve()
    command = [str(exe), '-m', str(args.model), '-ngl', '99', '-c', str(context),
               '-b','128','-ub','128','-fa','on','-ctk','q8_0','-ctv','q8_0',
               '-t','4','-tb','4','--host','127.0.0.1','--port','18333','--spec-type','none']
    if kvmem:
        command += ['--kvmem-budget','24576','--kvmem-gen-reserve','10240','--no-ui']
    else:
        command += ['--parallel','1','--fit','off','--reasoning','off','--cache-ram','0','--ctx-checkpoints','0','--perf']
    entry = results[name] = {'command':command, 'requests':[]}
    samples, stop = [], threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True, creationflags=creation, timeout=10)
                samples.append({'time':time.time(), 'device_mib':int(value.strip())})
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    with (args.out/f'{name}.log').open('w',encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, env=env, creationflags=creation)
        thread.start()
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    entry['startup_exit'] = proc.returncode
                    print(name, 'startup exit', proc.returncode, flush=True)
                    return
                try:
                    request('/health', timeout=2)
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError('startup did not become healthy')
            entry['healthy'] = True
            entry['warmup'] = chat([{'role':'user','content':'Reply with just the number: 17 + 25 = ?'}],16)
            if capacity:
                # Use the same archive, with one long reply to measure full-KV decode.
                evidence = Path('logs/bonsai-32k-smoke' if context == 32768 else 'logs/bonsai-64k-24k-10k')
                history = json.loads((evidence/'archive-request.json').read_text(encoding='utf-8'))
                history[0]['content'] = history[0]['content'].replace('Read this archive and remember its facts. Reply only OK.', 'Read this archive.')
                history[0]['content'] += '\nWrite a detailed 1000-word essay about how astronomical observatories work. Use complete paragraphs and keep writing until the explanation is finished.'
                entry['requests'].append({'case':f'{context}_full_kv', **chat(history,256)})
            else:
                for records in [240,1400]:
                    archive = '\n'.join(f'Archive entry {i:04d}: the warehouse stores ordinary green bamboo crates.' for i in range(records))
                    messages = [{'role':'user','content':archive+'\nWrite a detailed 1000-word essay about how astronomical observatories work. Use complete paragraphs and keep writing until the explanation is finished.'}]
                    response = chat(messages,256)
                    entry['requests'].append({'case':f'{records}_records', **response})
                    save()
                    print(name, records, json.dumps(response['response'].get('timings',{})), flush=True)
            entry['completed'] = True
        except Exception as exc:
            entry['error'] = repr(exc)
            print(name, 'error', repr(exc), flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            stop.set()
            thread.join(timeout=15)
            entry['peak_device_mib'] = max((s['device_mib'] for s in samples), default=None)
            (args.out/f'{name}-memory.json').write_text(json.dumps(samples),encoding='utf-8')
            save()

if args.profile == 'original-32k':
    run('prism-32k',32768,capacity=True)
else:
    run('prism-64k',65536,capacity=True)
    run('prism-matched',34816)
    run('kvmem-matched',65536,kvmem=True)
print('Comparison results saved:',args.out,flush=True)
if args.profile == 'original-32k' and not results['prism-32k'].get('completed'):
    raise SystemExit(1)
