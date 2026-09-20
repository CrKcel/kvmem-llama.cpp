"""Local no-MTP smoke test; needs an existing GGUF and CUDA build."""
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
ap.add_argument('--build', type=Path, default=Path('build-win-bonsai'))
ap.add_argument('--gpu', required=True, help='GPU UUID')
ap.add_argument('--port', type=int, default=18332)
ap.add_argument('--long', action='store_true', help='Exercise host spill and retrieval beyond the KV pool')
ap.add_argument('--out', type=Path, default=Path('logs/bonsai-smoke'))
ap.add_argument('--cuda', default=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = args.gpu
env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
env['PATH'] = str(Path(args.cuda)/'bin') + os.pathsep + env['PATH']
env['KVMEM_TRACE'] = '1'
exe = str((args.build/'bin/llama-kvmem-server.exe').resolve())
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
base = f'http://127.0.0.1:{args.port}'
results = {}
creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

def request(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base+path, data=body, headers={'Content-Type':'application/json'})
    with opener.open(req, timeout=300) as response:
        return json.load(response)

def chat(messages):
    result = request('/v1/chat/completions', {'model':'bonsai', 'messages':messages,
        'temperature':0, 'seed':42, 'max_tokens':96,
        'chat_template_kwargs':{'enable_thinking':False}, 'enable_thinking':False})
    return result

reject = subprocess.run([exe, '-m', str(args.model), '--spec-type', 'draft-mtp'], env=env,
    capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=creation, timeout=30)
results['mtp_rejected'] = reject.returncode != 0 and 'MTP is disabled' in reject.stderr
assert results['mtp_rejected'], reject.stderr

simple = [
    [{'role':'user','content':'Reply with just the number: 17 + 25 = ?'}],
    [{'role':'user','content':'Translate the English word bamboo into Chinese. Reply with only the translation.'}],
]
for mode in ['plain', 'kvmem']:
    command = [exe,'-m',str(args.model),'-ngl','99','--host','127.0.0.1','--port',str(args.port),
        '-c','32768' if mode=='kvmem' else '8192','-b','128','-ub','128','-fa','on','--kv-dtype','q8_0','--spec-type','none',
        '--kvmem-budget','2048','--kvmem-gen-reserve','1024','--kvmem-block-tokens','128',
        '--no-ui','--no-kvmem' if mode=='plain' else '--kvmem']
    peak = [0]
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                v = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True, creationflags=creation, timeout=10)
                peak[0] = max(peak[0], int(v.strip()))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    monitor = threading.Thread(target=sample, daemon=True)
    with (args.out/f'{mode}.log').open('w',encoding='utf-8') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=log, env=env, creationflags=creation)
        monitor.start()
        try:
            for _ in range(240):
                if proc.poll() is not None:
                    raise RuntimeError(f'{mode} exited {proc.returncode}; see {mode}.log')
                try:
                    request('/health')
                    break
                except Exception:
                    time.sleep(1)
            else:
                raise TimeoutError(f'{mode} startup')
            responses = [chat(messages) for messages in simple]
            contents = [r['choices'][0]['message'].get('content','') for r in responses]
            assert '42' in contents[0], contents
            assert '竹' in contents[1], contents
            results[mode] = {'short_responses':responses, 'answers':contents}
            print(mode, contents, flush=True)
            if mode == 'kvmem':
                history = [{'role':'user','content':'Remember this identifier: BAMBOO-7429. Reply OK.'}]
                first = chat(history)
                history.append(first['choices'][0]['message'])
                history.append({'role':'user','content':'What identifier did I ask you to remember? Reply only with it.'})
                second = chat(history)
                assert 'BAMBOO-7429' in second['choices'][0]['message'].get('content',''), second
                results[mode]['multi_turn'] = [first, second]
                if args.long:
                    records = [f'Archive entry {i:04d}: the warehouse stores ordinary green bamboo crates.'
                               for i in range(240)]
                    records[7] = 'Archive entry 0007: the secret access code for the lunar observatory is ORCHID-5831.'
                    history = [{'role':'user','content':'Read this archive and remember its facts. Reply only OK.\n'+'\n'.join(records)}]
                    first = chat(history)
                    history.append(first['choices'][0]['message'])
                    history.append({'role':'user','content':'What is the secret access code for the lunar observatory? Reply only with the code.'})
                    second = chat(history)
                    results[mode]['long_retrieval'] = [first, second]
                    assert first['usage']['prompt_tokens'] > 3072, first['usage']
                    assert 'ORCHID-5831' in second['choices'][0]['message'].get('content',''), second
            results[mode]['peak_device_mib'] = peak[0]
            results[mode]['command'] = command
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            stop.set()
            monitor.join(timeout=15)
            (args.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
results['short_greedy_answers_match'] = results['plain']['answers'] == results['kvmem']['answers']
assert results['short_greedy_answers_match'], results
(args.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'peaks_mib':{m:results[m]['peak_device_mib'] for m in ['plain','kvmem']} }),flush=True)
