"""Local Bonsai smoke test, optionally with MTP; needs an existing GGUF and CUDA build."""
import argparse
import json
import os
import re
import socket
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
ap.add_argument('--mtp', action='store_true', help='Enable snapshot MTP with one draft token; requires a model with an MTP head')
ap.add_argument('--cache-type-k', choices=('q8_0', 'q5_0', 'q4_0'), default='q8_0')
ap.add_argument('--cache-type-v', choices=('q8_0', 'q5_0', 'q4_0'), default='q8_0')
ap.add_argument('--draft-kv', choices=('f16', 'q8_0', 'q5_0', 'q4_0'), help='Override the default F16 MTP KV')
ap.add_argument('--reasoning-budget', type=int, default=4096)
ap.add_argument('--request-timeout', type=int, default=7200, help='Seconds per request, including long prefill')
ap.add_argument('--long', action='store_true', help='Exercise host spill and retrieval beyond the KV pool')
ap.add_argument('--records', type=int, default=240)
ap.add_argument('--kvmem-only', action='store_true')
ap.add_argument('--plain-only', action='store_true', help='Run the same workload at --context with KVMem disabled')
ap.add_argument('--decode-tokens', type=int, default=0, help='Also request a sustained prose response')
ap.add_argument('--context', type=int, default=32768)
ap.add_argument('--budget', type=int, default=2048)
ap.add_argument('--reserve', type=int, default=1024)
ap.add_argument('--sink-tokens', type=int, default=0, help='Pinned prefix size; 0 keeps the server default')
ap.add_argument('--out', type=Path, default=Path('logs/bonsai-smoke'))
ap.add_argument('--cuda', default=r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9')
args = ap.parse_args()
if args.kvmem_only and args.plain_only:
    ap.error('Choose either --kvmem-only or --plain-only')
if args.records < 8 or not 0 <= args.decode_tokens <= args.reserve:
    ap.error('records must be at least 8; decode-tokens must fit the generation reserve')
if min(args.budget, args.reserve) < 128 or args.budget % 128 or args.reserve % 128:
    ap.error('budget and reserve must be positive multiples of 128')
if args.sink_tokens < 0 or args.sink_tokens % 128 or args.sink_tokens > args.budget:
    ap.error('sink-tokens must be 0 or a positive multiple of 128 within the budget')
if args.budget + args.reserve > args.context:
    ap.error('budget + reserve must fit the context')
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
    with opener.open(req, timeout=args.request_timeout) as response:
        return json.load(response)

def chat(messages, max_tokens=96):
    started = time.monotonic()
    result = request('/v1/chat/completions', {'model':'bonsai', 'messages':messages,
        'temperature':0, 'seed':42, 'max_tokens':max_tokens,
        'chat_template_kwargs':{'enable_thinking':False}, 'enable_thinking':False})
    result['test_wall_seconds'] = time.monotonic() - started
    return result

simple = [
    [{'role':'user','content':'Reply with just the number: 17 + 25 = ?'}],
    [{'role':'user','content':'Translate the English word bamboo into Chinese. Reply with only the translation.'}],
]
for mode in (['plain'] if args.plain_only else ['kvmem'] if args.kvmem_only else ['plain', 'kvmem']):
    command = [exe,'-m',str(args.model),'-ngl','99','--host','127.0.0.1','--port',str(args.port),
        '-c',str(args.context) if mode=='kvmem' or args.plain_only else '8192','-b','128','-ub','128','-fa','on',
        '--cache-type-k',args.cache_type_k,'--cache-type-v',args.cache_type_v,'--spec-type','draft-mtp' if args.mtp else 'none',
        '--enable-thinking','--reasoning-budget',str(args.reasoning_budget),
        '--kvmem-budget',str(args.budget),'--kvmem-gen-reserve',str(args.reserve),'--kvmem-block-tokens','128',
        '--no-ui','--no-kvmem' if mode=='plain' else '--kvmem']
    if args.mtp:
        # Omit the override to exercise the server's F16 draft default.
        command += ['--spec-draft-n-max','1','--kvmem-mtp-state','snapshots']
        if args.draft_kv:
            command += ['--spec-kv-dtype', args.draft_kv]
    if args.sink_tokens:
        command += ['--kvmem-sink-tokens', str(args.sink_tokens)]
    results[mode] = {'command':command, 'request_thinking':False, 'server_reasoning_budget':args.reasoning_budget}
    print('starting', mode, 'mtp', args.mtp, 'context', args.context, 'pool', args.budget+args.reserve, flush=True)
    peak = [0]
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            try:
                v = subprocess.check_output(['nvidia-smi','-i',args.gpu,'--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True, creationflags=creation, timeout=10)
                peak[0] = max(peak[0], int(v.strip()))
                samples.append({'time':time.time(), 'device_mib':int(v.strip())})
            except (subprocess.SubprocessError, ValueError):
                pass
            stop.wait(1)
    monitor = threading.Thread(target=sample, daemon=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
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
            startup = (args.out/f'{mode}.log').read_text(encoding='utf-8', errors='replace')
            ready = re.search(r'KVMEM_STARTUP ready=(.+)', startup)
            assert ready, 'Missing startup configuration'
            ready_config = json.loads(ready.group(1))
            assert ready_config['kvmem']['enabled'] == (mode == 'kvmem'), ready_config
            actual_kv = ready_config['kv']
            assert actual_kv == {'k':args.cache_type_k, 'v':args.cache_type_v}, actual_kv
            if args.mtp:
                pool = re.search(r'KVMEM_TRACE mtp_pool [^\r\n]+', startup)
                draft_kv = args.draft_kv or 'f16'
                assert pool and f'type_k={draft_kv} type_v={draft_kv}' in pool.group(), 'Unexpected draft KV types'
            responses = [chat(messages) for messages in simple]
            contents = [r['choices'][0]['message'].get('content','') for r in responses]
            assert '42' in contents[0], contents
            assert '竹' in contents[1], contents
            results[mode].update({'short_responses':responses, 'answers':contents})
            print(mode, contents, flush=True)
            if mode == 'kvmem' or args.plain_only:
                history = [{'role':'user','content':'Remember this identifier: BAMBOO-7429. Reply OK.'}]
                first = chat(history)
                history.append(first['choices'][0]['message'])
                history.append({'role':'user','content':'What identifier did I ask you to remember? Reply only with it.'})
                second = chat(history)
                assert 'BAMBOO-7429' in second['choices'][0]['message'].get('content',''), second
                results[mode]['multi_turn'] = [first, second]
                if args.long:
                    records = [f'Archive entry {i:04d}: the warehouse stores ordinary green bamboo crates.'
                               for i in range(args.records)]
                    records[7] = 'Archive entry 0007: the secret access code for the lunar observatory is ORCHID-5831.'
                    expected = ['ORCHID-5831']
                    question = 'What is the secret access code for the lunar observatory? Reply only with the code.'
                    if args.records > 240:
                        middle, late = args.records//2, args.records*9//10
                        records[middle] = f'Archive entry {middle:04d}: the secret access code for the coral laboratory is MAPLE-2964.'
                        records[late] = f'Archive entry {late:04d}: the secret access code for the desert telescope is CEDAR-8172.'
                        expected += ['MAPLE-2964', 'CEDAR-8172']
                        question = 'List the secret access codes for the lunar observatory, coral laboratory, and desert telescope. Reply with each facility and its code.'
                    history = [{'role':'user','content':'Read this archive and remember its facts. Reply only OK.\n'+'\n'.join(records)}]
                    (args.out/'archive-request.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
                    print('submitting archive:', args.records, 'records', flush=True)
                    first = chat(history)
                    results[mode]['long_retrieval'] = [first]
                    print('archive', first['usage'], 'seconds', round(first['test_wall_seconds'],2), flush=True)
                    history.append(first['choices'][0]['message'])
                    history.append({'role':'user','content':question})
                    second = chat(history)
                    results[mode]['long_retrieval'] = [first, second]
                    answer = second['choices'][0]['message'].get('content','')
                    results[mode]['needle_matches'] = {code:code in answer for code in expected}
                    print('retrieval', json.dumps(results[mode]['needle_matches']), flush=True)
                    assert first['usage']['prompt_tokens'] > args.budget + args.reserve, first['usage']
                    if args.decode_tokens:
                        history.append(second['choices'][0]['message'])
                        history.append({'role':'user','content':'Write a detailed 1000-word essay about how astronomical observatories work. Use complete paragraphs and keep writing until the explanation is finished.'})
                        results[mode]['sustained_decode'] = chat(history, args.decode_tokens)
                        print('sustained', results[mode]['sustained_decode']['usage'], flush=True)
            results[mode]['peak_device_mib'] = peak[0]
            results[mode]['command'] = command
        except Exception as exc:
            results[mode]['error'] = repr(exc)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
            stop.set()
            monitor.join(timeout=15)
            results.setdefault(mode, {})['peak_device_mib'] = peak[0]
            (args.out/f'{mode}-memory.json').write_text(json.dumps(samples),encoding='utf-8')
            (args.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
if not args.kvmem_only and not args.plain_only:
    results['short_greedy_answers_match'] = results['plain']['answers'] == results['kvmem']['answers']
    assert results['short_greedy_answers_match'], results
(args.out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
if args.long:
    long_mode = 'plain' if args.plain_only else 'kvmem'
    assert all(results[long_mode]['needle_matches'].values()), results[long_mode]['needle_matches']
print(json.dumps({'passed':True,'peaks_mib':{m:results[m]['peak_device_mib'] for m in ['plain','kvmem'] if m in results} }),flush=True)
