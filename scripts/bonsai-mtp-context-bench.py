"""Sequential local MTP 0/1/2/3 benchmark; fresh prompts, fixed KV pool."""
import argparse
import hashlib
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
p.add_argument('--out', type=Path, required=True)
p.add_argument('--repeats', type=int, default=2)
p.add_argument('--port', type=int, default=18346)
p.add_argument('--drafts', type=int, nargs='+', default=[0, 1, 2, 3])
p.add_argument('--draft-kv', choices=['q8_0', 'f16'], help='Override draft KV (current server default: f16)')
a = p.parse_args()
assert a.repeats > 0 and all(d in range(4) for d in a.drafts)
a.out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', KVMEM_TRACE='1')
env.pop('KVMEM_AUDIT_RETRIEVAL', None)
env['PATH'] = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin' + os.pathsep + env['PATH']
creation = subprocess.CREATE_NO_WINDOW
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

def request(path, body=None, timeout=600):
    req = urllib.request.Request(f'http://127.0.0.1:{a.port}'+path,
        data=None if body is None else json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with opener.open(req, timeout=timeout) as r:
        return json.load(r)

def chat(text, count):
    return request('/v1/chat/completions', {'model':'bonsai',
        'messages':[{'role':'user','content':text}], 'temperature':0, 'seed':42,
        'presence_penalty':0, 'frequency_penalty':0, 'max_tokens':count,
        'chat_template_kwargs':{'enable_thinking':False}, 'enable_thinking':False})

report = {'settings':{'context':131072, 'budget':24576, 'reserve':10240,
    'kv':'q8_0/q8_0', 'draft_kv_requested':a.draft_kv, 'batch':128, 'ubatch':128, 'thinking':False,
    'output_tokens':512, 'repeats':a.repeats, 'gpu':a.gpu, 'trace':True,
    'audit':False, 'prefix':'default128', 'presence_penalty':0}, 'runs':[]}
for label, path in [('binary',a.binary),('model',a.model)]:
    with path.open('rb') as f:
        report[label+'_sha256'] = hashlib.file_digest(f,'sha256').hexdigest()

def save():
    (a.out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

for draft in a.drafts:
    with socket.socket() as s: s.bind(('127.0.0.1',a.port))
    cmd = [str(a.binary.resolve()), '-m',str(a.model.resolve()), '-ngl','99',
        '--host','127.0.0.1','--port',str(a.port), '-c','131072','-b','128','-ub','128',
        '-t','4','-tb','4','-fa','on','--cache-type-k','q8_0','--cache-type-v','q8_0',
        '--kvmem','--kvmem-budget','24576','--kvmem-gen-reserve','10240',
        '--kvmem-block-tokens','128','--no-ui','--enable-thinking','--reasoning-budget','4096',
        '--spec-type','draft-mtp' if draft else 'none']
    if draft:
        cmd += ['--spec-draft-n-max',str(draft),'--spec-draft-p-min','0','--kvmem-mtp-state','snapshots']
        if a.draft_kv:
            cmd += ['--spec-kv-dtype',a.draft_kv]
    entry = {'draft':draft,'command':cmd,'measurements':[]}
    report['runs'].append(entry)
    stop = threading.Event()
    samples = []
    def monitor():
        while not stop.is_set():
            try:
                values = subprocess.check_output(['nvidia-smi','-i',a.gpu,
                    '--query-gpu=memory.used,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv,noheader,nounits'],text=True,creationflags=creation,timeout=10)
                samples.append({'time':time.time(),'values':[x.strip() for x in values.split(',')]})
            except (OSError,ValueError,subprocess.SubprocessError): pass
            stop.wait(1)
    logpath = a.out/f'draft-{draft}.log'
    print(f'START draft={draft}',flush=True)
    with logpath.open('w',encoding='utf-8') as log:
        proc = subprocess.Popen(cmd,env=env,stdout=log,stderr=log,creationflags=creation)
        watcher = threading.Thread(target=monitor,daemon=True)
        watcher.start()
        try:
            for _ in range(240):
                if proc.poll() is not None: raise RuntimeError(f'startup exit {proc.returncode}')
                try: request('/health',timeout=2); break
                except OSError: time.sleep(1)
            else: raise TimeoutError('startup')
            startup=logpath.read_text(encoding='utf-8',errors='replace')
            ready=json.loads(re.search(r'KVMEM_STARTUP ready=(.+)',startup)[1])
            assert ready['kv']=={'k':'q8_0','v':'q8_0'} and ready['kvmem']['enabled']
            if draft:
                assert f'spec_start type=draft-mtp n_max={draft} p_min=0.000' in startup
                pool = re.search(r'KVMEM_TRACE mtp_pool ([^\r\n]+)', startup)
                expected_draft_kv = a.draft_kv or 'f16'
                assert pool and f'type_k={expected_draft_kv} type_v={expected_draft_kv}' in pool[1]
                entry['draft_kv_pool'] = pool[1]
            entry['startup']=ready
            warmup=chat('Write a detailed explanation of how reflecting telescopes focus light.',128)
            assert warmup['usage']['completion_tokens']>=100
            entry['warmup']=warmup
            for label, records in [('4k',240),('16k',950)]:
                for repeat in range(a.repeats):
                    # Different leading content forces a cold logical prompt;
                    # every draft mode receives the same prompt for each cell.
                    text=f'Benchmark {label} trial {repeat}. Read the following reference archive.\n'
                    text+='\n'.join(f'Archive entry {i:04d}: the warehouse stores ordinary green bamboo crates.' for i in range(records))
                    text+='\nNow write a detailed 2000-word guide to how astronomical observatories work, including telescope optics, detectors, calibration, atmospheric effects, data processing, and scientific observations. Use complete paragraphs and continue until every topic is fully explained.'
                    (a.out/f'prompt-{label}-{repeat}.txt').write_text(text,encoding='utf-8')
                    offset=len(logpath.read_text(encoding='utf-8',errors='replace'))
                    started=time.time()
                    response=chat(text,512)
                    elapsed=time.time()-started
                    for _ in range(30):
                        segment=logpath.read_text(encoding='utf-8',errors='replace')[offset:]
                        turns=re.findall(r'KVMEM_CHAT_TURN ([^\r\n]+)',segment)
                        if turns: break
                        time.sleep(.1)
                    assert turns, 'missing server timing'
                    timing={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',turns[-1])}
                    usage=response['usage']
                    assert usage.get('prompt_cache_hit_tokens',0)<=16, usage
                    assert usage['completion_tokens']==512, usage
                    assert 'prefill_pressure' not in segment, 'unexpected eviction'
                    spec=re.findall(r'KVMEM_TRACE spec_stats ([^\r\n]+)',segment)
                    stats={k:float(v) for k,v in re.findall(r'(\w+)=([\d.]+)',spec[-1])} if spec else None
                    if draft: assert stats and stats['n_gen']==512
                    histogram={}
                    for n in re.findall(r'spec_verify n_draft=(\d+)',segment):
                        histogram[n]=histogram.get(n,0)+1
                    row={'context_label':label,'repeat':repeat,'response':response,'timing':timing,
                        'wall_seconds':elapsed,'spec':stats,'draft_length_histogram':histogram,
                        'prefill_tps':usage['prompt_cache_miss_tokens']*1000/timing['prefill_ms'],
                        'peak_device_mib':max((int(s['values'][0]) for s in samples if s['time']>=started),default=0)}
                    entry['measurements'].append(row)
                    save()
                    print(json.dumps({'draft':draft,'context':label,'repeat':repeat,
                        'prompt':usage['prompt_tokens'],'cache':usage.get('prompt_cache_hit_tokens'),
                        'pp':round(row['prefill_tps'],2),'tg':timing['gen_toks'],
                        'accept':stats['accept_pct'] if stats else None,
                        'hist':histogram,'vram':row['peak_device_mib']}),flush=True)
        except Exception as exc:
            entry['error']=repr(exc)
            raise
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
            stop.set(); watcher.join(timeout=15)
            entry['peak_device_mib']=max((int(s['values'][0]) for s in samples),default=0)
            (a.out/f'draft-{draft}-telemetry.json').write_text(json.dumps(samples),encoding='utf-8')
            save()
print('DONE',flush=True)
