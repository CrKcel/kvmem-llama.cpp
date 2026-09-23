"""Run sequential, warmed-up Bonsai CUDA kernel comparisons on an idle GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--optimized', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--gpu', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--repeats', type=int, default=4)
p.add_argument('--port', type=int, default=18345)
a = p.parse_args()
if a.repeats < 3:
    p.error('Use at least one warmup and two measured runs')
a.out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ)
env.pop('GGML_CUDA_BATCH_INVARIANT', None)
env.pop('GGML_CUDA_PDL', None)
creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
metadata = {'gpu': a.gpu, 'model': str(a.model.resolve()), 'batch_invariant': False, 'pdl': 'default enabled',
            'repeats': a.repeats, 'warmup_runs': 1, 'binaries': {}, 'cases': []}
for name, path in [('baseline', a.baseline), ('optimized', a.optimized)]:
    metadata['binaries'][name] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def save():
    (a.out / 'comparison.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

for draft in [0, 1, 2]:
    order = [('baseline', a.baseline), ('optimized', a.optimized)]
    if draft == 1:
        order.reverse()
    for name, binary in order:
        state = subprocess.check_output(['nvidia-smi', '-i', a.gpu,
            '--query-gpu=name,memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], text=True, creationflags=creation).strip()
        parts = [part.strip() for part in state.split(',')]
        if int(parts[1]) > 1024 or int(parts[2]) > 10:
            raise SystemExit(f'GPU is busy; no process was stopped: {state}')
        folder = a.out / f'{name}-draft-{draft}'
        command = [sys.executable, str(Path(__file__).with_name('bonsai-mtp-smoke.py')),
            '--binary', str(binary), '--model', str(a.model), '--gpu', a.gpu,
            '--kvmem', '--long', '--drafts', str(draft), '--repeats', str(a.repeats),
            '--port', str(a.port), '--out', str(folder)]
        case = {'backend': name, 'draft': draft, 'preflight': state, 'command': command, 'started': time.time()}
        metadata['cases'].append(case)
        save()
        print(f'START {name} draft={draft}: {state}', flush=True)
        run = subprocess.run(command, env=env, creationflags=creation)
        case['exit_code'] = run.returncode
        case['finished'] = time.time()
        if (folder / 'results.json').exists():
            result = json.loads((folder / 'results.json').read_text(encoding='utf-8'))[str(draft)]
            case['checks'] = result['checks']
            if 'prose_runs' in result:
                runs = result['prose_runs'][1:]
                rates = [r['timings']['predicted_per_second'] for r in runs]
                case['decode_tps'] = {'median': statistics.median(rates), 'min': min(rates), 'max': max(rates), 'runs': rates}
                case['completion_tokens'] = [r['usage']['completion_tokens'] for r in runs]
                case['prose_sha256'] = [hashlib.sha256(r['choices'][0]['message']['content'].encode()).hexdigest() for r in runs]
            case['peak_device_mib'] = result.get('peak_device_mib')
            if 'long' in result:
                case['archive_prefill'] = result['long'][0].get('timings')
        save()
        print(json.dumps(case, ensure_ascii=False), flush=True)
        if run.returncode:
            raise SystemExit(run.returncode)
print('COMPARISON PASS', flush=True)
