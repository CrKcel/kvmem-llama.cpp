#!/usr/bin/env python3
"""Compare MTP F16/Q8/Q5 on the default IQ4 server; restore it after the experiment.

Requires the default server on port 18200 and exclusive use of the RTX 5060 Ti.
Each trial starts a fresh server on port 18201 and sends the same cold request.
"""
import argparse
import csv
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c'
URL = 'http://127.0.0.1:18201'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Memory(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint), ('total', ctypes.c_ulonglong),
                ('reserved', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]


class Sampler(threading.Thread):
    def __init__(self, nvml, device, folder):
        super().__init__(daemon=True)
        self.nvml, self.device, self.folder = nvml, device, folder
        self.phase = 'loading'
        self.stop_event = threading.Event()
        self.rows = []
        self.errors = []

    def run(self):
        start = time.monotonic()
        offset = 0
        with (self.folder / 'vram.csv').open('w') as fh:
            writer = csv.writer(fh)
            writer.writerow(['elapsed_s', 'phase', 'used_mib', 'free_mib', 'reserved_mib',
                             'temperature_c', 'sm_clock_mhz', 'power_w'])
            while not self.stop_event.is_set():
                if self.phase == 'prefill':
                    with (self.folder / 'server.stderr.log').open() as log:
                        log.seek(offset)
                        tail = log.read()
                        offset = log.tell()
                    if 'KVMEM_CHAT_PREFILL ' in tail:
                        self.phase = 'decode'
                m = Memory()
                m.version = ctypes.sizeof(Memory) | (2 << 24)
                status = self.nvml.nvmlDeviceGetMemoryInfo_v2(self.device, ctypes.byref(m))
                if status:
                    self.errors.append(status)
                else:
                    extra = []
                    for fn, args, scale in (
                        (self.nvml.nvmlDeviceGetTemperature, [self.device, 0], 1),
                        (self.nvml.nvmlDeviceGetClockInfo, [self.device, 1], 1),
                        (self.nvml.nvmlDeviceGetPowerUsage, [self.device], 1000),
                    ):
                        value = ctypes.c_uint()
                        extra.append(value.value / scale if fn(*args, ctypes.byref(value)) == 0 else None)
                    row = [time.monotonic() - start, self.phase, m.used / 2**20,
                           m.free / 2**20, m.reserved / 2**20, *extra]
                    self.rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                self.stop_event.wait(0.05)

    def finish(self):
        self.stop_event.set()
        self.join()
        phases = {}
        for phase in sorted({r[1] for r in self.rows}):
            rows = [r for r in self.rows if r[1] == phase]
            phases[phase] = {'peak_mib': max(r[2] for r in rows), 'last_mib': rows[-1][2],
                             'min_free_mib': min(r[3] for r in rows), 'samples': len(rows)}
            for i, name in ((5, 'temperature_c'), (6, 'sm_clock_mhz'), (7, 'power_w')):
                values = [r[i] for r in rows if r[i] is not None]
                if values:
                    phases[phase][name + '_mean'] = statistics.fmean(values)
        return {'peak_vram_mib': max(r[2] for r in self.rows), 'phase_metrics': phases,
                'sample_count': len(self.rows), 'sampling_errors': self.errors,
                'max_sample_gap_ms': max((b[0] - a[0] for a, b in zip(self.rows, self.rows[1:])), default=0) * 1000}


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def trial(folder, dtype, argv, request_bytes, env, nvml, device):
    folder.mkdir()
    result = {'mtp_kv_type': dtype, 'argv': argv, 'request_sha256': hashlib.sha256(request_bytes).hexdigest()}
    sampler = Sampler(nvml, device, folder)
    proc = None
    sampler.start()
    try:
        with (folder / 'server.stdout.log').open('w') as out, (folder / 'server.stderr.log').open('w') as err:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=err,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        result['pid'] = proc.pid
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            assert proc.poll() is None, 'server exited during startup'
            try:
                with OPENER.open(URL + '/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError('server health timeout')
        startup = (folder / 'server.stderr.log').read_text()
        pool = next(s for s in startup.splitlines() if 'KVMEM_TRACE mtp_pool' in s)
        assert f'type_k={dtype} type_v={dtype}' in pool and 'cells=43904' in pool, pool
        target = next(s for s in startup.splitlines() if 'slot-pool cells=' in s)
        assert 'type_k=q5_0 type_v=q5_0' in target and 'budget=32000 gen_reserve=11904' in target, target
        result['mtp_pool_log'] = pool
        result['target_pool_log'] = target
        sampler.phase = 'idle_ready'
        time.sleep(1)
        sampler.phase = 'prefill'
        print('START', folder.name, 'pid', proc.pid, flush=True)
        request = urllib.request.Request(URL + '/v1/chat/completions', data=request_bytes,
                                         headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        first = None
        content, reasoning = [], []
        usage, finish, done = None, None, False
        with OPENER.open(request, timeout=1800) as response, (folder / 'response.sse').open('w') as out:
            for line in response:
                line = line.decode()
                out.write(line)
                out.flush()
                if not line.startswith('data: '):
                    continue
                payload = line[6:].strip()
                if payload == '[DONE]':
                    done = True
                    break
                chunk = json.loads(payload)
                assert 'error' not in chunk, chunk
                if chunk.get('usage'):
                    usage = chunk['usage']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    c, r = delta.get('content') or '', delta.get('reasoning_content') or ''
                    if (c or r) and first is None:
                        first = time.monotonic()
                        print('FIRST_TOKEN', folder.name, 'ttft_s', round(first - start, 3), flush=True)
                    content.append(c)
                    reasoning.append(r)
                    finish = choice.get('finish_reason') or finish
        result.update(client_wall_s=time.monotonic() - start, ttft_s=None if first is None else first - start,
                      usage=usage, finish_reason=finish, done=done)
        sampler.phase = 'idle_after'
        time.sleep(1)
        code, thought = ''.join(content), ''.join(reasoning)
        (folder / 'generated.cpp').write_text(code)
        (folder / 'reasoning.txt').write_text(thought)
        result['code_sha256'] = hashlib.sha256(code.encode()).hexdigest()
        result['reasoning_sha256'] = hashlib.sha256(thought.encode()).hexdigest()
        log = (folder / 'server.stderr.log').read_text()
        pf = re.findall(r'KVMEM_CHAT_PREFILL ms=([\d.]+) n_prompt=(\d+)', log)[-1]
        gen = re.findall(r'KVMEM_GEN_WALL n=(\d+) ms=([\d.]+) toks=([\d.]+)', log)[-1]
        sp = re.findall(r'spec_stats n_gen=(\d+) n_drafted=(\d+) n_accept=(\d+) n_restore=(\d+) accept_pct=([\d.]+)', log)[-1]
        result.update(prompt_tokens=int(pf[1]), prefill_wall_ms=float(pf[0]), prefill_tps=int(pf[1]) * 1000 / float(pf[0]),
                      generated_tokens=int(gen[0]), decode_wall_ms=float(gen[1]), decode_tps=float(gen[2]),
                      drafted=int(sp[1]), accepted=int(sp[2]), accept_pct=float(sp[4]))
        result['mtp_follow'] = [s for s in log.splitlines() if 'KVMEM_TRACE mtp_follow' in s]
        assert done and usage and usage.get('prompt_cache_hit_tokens') == 0, 'incomplete response or cache hit'
        assert int(pf[1]) == 61159, 'input token count changed'
        assert all('n_no_raw=0' in s for s in result['mtp_follow']), 'MTP cold-block restore missed data'
    except Exception as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result.update(sampler.finish())
        (folder / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        if proc is not None:
            stop_server(proc)
    print('RESULT', folder.name, json.dumps({k: result[k] for k in (
        'prefill_tps', 'decode_tps', 'generated_tokens', 'accept_pct', 'peak_vram_mib')}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--request', type=Path, default=ROOT / 'logs/iq4_mtp_q5_32k12k_code_20260913_191415/request.json')
    args = ap.parse_args()
    if args.rounds < 1:
        ap.error('--rounds must be positive')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 18201))
    old_pid = int((ROOT / 'logs/iq4_18200.pid').read_text())
    old_argv = Path(f'/proc/{old_pid}/cmdline').read_bytes().decode().strip('\0').split('\0')
    assert old_argv[0] == str(ROOT / 'build/bin/llama-kvmem-server'), 'unexpected default server process'
    expected = {'--port': '18200', '--kv-dtype': 'q5_0', '--kvmem-budget': '32000',
                '--kvmem-gen-reserve': '12000', '--spec-draft-n-max': '2', '-b': '512', '-c': '262144'}
    for flag, value in expected.items():
        assert old_argv[old_argv.index(flag) + 1] == value, f'default server differs: {flag}'
    request_bytes = args.request.read_bytes()
    root = ROOT / 'logs' / ('mtp_kv_ab_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    root.mkdir()
    (root / 'request.json').write_bytes(request_bytes)
    shutil.copy2(__file__, root / 'benchmark.py')
    print('ARTIFACTS', root, flush=True)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=GPU,
               PATH='/usr/bin:/bin:/usr/lib/wsl/lib:/home/leye/kvmem_qw3/.cu13-env/bin',
               LD_LIBRARY_PATH=str(ROOT / 'build/bin') + ':/home/leye/kvmem_qw3/.cu13-env/lib')
    for key in ('KVMEM_TRACE', 'KVMEM_PERF'):
        env.pop(key, None)
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    assert nvml.nvmlInit_v2() == 0
    device = ctypes.c_void_p()
    assert nvml.nvmlDeviceGetHandleByUUID(GPU.encode(), ctypes.byref(device)) == 0
    summary = {'default_argv': old_argv, 'gpu_uuid': GPU, 'rounds': args.rounds,
               'server_sha256': hashlib.sha256(Path(old_argv[0]).read_bytes()).hexdigest(), 'trials': []}
    stopped = False
    try:
        os.kill(old_pid, signal.SIGTERM)
        stopped = True
        order = [t for r in range(args.rounds) for t in
                 (('f16', 'q8_0', 'q5_0') if r % 2 == 0 else ('q5_0', 'q8_0', 'f16'))]
        for index, dtype in enumerate(order, 1):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                m = Memory()
                m.version = ctypes.sizeof(Memory) | (2 << 24)
                assert nvml.nvmlDeviceGetMemoryInfo_v2(device, ctypes.byref(m)) == 0
                if m.used < 128 * 2**20:
                    break
                time.sleep(0.5)
            else:
                raise RuntimeError('GPU still occupied before trial')
            time.sleep(3)
            argv = old_argv.copy()
            argv[argv.index('--port') + 1] = '18201'
            while '--spec-kv-dtype' in argv:
                override = argv.index('--spec-kv-dtype')
                del argv[override:override + 2]
            argv += ['--spec-kv-dtype', dtype]
            row = trial(root / f'{index:02d}_{dtype}', dtype, argv, request_bytes, env, nvml, device)
            summary['trials'].append(row)
            (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        nvml.nvmlShutdown()
        if stopped:
            restore = subprocess.run(['bash', str(ROOT / 'scripts/start-iq4.sh')],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
            (root / 'restore.log').write_text(restore.stdout + restore.stderr)
            summary['restore_rc'] = restore.returncode
            print('RESTORE', restore.returncode, restore.stdout.strip(), flush=True)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
