#!/usr/bin/env python3
"""Shared Linux launcher for the IQ3/IQ4 recipes. Uses only the standard library."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def choose_gpu(env):
    if 'CUDA_VISIBLE_DEVICES' in env:
        if not env['CUDA_VISIBLE_DEVICES'].strip() or env['CUDA_VISIBLE_DEVICES'].strip() == '-1':
            raise ValueError('CUDA_VISIBLE_DEVICES disables GPUs; select a GPU to run this recipe')
        # Explicit selection (indices, UUIDs or lists) takes precedence unchanged.
        return env['CUDA_VISIBLE_DEVICES']
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    devices = [tuple(s.strip() for s in line.split(',', 1))
               for line in result.stdout.splitlines() if ',' in line]
    preferred = [uuid for uuid, name in devices if 'RTX 5060 Ti' in name]
    if len(preferred) == 1:
        return preferred[0]
    if len(devices) == 1:
        return devices[0][0]
    choices = '\n'.join(f'  {uuid}: {name}' for uuid, name in devices)
    raise ValueError('Cannot choose a GPU automatically. Set CUDA_VISIBLE_DEVICES to an index or UUID.\n' + choices)


def library_path(binary, env):
    # Honor caller settings and use the toolkit recorded by this build, without
    # embedding the developer's CUDA installation path in a portable launcher.
    directories = [str(binary.parent)]
    directories += [p for p in env.get('LD_LIBRARY_PATH', '').split(':') if p]
    roots = [Path(env[k]) for k in ('CUDA_HOME', 'CUDA_PATH') if env.get(k)]
    cache = binary.parent.parent / 'CMakeCache.txt'
    if cache.is_file():
        for key, value in re.findall(r'^(CMAKE_CUDA_COMPILER|CUDAToolkit_NVCC_EXECUTABLE):[^=]+=(.+)$',
                                     cache.read_text(), re.M):
            compiler = Path(value)
            if compiler.is_file():
                roots.append(compiler.parent.parent)
    nvcc = shutil.which('nvcc')
    if nvcc:
        roots.append(Path(nvcc).resolve().parent.parent)
    for root in roots:
        for suffix in ('lib64', 'lib', 'targets/x86_64-linux/lib'):
            directory = root / suffix
            if directory.is_dir():
                directories.append(str(directory))
    return ':'.join(dict.fromkeys(directories))


def process_info(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, start=stat[19], uid=proc.stat().st_uid,
                    exe=os.readlink(proc / 'exe'),
                    argv=[os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def owned(info, binary):
    return bool(info and info['uid'] == os.getuid() and
                info['exe'].removesuffix(' (deleted)') == str(binary))


def same_process(info):
    now = process_info(info['pid'])
    return bool(now and now['start'] == info['start'] and now['exe'] == info['exe'])


def stop_owned(info, binary):
    # A pidfd ties signals to the inspected process, even if a PID is reused.
    if not owned(info, binary):
        raise ValueError('Refusing to stop a process outside this project')
    try:
        fd = os.pidfd_open(info['pid'])
    except ProcessLookupError:
        return
    try:
        if not same_process(info):
            return
        print(f"stopping pid={info['pid']}", flush=True)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        if not select.select([fd], [], [], 10)[0]:
            signal.pidfd_send_signal(fd, signal.SIGKILL)
            if not select.select([fd], [], [], 10)[0]:
                raise RuntimeError(f"process {info['pid']} did not exit")
    except ProcessLookupError:
        # The inspected process may exit before either signal is sent.
        return
    finally:
        os.close(fd)


def listener(port):
    output = subprocess.run(['ss', '-H', '-ltnp', f'sport = :{port}'],
                            capture_output=True, text=True, check=True).stdout
    if not output.strip():
        return False, set()
    return True, {int(pid) for pid in re.findall(r'pid=(\d+)', output)}


def healthy(port):
    try:
        with OPENER.open(f'http://127.0.0.1:{port}/health', timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def stop_server(binary, port):
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    # Use the launcher's port lock so stopping cannot race startup/restart.
    with (logs / f'kvmem_{port}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'another launcher is working on port {port}; retry after it finishes')
        busy, pids = listener(port)
        targets = {}
        if busy:
            if len(pids) != 1:
                raise ValueError(f'port {port} is busy; cannot identify one owned service, leaving it running')
            info = process_info(next(iter(pids)))
            if not owned(info, binary):
                raise ValueError(f'port {port} belongs to another service; leaving it running')
            targets[info['pid']] = info
        # An interrupted launcher may leave a server loading without a listener.
        # A PID file alone is not sufficient proof of ownership.
        pidfiles = [logs / f'{recipe}_{port}.pid' for recipe in ('iq3', 'iq4')]
        for path in pidfiles:
            try:
                info = process_info(int(path.read_text().strip()))
            except (FileNotFoundError, ValueError):
                continue
            if owned(info, binary) and '--port' in info['argv']:
                index = info['argv'].index('--port')
                if info['argv'][index + 1:index + 2] == [str(port)]:
                    targets[info['pid']] = info
        for info in targets.values():
            stop_owned(info, binary)
        if listener(port)[0]:
            raise ValueError(f'port {port} is still busy; leaving PID files for inspection')
        for path in pidfiles:
            path.unlink(missing_ok=True)
        print(f'stopped server on port {port}' if targets else f'no owned server running on port {port}')


def same_config(info, argv, env):
    if info['exe'].endswith(' (deleted)') or info['argv'] != argv:
        return False
    try:
        raw = (Path('/proc') / str(info['pid']) / 'environ').read_bytes()
        values = dict(part.split(b'=', 1) for part in raw.split(b'\0') if b'=' in part)
        return all(values.get(key.encode()) == env[key].encode()
                   for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'))
    except (FileNotFoundError, PermissionError):
        return False


def launch(args, binary, argv, env, port):
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    pidfile = logs / f'{args.recipe}_{port}.pid'
    # Serialize IQ3/IQ4 decisions on this port, including startup and restart.
    with (logs / f'kvmem_{port}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'another launcher is working on port {port}; retry after it finishes')
        busy, pids = listener(port)
        if busy:
            if len(pids) != 1:
                raise ValueError(f'port {port} is busy; cannot identify one owned service, leaving it running')
            info = process_info(next(iter(pids)))
            if not owned(info, binary):
                raise ValueError(f'port {port} belongs to another service; leaving it running, even with --restart')
            if not args.restart:
                if same_config(info, argv, env) and healthy(port):
                    pidfile.write_text(str(info['pid']) + '\n')
                    print(f"already up recipe={args.recipe} pid={info['pid']} http://127.0.0.1:{port}/health")
                    return
                raise ValueError(f'port {port} has a different configuration or an unhealthy service; use --restart to switch')
            stop_owned(info, binary)
        # Also catch a previous launcher interrupted while its model was loading.
        # Stale PID files for unrelated processes are ignored, never signalled.
        for name in ('iq3', 'iq4'):
            previous = logs / f'{name}_{port}.pid'
            if previous.is_file():
                try:
                    info = process_info(int(previous.read_text().strip()))
                except ValueError:
                    info = None
                if owned(info, binary) and '--port' in info['argv']:
                    index = info['argv'].index('--port')
                    if info['argv'][index + 1:index + 2] == [str(port)]:
                        if not args.restart:
                            raise ValueError(f'owned server pid={info["pid"]} is still starting; wait or use --restart')
                        stop_owned(info, binary)
        if listener(port)[0]:
            raise ValueError(f'port {port} is still busy')
        port_suffix = '' if port == 18200 else f'_{port}'
        stdout = logs / f'opencode_kvmem_{args.recipe}_256k{port_suffix}.stdout.log'
        stderr = logs / f'opencode_kvmem_{args.recipe}_256k{port_suffix}.stderr.log'
        for path in (stdout, stderr):
            if path.exists():
                path.rename(path.with_name(path.name + f'.{time.time_ns()}'))
        with stdout.open('wb') as out, stderr.open('wb') as err:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=out, stderr=err, start_new_session=True)
        try:
            pidfile.write_text(str(proc.pid) + '\n')
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited with status {proc.returncode}')
                busy, pids = listener(port)
                if busy and pids == {proc.pid} and healthy(port):
                    print(f'started recipe={args.recipe} pid={proc.pid} gpu={env["CUDA_VISIBLE_DEVICES"]} '
                          f'vision={env["KVMEM_VISION_DEVICE"]} model={argv[2]} '
                          f'http://127.0.0.1:{port}/health')
                    return
                time.sleep(.5)
            raise RuntimeError('server startup timed out')
        except BaseException:
            # Only this child is cleaned up on startup failure or interruption.
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            pidfile.unlink(missing_ok=True)
            print(stderr.read_text(errors='replace')[-4000:], file=sys.stderr)
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, epilog=
        'Overrides: MODEL, MMPROJ, MMPROJ_DEVICE, CUDA_VISIBLE_DEVICES, PORT, BUILD_DIR, '
        'IMAGE_MAX_TOKENS, SPEC_KV_DTYPE, KVMEM_QUERY_REPLAY, KVMEM_QUERY_POLICY, CUDA_HOME, LD_LIBRARY_PATH.')
    ap.add_argument('--recipe', choices=('iq3', 'iq4'), required=True)
    ap.add_argument('--default-model', required=True)
    ap.add_argument('--default-mmproj', required=True)
    ap.add_argument('--default-vision-device', choices=('cpu', 'gpu'), required=True)
    ap.add_argument('--kv', required=True)
    ap.add_argument('--budget', type=int, required=True)
    ap.add_argument('--reserve', type=int, required=True)
    ap.add_argument('--explicit-sampling', action='store_true')
    action = ap.add_mutually_exclusive_group()
    action.add_argument('--restart', action='store_true')
    action.add_argument('--stop', action='store_true', help="stop this project's server on PORT, without loading models")
    action.add_argument('--dry-run', action='store_true', help='print resolved argv/environment without starting or stopping anything')
    ap.add_argument('--startup-timeout', type=float, default=180)
    args = ap.parse_args()
    env = os.environ.copy()
    binary = (Path(env.get('BUILD_DIR', str(ROOT / 'build'))) / 'bin/llama-kvmem-server').resolve()
    port = int(env.get('PORT', '18200'))
    if not 1 <= port <= 65535:
        raise ValueError('invalid PORT')
    if args.stop:
        stop_server(binary, port)
        return
    model = Path(env.get('MODEL', args.default_model)).resolve()
    mmproj = Path(env.get('MMPROJ', args.default_mmproj)).resolve()
    for path in (binary, model, mmproj):
        if not path.is_file():
            raise ValueError(f'file not found: {path}')
    if not os.access(binary, os.X_OK):
        raise ValueError(f'binary is not executable: {binary}')
    vision = env.get('MMPROJ_DEVICE', args.default_vision_device)
    replay = env.get('KVMEM_QUERY_REPLAY', 'auto')
    policy = env.get('KVMEM_QUERY_POLICY', 'user')
    draft_kv = env.get('SPEC_KV_DTYPE', 'f16')
    if vision not in ('cpu', 'gpu') or replay not in ('auto', 'legacy') or policy not in ('user', 'legacy'):
        raise ValueError('invalid MMPROJ_DEVICE, KVMEM_QUERY_REPLAY or KVMEM_QUERY_POLICY')
    if draft_kv not in ('f16', 'q8_0', 'q5_0', 'q4_0'):
        raise ValueError('SPEC_KV_DTYPE must be f16, q8_0, q5_0 or q4_0')
    image_tokens = int(env.get('IMAGE_MAX_TOKENS', '512'))
    if not 1 <= port <= 65535 or image_tokens <= 0 or not 0 < args.startup_timeout <= 3600:
        raise ValueError('invalid PORT, IMAGE_MAX_TOKENS or startup timeout')
    env['CUDA_VISIBLE_DEVICES'] = choose_gpu(env)
    env.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    env['LD_LIBRARY_PATH'] = library_path(binary, env)
    env['KVMEM_VISION_DEVICE'] = vision
    argv = [str(binary), '-m', str(model), '--mmproj', str(mmproj),
            '--mmproj-offload' if vision == 'gpu' else '--no-mmproj-offload',
            '--image-max-tokens', str(image_tokens), '--host', '127.0.0.1', '--port', str(port),
            '-c', '262144', '-n', str(args.reserve), '-b', '512', '-ngl', '99',
            '--kvmem', '--kvmem-method', 'retrieval',
            '--kvmem-query-replay', replay, '--kvmem-query-policy', policy,
            '--kvmem-budget', str(args.budget), '--kvmem-gen-reserve', str(args.reserve),
            '--kvmem-block-tokens', '128', '--kv-dtype', args.kv,
            '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2', '--spec-kv-dtype', draft_kv,
            '--enable-thinking', '--reasoning-budget', '4096']
    if args.explicit_sampling:
        argv += ['--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.0',
                 '--presence-penalty', '0.0', '--frequency-penalty', '0.0', '--repeat-penalty', '1.0']
    if args.dry_run:
        print(json.dumps(dict(argv=argv, environment={k: env[k] for k in
                         ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'LD_LIBRARY_PATH')}), indent=2))
        return
    # Resolve shared-library failures before stopping an existing working service.
    subprocess.run([str(binary), '--help'], env=env, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, check=True, timeout=30)
    launch(args, binary, argv, env, port)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'launch failed: {exc}', file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(os.fsdecode(exc.stderr), file=sys.stderr)
        sys.exit(1)
