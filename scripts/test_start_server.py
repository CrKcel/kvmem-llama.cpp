#!/usr/bin/env python3
"""Launcher regression tests with a tiny HTTP executable; no models or GPU work."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('launcher', SCRIPTS / 'start-server.py')
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)
STUB = r'''
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--help")) return getenv("TEST_BAD_HELP") ? 2 : 0;
    if (getenv("TEST_EXIT")) return 3;
    if (getenv("TEST_DELAY")) sleep(2);
    int port = 0;
    for (int i=1; i+1<argc; i++) if (!strcmp(argv[i], "--port")) port=atoi(argv[i+1]);
    int fd=socket(AF_INET, SOCK_STREAM, 0), yes=1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in address={.sin_family=AF_INET, .sin_port=htons(port)};
    address.sin_addr.s_addr=htonl(INADDR_LOOPBACK);
    if (bind(fd, (void *)&address, sizeof(address)) || listen(fd, 8)) return 4;
    fprintf(stderr, "test HTTP server listening\n"); fflush(stderr);
    for (;;) {
        int c=accept(fd, NULL, NULL);
        if (c<0) continue;
        char request[4096]; read(c, request, sizeof(request));
        const char *reply="HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        send(c, reply, strlen(reply), MSG_NOSIGNAL); close(c);
    }
}
'''


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='kvmem launcher ')
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'build/bin').mkdir(parents=True)
        for name in ('start-iq3.sh', 'start-iq4.sh', 'stop-iq3.sh', 'start-server.py'):
            shutil.copy2(SCRIPTS / name, self.root / 'scripts' / name)
        source = self.root / 'server.c'
        source.write_text(STUB)
        self.binary = self.root / 'build/bin/llama-kvmem-server'
        subprocess.run(['cc', str(source), '-o', str(self.binary)], check=True)
        self.model = self.root / 'model $(touch BAD).gguf'
        self.mmproj = self.root / 'projector.gguf'
        self.model.touch()
        self.mmproj.touch()
        gpu = self.root / 'nvidia-smi'
        gpu.write_text('#!/usr/bin/env python3\nimport os\nprint(os.environ["TEST_GPUS"])\n')
        gpu.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.env = os.environ.copy()
        for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'MODEL', 'MMPROJ', 'MMPROJ_DEVICE',
                    'BUILD_DIR', 'SPEC_KV_DTYPE', 'SPEC_DRAFT_N_MAX', 'KVMEM_MTP_STATE',
                    'KVMEM_QUERY_REPLAY', 'KVMEM_QUERY_POLICY',
                    'IMAGE_MAX_TOKENS', 'TEST_EXIT', 'TEST_DELAY', 'TEST_BAD_HELP'):
            self.env.pop(key, None)
        self.env.update(MODEL=str(self.model), MMPROJ=str(self.mmproj), PORT=str(self.port),
                        PATH=str(self.root)+':'+self.env['PATH'],
                        TEST_GPUS='GPU-small, NVIDIA GeForce RTX 5050 Laptop GPU\nGPU-big, NVIDIA GeForce RTX 5060 Ti')
        self.foreign = []
        self.pids = set()

    def tearDown(self):
        for path in (self.root / 'logs').glob('*.pid'):
            self.pids.add(int(path.read_text()))
        for pid in self.pids:
            info = LAUNCHER.process_info(pid)
            if LAUNCHER.owned(info, self.binary):
                LAUNCHER.stop_owned(info, self.binary)
        for proc in self.foreign:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        self.tmp.cleanup()

    def run_recipe(self, recipe='iq3', *args, overrides=None, success=True):
        env = self.env | (overrides or {})
        result = subprocess.run(['bash', str(self.root / 'scripts' / f'start-{recipe}.sh'), *args],
                                env=env, capture_output=True, text=True, timeout=20)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def pid(self, recipe='iq3'):
        pid = int((self.root / 'logs' / f'{recipe}_{self.port}.pid').read_text())
        self.pids.add(pid)
        return pid

    def run_stop(self, overrides=None, success=True):
        result = subprocess.run(['bash', str(self.root / 'scripts/stop-iq3.sh')],
                                env=self.env | (overrides or {}), capture_output=True,
                                text=True, timeout=25)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    def test_stop_owned_without_model_or_gpu_preflight(self):
        self.run_recipe()
        pid = self.pid()
        self.model.unlink()
        self.mmproj.unlink()
        self.run_stop(overrides={'CUDA_VISIBLE_DEVICES': '-1', 'TEST_BAD_HELP': '1'})
        self.assertIsNone(LAUNCHER.process_info(pid))
        self.assertFalse(LAUNCHER.listener(self.port)[0])
        self.assertFalse((self.root / 'logs' / f'iq3_{self.port}.pid').exists())
        self.run_stop()

    def test_stop_refuses_foreign_listener_and_ignores_stale_pid(self):
        other = self.root / 'unrelated-server'
        shutil.copy2(self.binary, other)
        proc = subprocess.Popen([str(other), '--port', str(self.port)], stderr=subprocess.DEVNULL)
        self.foreign.append(proc)
        for _ in range(50):
            if LAUNCHER.healthy(self.port):
                break
            time.sleep(.02)
        self.assertTrue(LAUNCHER.healthy(self.port))
        logs = self.root / 'logs'
        logs.mkdir()
        pidfile = logs / f'iq3_{self.port}.pid'
        pidfile.write_text(str(proc.pid))
        self.assertIn('another service', self.run_stop(success=False).stderr)
        self.assertIsNone(proc.poll())
        self.assertTrue(pidfile.exists())
        proc.terminate()
        proc.wait(timeout=5)
        sleeper = subprocess.Popen(['sleep', '30'])
        self.foreign.append(sleeper)
        pidfile.write_text(str(sleeper.pid))
        self.run_stop()
        self.assertIsNone(sleeper.poll())
        self.assertFalse(pidfile.exists())
        pidfile.write_text('invalid PID')
        self.run_stop()
        self.assertFalse(pidfile.exists())

    def test_stop_respects_build_dir(self):
        self.run_recipe()
        pid = self.pid()
        self.run_stop(overrides={'BUILD_DIR': str(self.root / 'another-build')}, success=False)
        self.assertIsNotNone(LAUNCHER.process_info(pid))
        self.run_stop(overrides={'BUILD_DIR': str(self.root / 'build')})
        self.assertIsNone(LAUNCHER.process_info(pid))

    def test_stop_ignores_pid_for_another_port(self):
        self.run_recipe()
        pid = self.pid()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            other_port = sock.getsockname()[1]
        stale = self.root / 'logs' / f'iq3_{other_port}.pid'
        stale.write_text(str(pid))
        self.run_stop(overrides={'PORT': str(other_port)})
        self.assertIsNotNone(LAUNCHER.process_info(pid))
        self.assertFalse(stale.exists())

    def test_stop_loading_server_and_respects_launcher_lock(self):
        runner = subprocess.Popen(['bash', str(self.root / 'scripts/start-iq3.sh')],
                                  env=self.env | {'TEST_DELAY': '1'}, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        self.foreign.append(runner)
        pidfile = self.root / 'logs' / f'iq3_{self.port}.pid'
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(.02)
        child = self.pid()
        self.assertIn('another launcher', self.run_stop(success=False).stderr)
        self.assertIsNotNone(LAUNCHER.process_info(child))
        runner.terminate()
        runner.wait(timeout=5)
        self.assertFalse(LAUNCHER.listener(self.port)[0])
        self.run_stop()
        self.assertIsNone(LAUNCHER.process_info(child))
        self.assertFalse(pidfile.exists())

    def test_stop_pid_reuse_and_exit_races(self):
        info = dict(pid=123, start='1', uid=os.getuid(), exe=str(self.binary))
        with mock.patch.object(LAUNCHER.os, 'pidfd_open', return_value=99), \
                mock.patch.object(LAUNCHER.os, 'close') as close, \
                mock.patch.object(LAUNCHER, 'process_info', return_value=info | {'start': '2'}), \
                mock.patch.object(LAUNCHER.signal, 'pidfd_send_signal') as send:
            LAUNCHER.stop_owned(info, self.binary)
            send.assert_not_called()
            close.assert_called_once_with(99)
        with mock.patch.object(LAUNCHER.os, 'pidfd_open', return_value=99), \
                mock.patch.object(LAUNCHER.os, 'close') as close, \
                mock.patch.object(LAUNCHER, 'process_info', return_value=info), \
                mock.patch.object(LAUNCHER.signal, 'pidfd_send_signal', side_effect=ProcessLookupError):
            LAUNCHER.stop_owned(info, self.binary)
            close.assert_called_once_with(99)

    def test_selection_overrides_and_quoting(self):
        data = json.loads(self.run_recipe('iq3', '--dry-run').stdout)
        self.assertEqual(data['environment']['CUDA_VISIBLE_DEVICES'], 'GPU-big')
        self.assertIn(str(self.model), data['argv'])
        self.assertIn('--mmproj-offload', data['argv'])
        self.assertNotIn('--spec-draft-n-max', data['argv'])
        self.assertNotIn('--kvmem-mtp-state', data['argv'])
        self.assertFalse((self.root / 'BAD').exists())
        self.assertFalse((self.root / 'logs').exists())
        data = json.loads(self.run_recipe('iq4', '--dry-run', overrides={
            'CUDA_VISIBLE_DEVICES': '3,1', 'MMPROJ_DEVICE': 'gpu', 'SPEC_KV_DTYPE': 'q4_0',
            'SPEC_DRAFT_N_MAX': '4', 'KVMEM_MTP_STATE': 'snapshots'}).stdout)
        self.assertEqual(data['environment']['CUDA_VISIBLE_DEVICES'], '3,1')
        self.assertIn('--mmproj-offload', data['argv'])
        self.assertEqual(data['argv'][data['argv'].index('--spec-kv-dtype')+1], 'q4_0')
        self.assertEqual(data['argv'][data['argv'].index('--spec-draft-n-max')+1], '4')
        self.assertEqual(data['argv'][data['argv'].index('--kvmem-mtp-state')+1], 'snapshots')
        self.run_recipe('iq3', '--dry-run', overrides={'SPEC_DRAFT_N_MAX': '6'}, success=False)
        self.run_recipe('iq3', '--dry-run', overrides={'KVMEM_MTP_STATE': 'invalid'}, success=False)
        data = json.loads(self.run_recipe('iq3', '--dry-run', overrides={'TEST_GPUS': 'GPU-only, Any NVIDIA'}).stdout)
        self.assertEqual(data['environment']['CUDA_VISIBLE_DEVICES'], 'GPU-only')
        self.run_recipe('iq3', '--dry-run', overrides={'TEST_GPUS': 'GPU-a, GPU A\nGPU-b, GPU B'}, success=False)
        self.run_recipe('iq3', '--dry-run', overrides={'CUDA_VISIBLE_DEVICES': ''}, success=False)

    def test_recipes_use_server_defaults(self):
        for recipe, kv, budget, reserve in [('iq3', 'q8_0', '36864', '16384'),
                                             ('iq4', 'q5_0', '32768', '12288')]:
            argv = json.loads(self.run_recipe(recipe, '--dry-run').stdout)['argv']
            for flag in ('--temp', '--top-p', '--top-k', '--min-p', '--presence-penalty',
                         '--frequency-penalty', '--repeat-penalty', '--kvmem', '--kvmem-method',
                         '--kvmem-block-tokens', '--kvmem-query-policy', '--kvmem-query-replay',
                         '--spec-kv-dtype', '--spec-draft-n-max', '--kvmem-mtp-state', '-b', '-ngl'):
                self.assertNotIn(flag, argv)
            for flag, value in [('--kv-dtype', kv), ('--kvmem-budget', budget),
                                ('--kvmem-gen-reserve', reserve), ('-n', reserve)]:
                self.assertEqual(argv[argv.index(flag) + 1], value)
        argv = json.loads(self.run_recipe('iq3', '--dry-run', overrides={
            'KVMEM_QUERY_REPLAY': 'legacy', 'KVMEM_QUERY_POLICY': 'legacy'}).stdout)['argv']
        for flag in ('--kvmem-query-replay', '--kvmem-query-policy'):
            self.assertEqual(argv[argv.index(flag) + 1], 'legacy')

    def test_reuse_and_switch(self):
        self.run_recipe()
        first = self.pid()
        self.assertIn('already up', self.run_recipe().stdout)
        self.assertEqual(self.pid(), first)
        result = self.run_recipe('iq4', success=False)
        self.assertIn('--restart', result.stderr)
        self.assertIsNotNone(LAUNCHER.process_info(first))
        self.run_recipe('iq4', '--restart')
        self.assertIsNone(LAUNCHER.process_info(first))
        self.assertNotEqual(self.pid('iq4'), first)
        self.assertIn('already up', self.run_recipe('iq4').stdout)
        result = self.run_recipe('iq4', overrides={'CUDA_VISIBLE_DEVICES': 'GPU-other'}, success=False)
        self.assertIn('different configuration', result.stderr)

    def test_preflight_failure_preserves_server(self):
        self.run_recipe()
        pid = self.pid()
        self.run_recipe('iq4', '--restart', overrides={'TEST_BAD_HELP': '1'}, success=False)
        self.assertIsNotNone(LAUNCHER.process_info(pid))
        self.run_recipe('iq4', '--restart', overrides={'MODEL': '/missing/file'}, success=False)
        self.assertIsNotNone(LAUNCHER.process_info(pid))

    def test_unrelated_listener_and_stale_pid_are_safe(self):
        other = self.root / 'unrelated-server'
        shutil.copy2(self.binary, other)
        proc = subprocess.Popen([str(other), '--port', str(self.port)], stderr=subprocess.DEVNULL)
        self.foreign.append(proc)
        for _ in range(50):
            if LAUNCHER.healthy(self.port):
                break
            time.sleep(.02)
        self.assertTrue(LAUNCHER.healthy(self.port))
        logs = self.root / 'logs'
        logs.mkdir()
        (logs / f'iq3_{self.port}.pid').write_text(str(proc.pid))
        result = self.run_recipe('iq3', '--restart', success=False)
        self.assertIn('another service', result.stderr)
        self.assertIsNone(proc.poll())
        proc.terminate()
        proc.wait(timeout=5)
        sleeper = subprocess.Popen(['sleep', '30'])
        self.foreign.append(sleeper)
        (logs / f'iq3_{self.port}.pid').write_text(str(sleeper.pid))
        self.run_recipe('iq3', '--restart')
        self.pid()
        self.assertIsNone(sleeper.poll())

    def test_startup_failure_cleanup(self):
        self.run_recipe('iq3', overrides={'TEST_EXIT': '1'}, success=False)
        self.assertFalse(LAUNCHER.listener(self.port)[0])
        self.assertFalse((self.root / 'logs' / f'iq3_{self.port}.pid').exists())
        self.run_recipe('iq3', '--startup-timeout', '.1', overrides={'TEST_DELAY': '1'}, success=False)
        self.assertFalse(LAUNCHER.listener(self.port)[0])
        self.assertFalse((self.root / 'logs' / f'iq3_{self.port}.pid').exists())

    def test_interrupted_launcher_recovery(self):
        runner = subprocess.Popen(['bash', str(self.root / 'scripts/start-iq3.sh')],
                                  env=self.env | {'TEST_DELAY': '1'}, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        self.foreign.append(runner)
        pidfile = self.root / 'logs' / f'iq3_{self.port}.pid'
        for _ in range(100):
            if pidfile.exists():
                break
            time.sleep(.02)
        child = self.pid()
        self.assertIn('another launcher', self.run_recipe(success=False).stderr)
        runner.terminate()
        runner.wait(timeout=5)
        self.run_recipe('iq4', '--restart')
        self.pid('iq4')
        self.assertIsNone(LAUNCHER.process_info(child))

    def test_template_options(self):
        template = self.root / 'custom $(touch BAD).jinja'
        template.write_text('{{ messages }}')
        for recipe in ('iq3', 'iq4'):
            data = json.loads(self.run_recipe(recipe, '--dry-run', '--chat-template-file', str(template),
                '--reasoning-effort', 'low', '--chat-template-kwargs', '{"flag":true,"count":2}', '--jinja').stdout)
            argv = data['argv']
            self.assertEqual(argv[argv.index('--chat-template-file') + 1], str(template))
            self.assertEqual(argv[argv.index('--reasoning-effort') + 1], 'low')
            self.assertEqual(json.loads(argv[argv.index('--chat-template-kwargs') + 1]), {'flag': True, 'count': 2})
            self.assertIn('--jinja', argv)
        self.assertFalse((self.root / 'BAD').exists())
        self.run_recipe('iq3', '--dry-run', '--chat-template-file', '/missing/template', success=False)
        self.run_recipe('iq3', '--dry-run', '--chat-template-kwargs', '[]', success=False)
        self.run_recipe('iq3', '--dry-run', '--chat-template', 'chatml', '--chat-template-file', str(template), success=False)

    def test_cuda_library_discovery(self):
        cuda = self.root / 'custom CUDA'
        (cuda / 'bin').mkdir(parents=True)
        (cuda / 'lib').mkdir()
        (cuda / 'bin/nvcc').touch()
        (self.root / 'build/CMakeCache.txt').write_text(f'CMAKE_CUDA_COMPILER:FILEPATH={cuda}/bin/nvcc\n')
        data = json.loads(self.run_recipe('iq3', '--dry-run', overrides={'LD_LIBRARY_PATH': '/caller/libs'}).stdout)
        paths = data['environment']['LD_LIBRARY_PATH'].split(':')
        self.assertEqual(paths[:2], [str(self.binary.parent), '/caller/libs'])
        self.assertIn(str(cuda / 'lib'), paths)

        bundled = self.binary.parent.parent / 'lib'
        bundled.mkdir()
        data = json.loads(self.run_recipe('iq3', '--dry-run', overrides={'LD_LIBRARY_PATH': '/caller/libs'}).stdout)
        self.assertEqual(data['environment']['LD_LIBRARY_PATH'].split(':')[:3],
                         [str(self.binary.parent), str(bundled), '/caller/libs'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
