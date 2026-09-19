"""Prompt-file lifetime regression tests; no model or GPU is required."""
import errno
import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/multi_needle.py'
spec = importlib.util.spec_from_file_location('multi_needle_under_test', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PromptFileTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='needle-test-')
        self.addCleanup(self.directory.cleanup)
        self.paths = []
        self.named_file = tempfile.NamedTemporaryFile
        self.real_run = subprocess.run
        for target, kwargs in (
            ('apply_gpu', {'side_effect': lambda env, which: env}),
            ('require_device', {'return_value': None}),
        ):
            mock = patch.object(module.gpu_env, target, **kwargs)
            mock.start()
            self.addCleanup(mock.stop)

    def temporary_file(self, *args, **kwargs):
        file = self.named_file(*args, **kwargs, dir=self.directory.name)
        self.paths.append(Path(file.name))
        return file

    def call(self, prompt='test prompt'):
        return module.run_cli(Path('unused-cli'), Path('unused-model'), [], prompt, 1)

    def assert_clean(self):
        self.assertTrue(self.paths)
        self.assertTrue(all(not path.exists() for path in self.paths))

    def test_large_prompt_exact_bytes_and_cleanup(self):
        prompt = 'line1\r\n中文\n' * 20000
        def run(cmd, **kwargs):
            self.assertNotIn(prompt, cmd)
            path = cmd[cmd.index('-f') + 1]
            # A real child reopens the closed file, including on Windows.
            code = ('import hashlib,sys; from pathlib import Path; '
                    'print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())')
            return self.real_run([sys.executable, '-c', code, path], **kwargs)
        with patch.object(module.tempfile, 'NamedTemporaryFile', self.temporary_file), \
             patch.object(module.subprocess, 'run', run):
            out, _ = self.call(prompt)
        self.assertEqual(out.strip(), hashlib.sha256(prompt.encode('utf-8')).hexdigest())
        self.assert_clean()

    def test_write_and_close_failures_cleanup_before_launch(self):
        for stage in ('write', 'close'):
            with self.subTest(stage=stage):
                def create(*args, **kwargs):
                    file = self.temporary_file(*args, **kwargs)
                    class FailingFile:
                        name = file.name
                        def __enter__(self):
                            return self
                        def write(self, text):
                            file.write(text[:5])
                            file.flush()
                            if stage == 'write':
                                raise OSError(errno.ENOSPC, 'simulated full disk')
                        def __exit__(self, *exc):
                            file.close()
                            if stage == 'close':
                                raise OSError(errno.ENOSPC, 'simulated flush failure')
                    return FailingFile()
                with patch.object(module.tempfile, 'NamedTemporaryFile', create), \
                     patch.object(module.subprocess, 'run') as run:
                    with self.assertRaises(OSError) as error:
                        self.call()
                    self.assertEqual(error.exception.errno, errno.ENOSPC)
                    run.assert_not_called()
                self.assert_clean()

    def test_creation_failure_preserves_original_error(self):
        with patch.object(module.tempfile, 'NamedTemporaryFile', side_effect=PermissionError('denied')), \
             patch.object(module.subprocess, 'run') as run:
            with self.assertRaises(PermissionError):
                self.call()
            run.assert_not_called()

    def test_process_start_failure_cleanup(self):
        with patch.object(module.tempfile, 'NamedTemporaryFile', self.temporary_file), \
             patch.object(module.subprocess, 'run', side_effect=FileNotFoundError('missing CLI')):
            with self.assertRaises(FileNotFoundError):
                self.call()
        self.assert_clean()

    def test_failed_process_cleanup_and_exit_code(self):
        with patch.object(module.tempfile, 'NamedTemporaryFile', self.temporary_file), \
             patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 7, '', 'failed')), \
             patch.object(module.sys, 'stderr', io.StringIO()):
            with self.assertRaisesRegex(SystemExit, 'rc=7'):
                self.call()
        self.assert_clean()


if __name__ == '__main__':
    unittest.main()
