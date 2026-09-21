"""Linux launcher regressions; no GPU, model loading or live service required."""
import importlib.util
from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    'launcher', Path(__file__).resolve().parents[1] / 'scripts/start-server.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class KeyFileTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.caller = self.root / 'caller'
        self.caller.mkdir()
        binary = self.root / 'build/bin/llama-kvmem-server'
        binary.parent.mkdir(parents=True)
        binary.write_text('#!/bin/sh\nexit 0\n')
        binary.chmod(0o755)
        self.model = self.root / 'model.gguf'
        self.model.touch()
        self.key = self.caller / 'keys with spaces.txt'
        self.key.write_text('launcher-test-key\n')
        previous = Path.cwd()
        os.chdir(self.caller)
        self.addCleanup(os.chdir, previous)
        stack = ExitStack()
        self.addCleanup(stack.close)
        env = {'BUILD_DIR': str(binary.parent.parent), 'CUDA_VISIBLE_DEVICES': '0'}
        stack.enter_context(mock.patch.dict(os.environ, env, clear=True))
        stack.enter_context(mock.patch.object(launcher, 'ROOT', self.root))
        stack.enter_context(mock.patch.object(launcher, 'library_path', return_value=''))
        self.run = stack.enter_context(mock.patch.object(launcher.subprocess, 'run'))
        self.launch = stack.enter_context(mock.patch.object(launcher, 'launch'))

    def invoke(self, path):
        argv = ['start-server.py', '--recipe', 'iq3', '--default-model', str(self.model),
                '--default-mmproj', str(self.model), '--default-vision-device', 'cpu',
                '--kv', 'q8_0', '--budget', '36864', '--reserve', '16384',
                '--api-key-file', str(path), '--restart']
        with mock.patch('sys.argv', argv):
            launcher.main()

    def test_relative_key_survives_child_working_directory(self):
        self.invoke(self.key.name)
        argv = self.launch.call_args.args[2]
        path = Path(argv[argv.index('--api-key-file') + 1])
        os.chdir(self.root)
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.read_text(), 'launcher-test-key\n')

    def test_absolute_key_is_preserved(self):
        self.invoke(self.key)
        argv = self.launch.call_args.args[2]
        self.assertEqual(argv[argv.index('--api-key-file') + 1], str(self.key))

    def test_missing_key_does_not_reach_restart(self):
        with self.assertRaises(FileNotFoundError):
            self.invoke('missing.txt')
        self.launch.assert_not_called()
        self.run.assert_not_called()

    def test_unreadable_key_does_not_reach_restart(self):
        with mock.patch.object(Path, 'open', side_effect=PermissionError('unreadable')):
            with self.assertRaises(PermissionError):
                self.invoke(self.key)
        self.launch.assert_not_called()
        self.run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
