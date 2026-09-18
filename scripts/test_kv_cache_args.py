"""Model-free regression checks against real server/CLI argument parsing.

Usage: python3 scripts/test_kv_cache_args.py build/bin/llama-kvmem-server build/bin/llama-kvmem-cli
Valid configurations must reach the missing-model check; invalid combinations
must fail before backend initialization/model loading. No model or GPU required.
"""
import pathlib
import subprocess
import sys
import tempfile


def check(binary, args, expected, missing):
    result = subprocess.run([str(binary), '-m', str(missing), *args],
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode != 0, (args, output)
    assert expected in output, (str(binary), args, expected, output)
    if expected != 'failed to load model':
        assert 'failed to load model' not in output, (args, output)
        assert 'ggml_cuda_init:' not in output, (args, output)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    count = 0
    with tempfile.TemporaryDirectory(prefix='kvmem-kv-args-') as directory:
        missing = pathlib.Path(directory) / 'missing.gguf'
        for argument in sys.argv[1:]:
            binary = pathlib.Path(argument).resolve(strict=True)
            for kind in ['f16', 'f32', 'q8_0', 'q5_0', 'q4_0']:
                for args in [['--kv-dtype', kind], ['-ctk', kind, '-ctv', kind],
                             ['--cache-type-k', kind, '--cache-type-v', kind]]:
                    check(binary, args, 'failed to load model', missing)
                    count += 1
            cases = [
                (['-ctk', 'q5_0'], 'incompatible KV cache types: K=q5_0, V=q8_0'),
                (['-ctv', 'f16'], 'incompatible KV cache types: K=q8_0, V=f16'),
                (['--kv-dtype', 'q5_0', '-ctv', 'q8_0'], 'incompatible KV cache types'),
                (['-ctk', 'q5_0', '--kv-dtype', 'q8_0'], 'failed to load model'),
                (['--kv-dtype', 'q5_0', '-ctk', 'q8_0', '-ctv', 'q8_0'], 'failed to load model'),
                (['-ctk', 'f16', '-ctv', 'f32'], 'failed to load model'),
            ]
            for flag in ['--kv-dtype', '-ctk', '-ctv', '--cache-type-k', '--cache-type-v']:
                cases.append(([flag, 'not-a-cache-type'], 'unsupported cache type'))
            for args, expected in cases:
                check(binary, args, expected, missing)
                count += 1
    print(f'PASS: {count} KV argument cases')


if __name__ == '__main__':
    main()
