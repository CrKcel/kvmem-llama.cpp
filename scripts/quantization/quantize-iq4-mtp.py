#!/usr/bin/env python3
"""Requantize the IQ4_XS model's eight MTP matrices using llama-quantize."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models/unsloth/Qwen3.8-27B-GGUF'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODELS / 'Qwen3.8-27B-UD-IQ4_XS.gguf')
    parser.add_argument('--output', type=Path, default=MODELS / 'Qwen3.8-27B-UD-IQ4_XS-mtp-q4_0.gguf')
    parser.add_argument('--imatrix', type=Path, default=MODELS / 'imatrix_unsloth.gguf')
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/bin/llama-quantize')
    parser.add_argument('--dry-run', action='store_true', help='Print the command without generating a model')
    args = parser.parse_args()
    type_map = Path(__file__).with_name('qwen3.8-27b-iq4-xs-mtp-q4_0.types')
    for path in (args.model, args.imatrix, args.binary, type_map):
        if not path.is_file():
            parser.error(f'File not found: {path}')
    if args.model.resolve() == args.output.resolve():
        parser.error('--output must differ from --model')
    if args.output.exists() and not args.dry_run:
        parser.error(f'Output already exists; select a new --output: {args.output}')
    cmd = [str(args.binary.resolve()), '--allow-requantize', '--max-buffer-size', '256',
           '--imatrix', str(args.imatrix.resolve()), '--tensor-type-file', str(type_map.resolve()),
           str(args.model.resolve()), str(args.output.resolve()), 'IQ4_XS']
    env = os.environ.copy()
    launcher = runpy.run_path(str(ROOT / 'scripts/start-server.py'))
    env['LD_LIBRARY_PATH'] = launcher['library_path'](args.binary.resolve(), env)
    if args.dry_run:
        print(json.dumps({'argv': cmd, 'LD_LIBRARY_PATH': env['LD_LIBRARY_PATH']}, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
