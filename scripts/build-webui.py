#!/usr/bin/env python3
"""Build a lightweight UI from the pinned llama.cpp sources and local entry points."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir', type=Path, default=ROOT / 'build-ui')
    ap.add_argument('--output', type=Path, default=ROOT / 'build/share/kvmem/ui')
    ap.add_argument('--full-ui', action='store_true',
                    help='build the complete upstream UI instead of the lightweight entry points')
    # 中文：--full-ui 构建完整上游 UI（含 PWA/manifest）；默认仅构建轻量级入口页面
    args = ap.parse_args()
    work = args.build_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if args.full_ui:
        # the workdir may have been populated by a previous lightweight build;
        # rebuild it from scratch so every file matches upstream
        # 中文：full-ui 需从头重建工作目录，确保产物与上游完全一致（避免残留轻量构建的文件）
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
    submodule_test = subprocess.run(['git', 'rev-parse', '--verify', 'HEAD:llama.cpp'], cwd=ROOT, capture_output=True)
    if (ROOT / '.git').exists() and submodule_test.returncode == 0:
        pin = subprocess.check_output(['git', 'rev-parse', 'HEAD:llama.cpp'], cwd=ROOT, text=True).strip()
        data = subprocess.check_output(['git', 'archive', pin, 'tools/ui'], cwd=ROOT / 'llama.cpp')
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    name = Path(member.name).relative_to('tools/ui')
                    dest = work / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.extractfile(member).read())
    else:
        pin = 'bundled-source'
        shutil.copytree(ROOT / 'llama.cpp/tools/ui', work, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('node_modules', 'dist', '.svelte-kit', '.git'))
    if not args.full_ui:
        shutil.rmtree(work / 'src/routes')
        shutil.copytree(ROOT / 'ui/lightweight', work / 'src/routes')
        # The lightweight entry uses upstream components without the full application's lifecycle.
        config = (work / 'vite.config.ts').read_text(encoding='utf-8')
        config = config.replace('\t\t\tSvelteKitPWA(SVELTEKIT_PWA_OPTIONS),', '')
        (work / 'vite.config.ts').write_text(config, encoding='utf-8')
        for path in (work / 'src/lib/constants').glob('*.ts'):
            text = path.read_text(encoding='utf-8')
            if 'export const STORAGE_APP_NAME =' in text:
                import re
                path.write_text(re.sub(r"export const STORAGE_APP_NAME = .*;", "export const STORAGE_APP_NAME = 'kvmem-chat';", text), encoding='utf-8')
        hook = work / 'src/lib/hooks/use-keyboard-shortcuts.svelte.ts'
        hook.write_text(hook.read_text(encoding='utf-8').replace("page.route.id === '/(chat)'", "(page.route.id as string | null) === '/(chat)'"), encoding='utf-8')
        tsconfig = work / 'tsconfig.json'
        tsconfig.write_text('\n'.join(line for line in tsconfig.read_text(encoding='utf-8').splitlines()
                                      if '"tests/' not in line and '".storybook/' not in line), encoding='utf-8')
    lock_hash = hashlib.sha256((work / 'package-lock.json').read_bytes()).hexdigest()
    stamp = work / '.installed-lock'
    if not (work / 'node_modules').is_dir() or not stamp.is_file() or stamp.read_text() != lock_hash:
        # npm.cmd / npx.cmd resolves the batch wrapper on Windows (plain 'npm' only works in a shell/POSIX).
        # 中文：Windows 上必须用 npm.cmd/npx.cmd 才能解析批处理包装器（裸 'npm' 仅适用于 POSIX shell）
        subprocess.run(['npm.cmd', 'ci', '--no-audit', '--no-fund'], cwd=work, check=True)
        stamp.write_text(lock_hash)
    subprocess.run(['npx.cmd', 'svelte-kit', 'sync'], cwd=work, check=True)
    subprocess.run(['npx.cmd', 'svelte-check', '--threshold', 'error'], cwd=work, check=True)
    subprocess.run(['npx.cmd', 'vite', 'build'], cwd=work, check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work / 'dist', output, dirs_exist_ok=True)
    shutil.copy2(ROOT / 'llama.cpp/LICENSE', output / 'llama.cpp-LICENSE')
    if args.full_ui:
        # Full UI: hash upstream src/lib entry files so the manifest records whether the page is genuinely upstream-built.
        # 中文：full-ui 模式下对上游 src/lib 的 TS 入口文件做哈希，写入 manifest 以证明产物非轻量构建
        entry_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'llama.cpp/tools/ui/src/lib').rglob('*.ts')) if p.is_file()}
        mode = 'full-ui'
    else:
        entry_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((ROOT / 'ui/lightweight').iterdir()) if p.is_file()}
        mode = 'lightweight'
    manifest = {'llama_commit': pin, 'mode': mode, 'entry_files': entry_files}
    (output / 'ui-build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)

if __name__ == '__main__':
    main()
