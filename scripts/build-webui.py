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
    args = ap.parse_args()
    work = args.build_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if (ROOT / '.git').exists():
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
    shutil.rmtree(work / 'src/routes')
    shutil.copytree(ROOT / 'ui/lightweight', work / 'src/routes')
    # The lightweight entry uses upstream components without the full application's lifecycle.
    config = (work / 'vite.config.ts').read_text()
    config = config.replace('\t\t\tSvelteKitPWA(SVELTEKIT_PWA_OPTIONS),', '')
    (work / 'vite.config.ts').write_text(config)
    for path in (work / 'src/lib/constants').glob('*.ts'):
        text = path.read_text()
        if 'export const STORAGE_APP_NAME =' in text:
            import re
            path.write_text(re.sub(r"export const STORAGE_APP_NAME = .*;", "export const STORAGE_APP_NAME = 'kvmem-chat';", text))
    hook = work / 'src/lib/hooks/use-keyboard-shortcuts.svelte.ts'
    hook.write_text(hook.read_text().replace("page.route.id === '/(chat)'", "(page.route.id as string | null) === '/(chat)'"))
    tsconfig = work / 'tsconfig.json'
    tsconfig.write_text('\n'.join(line for line in tsconfig.read_text().splitlines()
                                  if '"tests/' not in line and '".storybook/' not in line))
    lock_hash = hashlib.sha256((work / 'package-lock.json').read_bytes()).hexdigest()
    stamp = work / '.installed-lock'
    if not (work / 'node_modules').is_dir() or not stamp.is_file() or stamp.read_text() != lock_hash:
        subprocess.run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=work, check=True)
        stamp.write_text(lock_hash)
    subprocess.run(['npx', 'svelte-kit', 'sync'], cwd=work, check=True)
    subprocess.run(['npx', 'svelte-check', '--threshold', 'error'], cwd=work, check=True)
    subprocess.run(['npx', 'vite', 'build'], cwd=work, check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(work / 'dist', output, dirs_exist_ok=True)
    shutil.copy2(ROOT / 'llama.cpp/LICENSE', output / 'llama.cpp-LICENSE')
    manifest = {'llama_commit': pin, 'entry_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT / 'ui/lightweight').iterdir()) if p.is_file()}}
    (output / 'ui-build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output)

if __name__ == '__main__':
    main()
