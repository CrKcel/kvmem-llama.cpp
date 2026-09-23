"""Snapshot the integrated Bonsai sources for an isolated kernel build."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parents[1]
dest = root / 'build-win-bonsai-kernel' / 'source'
pin = '9a9394a895b96003ca842a6041cb28ac49a108f7'
actual = subprocess.check_output(['git', '-C', str(root / 'llama.cpp'), 'rev-parse', 'HEAD'], text=True).strip()
if actual != pin:
    raise SystemExit(f'Unexpected Prism pin: {actual}')
if dest.exists():
    raise SystemExit(f'Snapshot already exists: {dest}. Rebuild it with build.ps1; this script never overwrites snapshots.')
patch = root / 'patches/llama-kvmem-current.patch'
subprocess.run(['git', '-C', str(root / 'llama.cpp'), 'apply', '--reverse', '--check', str(patch)], check=True)
manifest = {'baseline': pin, 'files': {}, 'combined_patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest()}
for repo, prefix in [(root, ''), (root / 'llama.cpp', 'llama.cpp/')]:
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '--cached', '--others', '--exclude-standard', '-z']).decode().split('\0')
    for name in names:
        src = repo / name
        if not name or not src.is_file():
            continue
        rel = prefix + name
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest['files'][rel] = hashlib.sha256(src.read_bytes()).hexdigest()
# Kernel and PDL fixes are already included in the cumulative patch.
with (dest / 'CMakeLists.txt').open('a', encoding='utf-8') as cmake:
    cmake.write('\n# Reuse upstream numerical checks in this isolated kernel experiment.\n'
                'add_executable(test-backend-ops llama.cpp/tests/test-backend-ops.cpp)\n'
                'target_link_libraries(test-backend-ops PRIVATE llama llama-common)\n')
manifest['validation_target'] = 'test-backend-ops'
manifest['files']['CMakeLists.txt'] = hashlib.sha256((dest / 'CMakeLists.txt').read_bytes()).hexdigest()
(dest.parent / 'source-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(f'Prepared {dest}; {len(manifest["files"])} source files')
