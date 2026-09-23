"""Append the pinned ProCreations BF16 MTP head to Bonsai PTQ1 without changing its tensors."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'llama.cpp/gguf-py'))
import gguf

BASE_SHA = '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'
HEAD_SHA = '7a4a18b2d02116ef184d1b0ee4af46d829825ff2c042f79cf37ef8a03c399218'
REVISION = 'efffdea64c1f9e93cc7fa6bb24f72ae9d66ecf51'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--head', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    partial = a.output.with_suffix('.gguf.part')
    if a.output.exists() or partial.exists():
        p.error('Output or partial output already exists; choose another output path')
    for path, expected in ((a.base, BASE_SHA), (a.head, HEAD_SHA)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected SHA256: {path}')
    with a.head.open('rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        if header_size > 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        tensors = json.loads(f.read(header_size))
    tensors.pop('__metadata__', None)
    if len(tensors) != 15:
        raise ValueError('Expected exactly 15 MTP tensors')
    base = gguf.GGUFReader(a.base)
    if len(base.tensors) != 851:
        raise ValueError('Expected 851 base tensors')
    writer = gguf.GGUFWriter(str(partial), 'qwen35')
    for key, field in base.fields.items():
        if key.startswith('GGUF.') or key in ('general.architecture', 'general.name', 'qwen35.block_count'):
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             field.types[1] if len(field.types) > 1 else None)
    writer.add_uint32('qwen35.block_count', 65)
    writer.add_uint32('qwen35.nextn_predict_layers', 1)
    writer.add_string('general.name', 'Ternary Bonsai 2 27B PTQ1 with ProCreations r3 Q8 MTP')
    for tensor in base.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    names = {'fc': 'nextn.eh_proj', 'pre_fc_norm_embedding': 'nextn.enorm',
             'pre_fc_norm_hidden': 'nextn.hnorm', 'norm': 'nextn.shared_head_norm'}
    mapping = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 65)
    exported = []
    for name, tensor in tensors.items():
        if tensor['dtype'] != 'BF16' or not name.startswith('mtp.'):
            raise ValueError(f'Unexpected tensor: {name}')
        shape = tuple(tensor['shape'])
        start, end = tensor['data_offsets']
        if len(shape) not in (1, 2) or end - start != np.prod(shape) * 2 or end + 8 + header_size > a.head.stat().st_size:
            raise ValueError(f'Invalid tensor bounds: {name}')
        raw = np.memmap(a.head, mode='r', dtype='<u2', offset=8 + header_size + start, shape=shape)
        values = (raw.astype(np.uint32) << 16).view(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite tensor: {name}')
        if name.startswith('mtp.layers.0.'):
            target = mapping.get_name('model.layers.64.' + name.removeprefix('mtp.layers.0.'), try_suffixes=('.weight',))
        else:
            target = 'blk.64.' + names[name.removeprefix('mtp.').removesuffix('.weight')] + '.weight'
        if not target:
            raise ValueError(f'No GGUF mapping: {name}')
        if len(shape) == 1:
            # Qwen's BF16 norms are zero centered; GGUF stores effective multipliers.
            values += 1.0
            writer.add_tensor(target, values)
        else:
            writer.add_tensor(target, gguf.quantize(values, gguf.GGMLQuantizationType.Q8_0),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        exported.append({'source': name, 'target': target, 'shape': shape,
                         'type': 'F32' if len(shape) == 1 else 'Q8_0'})
        print(f'Converted {name} -> {target}', flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    result = gguf.GGUFReader(partial)
    lookup = {t.name: t for t in result.tensors}
    if len(lookup) != 866:
        raise ValueError('Unexpected output tensor count')
    for tensor in base.tensors:
        other = lookup[tensor.name]
        if tensor.tensor_type != other.tensor_type or not np.array_equal(tensor.shape, other.shape) or not np.array_equal(tensor.data.view(np.uint8), other.data.view(np.uint8)):
            raise ValueError(f'Base tensor changed: {tensor.name}')
    # Release the Windows file mapping before renaming the verified output.
    del lookup, other, result
    partial.rename(a.output)
    report = {'base': str(a.base), 'base_sha256': BASE_SHA, 'head': str(a.head),
              'head_sha256': HEAD_SHA, 'head_repository': 'ProCreations/Ternary-Bonsai-2-27B-MTP',
              'head_revision': REVISION, 'output': str(a.output), 'output_sha256': sha(a.output),
              'output_bytes': a.output.stat().st_size, 'unchanged_base_tensors': 851, 'mtp_tensors': exported}
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
