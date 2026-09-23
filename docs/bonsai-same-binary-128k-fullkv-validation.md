# Same-binary full-KV control on RTX 5060 Ti

Tested September 23, 2026. Same server binary, original PTQ1 model, GPU,
130103-token archive, warmups, questions, context 131072, K/V Q8_0, MTP off,
thinking off, batch/ubatch 128 and sampling settings. Command comparison confirms
the sole server argument change is `--kvmem` to `--no-kvmem`. Budget/reserve
arguments remain present but do not limit the native full KV when KVMem is off.
Startup confirms KVMem disabled, and no block selection occurred.

| Measurement | KVMem 24K + 10K | Same binary, full 128K KV |
|---|---:|---:|
| Recall | 1/3 | 3/3 |
| Prefill | 372.20 tok/s | 319.99 tok/s |
| Initial request wall time | 350.24 s | 407.23 s |
| Decode, 256 tokens | 37.50 tok/s | 20.10 tok/s |
| Sampled device VRAM peak | 7250 MiB | 10652 MiB |

No CUDA OOM or server crash. Recall response:

```text
- Lunar observatory: ORCHID-5831
- Coral laboratory: MAPLE-2964
- Desert telescope: CEDAR-8172
```

Disabling KVMem alone restores 3/3 recall with the identical binary, model, GPU and Q8/Q8 types. This isolates the observed failure to behavior enabled by KVMem (including sparse prefill, eviction/retrieval, masks and state handling), rather than requiring MTP, V Q4 or a different server/kernel binary. It does not yet identify the failing stage or prove that every shared kernel is correct in every sparse-input case.

The previous selected set included all three code blocks, so missing the block
IDs alone does not explain the failure. See the [Q8/Q8 sparse-KV control](bonsai-kvmem-128k-q8kv-5060-validation.md).
These are single runs of a synthetic archive. Decode prompts have the same user
instruction but include different preceding recall answers; rates are descriptive,
not a strict identical-token performance benchmark.

## Reproduce

```powershell
python scripts/bonsai-smoke.py --cache-type-k q8_0 --cache-type-v q8_0 --long --records 7650 --plain-only --decode-tokens 256 --request-timeout 7200 --context 131072 --budget 24576 --reserve 10240 --build build-win-bonsai-release --out logs/bonsai-128k-fullkv-5060-rerun --model "$env:LOCALAPPDATA/KVMem/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf" --gpu GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c
```

`--plain-only` runs the same workload at the requested context with KVMem off.
Raw results: `logs/bonsai-128k-no-kvmem-kq8-vq8-5060ti/`.
Binary SHA256: `047f88ed12c9b84a4e59c6972ca00d2f01f7f59c1be1a560a3286a9807bf6a95`.
Model SHA256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`.
